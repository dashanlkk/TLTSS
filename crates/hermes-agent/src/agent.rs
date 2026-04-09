use futures::StreamExt;
use hermes_cfg::message::Message;
use hermes_cfg::platform::SessionSource;
use hermes_cfg::traits::{LlmClient, StreamEvent, ToolContext};
use hermes_tools::ToolRegistry;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::memory::MemoryStore;
use crate::prompt::PromptBuilder;
use crate::session::Session;
use crate::trace::{TraceCollector, TraceEvent};

/// Agent 配置
pub struct AgentConfig {
    pub system_prompt: String,
    pub max_tool_rounds: u32,
    pub max_context_tokens: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            system_prompt: "You are Hermes, an intelligent AI assistant. You can use tools to help users.".to_string(),
            max_tool_rounds: 10,
            max_context_tokens: 8000,
        }
    }
}

/// 核心 Agent
pub struct Agent {
    config: AgentConfig,
    llm: Arc<dyn LlmClient>,
    registry: Arc<ToolRegistry>,
    memory: Arc<MemoryStore>,
    sessions: RwLock<Vec<Session>>,
    trace: Arc<TraceCollector>,
}

impl Agent {
    pub fn new(
        config: AgentConfig,
        llm: Arc<dyn LlmClient>,
        registry: Arc<ToolRegistry>,
        memory: Arc<MemoryStore>,
    ) -> Self {
        Self {
            config,
            llm,
            registry,
            memory,
            sessions: RwLock::new(Vec::new()),
            trace: Arc::new(TraceCollector::new()),
        }
    }

    /// 处理用户消息，返回 Agent 回复
    pub async fn chat(
        &self,
        user_message: &str,
        source: SessionSource,
    ) -> Result<Message, Box<dyn std::error::Error + Send + Sync>> {
        let trace_id = uuid::Uuid::new_v4().to_string();
        self.trace.start(&trace_id).await;

        // 获取或创建会话
        let mut sessions = self.sessions.write().await;
        let session = if let Some(s) = sessions.iter_mut().find(|s| s.source == source) {
            s
        } else {
            sessions.push(Session::new(source.clone()));
            sessions.last_mut().unwrap()
        };

        // 添加用户消息
        let user_msg = Message::new_user(user_message);
        session.push_message(user_msg.clone());

        self.trace.event(&trace_id, TraceEvent::UserMessage(user_message.to_string())).await;

        // 检索相关记忆
        let memories = self.memory.search(user_message).await;

        // 构建 prompt
        let builder = PromptBuilder::new(&self.config.system_prompt)
            .with_memories(memories);
        let messages = builder.build(&session.messages);
        self.trace.event(&trace_id, TraceEvent::PromptBuilt(messages.len())).await;

        // Agent loop
        let mut current_messages = messages;
        let mut rounds = 0;

        loop {
            if rounds >= self.config.max_tool_rounds {
                warn!("Max tool rounds reached");
                break;
            }

            // 调用 LLM
            self.trace.event(&trace_id, TraceEvent::LlmCallStart).await;
            let tools = self.registry.tool_definitions().await;
            let response = self.llm.complete(&current_messages, &tools).await?;
            self.trace.event(&trace_id, TraceEvent::LlmCallComplete(response.content.clone())).await;

            // 检查是否有工具调用
            let tool_calls = response.tool_calls.clone();
            session.push_message(response.clone());

            if let Some(calls) = tool_calls {
                if calls.is_empty() {
                    break;
                }

                debug!("Agent wants to call {} tool(s)", calls.len());

                // 并行执行工具
                let ctx = ToolContext::new(&session.id, source.clone());
                let mut results = Vec::new();

                // 使用 JoinSet 并行执行
                let mut join_set = tokio::task::JoinSet::new();
                for call in calls {
                    let registry = self.registry.clone();
                    let ctx = ctx.clone();
                    let tool_name = call.function.name.clone();
                    join_set.spawn(async move {
                        let result = registry.execute(&tool_name, &call.function.arguments, &ctx).await;
                        (call.id.clone(), tool_name, result)
                    });
                }

                while let Some(res) = join_set.join_next().await {
                    match res {
                        Ok((call_id, tool_name, result)) => {
                            match result {
                                Ok(tool_result) => {
                                    self.trace.event(&trace_id, TraceEvent::ToolExecuted {
                                        tool: tool_name,
                                        success: true,
                                    }).await;
                                    let msg = Message::new_tool_result(&call_id, &tool_result.content);
                                    results.push(msg);
                                }
                                Err(e) => {
                                    self.trace.event(&trace_id, TraceEvent::ToolExecuted {
                                        tool: tool_name,
                                        success: false,
                                    }).await;
                                    let msg = Message::new_tool_result(&call_id, &e.to_string());
                                    results.push(msg);
                                }
                            }
                        }
                        Err(e) => {
                            warn!("Tool task failed: {}", e);
                        }
                    }
                }

                // 将工具结果添加到消息列表
                for r in &results {
                    session.push_message(r.clone());
                    current_messages.push(r.clone());
                }

                rounds += 1;
            } else {
                // 没有工具调用，结束循环
                break;
            }
        }

        // 自动保存记忆（提取本轮对话关键信息）
        if !user_message.is_empty() {
            let _ = self.memory.add(user_message, vec![user_message.to_string()]).await;
        }

        self.trace.event(&trace_id, TraceEvent::Done).await;
        let final_msg = session.messages.last().cloned().unwrap_or_else(|| Message::new_assistant("No response"));
        Ok(final_msg)
    }

    /// 流式对话 — 逐 token 输出 Agent 回复
    pub async fn chat_stream(
        &self,
        user_message: &str,
        source: SessionSource,
        mut on_token: impl FnMut(&str),
    ) -> Result<Message, Box<dyn std::error::Error + Send + Sync>> {
        let trace_id = uuid::Uuid::new_v4().to_string();
        self.trace.start(&trace_id).await;

        // 获取或创建会话
        let mut sessions = self.sessions.write().await;
        let session = if let Some(s) = sessions.iter_mut().find(|s| s.source == source) {
            s
        } else {
            sessions.push(Session::new(source.clone()));
            sessions.last_mut().unwrap()
        };

        let user_msg = Message::new_user(user_message);
        session.push_message(user_msg.clone());
        self.trace.event(&trace_id, TraceEvent::UserMessage(user_message.to_string())).await;

        let memories = self.memory.search(user_message).await;
        let builder = PromptBuilder::new(&self.config.system_prompt).with_memories(memories);
        let messages = builder.build(&session.messages);
        self.trace.event(&trace_id, TraceEvent::PromptBuilt(messages.len())).await;

        // 流式调用 LLM
        let tools = self.registry.tool_definitions().await;

        match self.llm.complete_stream(&messages, &tools).await {
            Ok(mut stream) => {
                let mut full_content = String::new();
                while let Some(event) = stream.next().await {
                    match event {
                        Ok(StreamEvent::Delta(token)) => {
                            on_token(&token);
                            full_content.push_str(&token);
                        }
                        Ok(StreamEvent::Done) => break,
                        Ok(StreamEvent::ToolCall { .. }) => {} // 流式中暂不处理
                        Err(e) => {
                            warn!("Stream error: {}", e);
                            break;
                        }
                    }
                }

                let response = Message::new_assistant(&full_content);
                session.push_message(response.clone());

                if !user_message.is_empty() {
                    let _ = self.memory.add(user_message, vec![user_message.to_string()]).await;
                }

                self.trace.event(&trace_id, TraceEvent::Done).await;
                Ok(response)
            }
            Err(_) => {
                // 流式失败时回退到普通调用
                drop(sessions);
                self.chat(user_message, source).await
            }
        }
    }

    /// 获取所有会话
    pub async fn list_sessions(&self) -> Vec<Session> {
        self.sessions.read().await.clone()
    }

    /// 获取 trace 收集器
    pub fn trace_collector(&self) -> Arc<TraceCollector> {
        self.trace.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::TraceEvent;
    use hermes_cfg::platform::SessionSource;

    #[tokio::test]
    async fn test_agent_chat_basic() {
        let llm = Arc::new(hermes_llm::FakeClient::new("Hello! I'm Hermes."));
        let registry = Arc::new(ToolRegistry::new());
        let memory = Arc::new(MemoryStore::new());

        let agent = Agent::new(
            AgentConfig::default(),
            llm,
            registry,
            memory,
        );

        let response = agent.chat("Hi there", SessionSource::cli()).await.unwrap();
        assert!(response.content.contains("Hermes"));
    }

    #[tokio::test]
    async fn test_agent_memory_autowrite() {
        let llm = Arc::new(hermes_llm::FakeClient::new("OK"));
        let registry = Arc::new(ToolRegistry::new());
        let memory = Arc::new(MemoryStore::new());

        let agent = Agent::new(
            AgentConfig::default(),
            llm,
            registry,
            memory.clone(),
        );

        agent.chat("Remember my name is Alice", SessionSource::cli()).await.unwrap();

        // Memory 应该被自动写入
        let memories = memory.list().await;
        assert_eq!(memories.len(), 1);
        assert!(memories[0].content.contains("Alice"));
    }

    #[tokio::test]
    async fn test_agent_trace_collected() {
        let llm = Arc::new(hermes_llm::FakeClient::new("Response"));
        let registry = Arc::new(ToolRegistry::new());
        let memory = Arc::new(MemoryStore::new());

        let agent = Agent::new(
            AgentConfig::default(),
            llm,
            registry,
            memory,
        );

        // 直接验证 trace collector 可用
        let collector = agent.trace_collector();
        collector.start("test-trace").await;
        collector.event("test-trace", TraceEvent::UserMessage("test".into())).await;
        collector.event("test-trace", TraceEvent::Done).await;

        let traces = collector.list().await;
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].events.len(), 2);

        // 再验证 chat 调用后的 trace
        agent.chat("test", SessionSource::cli()).await.unwrap();
        let traces = collector.list().await;
        assert_eq!(traces.len(), 2, "Should have 2 traces after chat");
    }

    #[tokio::test]
    async fn test_agent_session_tracking() {
        let llm = Arc::new(hermes_llm::FakeClient::new("Reply"));
        let registry = Arc::new(ToolRegistry::new());
        let memory = Arc::new(MemoryStore::new());

        let agent = Agent::new(
            AgentConfig::default(),
            llm,
            registry,
            memory,
        );

        agent.chat("msg1", SessionSource::cli()).await.unwrap();
        agent.chat("msg2", SessionSource::cli()).await.unwrap();

        let sessions = agent.list_sessions().await;
        assert_eq!(sessions.len(), 1);
        // 2 user + 2 assistant = 4 messages
        assert_eq!(sessions[0].messages.len(), 4);
    }

    #[tokio::test]
    async fn test_agent_chat_stream() {
        let llm = Arc::new(hermes_llm::FakeClient::new("Stream test"));
        let registry = Arc::new(ToolRegistry::new());
        let memory = Arc::new(MemoryStore::new());

        let agent = Agent::new(
            AgentConfig::default(),
            llm,
            registry,
            memory,
        );

        let mut tokens = Vec::new();
        let response = agent.chat_stream("stream me", SessionSource::cli(), |t| {
            tokens.push(t.to_string());
        }).await.unwrap();

        let full: String = tokens.join("");
        assert_eq!(full, "Stream test");
        assert_eq!(response.content, "Stream test");
    }
}
