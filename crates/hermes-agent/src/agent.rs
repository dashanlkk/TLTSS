use hermes_cfg::message::Message;
use hermes_cfg::platform::SessionSource;
use hermes_cfg::prelude::*;
use hermes_cfg::traits::{LlmClient, ToolContext};
use hermes_tools::ToolRegistry;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

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
        self.trace.start(&trace_id);

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

        self.trace.event(&trace_id, TraceEvent::UserMessage(user_message.to_string()));

        // 检索相关记忆
        let memories = self.memory.search(user_message).await;

        // 构建 prompt
        let builder = PromptBuilder::new(&self.config.system_prompt)
            .with_memories(memories);
        let messages = builder.build(&session.messages);
        self.trace.event(&trace_id, TraceEvent::PromptBuilt(messages.len()));

        // Agent loop
        let mut current_messages = messages;
        let mut rounds = 0;

        loop {
            if rounds >= self.config.max_tool_rounds {
                warn!("Max tool rounds reached");
                break;
            }

            // 调用 LLM
            self.trace.event(&trace_id, TraceEvent::LlmCallStart);
            let tools = self.registry.tool_definitions().await;
            let response = self.llm.complete(&current_messages, &tools).await?;
            self.trace.event(&trace_id, TraceEvent::LlmCallComplete(response.content.clone()));

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
                    join_set.spawn(async move {
                        let result = registry.execute(&call.function.name, &call.function.arguments, &ctx).await;
                        (call.id.clone(), result)
                    });
                }

                while let Some(res) = join_set.join_next().await {
                    match res {
                        Ok((call_id, result)) => {
                            match result {
                                Ok(tool_result) => {
                                    self.trace.event(&trace_id, TraceEvent::ToolExecuted {
                                        tool: "unknown".into(),
                                        success: true,
                                    });
                                    let msg = Message::new_tool_result(&call_id, &tool_result.content);
                                    results.push(msg);
                                }
                                Err(e) => {
                                    self.trace.event(&trace_id, TraceEvent::ToolExecuted {
                                        tool: "unknown".into(),
                                        success: false,
                                    });
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

        self.trace.event(&trace_id, TraceEvent::Done);
        let final_msg = session.messages.last().cloned().unwrap_or_else(|| Message::new_assistant("No response"));
        Ok(final_msg)
    }

    /// 流式对话
    pub async fn chat_stream(
        &self,
        _user_message: &str,
        _source: SessionSource,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // TODO: 实现流式 Agent loop
        unimplemented!("Stream chat not yet implemented")
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
