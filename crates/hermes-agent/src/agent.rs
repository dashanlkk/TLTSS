use futures::StreamExt;
use hermes_cfg::message::Message;
use hermes_cfg::platform::SessionSource;
use hermes_cfg::traits::{LlmClient, StreamEvent, ToolContext};
use hermes_skill::manifest::SkillManifest;
use hermes_tools::ToolRegistry;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::memory::MemoryStore;
use crate::prompt::PromptBuilder;
use crate::session::Session;
use crate::trace::{TraceCollector, TraceEvent};
use crate::compressor::{self, CompressionConfig};
use crate::retry::{self, RecoveryStrategy};

/// Agent 配置
pub struct AgentConfig {
    pub system_prompt: String,
    pub max_tool_rounds: u32,
    /// Maximum total API call iterations per conversation (default: 90)
    pub max_iterations: u32,
    pub max_context_tokens: usize,
    /// Maximum LLM retries per call (default: 3)
    pub max_retries: u32,
    pub data_dir: Option<PathBuf>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            system_prompt: "You are Hermes, an intelligent AI assistant. You can use tools to help users.".to_string(),
            max_tool_rounds: 10,
            max_iterations: 90,
            max_context_tokens: 128_000,
            max_retries: 3,
            data_dir: None,
        }
    }
}

/// 核心 Agent
pub struct Agent {
    config: AgentConfig,
    llm: Arc<dyn LlmClient>,
    registry: Arc<ToolRegistry>,
    memory: Arc<MemoryStore>,
    skills: Vec<SkillManifest>,
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
            skills: Vec::new(),
            sessions: RwLock::new(Vec::new()),
            trace: Arc::new(TraceCollector::new()),
        }
    }

    /// 设置技能列表
    pub fn with_skills(mut self, skills: Vec<SkillManifest>) -> Self {
        self.skills = skills;
        self
    }

    /// 从持久化目录加载 sessions
    pub async fn load_sessions(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(dir) = &self.config.data_dir {
            let session_dir = dir.join("sessions");
            let loaded = Session::load_all_from_dir(&session_dir).await?;
            if !loaded.is_empty() {
                *self.sessions.write().await = loaded;
                info!("Loaded {} sessions from {}", self.sessions.read().await.len(), session_dir.display());
            }
        }
        Ok(())
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

        self.trace
            .event(
                &trace_id,
                TraceEvent::UserMessage(user_message.to_string()),
            )
            .await;

        // 检查技能触发
        if let Some(skill_result) = self.try_skill_trigger(user_message, &source).await {
            self.trace.event(&trace_id, TraceEvent::Done).await;
            let response = Message::new_assistant(&skill_result);
            session.push_message(response.clone());
            self.persist_session(session).await;
            return Ok(response);
        }

        // 检索相关记忆
        let memories = self.memory.search(user_message).await;

        // 构建 prompt
        let builder = PromptBuilder::new(&self.config.system_prompt).with_memories(memories);
        let messages = builder.build(&session.messages);
        self.trace
            .event(&trace_id, TraceEvent::PromptBuilt(messages.len()))
            .await;

        // Agent loop with retry and compression
        let mut current_messages = messages;
        let mut rounds = 0;
        let mut iterations = 0;
        let compression_config = CompressionConfig::default();

        loop {
            if rounds >= self.config.max_tool_rounds {
                warn!("Max tool rounds reached");
                break;
            }
            if iterations >= self.config.max_iterations {
                warn!("Max iterations ({}) reached", self.config.max_iterations);
                break;
            }

            // Pre-call compression check
            if compressor::should_compress(&current_messages, self.config.max_context_tokens, &compression_config) {
                info!("Context compression triggered");
                let result = compressor::compress(&current_messages, self.config.max_context_tokens, &compression_config);
                current_messages = result.messages;
            }

            // 调用 LLM with retry
            self.trace.event(&trace_id, TraceEvent::LlmCallStart).await;
            let tools = self.registry.tool_definitions().await;
            iterations += 1;

            let response = {
                let llm = &self.llm;
                let msgs = &current_messages;
                let tool_defs = &tools;
                let max_retries = self.config.max_retries;

                retry::retry_llm_call(max_retries, || {
                    let msgs = msgs.clone();
                    let tool_defs = tool_defs.clone();
                    async move { llm.complete(&msgs, &tool_defs).await }
                }).await
            };

            let response = match response {
                Ok(r) => r,
                Err(e) => {
                    let strategy = retry::classify_error(&e);
                    match strategy {
                        RecoveryStrategy::Compress => {
                            // Force compress and retry once more
                            info!("Context overflow, forcing compression");
                            let compressed = compressor::compress(
                                &current_messages,
                                self.config.max_context_tokens / 2,
                                &compression_config,
                            );
                            current_messages = compressed.messages;
                            continue;
                        }
                        _ => return Err(e.into()),
                    }
                }
            };

            self.trace
                .event(
                    &trace_id,
                    TraceEvent::LlmCallComplete(response.content.clone()),
                )
                .await;

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

                // 使用 JoinSet 并行执行
                let mut join_set = tokio::task::JoinSet::new();
                for call in calls {
                    let registry = self.registry.clone();
                    let ctx = ctx.clone();
                    let tool_name = call.function.name.clone();
                    join_set.spawn(async move {
                        let result = registry
                            .execute(&tool_name, &call.function.arguments, &ctx)
                            .await;
                        (call.id.clone(), tool_name, result)
                    });
                }

                while let Some(res) = join_set.join_next().await {
                    match res {
                        Ok((call_id, tool_name, result)) => match result {
                            Ok(tool_result) => {
                                self.trace
                                    .event(
                                        &trace_id,
                                        TraceEvent::ToolExecuted {
                                            tool: tool_name,
                                            success: true,
                                        },
                                    )
                                    .await;
                                let msg = Message::new_tool_result(&call_id, &tool_result.content);
                                current_messages.push(msg.clone());
                                session.push_message(msg);
                            }
                            Err(e) => {
                                self.trace
                                    .event(
                                        &trace_id,
                                        TraceEvent::ToolExecuted {
                                            tool: tool_name,
                                            success: false,
                                        },
                                    )
                                    .await;
                                let msg = Message::new_tool_result(&call_id, e.to_string());
                                current_messages.push(msg.clone());
                                session.push_message(msg);
                            }
                        },
                        Err(e) => {
                            warn!("Tool task failed: {}", e);
                        }
                    }
                }

                rounds += 1;
            } else {
                // 没有工具调用，结束循环
                break;
            }
        }

        // 自动保存记忆
        if !user_message.is_empty() {
            let _ = self
                .memory
                .add(user_message, vec![user_message.to_string()])
                .await;
        }

        self.trace.event(&trace_id, TraceEvent::Done).await;
        let final_msg = session
            .messages
            .last()
            .cloned()
            .unwrap_or_else(|| Message::new_assistant("No response"));

        self.persist_session(session).await;
        Ok(final_msg)
    }

    /// 流式对话 — 逐 token 输出 Agent 回复（支持 tool_call）
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
        self.trace
            .event(
                &trace_id,
                TraceEvent::UserMessage(user_message.to_string()),
            )
            .await;

        // 检查技能触发
        if let Some(skill_result) = self.try_skill_trigger(user_message, &source).await {
            self.trace.event(&trace_id, TraceEvent::Done).await;
            let response = Message::new_assistant(&skill_result);
            session.push_message(response.clone());
            self.persist_session(session).await;
            return Ok(response);
        }

        let memories = self.memory.search(user_message).await;
        let builder = PromptBuilder::new(&self.config.system_prompt).with_memories(memories);
        let messages = builder.build(&session.messages);
        self.trace
            .event(&trace_id, TraceEvent::PromptBuilt(messages.len()))
            .await;

        // 流式调用 LLM
        let tools = self.registry.tool_definitions().await;

        match self.llm.complete_stream(&messages, &tools).await {
            Ok(mut stream) => {
                let mut full_content = String::new();
                let mut full_reasoning = String::new();
                let mut pending_tool_calls: Vec<hermes_cfg::tool::ToolCall> = Vec::new();

                while let Some(event) = stream.next().await {
                    match event {
                        Ok(StreamEvent::Delta(token)) => {
                            on_token(&token);
                            full_content.push_str(&token);
                        }
                        Ok(StreamEvent::Reasoning(token)) => {
                            full_reasoning.push_str(&token);
                        }
                        Ok(StreamEvent::Done) => break,
                        Ok(StreamEvent::ToolCall {
                            id,
                            name,
                            arguments,
                        }) => {
                            debug!("Stream tool call: {} ({})", name, id);
                            pending_tool_calls.push(hermes_cfg::tool::ToolCall {
                                id,
                                call_type: "function".to_string(),
                                function: hermes_cfg::tool::FunctionCall {
                                    name,
                                    arguments,
                                },
                            });
                        }
                        Err(e) => {
                            warn!("Stream error: {}", e);
                            break;
                        }
                    }
                }

                let mut response = Message::new_assistant(&full_content);

                // 处理流式中的 tool calls
                if !pending_tool_calls.is_empty() {
                    response = response.with_tool_calls(pending_tool_calls.clone());

                    // 执行工具并收集结果
                    let ctx = ToolContext::new(&session.id, source.clone());
                    for call in &pending_tool_calls {
                        match self
                            .registry
                            .execute(&call.function.name, &call.function.arguments, &ctx)
                            .await
                        {
                            Ok(tool_result) => {
                                let msg =
                                    Message::new_tool_result(&call.id, &tool_result.content);
                                session.push_message(msg);
                                self.trace
                                    .event(
                                        &trace_id,
                                        TraceEvent::ToolExecuted {
                                            tool: call.function.name.clone(),
                                            success: true,
                                        },
                                    )
                                    .await;
                            }
                            Err(e) => {
                                let msg = Message::new_tool_result(&call.id, e.to_string());
                                session.push_message(msg);
                                self.trace
                                    .event(
                                        &trace_id,
                                        TraceEvent::ToolExecuted {
                                            tool: call.function.name.clone(),
                                            success: false,
                                        },
                                    )
                                    .await;
                            }
                        }
                    }
                }

                session.push_message(response.clone());

                if !user_message.is_empty() {
                    let _ = self
                        .memory
                        .add(user_message, vec![user_message.to_string()])
                        .await;
                }

                self.trace.event(&trace_id, TraceEvent::Done).await;
                self.persist_session(session).await;
                Ok(response)
            }
            Err(_) => {
                // 流式失败时回退到普通调用
                drop(sessions);
                self.chat(user_message, source).await
            }
        }
    }

    /// 尝试技能触发，返回匹配技能的执行结果
    async fn try_skill_trigger(
        &self,
        user_message: &str,
        source: &SessionSource,
    ) -> Option<String> {
        let matched_skill = self.skills.iter().find(|s| s.matches(user_message))?;

        info!("Skill triggered: {}", matched_skill.name);
        let executor = hermes_skill::executor::SkillExecutor::new(self.registry.clone());
        let ctx = ToolContext::new("skill-trigger", source.clone());

        match executor.execute(matched_skill, &ctx).await {
            Ok(results) => {
                let output: Vec<String> = results
                    .iter()
                    .map(|r| {
                        if r.is_error {
                            format!("[ERROR] {}", r.content)
                        } else {
                            r.content.clone()
                        }
                    })
                    .collect();
                Some(format!(
                    "Skill '{}' executed:\n{}",
                    matched_skill.name,
                    output.join("\n")
                ))
            }
            Err(e) => {
                warn!("Skill '{}' execution failed: {}", matched_skill.name, e);
                None
            }
        }
    }

    /// 持久化 session 到文件
    async fn persist_session(&self, session: &Session) {
        if let Some(dir) = &self.config.data_dir {
            let session_dir = dir.join("sessions");
            if let Err(e) = session.save_to_file(&session_dir).await {
                warn!("Failed to persist session: {}", e);
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

    /// Clone for gateway use — creates a new Agent sharing the same
    /// LLM, registry, and memory, but with a fresh session state.
    /// Used by GatewayRunner to create per-session agents that share
    /// the same underlying resources.
    pub fn clone_for_gateway(&self) -> Self {
        Self {
            config: AgentConfig {
                system_prompt: self.config.system_prompt.clone(),
                ..AgentConfig::default()
            },
            llm: self.llm.clone(),
            registry: self.registry.clone(),
            memory: self.memory.clone(),
            skills: self.skills.clone(),
            sessions: RwLock::new(Vec::new()),
            trace: Arc::new(TraceCollector::new()),
        }
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

        let agent = Agent::new(AgentConfig::default(), llm, registry, memory);

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

        agent
            .chat("Remember my name is Alice", SessionSource::cli())
            .await
            .unwrap();

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

        let agent = Agent::new(AgentConfig::default(), llm, registry, memory);

        // 直接验证 trace collector 可用
        let collector = agent.trace_collector();
        collector.start("test-trace").await;
        collector
            .event("test-trace", TraceEvent::UserMessage("test".into()))
            .await;
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
        let response = agent
            .chat_stream("stream me", SessionSource::cli(), |t| {
                tokens.push(t.to_string());
            })
            .await
            .unwrap();

        let full: String = tokens.join("");
        assert_eq!(full, "Stream test");
        assert_eq!(response.content, "Stream test");
    }

    #[tokio::test]
    async fn test_agent_skill_trigger() {
        let llm = Arc::new(hermes_llm::FakeClient::new("Reply"));
        let registry = Arc::new(ToolRegistry::new());
        let memory = Arc::new(MemoryStore::new());

        // 注册一个 dummy tool 以便 skill 可以执行
        struct EchoTool;
        #[async_trait::async_trait]
        impl hermes_cfg::traits::ToolHandler for EchoTool {
            fn name(&self) -> &str { "echo_action" }
            fn description(&self) -> &str { "Echo" }
            fn parameters_schema(&self) -> serde_json::Value {
                serde_json::json!({"type": "object"})
            }
            async fn execute(
                &self,
                _args: &str,
                _ctx: &hermes_cfg::traits::ToolContext,
            ) -> Result<hermes_cfg::tool::ToolResult, hermes_cfg::error::ToolError> {
                Ok(hermes_cfg::tool::ToolResult::success("echo_action", "echo ok"))
            }
        }
        registry.register(Arc::new(EchoTool)).await;

        let skill_yaml = r#"
name: test_echo
version: "1.0"
description: Test echo skill
trigger_patterns:
  - "echo"
steps:
  - action: echo_action
    params: {}
"#;
        let skill = hermes_skill::manifest::SkillManifest::from_yaml(skill_yaml).unwrap();

        let agent = Agent::new(
            AgentConfig::default(),
            llm,
            registry,
            memory,
        )
        .with_skills(vec![skill]);

        let response = agent.chat("please echo something", SessionSource::cli()).await.unwrap();
        assert!(response.content.contains("test_echo"));
        assert!(response.content.contains("echo ok"));
    }
}
