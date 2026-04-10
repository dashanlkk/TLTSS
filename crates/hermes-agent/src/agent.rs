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
use hermes_tools::parallel::{should_parallelize, ToolCallInfo, MAX_TOOL_WORKERS};

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
    memory_manager: Option<Arc<crate::memory_manager::MemoryManager>>,
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
            memory_manager: None,
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

    /// 设置 MemoryManager（文件持久化记忆）
    pub fn with_memory_manager(mut self, mgr: Arc<crate::memory_manager::MemoryManager>) -> Self {
        self.memory_manager = Some(mgr);
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

        // Prompt 注入扫描
        if let hermes_security::prompt::ScanResult::Suspicious { matched_pattern } =
            hermes_security::prompt::scan_prompt(user_message)
        {
            warn!("Prompt injection pattern detected: '{}'", matched_pattern);
            let response = Message::new_assistant(
                "I detected a potentially harmful pattern in your input. Please rephrase your request.",
            );
            session.push_message(response.clone());
            self.persist_session(session).await;
            return Ok(response);
        }

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

        // 发现项目上下文文件 (.hermes.md / AGENTS.md / CLAUDE.md / .cursorrules)
        let context_files = crate::context_files::discover_context_files(
            &std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
        );

        // 构建 prompt
        let memory_context = if let Some(mgr) = &self.memory_manager {
            mgr.build_memory_context_block().await
        } else {
            String::new()
        };

        let builder = PromptBuilder::new(&self.config.system_prompt)
            .with_memories(memories)
            .with_context_files(context_files)
            .with_memory_context(memory_context);
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
                warn!("Max tool rounds reached, requesting final text response");
                // Make one final call without tools to get a text summary
                let final_response = {
                    let llm = &self.llm;
                    let msgs = &current_messages;
                    retry::retry_llm_call(self.config.max_retries, || {
                        let msgs = msgs.clone();
                        async move { llm.complete(&msgs, &[]).await }
                    }).await
                };
                if let Ok(resp) = final_response {
                    current_messages.push(resp.clone());
                    session.push_message(resp);
                }
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
            current_messages.push(response.clone());
            session.push_message(response.clone());

            if let Some(calls) = tool_calls {
                if calls.is_empty() {
                    break;
                }

                debug!("Agent wants to call {} tool(s)", calls.len());

                // 并行/顺序执行决策
                let ctx = ToolContext::new(&session.id, source.clone());
                let call_infos: Vec<ToolCallInfo> = calls.iter().map(|c| ToolCallInfo {
                    name: c.function.name.clone(),
                    arguments: c.function.arguments.clone(),
                }).collect();
                let parallel = should_parallelize(&call_infos);

                if parallel {
                    // 安全并行：使用 JoinSet
                    debug!("Parallel execution: {} tools", calls.len());
                    let max_workers = calls.len().min(MAX_TOOL_WORKERS);
                    let mut join_set = tokio::task::JoinSet::new();
                    for call in calls.iter().take(max_workers) {
                        let registry = self.registry.clone();
                        let ctx = ctx.clone();
                        let call_id = call.id.clone();
                        let tool_name = call.function.name.clone();
                        let args = call.function.arguments.clone();
                        join_set.spawn(async move {
                            let result = registry
                                .execute(&tool_name, &args, &ctx)
                                .await;
                            (call_id, tool_name, result)
                        });
                    }

                    while let Some(res) = join_set.join_next().await {
                        match res {
                            Ok((call_id, tool_name, result)) => match result {
                                Ok(tool_result) => {
                                    self.trace.event(&trace_id, TraceEvent::ToolExecuted {
                                        tool: tool_name,
                                        success: true,
                                    }).await;
                                    let msg = Message::new_tool_result(&call_id, &tool_result.content);
                                    current_messages.push(msg.clone());
                                    session.push_message(msg);
                                }
                                Err(e) => {
                                    self.trace.event(&trace_id, TraceEvent::ToolExecuted {
                                        tool: tool_name,
                                        success: false,
                                    }).await;
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
                } else {
                    // 顺序执行
                    debug!("Sequential execution: {} tools", calls.len());
                    for call in &calls {
                        let tool_name = call.function.name.clone();
                        let result = self.registry
                            .execute(&tool_name, &call.function.arguments, &ctx)
                            .await;
                        match result {
                            Ok(tool_result) => {
                                self.trace.event(&trace_id, TraceEvent::ToolExecuted {
                                    tool: tool_name,
                                    success: true,
                                }).await;
                                let msg = Message::new_tool_result(&call.id, &tool_result.content);
                                current_messages.push(msg.clone());
                                session.push_message(msg);
                            }
                            Err(e) => {
                                self.trace.event(&trace_id, TraceEvent::ToolExecuted {
                                    tool: tool_name,
                                    success: false,
                                }).await;
                                let msg = Message::new_tool_result(&call.id, e.to_string());
                                current_messages.push(msg.clone());
                                session.push_message(msg);
                            }
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

    /// 流式对话 — 逐 token 输出 Agent 回复（支持多轮 tool_call 循环）
    ///
    /// 与 `chat` 方法对齐的循环结构：
    /// 1. 流式调用 LLM，收集 content + tool_calls
    /// 2. 若有 tool_calls → 执行工具 → 将结果加入 current_messages → 回到步骤 1
    /// 3. 若无 tool_calls → 返回最终回复
    /// 4. 支持 compression、iteration budget、max_tool_rounds 限制
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

        // Prompt 注入扫描
        if let hermes_security::prompt::ScanResult::Suspicious { matched_pattern } =
            hermes_security::prompt::scan_prompt(user_message)
        {
            warn!("Prompt injection pattern detected: '{}'", matched_pattern);
            let response = Message::new_assistant(
                "I detected a potentially harmful pattern in your input. Please rephrase your request.",
            );
            on_token("I detected a potentially harmful pattern in your input. Please rephrase your request.");
            session.push_message(response.clone());
            self.persist_session(session).await;
            return Ok(response);
        }

        // 检查技能触发
        if let Some(skill_result) = self.try_skill_trigger(user_message, &source).await {
            self.trace.event(&trace_id, TraceEvent::Done).await;
            let response = Message::new_assistant(&skill_result);
            session.push_message(response.clone());
            self.persist_session(session).await;
            return Ok(response);
        }

        let memories = self.memory.search(user_message).await;
        let context_files = crate::context_files::discover_context_files(
            &std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
        );
        let memory_context = if let Some(mgr) = &self.memory_manager {
            mgr.build_memory_context_block().await
        } else {
            String::new()
        };
        let builder = PromptBuilder::new(&self.config.system_prompt)
            .with_memories(memories)
            .with_context_files(context_files)
            .with_memory_context(memory_context);
        let mut current_messages = builder.build(&session.messages);
        self.trace
            .event(&trace_id, TraceEvent::PromptBuilt(current_messages.len()))
            .await;

        let mut rounds = 0u32;
        let mut iterations = 0u32;
        let compression_config = CompressionConfig::default();

        loop {
            // Iteration budget check
            if iterations >= self.config.max_iterations {
                warn!("Stream: max iterations ({}) reached", self.config.max_iterations);
                break;
            }
            if rounds >= self.config.max_tool_rounds {
                warn!("Stream: max tool rounds reached, requesting final text response");
                // Final non-streaming call without tools
                let final_response = {
                    let llm = &self.llm;
                    let msgs = &current_messages;
                    retry::retry_llm_call(self.config.max_retries, || {
                        let msgs = msgs.clone();
                        async move { llm.complete(&msgs, &[]).await }
                    }).await
                };
                if let Ok(resp) = final_response {
                    on_token(&resp.content);
                    current_messages.push(resp.clone());
                    session.push_message(resp);
                }
                break;
            }

            // Pre-call compression check
            if compressor::should_compress(&current_messages, self.config.max_context_tokens, &compression_config) {
                info!("Stream: context compression triggered");
                let result = compressor::compress(&current_messages, self.config.max_context_tokens, &compression_config);
                current_messages = result.messages;
            }

            // 流式调用 LLM
            let tools = self.registry.tool_definitions().await;
            iterations += 1;
            self.trace.event(&trace_id, TraceEvent::LlmCallStart).await;

            let stream_result = {
                let llm = &self.llm;
                let msgs = &current_messages;
                let tool_defs = &tools;
                // Stream retry — uses retry_stream_call which handles LlmError
                retry::retry_stream_call(self.config.max_retries, || {
                    let msgs = msgs.clone();
                    let tool_defs = tool_defs.clone();
                    async move { llm.complete_stream(&msgs, &tool_defs).await }
                }).await
            };

            let mut stream = match stream_result {
                Ok(s) => s,
                Err(e) => {
                    let strategy = retry::classify_error(&e);
                    match strategy {
                        RecoveryStrategy::Compress => {
                            info!("Stream: context overflow, forcing compression");
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

            // Collect stream events
            let mut full_content = String::new();
            let mut pending_tool_calls: Vec<hermes_cfg::tool::ToolCall> = Vec::new();

            while let Some(event) = stream.next().await {
                match event {
                    Ok(StreamEvent::Delta(token)) => {
                        on_token(&token);
                        full_content.push_str(&token);
                    }
                    Ok(StreamEvent::Reasoning(_token)) => {
                        // Reasoning collected but not streamed to user
                    }
                    Ok(StreamEvent::Done) => break,
                    Ok(StreamEvent::ToolCall { id, name, arguments }) => {
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
                    Ok(_) => {} // Unknown StreamEvent variants
                }
            }

            // Build response message
            let mut response = Message::new_assistant(&full_content);
            let tool_calls = if !pending_tool_calls.is_empty() {
                response = response.with_tool_calls(pending_tool_calls);
                response.tool_calls.clone()
            } else {
                None
            };

            current_messages.push(response.clone());
            session.push_message(response.clone());

            self.trace
                .event(&trace_id, TraceEvent::LlmCallComplete(full_content))
                .await;

            // Check tool calls
            if let Some(calls) = tool_calls {
                if calls.is_empty() {
                    break;
                }

                debug!("Stream: agent calls {} tool(s)", calls.len());

                // 并行/顺序执行决策
                let ctx = ToolContext::new(&session.id, source.clone());
                let call_infos: Vec<ToolCallInfo> = calls.iter().map(|c| ToolCallInfo {
                    name: c.function.name.clone(),
                    arguments: c.function.arguments.clone(),
                }).collect();
                let parallel = should_parallelize(&call_infos);

                if parallel {
                    debug!("Stream parallel execution: {} tools", calls.len());
                    let max_workers = calls.len().min(MAX_TOOL_WORKERS);
                    let mut join_set = tokio::task::JoinSet::new();
                    for call in calls.iter().take(max_workers) {
                        let registry = self.registry.clone();
                        let ctx = ctx.clone();
                        let call_id = call.id.clone();
                        let tool_name = call.function.name.clone();
                        let args = call.function.arguments.clone();
                        join_set.spawn(async move {
                            let result = registry
                                .execute(&tool_name, &args, &ctx)
                                .await;
                            (call_id, tool_name, result)
                        });
                    }

                    while let Some(res) = join_set.join_next().await {
                        match res {
                            Ok((call_id, tool_name, result)) => match result {
                                Ok(tool_result) => {
                                    self.trace
                                        .event(&trace_id, TraceEvent::ToolExecuted {
                                            tool: tool_name,
                                            success: true,
                                        })
                                        .await;
                                    let msg = Message::new_tool_result(&call_id, &tool_result.content);
                                    current_messages.push(msg.clone());
                                    session.push_message(msg);
                                }
                                Err(e) => {
                                    self.trace
                                        .event(&trace_id, TraceEvent::ToolExecuted {
                                            tool: tool_name,
                                            success: false,
                                        })
                                        .await;
                                    let msg = Message::new_tool_result(&call_id, e.to_string());
                                    current_messages.push(msg.clone());
                                    session.push_message(msg);
                                }
                            },
                            Err(e) => {
                                warn!("Stream tool task failed: {}", e);
                            }
                        }
                    }
                } else {
                    debug!("Stream sequential execution: {} tools", calls.len());
                    for call in &calls {
                        let tool_name = call.function.name.clone();
                        let result = self.registry
                            .execute(&tool_name, &call.function.arguments, &ctx)
                            .await;
                        match result {
                            Ok(tool_result) => {
                                self.trace
                                    .event(&trace_id, TraceEvent::ToolExecuted {
                                        tool: tool_name,
                                        success: true,
                                    })
                                    .await;
                                let msg = Message::new_tool_result(&call.id, &tool_result.content);
                                current_messages.push(msg.clone());
                                session.push_message(msg);
                            }
                            Err(e) => {
                                self.trace
                                    .event(&trace_id, TraceEvent::ToolExecuted {
                                        tool: tool_name,
                                        success: false,
                                    })
                                    .await;
                                let msg = Message::new_tool_result(&call.id, e.to_string());
                                current_messages.push(msg.clone());
                                session.push_message(msg);
                            }
                        }
                    }
                }

                rounds += 1;
                // Loop back: next iteration will call LLM again with tool results
            } else {
                // No tool calls — final response
                break;
            }
        }

        // Auto-save memory
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
            memory_manager: self.memory_manager.clone(),
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

    // ── Edge case tests ──────────────────────────────────────────────

    /// Helper: create an agent with a tool-calling FakeClient
    fn make_tool_calling_agent(
        response_text: &str,
        tool_call_sequence: Vec<Vec<hermes_cfg::tool::ToolCall>>,
        config: AgentConfig,
    ) -> (Agent, Arc<MemoryStore>) {
        let llm = Arc::new(
            hermes_llm::FakeClient::new(response_text)
                .with_tool_calls_sequence(tool_call_sequence),
        );
        let registry = Arc::new(ToolRegistry::new());
        let memory = Arc::new(MemoryStore::new());
        let agent = Agent::new(config, llm, registry, memory.clone());
        (agent, memory)
    }

    /// Helper: register a no-op tool by name
    async fn register_noop_tool(registry: &Arc<ToolRegistry>, name: &str) {
        struct Noop(String);
        #[async_trait::async_trait]
        impl hermes_cfg::traits::ToolHandler for Noop {
            fn name(&self) -> &str { &self.0 }
            fn description(&self) -> &str { "Noop" }
            fn parameters_schema(&self) -> serde_json::Value {
                serde_json::json!({"type": "object"})
            }
            async fn execute(
                &self,
                _args: &str,
                _ctx: &hermes_cfg::traits::ToolContext,
            ) -> Result<hermes_cfg::tool::ToolResult, hermes_cfg::error::ToolError> {
                Ok(hermes_cfg::tool::ToolResult::success(&self.0, "ok"))
            }
        }
        registry.register(Arc::new(Noop(name.to_string()))).await;
    }

    fn make_tool_call(name: &str, args: &str) -> hermes_cfg::tool::ToolCall {
        hermes_cfg::tool::ToolCall {
            id: format!("call_{}", name),
            call_type: "function".to_string(),
            function: hermes_cfg::tool::FunctionCall {
                name: name.to_string(),
                arguments: args.to_string(),
            },
        }
    }

    #[tokio::test]
    async fn test_agent_chat_with_tool_calls() {
        let config = AgentConfig::default();
        let (agent, _) = make_tool_calling_agent(
            "Done",
            vec![vec![make_tool_call("noop", "{}")]],
            config,
        );
        register_noop_tool(&agent.registry, "noop").await;

        let response = agent.chat("use noop tool", SessionSource::cli()).await.unwrap();
        assert_eq!(response.content, "Done");

        let sessions = agent.list_sessions().await;
        // user + assistant(tool_call) + tool_result + assistant(final) = 4
        assert_eq!(sessions[0].messages.len(), 4);
    }

    #[tokio::test]
    async fn test_agent_max_iterations() {
        let config = AgentConfig {
            max_iterations: 1,
            ..AgentConfig::default()
        };
        let llm = Arc::new(hermes_llm::FakeClient::new("Response"));
        let registry = Arc::new(ToolRegistry::new());
        let memory = Arc::new(MemoryStore::new());
        let agent = Agent::new(config, llm, registry, memory);

        // max_iterations=1 means only 1 LLM call; should return normally
        let response = agent.chat("test", SessionSource::cli()).await.unwrap();
        assert_eq!(response.content, "Response");
    }

    #[tokio::test]
    async fn test_agent_max_tool_rounds() {
        let config = AgentConfig {
            max_tool_rounds: 1,
            max_iterations: 90,
            ..AgentConfig::default()
        };

        // FakeClient will always return tool_calls (even after tool results)
        let llm = Arc::new(
            hermes_llm::FakeClient::new("Final")
                .with_tool_calls_sequence(vec![
                    vec![make_tool_call("noop", "{}")],
                    // After round 1, the tool_call sequence is exhausted,
                    // so the next call returns plain text "Final"
                ]),
        );
        let registry = Arc::new(ToolRegistry::new());
        let memory = Arc::new(MemoryStore::new());
        let agent = Agent::new(config, llm, registry, memory);
        register_noop_tool(&agent.registry, "noop").await;

        let response = agent.chat("test", SessionSource::cli()).await.unwrap();
        // Should complete after 1 tool round + 1 final text response
        assert_eq!(response.content, "Final");
    }

    #[tokio::test]
    async fn test_agent_empty_message() {
        let llm = Arc::new(hermes_llm::FakeClient::new("Reply"));
        let registry = Arc::new(ToolRegistry::new());
        let memory = Arc::new(MemoryStore::new());
        let agent = Agent::new(AgentConfig::default(), llm, registry, memory);

        let response = agent.chat("", SessionSource::cli()).await.unwrap();
        assert_eq!(response.content, "Reply");
    }

    #[tokio::test]
    async fn test_agent_stream_with_tool_calls() {
        let config = AgentConfig::default();
        let llm = Arc::new(
            hermes_llm::FakeClient::new("Final answer")
                .with_tool_calls_sequence(vec![
                    vec![make_tool_call("noop", "{}")],
                ]),
        );
        let registry = Arc::new(ToolRegistry::new());
        let memory = Arc::new(MemoryStore::new());
        let agent = Agent::new(config, llm, registry, memory);
        register_noop_tool(&agent.registry, "noop").await;

        let mut tokens = Vec::new();
        let response = agent
            .chat_stream("use tool", SessionSource::cli(), |t| {
                tokens.push(t.to_string());
            })
            .await
            .unwrap();

        assert_eq!(response.content, "Final answer");
        let sessions = agent.list_sessions().await;
        // user + assistant(tool_call) + tool_result + assistant(final) = 4
        assert_eq!(sessions[0].messages.len(), 4);
    }

    #[tokio::test]
    async fn test_agent_clone_for_gateway_isolation() {
        let llm = Arc::new(hermes_llm::FakeClient::new("Reply"));
        let registry = Arc::new(ToolRegistry::new());
        let memory = Arc::new(MemoryStore::new());
        let agent = Agent::new(AgentConfig::default(), llm, registry, memory);

        // Chat on original agent
        agent.chat("msg1", SessionSource::cli()).await.unwrap();
        assert_eq!(agent.list_sessions().await.len(), 1);

        // Clone should have empty sessions
        let cloned = agent.clone_for_gateway();
        assert_eq!(cloned.list_sessions().await.len(), 0);
    }

    #[tokio::test]
    async fn test_agent_multi_round_tool_calls() {
        let config = AgentConfig {
            max_tool_rounds: 10,
            ..AgentConfig::default()
        };
        // 2 rounds of tool calls then a final text response
        let llm = Arc::new(
            hermes_llm::FakeClient::new("All done")
                .with_tool_calls_sequence(vec![
                    vec![make_tool_call("noop", "{}")],
                    vec![make_tool_call("noop", "{}")],
                ]),
        );
        let registry = Arc::new(ToolRegistry::new());
        let memory = Arc::new(MemoryStore::new());
        let agent = Agent::new(config, llm, registry, memory);
        register_noop_tool(&agent.registry, "noop").await;

        let response = agent.chat("multi-step task", SessionSource::cli()).await.unwrap();
        assert_eq!(response.content, "All done");

        let sessions = agent.list_sessions().await;
        // user + asst(tool1) + tool_result + asst(tool2) + tool_result + asst(final) = 6
        assert_eq!(sessions[0].messages.len(), 6);
    }
}
