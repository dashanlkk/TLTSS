//! Gateway Runner — message processing loop that integrates Agent with Gateway.
//!
//! Reads messages from the gateway channel, processes them through
//! per-session Agent instances, and routes replies back to the
//! originating platform via GatewayManager.

use hermes_agent::{Agent, MemoryStore};
use hermes_cfg::platform::SessionSource;
use hermes_cfg::traits::LlmClient;
use hermes_tools::ToolRegistry;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

use crate::channel::{GatewayChannel, GatewayMessage};
use crate::telegram::GatewayManager;

/// Gateway Runner — connects the Agent to the messaging gateway.
///
/// Responsibilities:
/// 1. Read incoming messages from the gateway channel
/// 2. Route each message to a per-session Agent
/// 3. Send Agent replies back to the originating platform
pub struct GatewayRunner {
    /// Shared LLM client
    llm: Arc<dyn LlmClient>,
    /// Shared tool registry
    registry: Arc<ToolRegistry>,
    /// Shared memory store
    memory: Arc<MemoryStore>,
    /// Per-session Agent cache (session_key → Agent)
    agents: Arc<RwLock<HashMap<String, Agent>>>,
    /// Platform adapter manager (for sending replies)
    manager: Arc<RwLock<GatewayManager>>,
    /// System prompt for gateway agents
    system_prompt: String,
}

impl GatewayRunner {
    /// Create a new GatewayRunner.
    pub fn new(
        llm: Arc<dyn LlmClient>,
        registry: Arc<ToolRegistry>,
        memory: Arc<MemoryStore>,
        manager: GatewayManager,
    ) -> Self {
        Self {
            llm,
            registry,
            memory,
            agents: Arc::new(RwLock::new(HashMap::new())),
            manager: Arc::new(RwLock::new(manager)),
            system_prompt: "You are Hermes, an intelligent AI assistant. You help users through various messaging platforms. Be concise and helpful.".to_string(),
        }
    }

    /// Set custom system prompt for gateway agents.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Run the message processing loop.
    ///
    /// Reads from the gateway channel, processes messages through Agents,
    /// and sends replies back to platforms.
    pub async fn run(
        &self,
        channel: &mut GatewayChannel,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let rx = channel
            .take_receiver()
            .ok_or("Gateway channel receiver already taken")?;

        info!("GatewayRunner started, waiting for messages...");

        use tokio_stream::wrappers::UnboundedReceiverStream;
        use futures::StreamExt;

        let mut stream = UnboundedReceiverStream::new(rx);

        while let Some(msg) = stream.next().await {
            if let Err(e) = self.handle_message(msg).await {
                error!("Error processing gateway message: {}", e);
            }
        }

        info!("GatewayRunner stopped (channel closed)");
        Ok(())
    }

    /// Process a single incoming gateway message.
    async fn handle_message(&self, msg: GatewayMessage) -> Result<(), String> {
        let session_key = format!(
            "agent:main:{}:{}",
            msg.source.platform.as_str(),
            msg.chat_id
        );

        info!(
            "Processing message from {} (session={}): {}",
            msg.source.platform.as_str(),
            session_key,
            if msg.content.len() > 100 {
                format!("{}...", &msg.content[..100])
            } else {
                msg.content.clone()
            }
        );

        // Get or create agent for this session
        let agent = self.get_or_create_agent(&session_key, &msg.source).await;

        // Process through Agent
        match agent.chat(&msg.content, msg.source.clone()).await {
            Ok(response) => {
                info!(
                    "Agent replied to session={} ({} chars)",
                    session_key,
                    response.content.len()
                );

                // Send reply back to the originating platform
                let platform = msg.source.platform.as_str();
                let chat_id = &msg.chat_id;
                let reply = &response.content;

                let manager = self.manager.read().await;
                if let Err(e) = manager.send_to(platform, chat_id, reply).await {
                    warn!(
                        "Failed to send reply to {} (chat_id={}): {}",
                        platform, chat_id, e
                    );
                }
            }
            Err(e) => {
                error!("Agent error for session={}: {}", session_key, e);

                // Send error message to user
                let manager = self.manager.read().await;
                let _ = manager
                    .send_to(
                        msg.source.platform.as_str(),
                        &msg.chat_id,
                        "Sorry, I encountered an error processing your message. Please try again.",
                    )
                    .await;
            }
        }

        Ok(())
    }

    /// Get an existing Agent for the session, or create a new one.
    async fn get_or_create_agent(
        &self,
        session_key: &str,
        _source: &SessionSource,
    ) -> Agent {
        // Check cache
        {
            let agents = self.agents.read().await;
            if let Some(agent) = agents.get(session_key) {
                return agent.clone_for_gateway();
            }
        }

        // Create new agent
        info!("Creating new Agent for session={}", session_key);
        let agent_config = hermes_agent::agent::AgentConfig {
            system_prompt: self.system_prompt.clone(),
            max_tool_rounds: 10,
            max_iterations: 60,
            max_context_tokens: 128_000,
            max_retries: 3,
            data_dir: None,
        };

        let agent = Agent::new(
            agent_config,
            self.llm.clone(),
            self.registry.clone(),
            self.memory.clone(),
        );

        // Cache it
        {
            let mut agents = self.agents.write().await;
            agents.insert(session_key.to_string(), agent.clone_for_gateway());
        }

        agent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_key_format() {
        let source = SessionSource::api();
        let key = format!("agent:main:{}:chat-123", source.platform.as_str());
        assert!(key.contains("api"));
        assert!(key.contains("chat-123"));
    }
}
