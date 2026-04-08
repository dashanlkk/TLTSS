use async_trait::async_trait;
use futures::Stream;
use hermes_cfg::prelude::*;
use hermes_cfg::traits::{LlmClient, StreamEvent};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tracing::info;

/// Fallback 客户端：主 provider 失败时切换到备用
pub struct FallbackClient {
    primary: Arc<dyn LlmClient>,
    fallback: Arc<dyn LlmClient>,
}

impl FallbackClient {
    pub fn new(primary: Arc<dyn LlmClient>, fallback: Arc<dyn LlmClient>) -> Self {
        Self { primary, fallback }
    }
}

#[async_trait]
impl LlmClient for FallbackClient {
    async fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<Message, LlmError> {
        match self.primary.complete(messages, tools).await {
            Ok(msg) => {
                info!("using primary");
                Ok(msg)
            }
            Err(e) => {
                info!("primary failed ({}), switching to fallback", e);
                self.fallback.complete(messages, tools).await
            }
        }
    }

    async fn complete_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send>>, LlmError>
    {
        match self.primary.complete_stream(messages, tools).await {
            Ok(s) => {
                info!("using primary stream");
                Ok(s)
            }
            Err(e) => {
                info!("primary stream failed ({}), switching to fallback", e);
                self.fallback.complete_stream(messages, tools).await
            }
        }
    }

    async fn ping(&self) -> Result<Duration, LlmError> {
        self.primary.ping().await
    }
}
