use async_trait::async_trait;
use hermes_cfg::traits::PlatformAdapter;
use hermes_cfg::platform::SessionSource;
use hermes_cfg::error::LlmError;

use crate::channel::GatewayChannel;

/// HTTP API Gateway 适配器
pub struct ApiAdapter {
    channel: GatewayChannel,
    handler: Option<Box<dyn Fn(String, SessionSource) + Send + Sync>>,
}

impl ApiAdapter {
    pub fn new(channel: GatewayChannel) -> Self {
        Self {
            channel,
            handler: None,
        }
    }
}

#[async_trait]
impl PlatformAdapter for ApiAdapter {
    async fn run(&mut self) -> Result<(), LlmError> {
        // API adapter 不需要主动轮询，被动接收 HTTP 请求
        Ok(())
    }

    async fn send(&self, chat_id: &str, message: &str) -> Result<(), LlmError> {
        // 通过 channel 转发到 Agent
        self.channel.send(crate::channel::GatewayMessage {
            chat_id: chat_id.to_string(),
            content: message.to_string(),
            source: SessionSource::api(),
        }).map_err(LlmError::ProviderError)
    }

    fn set_message_handler(&mut self, handler: Box<dyn Fn(String, SessionSource) + Send + Sync>) {
        self.handler = Some(handler);
    }

    fn platform_name(&self) -> &str {
        "api"
    }
}
