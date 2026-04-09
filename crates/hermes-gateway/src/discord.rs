use async_trait::async_trait;
use hermes_cfg::error::LlmError;
use hermes_cfg::platform::SessionSource;
use hermes_cfg::traits::PlatformAdapter;
use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};

use crate::channel::GatewayMessage;

// --- Discord REST API 类型 ---

#[derive(Debug, Deserialize)]
struct DiscordMessage {
    id: String,
    #[allow(dead_code)]
    channel_id: String,
    author: DiscordUser,
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DiscordUser {
    #[allow(dead_code)]
    id: String,
    #[allow(dead_code)]
    username: String,
    bot: Option<bool>,
}

#[derive(Debug, Serialize)]
struct CreateMessageRequest {
    content: String,
}

/// Discord Bot 适配器（基于 REST 轮询）
pub struct DiscordAdapter {
    token: String,
    channel_id: String,
    gateway_tx: tokio::sync::mpsc::UnboundedSender<GatewayMessage>,
    handler: Option<Box<dyn Fn(String, SessionSource) + Send + Sync>>,
    client: reqwest::Client,
    last_message_id: std::sync::Arc<tokio::sync::Mutex<String>>,
}

impl DiscordAdapter {
    pub fn new(
        token: &str,
        channel_id: &str,
        gateway_tx: tokio::sync::mpsc::UnboundedSender<GatewayMessage>,
    ) -> Self {
        Self {
            token: token.to_string(),
            channel_id: channel_id.to_string(),
            gateway_tx,
            handler: None,
            client: reqwest::Client::new(),
            last_message_id: std::sync::Arc::new(tokio::sync::Mutex::new(String::new())),
        }
    }

    /// Discord API 请求头
    fn auth_header(&self) -> reqwest::header::HeaderValue {
        reqwest::header::HeaderValue::from_str(&format!("Bot {}", self.token))
            .unwrap_or_else(|_| reqwest::header::HeaderValue::from_static(""))
    }

    /// 获取频道最新消息（REST 轮询）
    async fn get_recent_messages(&self, after: &str) -> Result<Vec<DiscordMessage>, LlmError> {
        let url = format!(
            "https://discord.com/api/v10/channels/{}/messages?limit=10{}",
            self.channel_id,
            if after.is_empty() {
                String::new()
            } else {
                format!("&after={}", after)
            }
        );

        let resp = self
            .client
            .get(&url)
            .header("Authorization", self.auth_header())
            .send()
            .await
            .map_err(|e| LlmError::ProviderError(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(LlmError::ProviderError(format!(
                "Discord API error {}: {}",
                status, text
            )));
        }

        resp.json()
            .await
            .map_err(|e| LlmError::ProviderError(e.to_string()))
    }

    /// 发送消息到 Discord 频道
    pub async fn send_message(&self, content: &str) -> Result<(), LlmError> {
        let url = format!(
            "https://discord.com/api/v10/channels/{}/messages",
            self.channel_id
        );

        let body = CreateMessageRequest {
            content: content.to_string(),
        };

        let resp = self
            .client
            .post(&url)
            .header("Authorization", self.auth_header())
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::ProviderError(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(LlmError::ProviderError(format!(
                "Discord send error {}: {}",
                status, text
            )));
        }

        Ok(())
    }

    /// 验证 Bot Token 有效性
    async fn validate_token(&self) -> Result<(), LlmError> {
        let resp = self
            .client
            .get("https://discord.com/api/v10/users/@me")
            .header("Authorization", self.auth_header())
            .send()
            .await
            .map_err(|e| LlmError::ProviderError(format!("Discord getMe failed: {}", e)))?;

        if !resp.status().is_success() {
            return Err(LlmError::ProviderError(
                "Invalid Discord bot token".into(),
            ));
        }

        Ok(())
    }

    /// 运行轮询循环
    async fn poll_loop(&self) {
        // 先获取最新消息 ID 作为起点
        match self.get_recent_messages("").await {
            Ok(messages) => {
                if let Some(first) = messages.first() {
                    *self.last_message_id.lock().await = first.id.clone();
                    info!("Discord: starting after message {}", first.id);
                }
            }
            Err(e) => {
                warn!("Discord initial fetch failed: {}", e);
            }
        }

        loop {
            let last_id = self.last_message_id.lock().await.clone();

            match self.get_recent_messages(&last_id).await {
                Ok(messages) => {
                    // Discord 返回最新在前，需要反转
                    let messages: Vec<_> = messages.into_iter().rev().collect();

                    for msg in messages {
                        // 跳过 bot 自己的消息
                        if msg.author.bot.unwrap_or(false) {
                            continue;
                        }

                        if let Some(text) = msg.content {
                            if text.trim().is_empty() {
                                continue;
                            }

                            let channel_id = msg.channel_id.clone();
                            info!("Discord message from {}: {}", msg.author.username, text);

                            let gateway_msg = GatewayMessage {
                                chat_id: channel_id.clone(),
                                content: text.clone(),
                                source: SessionSource::discord(&channel_id),
                            };

                            if let Err(e) = self.gateway_tx.send(gateway_msg) {
                                error!("Failed to forward Discord message: {}", e);
                            }

                            if let Some(ref handler) = self.handler {
                                handler(text, SessionSource::discord(&channel_id));
                            }
                        }

                        *self.last_message_id.lock().await = msg.id.clone();
                    }
                }
                Err(e) => {
                    warn!("Discord polling error: {}", e);
                }
            }

            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        }
    }
}

#[async_trait]
impl PlatformAdapter for DiscordAdapter {
    async fn run(&mut self) -> Result<(), LlmError> {
        info!("Discord adapter starting...");
        self.validate_token().await?;

        let client = self.client.clone();
        let token = self.token.clone();
        let channel_id = self.channel_id.clone();
        let gateway_tx = self.gateway_tx.clone();
        let last_message_id = self.last_message_id.clone();

        tokio::spawn(async move {
            let adapter = DiscordAdapter {
                token,
                channel_id,
                gateway_tx,
                handler: None,
                client,
                last_message_id,
            };
            adapter.poll_loop().await;
        });

        info!("Discord adapter started (polling channel {})", self.channel_id);
        Ok(())
    }

    async fn send(&self, chat_id: &str, message: &str) -> Result<(), LlmError> {
        // Discord 适配器针对单频道，chat_id 作为校验
        let _ = chat_id;
        self.send_message(message).await
    }

    fn set_message_handler(
        &mut self,
        handler: Box<dyn Fn(String, SessionSource) + Send + Sync>,
    ) {
        self.handler = Some(handler);
    }

    fn platform_name(&self) -> &str {
        "discord"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discord_adapter_name() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let adapter = DiscordAdapter::new("test-token", "channel-123", tx);
        assert_eq!(adapter.platform_name(), "discord");
    }

    #[test]
    fn test_discord_auth_header() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let adapter = DiscordAdapter::new("mytoken123", "channel-123", tx);
        let header = adapter.auth_header();
        assert_eq!(header.to_str().unwrap(), "Bot mytoken123");
    }
}
