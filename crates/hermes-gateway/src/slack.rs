use async_trait::async_trait;
use hermes_cfg::error::LlmError;
use hermes_cfg::platform::SessionSource;
use hermes_cfg::traits::PlatformAdapter;
use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};

use crate::channel::GatewayMessage;

// --- Slack Web API 类型 ---

#[derive(Debug, Deserialize)]
struct SlackConversationsHistoryResponse {
    ok: bool,
    messages: Option<Vec<SlackMessage>>,
    #[allow(dead_code)]
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SlackMessage {
    ts: String,
    text: Option<String>,
    #[serde(default)]
    subtype: Option<String>,
    bot_id: Option<String>,
    #[allow(dead_code)]
    user: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SlackAuthTestResponse {
    ok: bool,
    #[allow(dead_code)]
    bot_id: Option<String>,
    #[allow(dead_code)]
    user: Option<String>,
    #[allow(dead_code)]
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct SlackPostMessageRequest {
    channel: String,
    text: String,
}

/// Slack Bot 适配器（基于 REST 轮询 conversations.history）
pub struct SlackAdapter {
    token: String,
    channel_id: String,
    gateway_tx: tokio::sync::mpsc::UnboundedSender<GatewayMessage>,
    handler: Option<Box<dyn Fn(String, SessionSource) + Send + Sync>>,
    client: reqwest::Client,
    last_ts: std::sync::Arc<tokio::sync::Mutex<String>>,
}

impl SlackAdapter {
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
            last_ts: std::sync::Arc::new(tokio::sync::Mutex::new(String::new())),
        }
    }

    /// 获取频道历史消息
    async fn get_history(&self, oldest: &str) -> Result<Vec<SlackMessage>, LlmError> {
        let url = "https://slack.com/api/conversations.history";
        let mut params = vec![
            ("channel", self.channel_id.as_str()),
            ("limit", "10"),
        ];

        let oldest_owned;
        if !oldest.is_empty() {
            oldest_owned = oldest.to_string();
            params.push(("oldest", oldest_owned.as_str()));
        }

        let resp = self
            .client
            .get(url)
            .header(
                "Authorization",
                format!("Bearer {}", self.token)
                    .parse::<reqwest::header::HeaderValue>()
                    .map_err(|e| LlmError::ProviderError(e.to_string()))?,
            )
            .query(&params)
            .send()
            .await
            .map_err(|e| LlmError::ProviderError(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(LlmError::ProviderError(format!(
                "Slack API HTTP error: {}",
                resp.status()
            )));
        }

        let result: SlackConversationsHistoryResponse = resp
            .json()
            .await
            .map_err(|e| LlmError::ProviderError(e.to_string()))?;

        if !result.ok {
            return Err(LlmError::ProviderError("Slack API returned ok=false".into()));
        }

        Ok(result.messages.unwrap_or_default())
    }

    /// 发送消息到 Slack 频道
    pub async fn send_message(&self, text: &str) -> Result<(), LlmError> {
        let url = "https://slack.com/api/chat.postMessage";
        let body = SlackPostMessageRequest {
            channel: self.channel_id.clone(),
            text: text.to_string(),
        };

        let resp = self
            .client
            .post(url)
            .header(
                "Authorization",
                format!("Bearer {}", self.token)
                    .parse::<reqwest::header::HeaderValue>()
                    .map_err(|e| LlmError::ProviderError(e.to_string()))?,
            )
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::ProviderError(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(LlmError::ProviderError(format!(
                "Slack send error {}: {}",
                status, text
            )));
        }

        Ok(())
    }

    /// 验证 Token 有效性
    async fn validate_token(&self) -> Result<(), LlmError> {
        let resp = self
            .client
            .post("https://slack.com/api/auth.test")
            .header(
                "Authorization",
                format!("Bearer {}", self.token)
                    .parse::<reqwest::header::HeaderValue>()
                    .map_err(|e| LlmError::ProviderError(e.to_string()))?,
            )
            .send()
            .await
            .map_err(|e| LlmError::ProviderError(format!("Slack auth.test failed: {}", e)))?;

        if !resp.status().is_success() {
            return Err(LlmError::ProviderError(
                "Invalid Slack bot token".into(),
            ));
        }

        let result: SlackAuthTestResponse = resp
            .json()
            .await
            .map_err(|e| LlmError::ProviderError(e.to_string()))?;

        if !result.ok {
            return Err(LlmError::ProviderError(
                "Slack auth.test returned ok=false".into(),
            ));
        }

        Ok(())
    }

    /// 运行轮询循环
    async fn poll_loop(&self) {
        loop {
            let last_ts = self.last_ts.lock().await.clone();

            match self.get_history(&last_ts).await {
                Ok(messages) => {
                    // Slack history 返回最新在前，需要反转
                    let messages: Vec<_> = messages.into_iter().rev().collect();

                    for msg in messages {
                        // 跳过 bot 消息和子类型消息（如 channel_join）
                        if msg.bot_id.is_some() || msg.subtype.is_some() {
                            continue;
                        }

                        if let Some(text) = msg.text {
                            if text.trim().is_empty() {
                                continue;
                            }

                            let channel_id = self.channel_id.clone();
                            info!("Slack message (ts={}): {}", msg.ts, text);

                            let gateway_msg = GatewayMessage {
                                chat_id: channel_id.clone(),
                                content: text.clone(),
                                source: SessionSource::slack(&channel_id),
                            };

                            if let Err(e) = self.gateway_tx.send(gateway_msg) {
                                error!("Failed to forward Slack message: {}", e);
                            }

                            if let Some(ref handler) = self.handler {
                                handler(text, SessionSource::slack(&channel_id));
                            }
                        }

                        *self.last_ts.lock().await = msg.ts.clone();
                    }
                }
                Err(e) => {
                    warn!("Slack polling error: {}", e);
                }
            }

            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        }
    }
}

#[async_trait]
impl PlatformAdapter for SlackAdapter {
    async fn run(&mut self) -> Result<(), LlmError> {
        info!("Slack adapter starting...");
        self.validate_token().await?;

        let client = self.client.clone();
        let token = self.token.clone();
        let channel_id = self.channel_id.clone();
        let gateway_tx = self.gateway_tx.clone();
        let last_ts = self.last_ts.clone();

        tokio::spawn(async move {
            let adapter = SlackAdapter {
                token,
                channel_id,
                gateway_tx,
                handler: None,
                client,
                last_ts,
            };
            adapter.poll_loop().await;
        });

        info!("Slack adapter started (polling channel {})", self.channel_id);
        Ok(())
    }

    async fn send(&self, chat_id: &str, message: &str) -> Result<(), LlmError> {
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
        "slack"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slack_adapter_name() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let adapter = SlackAdapter::new("xoxb-test-token", "C12345678", tx);
        assert_eq!(adapter.platform_name(), "slack");
    }
}
