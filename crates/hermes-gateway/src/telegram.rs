use async_trait::async_trait;
use hermes_cfg::error::LlmError;
use hermes_cfg::platform::SessionSource;
use hermes_cfg::traits::PlatformAdapter;
use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};

use crate::channel::GatewayMessage;

// --- Telegram Bot API 类型 ---

#[derive(Debug, Deserialize)]
struct TelegramUpdate {
    update_id: i64,
    message: Option<TelegramMessage>,
}

#[derive(Debug, Deserialize)]
struct TelegramMessage {
    #[allow(dead_code)]
    message_id: i64,
    chat: TelegramChat,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TelegramChat {
    id: i64,
}

#[derive(Debug, Serialize)]
struct SendMessageRequest {
    chat_id: i64,
    text: String,
}

#[derive(Debug, Deserialize)]
struct TelegramResponse<T> {
    #[allow(dead_code)]
    ok: bool,
    result: Option<T>,
    #[allow(dead_code)]
    description: Option<String>,
}

/// Telegram Bot 适配器（基于 long polling）
pub struct TelegramAdapter {
    token: String,
    gateway_tx: tokio::sync::mpsc::UnboundedSender<GatewayMessage>,
    handler: Option<Box<dyn Fn(String, SessionSource) + Send + Sync>>,
    client: reqwest::Client,
    offset: std::sync::Arc<tokio::sync::Mutex<i64>>,
}

impl TelegramAdapter {
    pub fn new(token: &str, gateway_tx: tokio::sync::mpsc::UnboundedSender<GatewayMessage>) -> Self {
        Self {
            token: token.to_string(),
            gateway_tx,
            handler: None,
            client: reqwest::Client::new(),
            offset: std::sync::Arc::new(tokio::sync::Mutex::new(0)),
        }
    }

    fn api_url(&self, method: &str) -> String {
        format!("https://api.telegram.org/bot{}/{}", self.token, method)
    }

    /// 发送消息到 Telegram chat
    pub async fn send_message(&self, chat_id: i64, text: &str) -> Result<(), LlmError> {
        let body = SendMessageRequest {
            chat_id,
            text: text.to_string(),
        };

        let resp = self.client
            .post(self.api_url("sendMessage"))
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::ProviderError(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(LlmError::ProviderError(format!("Telegram API error {}: {}", status, text)));
        }

        Ok(())
    }

    /// 获取 updates（long polling）
    async fn get_updates(&self, offset: i64) -> Result<Vec<TelegramUpdate>, LlmError> {
        let resp = self.client
            .post(self.api_url("getUpdates"))
            .json(&serde_json::json!({
                "offset": offset,
                "timeout": 10,
                "allowed_updates": ["message"]
            }))
            .send()
            .await
            .map_err(|e| LlmError::ProviderError(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(LlmError::ProviderError("Telegram getUpdates failed".into()));
        }

        let result: TelegramResponse<Vec<TelegramUpdate>> = resp
            .json()
            .await
            .map_err(|e| LlmError::ProviderError(e.to_string()))?;

        Ok(result.result.unwrap_or_default())
    }

    /// 运行 polling 循环
    async fn poll_loop(&self) {
        loop {
            let offset = *self.offset.lock().await;

            match self.get_updates(offset).await {
                Ok(updates) => {
                    for update in updates {
                        // 更新 offset（confirm receipt）
                        *self.offset.lock().await = update.update_id + 1;

                        if let Some(msg) = update.message {
                            if let Some(text) = msg.text {
                                let chat_id = msg.chat.id.to_string();
                                info!("Telegram message from chat_id={}: {}", chat_id, text);

                                let gateway_msg = GatewayMessage {
                                    chat_id: chat_id.clone(),
                                    content: text.clone(),
                                    source: SessionSource::telegram(&chat_id),
                                };

                                if let Err(e) = self.gateway_tx.send(gateway_msg) {
                                    error!("Failed to forward Telegram message: {}", e);
                                }

                                if let Some(ref handler) = self.handler {
                                    handler(text, SessionSource::telegram(&chat_id));
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("Telegram polling error: {}", e);
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                }
            }
        }
    }
}

#[async_trait]
impl PlatformAdapter for TelegramAdapter {
    async fn run(&mut self) -> Result<(), LlmError> {
        info!("Telegram adapter starting (long polling)...");

        // 验证 bot token 有效性
        let resp = self.client
            .post(self.api_url("getMe"))
            .send()
            .await
            .map_err(|e| LlmError::ProviderError(format!("Telegram getMe failed: {}", e)))?;

        if !resp.status().is_success() {
            return Err(LlmError::ProviderError("Invalid Telegram bot token".into()));
        }

        // 启动 polling 循环
        let client = self.client.clone();
        let token = self.token.clone();
        let gateway_tx = self.gateway_tx.clone();
        let offset = self.offset.clone();

        tokio::spawn(async move {
            let adapter = TelegramAdapter {
                token,
                gateway_tx,
                handler: None,
                client,
                offset,
            };
            adapter.poll_loop().await;
        });

        info!("Telegram adapter started");
        Ok(())
    }

    async fn send(&self, chat_id: &str, message: &str) -> Result<(), LlmError> {
        let chat_id: i64 = chat_id.parse()
            .map_err(|e| LlmError::ProviderError(format!("Invalid chat_id: {}", e)))?;

        self.send_message(chat_id, message).await
    }

    fn set_message_handler(&mut self, handler: Box<dyn Fn(String, SessionSource) + Send + Sync>) {
        self.handler = Some(handler);
    }

    fn platform_name(&self) -> &str {
        "telegram"
    }
}

/// Gateway 管理器：根据配置启动对应的适配器
pub struct GatewayManager {
    adapters: Vec<Box<dyn PlatformAdapter>>,
}

impl GatewayManager {
    pub fn new() -> Self {
        Self {
            adapters: Vec::new(),
        }
    }

    /// 注册 API 适配器
    pub fn with_api(&mut self, gateway_tx: tokio::sync::mpsc::UnboundedSender<GatewayMessage>) {
        let adapter = crate::adapter::ApiAdapter::new(
            crate::channel::GatewayChannel::from_sender(gateway_tx),
        );
        self.adapters.push(Box::new(adapter));
        info!("API adapter registered");
    }

    /// 注册 Telegram 适配器
    pub fn with_telegram(&mut self, token: &str, gateway_tx: tokio::sync::mpsc::UnboundedSender<GatewayMessage>) {
        let adapter = TelegramAdapter::new(token, gateway_tx);
        self.adapters.push(Box::new(adapter));
        info!("Telegram adapter registered");
    }

    /// 注册 Discord 适配器
    pub fn with_discord(&mut self, token: &str, channel_id: &str, gateway_tx: tokio::sync::mpsc::UnboundedSender<GatewayMessage>) {
        let adapter = crate::discord::DiscordAdapter::new(token, channel_id, gateway_tx);
        self.adapters.push(Box::new(adapter));
        info!("Discord adapter registered");
    }

    /// 注册 Slack 适配器
    pub fn with_slack(&mut self, token: &str, channel_id: &str, gateway_tx: tokio::sync::mpsc::UnboundedSender<GatewayMessage>) {
        let adapter = crate::slack::SlackAdapter::new(token, channel_id, gateway_tx);
        self.adapters.push(Box::new(adapter));
        info!("Slack adapter registered");
    }

    /// 启动所有已注册的适配器
    pub async fn start_all(&mut self) -> Result<(), LlmError> {
        for adapter in &mut self.adapters {
            info!("Starting {} adapter...", adapter.platform_name());
            adapter.run().await?;
            info!("{} adapter started", adapter.platform_name());
        }
        Ok(())
    }

    /// 向指定平台发送回复
    pub async fn send_to(&self, platform: &str, chat_id: &str, message: &str) -> Result<(), LlmError> {
        for adapter in &self.adapters {
            if adapter.platform_name() == platform {
                return adapter.send(chat_id, message).await;
            }
        }
        Err(LlmError::ProviderError(format!("No adapter for platform: {}", platform)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telegram_adapter_name() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let adapter = TelegramAdapter::new("test-token", tx);
        assert_eq!(adapter.platform_name(), "telegram");
    }

    #[test]
    fn test_gateway_manager_register() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut manager = GatewayManager::new();
        manager.with_api(tx.clone());
        assert_eq!(manager.adapters.len(), 1);
        assert_eq!(manager.adapters[0].platform_name(), "api");
    }

    #[test]
    fn test_telegram_api_url() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let adapter = TelegramAdapter::new("123456:ABC", tx);
        assert_eq!(adapter.api_url("sendMessage"), "https://api.telegram.org/bot123456:ABC/sendMessage");
    }

    #[tokio::test]
    async fn test_gateway_manager_multiple_adapters() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut manager = GatewayManager::new();
        manager.with_api(tx.clone());
        manager.with_telegram("fake-token", tx.clone());
        assert_eq!(manager.adapters.len(), 2);
        assert_eq!(manager.adapters[0].platform_name(), "api");
        assert_eq!(manager.adapters[1].platform_name(), "telegram");
    }
}
