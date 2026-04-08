use hermes_cfg::platform::SessionSource;
use tokio::sync::mpsc;

/// 平台消息信封
#[derive(Debug, Clone)]
pub struct GatewayMessage {
    pub chat_id: String,
    pub content: String,
    pub source: SessionSource,
}

/// Gateway 与 Agent 之间的通信通道
pub struct GatewayChannel {
    tx: mpsc::UnboundedSender<GatewayMessage>,
    rx: Option<mpsc::UnboundedReceiver<GatewayMessage>>,
}

impl GatewayChannel {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        Self { tx, rx: Some(rx) }
    }

    /// 发送消息到 Agent
    pub fn send(&self, msg: GatewayMessage) -> Result<(), String> {
        self.tx.send(msg).map_err(|e| e.to_string())
    }

    /// 获取接收端（只能调用一次）
    pub fn take_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<GatewayMessage>> {
        self.rx.take()
    }

    /// 克隆发送端
    pub fn sender(&self) -> mpsc::UnboundedSender<GatewayMessage> {
        self.tx.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_channel_communication() {
        let mut channel = GatewayChannel::new();
        let mut rx = channel.take_receiver().unwrap();

        channel.send(GatewayMessage {
            chat_id: "test".into(),
            content: "hello".into(),
            source: SessionSource::cli(),
        }).unwrap();

        let msg = rx.recv().await.unwrap();
        assert_eq!(msg.content, "hello");
    }
}
