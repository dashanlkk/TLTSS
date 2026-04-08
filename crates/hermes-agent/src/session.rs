use chrono::{DateTime, Utc};
use hermes_cfg::message::Message;
use hermes_cfg::platform::SessionSource;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// 会话
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub source: SessionSource,
    pub messages: Vec<Message>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Session {
    pub fn new(source: SessionSource) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            source,
            messages: Vec::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// 追加消息
    pub fn push_message(&mut self, message: Message) {
        self.updated_at = Utc::now();
        self.messages.push(message);
    }

    /// 获取最近 N 条消息（用于上下文窗口）
    pub fn recent_messages(&self, n: usize) -> &[Message] {
        if self.messages.len() <= n {
            &self.messages
        } else {
            &self.messages[self.messages.len() - n..]
        }
    }

    /// 估算 token 数（粗略：4 chars ≈ 1 token）
    pub fn estimate_tokens(&self) -> usize {
        self.messages.iter().map(|m| m.content.len() / 4).sum()
    }
}
