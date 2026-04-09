use chrono::{DateTime, Utc};
use hermes_cfg::message::Message;
use hermes_cfg::platform::SessionSource;
use serde::{Deserialize, Serialize};
use tracing::info;
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

    /// 序列化为 JSON 字符串
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// 从 JSON 反序列化
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// 保存到文件
    pub async fn save_to_file(&self, dir: &std::path::Path) -> std::io::Result<()> {
        tokio::fs::create_dir_all(dir).await?;
        let path = dir.join(format!("{}.json", self.id));
        let json = self.to_json().map_err(|e| std::io::Error::other(e.to_string()))?;
        tokio::fs::write(&path, json).await
    }

    /// 从目录加载所有 session 文件
    pub async fn load_all_from_dir(dir: &std::path::Path) -> std::io::Result<Vec<Self>> {
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let mut sessions = Vec::new();
        let mut entries = tokio::fs::read_dir(dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().map(|e| e == "json").unwrap_or(false) {
                let content = tokio::fs::read_to_string(&path).await?;
                match Self::from_json(&content) {
                    Ok(session) => sessions.push(session),
                    Err(e) => {
                        tracing::warn!("Failed to parse session {:?}: {}", path, e);
                    }
                }
            }
        }
        if !sessions.is_empty() {
            info!("Loaded {} sessions from {}", sessions.len(), dir.display());
        }
        Ok(sessions)
    }
}
