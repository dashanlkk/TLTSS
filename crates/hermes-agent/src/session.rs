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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_new() {
        let session = Session::new(SessionSource::cli());
        assert!(!session.id.is_empty());
        assert_eq!(session.source, SessionSource::cli());
        assert!(session.messages.is_empty());
        assert!(session.created_at <= Utc::now());
        assert_eq!(session.created_at, session.updated_at);
    }

    #[test]
    fn test_push_message_updates_timestamp() {
        let mut session = Session::new(SessionSource::cli());
        let before = session.updated_at;
        // small sleep to ensure timestamp differs
        std::thread::sleep(std::time::Duration::from_millis(1));
        session.push_message(Message::new_user("hello"));
        assert_eq!(session.messages.len(), 1);
        assert!(session.updated_at >= before);
    }

    #[test]
    fn test_recent_messages_fewer_than_n() {
        let mut session = Session::new(SessionSource::cli());
        session.push_message(Message::new_user("msg1"));
        session.push_message(Message::new_user("msg2"));
        let recent = session.recent_messages(5);
        assert_eq!(recent.len(), 2);
    }

    #[test]
    fn test_recent_messages_more_than_n() {
        let mut session = Session::new(SessionSource::cli());
        for i in 0..10 {
            session.push_message(Message::new_user(format!("msg{}", i)));
        }
        let recent = session.recent_messages(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].content, "msg7");
        assert_eq!(recent[2].content, "msg9");
    }

    #[test]
    fn test_recent_messages_zero() {
        let mut session = Session::new(SessionSource::cli());
        session.push_message(Message::new_user("msg"));
        assert!(session.recent_messages(0).is_empty());
    }

    #[test]
    fn test_estimate_tokens() {
        let mut session = Session::new(SessionSource::cli());
        // 4 chars = 1 token
        session.push_message(Message::new_user("abcd"));
        assert_eq!(session.estimate_tokens(), 1);

        session.push_message(Message::new_user("abcdefgh"));
        assert_eq!(session.estimate_tokens(), 3);
    }

    #[test]
    fn test_estimate_tokens_empty() {
        let session = Session::new(SessionSource::cli());
        assert_eq!(session.estimate_tokens(), 0);
    }

    #[test]
    fn test_json_roundtrip() {
        let mut session = Session::new(SessionSource::cli());
        session.push_message(Message::new_user("hello"));
        session.push_message(Message::new_assistant("hi"));

        let json = session.to_json().unwrap();
        let restored = Session::from_json(&json).unwrap();

        assert_eq!(restored.id, session.id);
        assert_eq!(restored.source, session.source);
        assert_eq!(restored.messages.len(), 2);
        assert_eq!(restored.messages[0].content, "hello");
        assert_eq!(restored.messages[1].content, "hi");
    }

    #[test]
    fn test_from_json_invalid() {
        let result = Session::from_json("not valid json");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_save_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let session_dir = dir.path().join("sessions");

        let mut session = Session::new(SessionSource::cli());
        session.push_message(Message::new_user("test message"));
        session.save_to_file(&session_dir).await.unwrap();

        let loaded = Session::load_all_from_dir(&session_dir).await.unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, session.id);
        assert_eq!(loaded[0].messages[0].content, "test message");
    }

    #[tokio::test]
    async fn test_load_from_nonexistent_dir() {
        let loaded = Session::load_all_from_dir(std::path::Path::new("/tmp/no_such_dir_hermes_test"))
            .await
            .unwrap();
        assert!(loaded.is_empty());
    }

    #[tokio::test]
    async fn test_load_skips_corrupt_files() {
        let dir = tempfile::tempdir().unwrap();
        let session_dir = dir.path().join("sessions");
        tokio::fs::create_dir_all(&session_dir).await.unwrap();

        // Valid session
        let mut session = Session::new(SessionSource::cli());
        session.push_message(Message::new_user("good"));
        session.save_to_file(&session_dir).await.unwrap();

        // Corrupt file
        tokio::fs::write(session_dir.join("bad.json"), "not json{{{")
            .await
            .unwrap();

        let loaded = Session::load_all_from_dir(&session_dir).await.unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].messages[0].content, "good");
    }

    #[tokio::test]
    async fn test_load_multiple_sessions() {
        let dir = tempfile::tempdir().unwrap();
        let session_dir = dir.path().join("sessions");

        for i in 0..3 {
            let mut s = Session::new(SessionSource::cli());
            s.push_message(Message::new_user(format!("session{}", i)));
            s.save_to_file(&session_dir).await.unwrap();
        }

        let loaded = Session::load_all_from_dir(&session_dir).await.unwrap();
        assert_eq!(loaded.len(), 3);
    }
}
