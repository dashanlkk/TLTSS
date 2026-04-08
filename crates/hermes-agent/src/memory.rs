use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// 记忆条目
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub content: String,
    pub keywords: Vec<String>,
    pub created_at: DateTime<Utc>,
}

/// 长期记忆存储
pub struct MemoryStore {
    entries: RwLock<Vec<MemoryEntry>>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(Vec::new()),
        }
    }

    /// 添加记忆
    pub async fn add(&self, content: impl Into<String>, keywords: Vec<String>) -> String {
        let id = Uuid::new_v4().to_string();
        let entry = MemoryEntry {
            id: id.clone(),
            content: content.into(),
            keywords,
            created_at: Utc::now(),
        };
        self.entries.write().await.push(entry);
        id
    }

    /// 关键词搜索记忆
    pub async fn search(&self, query: &str) -> Vec<String> {
        let query_lower = query.to_lowercase();
        let entries = self.entries.read().await;
        entries
            .iter()
            .filter(|e| {
                let content_lower = e.content.to_lowercase();
                e.keywords.iter().any(|k| query_lower.contains(&k.to_lowercase()))
                    || content_lower.contains(&query_lower)
            })
            .map(|e| e.content.clone())
            .collect()
    }

    /// 列出所有记忆
    pub async fn list(&self) -> Vec<MemoryEntry> {
        self.entries.read().await.clone()
    }

    /// 删除记忆
    pub async fn delete(&self, id: &str) -> bool {
        let mut entries = self.entries.write().await;
        let before = entries.len();
        entries.retain(|e| e.id != id);
        entries.len() < before
    }
}
