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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_add_and_search() {
        let store = MemoryStore::new();
        store.add("User prefers Chinese", vec!["Chinese".into()]).await;
        store.add("Dark mode enabled", vec!["dark".into(), "mode".into()]).await;

        let results = store.search("Chinese").await;
        assert_eq!(results.len(), 1);
        assert!(results[0].contains("Chinese"));
    }

    #[tokio::test]
    async fn test_list_and_delete() {
        let store = MemoryStore::new();
        let id = store.add("test memory", vec![]).await;
        assert_eq!(store.list().await.len(), 1);

        assert!(store.delete(&id).await);
        assert!(store.list().await.is_empty());
        assert!(!store.delete("nonexistent").await);
    }

    #[tokio::test]
    async fn test_search_by_content() {
        let store = MemoryStore::new();
        store.add("Rust is fast and safe", vec![]).await;

        let results = store.search("fast").await;
        assert_eq!(results.len(), 1);
    }
}
