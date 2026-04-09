use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::sync::RwLock;
use tracing::info;
use uuid::Uuid;

/// 记忆条目
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub content: String,
    pub keywords: Vec<String>,
    pub created_at: DateTime<Utc>,
}

/// 长期记忆存储（支持持久化）
pub struct MemoryStore {
    entries: RwLock<Vec<MemoryEntry>>,
    data_dir: Option<PathBuf>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(Vec::new()),
            data_dir: None,
        }
    }

    /// 设置持久化目录
    pub fn with_data_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.data_dir = Some(dir.into());
        self
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
        let _ = self.persist().await;
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
                e.keywords
                    .iter()
                    .any(|k| query_lower.contains(&k.to_lowercase()))
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
        let deleted = entries.len() < before;
        drop(entries);
        if deleted {
            let _ = self.persist().await;
        }
        deleted
    }

    /// 持久化到 JSON 文件
    async fn persist(&self) -> Result<(), std::io::Error> {
        let dir = match &self.data_dir {
            Some(d) => d,
            None => return Ok(()),
        };

        tokio::fs::create_dir_all(dir).await?;
        let entries = self.entries.read().await;
        let json = serde_json::to_string_pretty(&*entries)?;
        tokio::fs::write(dir.join("memory.json"), json).await?;
        Ok(())
    }

    /// 从文件加载
    pub async fn load_from_dir(&self) -> Result<(), std::io::Error> {
        let dir = match &self.data_dir {
            Some(d) => d,
            None => return Ok(()),
        };

        let path = dir.join("memory.json");
        if !path.exists() {
            return Ok(());
        }

        let content = tokio::fs::read_to_string(&path).await?;
        let entries: Vec<MemoryEntry> = serde_json::from_str(&content)?;
        let count = entries.len();
        *self.entries.write().await = entries;
        info!("Loaded {} memories from {}", count, path.display());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_add_and_search() {
        let store = MemoryStore::new();
        store
            .add("User prefers Chinese", vec!["Chinese".into()])
            .await;
        store
            .add("Dark mode enabled", vec!["dark".into(), "mode".into()])
            .await;

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

    #[tokio::test]
    async fn test_persist_and_load() {
        let dir = std::env::temp_dir().join("hermes_memory_test_persist");
        let _ = std::fs::remove_dir_all(&dir);

        let id;
        // 写入
        {
            let store = MemoryStore::new().with_data_dir(&dir);
            id = store.add("persisted memory", vec!["test".into()]).await;
        }

        // 读取
        {
            let store = MemoryStore::new().with_data_dir(&dir);
            store.load_from_dir().await.unwrap();
            let entries = store.list().await;
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].id, id);
        }

        std::fs::remove_dir_all(&dir).ok();
    }
}
