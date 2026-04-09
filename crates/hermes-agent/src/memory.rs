use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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
    /// 可选的向量 embedding（预留，当前未使用）
    #[serde(default)]
    pub embedding: Option<Vec<f64>>,
}

/// 搜索结果（带得分）
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub content: String,
    pub score: f64,
    pub id: String,
}

/// 长期记忆存储（支持持久化）
pub struct MemoryStore {
    entries: RwLock<Vec<MemoryEntry>>,
    data_dir: Option<PathBuf>,
    /// IDF 缓存：token → 文档频率
    idf_cache: RwLock<HashMap<String, f64>>,
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryStore {
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(Vec::new()),
            data_dir: None,
            idf_cache: RwLock::new(HashMap::new()),
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
            embedding: None,
        };
        self.entries.write().await.push(entry);
        // 清除 IDF 缓存（文档集合已变化）
        self.idf_cache.write().await.clear();
        let _ = self.persist().await;
        id
    }

    /// 关键词搜索记忆（简单匹配，向后兼容）
    pub async fn search(&self, query: &str) -> Vec<String> {
        self.search_ranked(query)
            .await
            .into_iter()
            .map(|r| r.content)
            .collect()
    }

    /// TF-IDF 加权搜索，返回带得分的结果
    pub async fn search_ranked(&self, query: &str) -> Vec<SearchResult> {
        let entries = self.entries.read().await;
        if entries.is_empty() {
            return Vec::new();
        }

        let query_tokens = tokenize(query);
        if query_tokens.is_empty() {
            return Vec::new();
        }

        let n_docs = entries.len() as f64;

        // 计算 IDF（惰性，每次搜索时基于当前文档集计算）
        let idf = compute_idf(&entries, &query_tokens, n_docs);

        // 对每个 entry 计算 TF-IDF 得分
        let mut scored: Vec<(f64, &MemoryEntry)> = entries
            .iter()
            .filter_map(|entry| {
                let score = tfidf_score(entry, &query_tokens, &idf);
                if score > 0.0 {
                    Some((score, entry))
                } else {
                    None
                }
            })
            .collect();

        // 按得分降序排列
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .map(|(score, entry)| SearchResult {
                content: entry.content.clone(),
                score,
                id: entry.id.clone(),
            })
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
            self.idf_cache.write().await.clear();
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
        self.idf_cache.write().await.clear();
        info!("Loaded {} memories from {}", count, path.display());
        Ok(())
    }
}

// ── TF-IDF 内部实现 ─────────────────────────────────────────────────

/// 简易分词：按空格/标点分割，ASCII/CJK 交界处额外拆分，中文按字符拆分
pub fn tokenize(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let mut tokens = Vec::new();

    for word in lower.split(|c: char| !c.is_alphanumeric()) {
        if word.is_empty() {
            continue;
        }
        if word.is_ascii() {
            tokens.push(word.to_string());
        } else {
            // 在 ASCII 和非 ASCII 交界处拆分
            let mut current = String::new();
            let mut prev_is_ascii = None;

            for ch in word.chars() {
                let is_ascii = ch.is_ascii();
                if prev_is_ascii != Some(is_ascii) && !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
                current.push(ch);
                prev_is_ascii = Some(is_ascii);
            }
            if !current.is_empty() {
                tokens.push(current);
            }

            // CJK 部分：逐字符也作为 token
            for ch in word.chars() {
                if ch.is_alphanumeric() && !ch.is_ascii() {
                    tokens.push(ch.to_string());
                }
            }
        }
    }

    tokens
}

/// 计算查询 token 的 IDF 值
fn compute_idf(entries: &[MemoryEntry], query_tokens: &[String], n_docs: f64) -> HashMap<String, f64> {
    let mut doc_freq: HashMap<&str, usize> = HashMap::new();

    for token in query_tokens {
        if doc_freq.contains_key(token.as_str()) {
            continue;
        }
        let count = entries.iter().filter(|e| {
            let tokens = tokenize(&format!("{} {}", e.content, e.keywords.join(" ")));
            tokens.iter().any(|t| t == token)
        }).count();
        doc_freq.insert(token.as_str(), count);
    }

    query_tokens.iter()
        .map(|token| {
            let df = *doc_freq.get(token.as_str()).unwrap_or(&0) as f64;
            let idf = if df > 0.0 {
                (n_docs / df).ln() + 1.0
            } else {
                0.0
            };
            (token.clone(), idf)
        })
        .collect()
}

/// 计算 entry 对查询的 TF-IDF 得分
fn tfidf_score(entry: &MemoryEntry, query_tokens: &[String], idf: &HashMap<String, f64>) -> f64 {
    // 将 content + keywords 合并为文档文本
    let doc_text = format!("{} {}", entry.content, entry.keywords.join(" "));
    let doc_tokens = tokenize(&doc_text);

    if doc_tokens.is_empty() {
        return 0.0;
    }

    // 计算 TF：每个 query token 在文档中出现的次数 / 文档 token 总数
    let doc_len = doc_tokens.len() as f64;
    let mut score = 0.0;

    for token in query_tokens {
        let tf = doc_tokens.iter().filter(|t| *t == token).count() as f64 / doc_len;
        let idf_val = idf.get(token).copied().unwrap_or(0.0);
        score += tf * idf_val;
    }

    // 对 keywords 中的精确匹配额外加分
    for kw in &entry.keywords {
        let kw_lower = kw.to_lowercase();
        if query_tokens.iter().any(|t| kw_lower.contains(t)) {
            score += 0.5;
        }
    }

    score
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

    // ── TF-IDF 搜索测试 ──

    #[test]
    fn test_tokenize_ascii() {
        let tokens = tokenize("hello world rust");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"rust".to_string()));
    }

    #[test]
    fn test_tokenize_chinese() {
        let tokens = tokenize("你好世界");
        // CJK 字符应该逐字符拆分 + 完整词
        assert!(tokens.contains(&"你好世界".to_string()));
        assert!(tokens.contains(&"你".to_string()));
        assert!(tokens.contains(&"好".to_string()));
    }

    #[test]
    fn test_tokenize_mixed() {
        let tokens = tokenize("Rust语言 fast");
        assert!(tokens.contains(&"rust".to_string()));
        assert!(tokens.contains(&"fast".to_string()));
        assert!(tokens.contains(&"语言".to_string()));
    }

    #[tokio::test]
    async fn test_tfidf_ranked_search() {
        let store = MemoryStore::new();
        store
            .add("Rust is a systems programming language", vec!["rust".into(), "programming".into()])
            .await;
        store
            .add("Python is a scripting language", vec!["python".into(), "scripting".into()])
            .await;
        store
            .add("Rust programming best practices", vec!["rust".into(), "best".into()])
            .await;

        let results = store.search_ranked("rust programming").await;
        assert!(!results.is_empty());
        // "Rust programming best practices" 应该排名最高（两个关键词都出现）
        assert!(results[0].content.contains("Rust"));
        assert!(results[0].score > 0.0);
    }

    #[tokio::test]
    async fn test_tfidf_chinese_search() {
        let store = MemoryStore::new();
        store.add("用户偏好中文回复", vec!["中文".into()]).await;
        store.add("暗色模式已启用", vec!["暗色".into()]).await;

        let results = store.search("中文").await;
        assert_eq!(results.len(), 1);
        assert!(results[0].contains("中文"));
    }

    #[tokio::test]
    async fn test_tfidf_no_results() {
        let store = MemoryStore::new();
        store.add("hello world", vec![]).await;

        let results = store.search_ranked("zzznonexistent").await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_tfidf_empty_store() {
        let store = MemoryStore::new();
        let results = store.search_ranked("anything").await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_search_result_has_score_and_id() {
        let store = MemoryStore::new();
        store.add("test content here", vec!["test".into()]).await;

        let results = store.search_ranked("test").await;
        assert_eq!(results.len(), 1);
        assert!(!results[0].id.is_empty());
        assert!(results[0].score > 0.0);
    }

    #[tokio::test]
    async fn test_keyword_bonus_scoring() {
        let store = MemoryStore::new();
        // Entry with matching keyword should score higher
        store
            .add("The weather is nice today", vec!["weather".into()])
            .await;
        store
            .add("She said weather is a topic", vec![])
            .await;

        let results = store.search_ranked("weather").await;
        assert_eq!(results.len(), 2);
        // First result should be the one with the keyword tag
        assert!(results[0].content.contains("nice today"));
    }
}
