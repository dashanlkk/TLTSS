//! Session 全文搜索
//!
//! 跨 session 的历史消息全文搜索，返回匹配片段、session ID 和时间信息。

use crate::session::Session;
use chrono::{DateTime, Utc};
use std::path::Path;

/// 搜索结果片段
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub session_id: String,
    pub message_index: usize,
    /// 匹配片段（上下文裁剪后的内容）
    pub snippet: String,
    pub created_at: DateTime<Utc>,
    /// 相关度得分
    pub score: f64,
}

/// Session 搜索器
pub struct SessionSearcher {
    sessions: Vec<Session>,
}

impl SessionSearcher {
    /// 从内存中的 sessions 创建搜索器
    pub fn new(sessions: Vec<Session>) -> Self {
        Self { sessions }
    }

    /// 从目录加载 sessions 并创建搜索器
    pub async fn from_dir(dir: &Path) -> Result<Self, std::io::Error> {
        let sessions = Session::load_all_from_dir(dir).await?;
        Ok(Self { sessions })
    }

    /// 全文搜索，返回匹配片段
    ///
    /// `query` - 搜索关键词
    /// `max_results` - 最大返回条数
    /// `snippet_radius` - 匹配位置前后各取多少字符作为片段
    pub fn search(&self, query: &str, max_results: usize, snippet_radius: usize) -> Vec<SearchHit> {
        let query_lower = query.to_lowercase();
        let query_tokens = crate::memory::tokenize(query);
        let mut hits: Vec<SearchHit> = Vec::new();

        for session in &self.sessions {
            for (idx, msg) in session.messages.iter().enumerate() {
                let content_lower = msg.content.to_lowercase();
                let mut score: f64 = 0.0;

                // 精确子串匹配加分
                if content_lower.contains(&query_lower) {
                    score += 1.0;
                }

                // TF-IDF 词频匹配加分
                let doc_tokens = crate::memory::tokenize(&msg.content);
                let doc_len = doc_tokens.len() as f64;
                if doc_len > 0.0 {
                    for token in &query_tokens {
                        let tf = doc_tokens.iter().filter(|t| *t == token).count() as f64 / doc_len;
                        score += tf * 2.0;
                    }
                }

                if score > 0.0 {
                    let snippet = make_snippet(&msg.content, &query_lower, snippet_radius);
                    hits.push(SearchHit {
                        session_id: session.id.clone(),
                        message_index: idx,
                        snippet,
                        created_at: session.created_at,
                        score,
                    });
                }
            }
        }

        // 按得分降序排列
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(max_results);
        hits
    }

    /// 获取搜索器中的 session 数量
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }
}

/// 生成匹配片段：定位第一个匹配位置，取前后 `radius` 字符
fn make_snippet(content: &str, query_lower: &str, radius: usize) -> String {
    let content_lower = content.to_lowercase();
    if let Some(pos) = content_lower.find(query_lower) {
        let start = pos.saturating_sub(radius);
        let end = (pos + query_lower.len() + radius).min(content.len());

        let mut snippet = String::new();
        if start > 0 {
            snippet.push_str("...");
        }
        snippet.push_str(&content[start..end]);
        if end < content.len() {
            snippet.push_str("...");
        }
        snippet
    } else {
        // 没有精确匹配（token 匹配），取前 radius*2 字符
        let end = (radius * 2).min(content.len());
        let mut snippet = content[..end].to_string();
        if end < content.len() {
            snippet.push_str("...");
        }
        snippet
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hermes_cfg::message::Message;
    use hermes_cfg::platform::SessionSource;

    fn make_session_with_messages(messages: Vec<&str>) -> Session {
        let mut session = Session::new(SessionSource::cli());
        for msg in messages {
            session.push_message(Message::new_user(msg));
        }
        session
    }

    #[test]
    fn test_search_basic() {
        let sessions = vec![
            make_session_with_messages(vec!["hello world", "foo bar"]),
            make_session_with_messages(vec!["another hello", "no match"]),
        ];

        let searcher = SessionSearcher::new(sessions);
        let hits = searcher.search("hello", 10, 50);
        assert_eq!(hits.len(), 2);
        // Both should have positive score
        assert!(hits[0].score > 0.0);
        assert!(hits[1].score > 0.0);
    }

    #[test]
    fn test_search_max_results() {
        let sessions = vec![
            make_session_with_messages(vec!["match one", "match two", "match three"]),
        ];

        let searcher = SessionSearcher::new(sessions);
        let hits = searcher.search("match", 2, 50);
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn test_search_no_match() {
        let sessions = vec![
            make_session_with_messages(vec!["hello world"]),
        ];

        let searcher = SessionSearcher::new(sessions);
        let hits = searcher.search("zzznonexistent", 10, 50);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_search_hit_has_session_id() {
        let sessions = vec![
            make_session_with_messages(vec!["find me"]),
        ];
        let expected_id = sessions[0].id.clone();

        let searcher = SessionSearcher::new(sessions);
        let hits = searcher.search("find", 10, 50);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].session_id, expected_id);
    }

    #[test]
    fn test_snippet_exact_match() {
        let snippet = make_snippet("the quick brown fox jumps over the lazy dog", "brown fox", 10);
        assert!(snippet.contains("brown fox"));
        assert!(snippet.contains("..."));
    }

    #[test]
    fn test_snippet_no_match() {
        let snippet = make_snippet("short content", "xyz", 20);
        // Should return truncated content
        assert!(snippet.contains("short content"));
    }

    #[test]
    fn test_snippet_at_beginning() {
        let snippet = make_snippet("hello world this is a test", "hello", 10);
        // Match at position 0, no leading "..."
        assert!(snippet.starts_with("hello"));
    }

    #[test]
    fn test_session_count() {
        let sessions = vec![
            make_session_with_messages(vec!["a"]),
            make_session_with_messages(vec!["b"]),
        ];
        let searcher = SessionSearcher::new(sessions);
        assert_eq!(searcher.session_count(), 2);
    }

    #[test]
    fn test_search_chinese() {
        let sessions = vec![
            make_session_with_messages(vec!["用户偏好中文回复", "其他内容"]),
        ];

        let searcher = SessionSearcher::new(sessions);
        let hits = searcher.search("中文", 10, 50);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].snippet.contains("中文"));
    }

    #[test]
    fn test_search_ranked_order() {
        let sessions = vec![
            make_session_with_messages(vec![
                "rust is great for rust programming",
                "I like rust",
                "something else entirely",
            ]),
        ];

        let searcher = SessionSearcher::new(sessions);
        let hits = searcher.search("rust", 10, 50);
        assert_eq!(hits.len(), 2);
        // First hit should have higher score (more occurrences)
        assert!(hits[0].score >= hits[1].score);
    }
}
