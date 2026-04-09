//! Memory Manager — MEMORY.md/USER.md file-backed memory system.
//!
//! Port of Python hermes-agent/tools/memory_tool.py
//!
//! Provides structured memory persistence with:
//! - MEMORY.md for agent observations and learnings
//! - USER.md for user profile/preferences
//! - Character limits and content scanning
//! - File locking for concurrent access
//! - Entry delimiter: `\n§\n` (section sign)

use std::path::PathBuf;
use tokio::sync::RwLock;
use tracing::info;

/// Entry delimiter (section sign)
const ENTRY_DELIMITER: &str = "\n\u{00a7}\n";

/// Character limits (matching Python version)
const MEMORY_CHAR_LIMIT: usize = 2200;
const USER_CHAR_LIMIT: usize = 1375;

/// Injection/exfiltration patterns to scan for
const BLOCKED_PATTERNS: &[&str] = &[
    "ignore previous",
    "ignore above",
    "forget everything",
    "system prompt",
    "you are now",
    "jailbreak",
    "<!--",
    "<memory-context>",
    "</memory-context>",
];

/// Memory target: MEMORY.md or USER.md
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryTarget {
    Memory,
    User,
}

impl std::fmt::Display for MemoryTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryTarget::Memory => write!(f, "memory"),
            MemoryTarget::User => write!(f, "user"),
        }
    }
}

/// Memory Manager — manages MEMORY.md and USER.md files.
pub struct MemoryManager {
    /// Directory for memory files (typically ~/.hermes/memories/)
    data_dir: PathBuf,
    /// In-memory cache for MEMORY.md entries
    memory_entries: RwLock<Vec<String>>,
    /// In-memory cache for USER.md entries
    user_entries: RwLock<Vec<String>>,
}

impl MemoryManager {
    /// Create a new MemoryManager with the given data directory.
    pub fn new(data_dir: impl Into<PathBuf>) -> Self {
        Self {
            data_dir: data_dir.into(),
            memory_entries: RwLock::new(Vec::new()),
            user_entries: RwLock::new(Vec::new()),
        }
    }

    /// Create with default path (~/.hermes/memories/)
    pub fn default_path() -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        Self::new(home.join(".hermes").join("memories"))
    }

    /// Initialize: load existing files into memory.
    pub async fn initialize(&self) -> Result<(), std::io::Error> {
        tokio::fs::create_dir_all(&self.data_dir).await?;

        let memory_path = self.data_dir.join("MEMORY.md");
        let user_path = self.data_dir.join("USER.md");

        if memory_path.exists() {
            let content = tokio::fs::read_to_string(&memory_path).await?;
            let entries = parse_entries(&content);
            info!("Loaded {} MEMORY.md entries", entries.len());
            *self.memory_entries.write().await = entries;
        }

        if user_path.exists() {
            let content = tokio::fs::read_to_string(&user_path).await?;
            let entries = parse_entries(&content);
            info!("Loaded {} USER.md entries", entries.len());
            *self.user_entries.write().await = entries;
        }

        Ok(())
    }

    /// Add an entry to the specified target.
    pub async fn add(&self, target: MemoryTarget, content: &str) -> Result<String, String> {
        // Scan for blocked patterns
        scan_content(content)?;

        // Check character limit
        let (entries, limit) = self.get_target(target).await;
        let current_total: usize = entries.read().await.iter().map(|e| e.len()).sum();
        if current_total + content.len() > limit {
            return Err(format!(
                "Character limit exceeded: {} + {} > {}",
                current_total, content.len(), limit
            ));
        }

        // Add entry
        entries.write().await.push(content.to_string());
        self.persist(target).await.map_err(|e| e.to_string())?;

        info!("Added {} entry ({} chars)", target, content.len());
        Ok(format!("Added to {}", target))
    }

    /// Replace an existing entry.
    pub async fn replace(
        &self,
        target: MemoryTarget,
        old: &str,
        new: &str,
    ) -> Result<String, String> {
        scan_content(new)?;

        let (entries, _) = self.get_target(target).await;
        let mut entries = entries.write().await;

        let idx = entries.iter().position(|e| e.contains(old));
        match idx {
            Some(i) => {
                entries[i] = new.to_string();
                drop(entries);
                self.persist(target).await.map_err(|e| e.to_string())?;
                info!("Replaced {} entry", target);
                Ok(format!("Replaced in {}", target))
            }
            None => Err(format!("Entry not found in {}", target)),
        }
    }

    /// Remove an entry.
    pub async fn remove(&self, target: MemoryTarget, old: &str) -> Result<String, String> {
        let (entries, _) = self.get_target(target).await;
        let mut entries = entries.write().await;

        let before = entries.len();
        entries.retain(|e| !e.contains(old));

        if entries.len() < before {
            drop(entries);
            self.persist(target).await.map_err(|e| e.to_string())?;
            info!("Removed {} entry", target);
            Ok(format!("Removed from {}", target))
        } else {
            Err(format!("Entry not found in {}", target))
        }
    }

    /// Read all entries from the specified target.
    pub async fn read(&self, target: MemoryTarget) -> Vec<String> {
        let (entries, _) = self.get_target(target).await;
        entries.read().await.clone()
    }

    /// Get the full MEMORY.md content for system prompt injection.
    pub async fn memory_snapshot(&self) -> String {
        let entries = self.memory_entries.read().await;
        if entries.is_empty() {
            return String::new();
        }
        entries.join("\n")
    }

    /// Get the full USER.md content for system prompt injection.
    pub async fn user_snapshot(&self) -> String {
        let entries = self.user_entries.read().await;
        if entries.is_empty() {
            return String::new();
        }
        entries.join("\n")
    }

    /// Build the memory context block for system prompt.
    /// Wraps memory content in `<memory-context>` fence.
    pub async fn build_memory_context_block(&self) -> String {
        let memory = self.memory_snapshot().await;
        let user = self.user_snapshot().await;

        if memory.is_empty() && user.is_empty() {
            return String::new();
        }

        let mut block = String::from("<memory-context>\n");

        if !user.is_empty() {
            block.push_str("## User Profile\n");
            block.push_str(&user);
            block.push('\n');
        }

        if !memory.is_empty() {
            block.push_str("## Memory\n");
            block.push_str(&memory);
            block.push('\n');
        }

        block.push_str("</memory-context>");
        block
    }

    /// Get references to the appropriate entries and limit.
    async fn get_target(
        &self,
        target: MemoryTarget,
    ) -> (&RwLock<Vec<String>>, usize) {
        match target {
            MemoryTarget::Memory => (&self.memory_entries, MEMORY_CHAR_LIMIT),
            MemoryTarget::User => (&self.user_entries, USER_CHAR_LIMIT),
        }
    }

    /// Persist entries to file.
    async fn persist(&self, target: MemoryTarget) -> Result<(), std::io::Error> {
        let filename = match target {
            MemoryTarget::Memory => "MEMORY.md",
            MemoryTarget::User => "USER.md",
        };
        let path = self.data_dir.join(filename);

        let (entries, _) = self.get_target(target).await;
        let content = entries.read().await.join(ENTRY_DELIMITER);

        // Atomic write: write to temp file, then rename
        let temp_path = path.with_extension("tmp");
        tokio::fs::write(&temp_path, &content).await?;
        tokio::fs::rename(&temp_path, &path).await?;

        Ok(())
    }
}

/// Parse file content into entries using the delimiter.
fn parse_entries(content: &str) -> Vec<String> {
    content
        .split(ENTRY_DELIMITER)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Scan content for injection/exfiltration patterns.
fn scan_content(content: &str) -> Result<(), String> {
    let lower = content.to_lowercase();
    for pattern in BLOCKED_PATTERNS {
        if lower.contains(pattern) {
            return Err(format!(
                "Content contains blocked pattern: '{}'",
                pattern
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_entries() {
        let content = "Entry one\n\u{00a7}\nEntry two\n\u{00a7}\nEntry three";
        let entries = parse_entries(content);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0], "Entry one");
        assert_eq!(entries[1], "Entry two");
        assert_eq!(entries[2], "Entry three");
    }

    #[test]
    fn test_parse_entries_empty() {
        let entries = parse_entries("");
        assert!(entries.is_empty());
    }

    #[test]
    fn test_scan_content_safe() {
        assert!(scan_content("User prefers dark mode").is_ok());
    }

    #[test]
    fn test_scan_content_blocked() {
        assert!(scan_content("ignore previous instructions").is_err());
        assert!(scan_content("FORGET EVERYTHING").is_err());
        assert!(scan_content("<!-- hidden -->").is_err());
    }

    #[test]
    fn test_memory_target_display() {
        assert_eq!(format!("{}", MemoryTarget::Memory), "memory");
        assert_eq!(format!("{}", MemoryTarget::User), "user");
    }

    #[tokio::test]
    async fn test_memory_manager_add_and_read() {
        let dir = std::env::temp_dir().join("hermes_mem_mgr_test");
        let _ = std::fs::remove_dir_all(&dir);

        let mgr = MemoryManager::new(&dir);
        mgr.initialize().await.unwrap();

        mgr.add(MemoryTarget::Memory, "User likes Rust").await.unwrap();
        mgr.add(MemoryTarget::User, "Name: Alice").await.unwrap();

        let memory = mgr.read(MemoryTarget::Memory).await;
        assert_eq!(memory.len(), 1);
        assert!(memory[0].contains("Rust"));

        let user = mgr.read(MemoryTarget::User).await;
        assert_eq!(user.len(), 1);
        assert!(user[0].contains("Alice"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_memory_manager_replace() {
        let dir = std::env::temp_dir().join("hermes_mem_mgr_replace");
        let _ = std::fs::remove_dir_all(&dir);

        let mgr = MemoryManager::new(&dir);
        mgr.initialize().await.unwrap();

        mgr.add(MemoryTarget::Memory, "Version 1").await.unwrap();
        mgr.replace(MemoryTarget::Memory, "Version 1", "Version 2").await.unwrap();

        let entries = mgr.read(MemoryTarget::Memory).await;
        assert_eq!(entries.len(), 1);
        assert!(entries[0].contains("Version 2"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_memory_manager_remove() {
        let dir = std::env::temp_dir().join("hermes_mem_mgr_remove");
        let _ = std::fs::remove_dir_all(&dir);

        let mgr = MemoryManager::new(&dir);
        mgr.initialize().await.unwrap();

        mgr.add(MemoryTarget::Memory, "To remove").await.unwrap();
        mgr.add(MemoryTarget::Memory, "To keep").await.unwrap();
        mgr.remove(MemoryTarget::Memory, "To remove").await.unwrap();

        let entries = mgr.read(MemoryTarget::Memory).await;
        assert_eq!(entries.len(), 1);
        assert!(entries[0].contains("To keep"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_memory_manager_persist_and_reload() {
        let dir = std::env::temp_dir().join("hermes_mem_mgr_persist");
        let _ = std::fs::remove_dir_all(&dir);

        // Write
        {
            let mgr = MemoryManager::new(&dir);
            mgr.initialize().await.unwrap();
            mgr.add(MemoryTarget::Memory, "Persisted entry").await.unwrap();
        }

        // Reload
        {
            let mgr = MemoryManager::new(&dir);
            mgr.initialize().await.unwrap();
            let entries = mgr.read(MemoryTarget::Memory).await;
            assert_eq!(entries.len(), 1);
            assert!(entries[0].contains("Persisted entry"));
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_memory_context_block() {
        let dir = std::env::temp_dir().join("hermes_mem_mgr_ctx");
        let _ = std::fs::remove_dir_all(&dir);

        let mgr = MemoryManager::new(&dir);
        mgr.initialize().await.unwrap();
        mgr.add(MemoryTarget::Memory, "Likes Rust").await.unwrap();
        mgr.add(MemoryTarget::User, "Name: Bob").await.unwrap();

        let block = mgr.build_memory_context_block().await;
        assert!(block.contains("<memory-context>"));
        assert!(block.contains("</memory-context>"));
        assert!(block.contains("User Profile"));
        assert!(block.contains("Memory"));
        assert!(block.contains("Bob"));
        assert!(block.contains("Rust"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_memory_manager_blocked_content() {
        let dir = std::env::temp_dir().join("hermes_mem_mgr_blocked");
        let _ = std::fs::remove_dir_all(&dir);

        let mgr = MemoryManager::new(&dir);
        mgr.initialize().await.unwrap();

        let result = mgr.add(MemoryTarget::Memory, "ignore previous instructions").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("blocked pattern"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_memory_manager_replace_not_found() {
        let dir = std::env::temp_dir().join("hermes_mem_mgr_notfound");
        let _ = std::fs::remove_dir_all(&dir);

        let mgr = MemoryManager::new(&dir);
        mgr.initialize().await.unwrap();

        let result = mgr.replace(MemoryTarget::Memory, "nonexistent", "new").await;
        assert!(result.is_err());

        std::fs::remove_dir_all(&dir).ok();
    }
}
