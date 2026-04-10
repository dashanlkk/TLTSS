//! Context file hierarchy discovery and loading.
//!
//! Ported from Python hermes-agent `agent/prompt_builder.py`:
//! Discovers and loads project context files with priority:
//! .hermes.md (walk to git root) > AGENTS.md (cwd) > CLAUDE.md (cwd) > .cursorrules + .cursor/rules/*.mdc (cwd)
//!
//! Each file is capped at `MAX_CONTEXT_FILE_CHARS` (20K) with head/tail truncation.

use std::path::{Path, PathBuf};
use tracing::debug;

/// Maximum characters per context file (20K, aligned with Python version)
pub const MAX_CONTEXT_FILE_CHARS: usize = 20_000;

/// Context file candidates in priority order (highest priority first)
const CONTEXT_FILES: &[&str] = &[
    ".hermes.md",
    "AGENTS.md",
    "CLAUDE.md",
    ".cursorrules",
];

/// A discovered context file
#[derive(Debug, Clone)]
pub struct ContextFile {
    pub path: PathBuf,
    pub content: String,
    pub source: String,
}

/// Discover and load context files from the working directory.
///
/// Priority order:
/// 1. `.hermes.md` — walk up to git root, use highest found
/// 2. `AGENTS.md` — current directory only
/// 3. `CLAUDE.md` — current directory only
/// 4. `.cursorrules` — current directory only
/// 5. `.cursor/rules/*.mdc` — current directory only
///
/// Returns discovered context files, each capped at MAX_CONTEXT_FILE_CHARS.
pub fn discover_context_files(work_dir: &Path) -> Vec<ContextFile> {
    let mut results = Vec::new();
    let mut seen_contents: Vec<String> = Vec::new();

    // 1. .hermes.md — walk up to git root
    if let Some(content) = find_hermes_md(work_dir) {
        let dedup_key = content.content_hash();
        seen_contents.push(dedup_key);
        results.push(content);
    }

    // 2-4. Fixed location files in cwd
    for filename in &CONTEXT_FILES[1..] {
        let path = work_dir.join(filename);
        if let Some(content) = load_and_cap(&path) {
            let hash = content.content_hash();
            if !seen_contents.contains(&hash) {
                seen_contents.push(hash);
                results.push(content);
            }
        }
    }

    // 5. .cursor/rules/*.mdc
    let cursor_rules_dir = work_dir.join(".cursor").join("rules");
    if cursor_rules_dir.is_dir() {
        if let Ok(entries) = std::fs::read_dir(&cursor_rules_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "mdc").unwrap_or(false) {
                    if let Some(content) = load_and_cap(&path) {
                        let hash = content.content_hash();
                        if !seen_contents.contains(&hash) {
                            seen_contents.push(hash);
                            results.push(content);
                        }
                    }
                }
            }
        }
    }

    if !results.is_empty() {
        debug!("Discovered {} context files", results.len());
    }

    results
}

/// Walk up from work_dir to find `.hermes.md`, stopping at git root.
fn find_hermes_md(work_dir: &Path) -> Option<ContextFile> {
    let mut dir = work_dir.to_path_buf();

    loop {
        let candidate = dir.join(".hermes.md");
        if candidate.is_file() {
            return load_and_cap(&candidate);
        }

        // Stop at git root (directory containing .git)
        if dir.join(".git").exists() {
            break;
        }

        // Walk up
        match dir.parent() {
            Some(parent) => dir = parent.to_path_buf(),
            None => break,
        }
    }

    None
}

/// Load a file and cap at MAX_CONTEXT_FILE_CHARS.
fn load_and_cap(path: &Path) -> Option<ContextFile> {
    let content = std::fs::read_to_string(path).ok()?;

    if content.trim().is_empty() {
        return None;
    }

    let source = path.file_name()?.to_string_lossy().to_string();
    let capped = cap_content(&content, MAX_CONTEXT_FILE_CHARS);

    Some(ContextFile {
        path: path.to_path_buf(),
        content: capped,
        source,
    })
}

/// Truncate content to max_chars, keeping head and tail.
fn cap_content(content: &str, max_chars: usize) -> String {
    if content.len() <= max_chars {
        return content.to_string();
    }

    let head_size = max_chars / 2;
    let tail_size = max_chars / 2;

    let head = &content[..head_size];
    // Find a safe UTF-8 boundary for tail
    let tail_start = content.len() - tail_size;
    let tail_start = content.floor_char_boundary(tail_start);
    let tail = &content[tail_start..];

    format!(
        "{}\n\n... [truncated, showing first {} and last {} of {} chars] ...\n\n{}",
        head,
        head_size,
        content.len() - tail_start,
        content.len(),
        tail,
    )
}

/// Format context files for injection into the system prompt.
pub fn format_context_block(files: &[ContextFile]) -> String {
    if files.is_empty() {
        return String::new();
    }

    let mut block = String::from("<project-context>\n");
    for file in files {
        block.push_str(&format!("--- {} ---\n{}\n\n", file.source, file.content));
    }
    block.push_str("</project-context>");
    block
}

trait ContentHash {
    fn content_hash(&self) -> String;
}

impl ContentHash for ContextFile {
    fn content_hash(&self) -> String {
        // Simple dedup: first 100 chars of trimmed content
        let trimmed = self.content.trim();
        trimmed.chars().take(100).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn test_cap_content_under_limit() {
        let content = "Hello world";
        let capped = cap_content(content, 100);
        assert_eq!(capped, content);
    }

    #[test]
    fn test_cap_content_over_limit() {
        let content = "x".repeat(1000);
        let capped = cap_content(&content, 100);
        assert!(capped.contains("truncated"));
        assert!(capped.len() < 1200); // Some overhead for the truncation message
    }

    #[test]
    fn test_cap_content_empty() {
        let capped = cap_content("", 100);
        assert!(capped.is_empty());
    }

    #[test]
    fn test_discover_no_files() {
        let dir = temp_dir();
        let results = discover_context_files(dir.path());
        assert!(results.is_empty());
    }

    #[test]
    fn test_discover_claude_md() {
        let dir = temp_dir();
        std::fs::write(dir.path().join("CLAUDE.md"), "# Project rules\nUse Rust.").unwrap();

        let results = discover_context_files(dir.path());
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source, "CLAUDE.md");
        assert!(results[0].content.contains("Use Rust"));
    }

    #[test]
    fn test_discover_agents_md() {
        let dir = temp_dir();
        std::fs::write(dir.path().join("AGENTS.md"), "# Agent instructions").unwrap();

        let results = discover_context_files(dir.path());
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source, "AGENTS.md");
    }

    #[test]
    fn test_discover_cursorrules() {
        let dir = temp_dir();
        std::fs::write(dir.path().join(".cursorrules"), "Always use tabs").unwrap();

        let results = discover_context_files(dir.path());
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source, ".cursorrules");
    }

    #[test]
    fn test_discover_cursor_rules_mdc() {
        let dir = temp_dir();
        let cursor_dir = dir.path().join(".cursor").join("rules");
        std::fs::create_dir_all(&cursor_dir).unwrap();
        std::fs::write(cursor_dir.join("rust.mdc"), "Rust rules here").unwrap();

        let results = discover_context_files(dir.path());
        assert_eq!(results.len(), 1);
        assert!(results[0].source.contains("rust.mdc"));
    }

    #[test]
    fn test_discover_priority_order() {
        let dir = temp_dir();
        std::fs::write(dir.path().join("AGENTS.md"), "Agents content").unwrap();
        std::fs::write(dir.path().join("CLAUDE.md"), "Claude content").unwrap();

        let results = discover_context_files(dir.path());
        assert_eq!(results.len(), 2);
        // AGENTS.md should come first (higher priority)
        assert_eq!(results[0].source, "AGENTS.md");
        assert_eq!(results[1].source, "CLAUDE.md");
    }

    #[test]
    fn test_discover_deduplication() {
        let dir = temp_dir();
        // Same content in two files — only first should be kept
        let content = "Same content here";
        std::fs::write(dir.path().join("AGENTS.md"), content).unwrap();
        std::fs::write(dir.path().join("CLAUDE.md"), content).unwrap();

        let results = discover_context_files(dir.path());
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_discover_hermes_md_walking() {
        let dir = temp_dir();
        let subdir = dir.path().join("src").join("module");
        std::fs::create_dir_all(&subdir).unwrap();
        // .hermes.md at root level
        std::fs::write(dir.path().join(".hermes.md"), "# Hermes config").unwrap();
        // .git at root to stop walking
        std::fs::create_dir(dir.path().join(".git")).unwrap();

        let results = discover_context_files(&subdir);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source, ".hermes.md");
    }

    #[test]
    fn test_discover_empty_file_ignored() {
        let dir = temp_dir();
        std::fs::write(dir.path().join("CLAUDE.md"), "   \n  \n").unwrap();

        let results = discover_context_files(dir.path());
        assert!(results.is_empty());
    }

    #[test]
    fn test_format_context_block_empty() {
        assert!(format_context_block(&[]).is_empty());
    }

    #[test]
    fn test_format_context_block_content() {
        let files = vec![ContextFile {
            path: PathBuf::from("CLAUDE.md"),
            content: "Use Rust".to_string(),
            source: "CLAUDE.md".to_string(),
        }];

        let block = format_context_block(&files);
        assert!(block.starts_with("<project-context>"));
        assert!(block.ends_with("</project-context>"));
        assert!(block.contains("Use Rust"));
    }

    #[test]
    fn test_large_file_truncated() {
        let dir = temp_dir();
        let content = "x".repeat(50_000);
        std::fs::write(dir.path().join("CLAUDE.md"), &content).unwrap();

        let results = discover_context_files(dir.path());
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("truncated"));
        assert!(results[0].content.len() < 30_000);
    }
}
