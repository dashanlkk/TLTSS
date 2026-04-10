//! Parallel tool execution safety tiers.
//!
//! Ported from Python hermes-agent `run_agent.py`:
//! Three-tier classification for concurrent tool execution:
//! 1. NEVER_PARALLEL — interactive/user-facing tools that must run sequentially
//! 2. PARALLEL_SAFE — read-only tools with no shared mutable state
//! 3. PATH_SCOPED — file tools that can run concurrently when targeting independent paths
//!
//! Default: sequential (unsafe to parallelize)

use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Tools that must never run concurrently (interactive / user-facing).
/// When any of these appear in a batch, the entire batch runs sequentially.
const NEVER_PARALLEL_TOOLS: &[&str] = &["clarify"];

/// Read-only tools with no shared mutable session state.
/// These can always run concurrently.
const PARALLEL_SAFE_TOOLS: &[&str] = &[
    "read_file",
    "search_files",
    "list_dir",
    "web_fetch",
    "web_search",
    "web_extract",
    "session_search",
    "skill_view",
    "skills_list",
    "vision_analyze",
];

/// File tools that can run concurrently when targeting independent paths.
const PATH_SCOPED_TOOLS: &[&str] = &["read_file", "write_file", "patch"];

/// Maximum number of concurrent workers for parallel tool execution.
pub const MAX_TOOL_WORKERS: usize = 8;

/// A single tool call to evaluate for parallel safety.
#[derive(Debug, Clone)]
pub struct ToolCallInfo {
    pub name: String,
    pub arguments: String,
}

/// Determine whether a batch of tool calls is safe to run concurrently.
///
/// Decision flow (aligned with Python `_should_parallelize_tool_batch`):
/// 1. If batch has <= 1 tool → sequential (no benefit from parallel)
/// 2. If any tool is in NEVER_PARALLEL → sequential
/// 3. For each tool:
///    - If PATH_SCOPED: extract path, check overlap with previous paths → sequential if overlap
///    - If not PARALLEL_SAFE and not PATH_SCOPED → sequential
/// 4. If all checks pass → parallel
pub fn should_parallelize(calls: &[ToolCallInfo]) -> bool {
    if calls.len() <= 1 {
        return false;
    }

    let never_set: HashSet<&str> = NEVER_PARALLEL_TOOLS.iter().copied().collect();
    let safe_set: HashSet<&str> = PARALLEL_SAFE_TOOLS.iter().copied().collect();
    let scoped_set: HashSet<&str> = PATH_SCOPED_TOOLS.iter().copied().collect();

    // Check for never-parallel tools
    for call in calls {
        if never_set.contains(call.name.as_str()) {
            return false;
        }
    }

    // Check each tool's safety
    let mut reserved_paths: Vec<PathBuf> = Vec::new();

    for call in calls {
        let name = call.name.as_str();

        // Parse arguments
        let args: serde_json::Value = match serde_json::from_str::<serde_json::Value>(&call.arguments) {
            Ok(v) if v.is_object() => v,
            _ => return false, // Non-dict or parse error → sequential
        };

        // Path-scoped tools: check for path overlap
        if scoped_set.contains(name) {
            let scoped_path = extract_path(name, &args);
            match scoped_path {
                Some(path) => {
                    if reserved_paths.iter().any(|p| paths_overlap(p, &path)) {
                        return false;
                    }
                    reserved_paths.push(path);
                    continue;
                }
                None => return false, // Could not extract path → sequential
            }
        }

        // Non-safe, non-scoped tools → sequential
        if !safe_set.contains(name) {
            return false;
        }
    }

    true
}

/// Extract the normalized file target for path-scoped tools.
fn extract_path(tool_name: &str, args: &serde_json::Value) -> Option<PathBuf> {
    if !PATH_SCOPED_TOOLS.contains(&tool_name) {
        return None;
    }

    let raw_path = args.get("path").and_then(|v| v.as_str())?;
    let raw_path = raw_path.trim();
    if raw_path.is_empty() {
        return None;
    }

    // Expand ~ to home directory
    let expanded = if raw_path.starts_with("~/") || raw_path == "~" {
        let home = dirs::home_dir()?;
        if raw_path == "~" {
            home
        } else {
            home.join(&raw_path[2..])
        }
    } else {
        PathBuf::from(raw_path)
    };

    Some(if expanded.is_absolute() {
        expanded
    } else {
        std::env::current_dir().ok()?.join(expanded)
    })
}

/// Check if two paths may refer to the same subtree.
/// Two paths overlap if one is a prefix of the other.
fn paths_overlap(left: &Path, right: &Path) -> bool {
    let left_parts: Vec<_> = left.components().collect();
    let right_parts: Vec<_> = right.components().collect();

    if left_parts.is_empty() || right_parts.is_empty() {
        return false;
    }

    let common_len = left_parts.len().min(right_parts.len());
    left_parts[..common_len] == right_parts[..common_len]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_call(name: &str, args: &str) -> ToolCallInfo {
        ToolCallInfo {
            name: name.to_string(),
            arguments: args.to_string(),
        }
    }

    #[test]
    fn test_single_tool_not_parallel() {
        let calls = vec![make_call("read_file", r#"{"path":"a.txt"}"#)];
        assert!(!should_parallelize(&calls));
    }

    #[test]
    fn test_empty_batch_not_parallel() {
        assert!(!should_parallelize(&[]));
    }

    #[test]
    fn test_never_parallel_forces_sequential() {
        let calls = vec![
            make_call("read_file", r#"{"path":"a.txt"}"#),
            make_call("clarify", r#"{"question":"?"}"#),
        ];
        assert!(!should_parallelize(&calls));
    }

    #[test]
    fn test_two_safe_tools_parallel() {
        let calls = vec![
            make_call("read_file", r#"{"path":"a.txt"}"#),
            make_call("search_files", r#"{"pattern":"TODO"}"#),
        ];
        assert!(should_parallelize(&calls));
    }

    #[test]
    fn test_unknown_tool_forces_sequential() {
        let calls = vec![
            make_call("read_file", r#"{"path":"a.txt"}"#),
            make_call("execute_command", r#"{"command":"ls"}"#),
        ];
        assert!(!should_parallelize(&calls));
    }

    #[test]
    fn test_path_scoped_independent_paths_parallel() {
        let calls = vec![
            make_call("read_file", r#"{"path":"a.txt"}"#),
            make_call("write_file", r#"{"path":"b.txt","content":"x"}"#),
        ];
        assert!(should_parallelize(&calls));
    }

    #[test]
    fn test_path_scoped_overlapping_paths_sequential() {
        let calls = vec![
            make_call("read_file", r#"{"path":"src/main.rs"}"#),
            make_call("write_file", r#"{"path":"src/main.rs","content":"x"}"#),
        ];
        assert!(!should_parallelize(&calls));
    }

    #[test]
    fn test_path_scoped_parent_child_overlap() {
        let calls = vec![
            make_call("read_file", r#"{"path":"src"}"#),
            make_call("write_file", r#"{"path":"src/main.rs","content":"x"}"#),
        ];
        assert!(!should_parallelize(&calls));
    }

    #[test]
    fn test_path_scoped_missing_path_sequential() {
        let calls = vec![
            make_call("read_file", r#"{}"#),
            make_call("read_file", r#"{"path":"b.txt"}"#),
        ];
        assert!(!should_parallelize(&calls));
    }

    #[test]
    fn test_invalid_json_sequential() {
        let calls = vec![
            make_call("read_file", r#"not json"#),
            make_call("search_files", r#"{"pattern":"x"}"#),
        ];
        assert!(!should_parallelize(&calls));
    }

    #[test]
    fn test_multiple_safe_tools_parallel() {
        let calls = vec![
            make_call("read_file", r#"{"path":"a.txt"}"#),
            make_call("search_files", r#"{"pattern":"TODO"}"#),
            make_call("list_dir", r#"{"path":"."}"#),
        ];
        assert!(should_parallelize(&calls));
    }

    #[test]
    fn test_paths_overlap_same_path() {
        let cwd = std::env::current_dir().unwrap();
        let a = cwd.join("foo.txt");
        let b = cwd.join("foo.txt");
        assert!(paths_overlap(&a, &b));
    }

    #[test]
    fn test_paths_overlap_different() {
        let cwd = std::env::current_dir().unwrap();
        let a = cwd.join("foo.txt");
        let b = cwd.join("bar.txt");
        assert!(!paths_overlap(&a, &b));
    }

    #[test]
    fn test_paths_overlap_parent_child() {
        let cwd = std::env::current_dir().unwrap();
        let parent = cwd.join("src");
        let child = cwd.join("src").join("main.rs");
        assert!(paths_overlap(&parent, &child));
    }

    #[test]
    fn test_extract_path_valid() {
        let args = serde_json::json!({"path": "test.txt"});
        let path = extract_path("read_file", &args).unwrap();
        assert!(path.to_string_lossy().contains("test.txt"));
    }

    #[test]
    fn test_extract_path_missing() {
        let args = serde_json::json!({});
        assert!(extract_path("read_file", &args).is_none());
    }

    #[test]
    fn test_extract_path_empty_string() {
        let args = serde_json::json!({"path": ""});
        assert!(extract_path("read_file", &args).is_none());
    }

    #[test]
    fn test_extract_path_non_scoped_tool() {
        let args = serde_json::json!({"path": "test.txt"});
        assert!(extract_path("search_files", &args).is_none());
    }
}
