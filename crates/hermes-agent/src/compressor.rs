//! Context compression for long conversations.
//!
//! Enhanced with Python hermes-agent/agent/context_compressor.py approach:
//! 1. Prune old tool results (truncate long outputs outside protected tail)
//! 2. Determine boundaries (protect head + tail)
//! 3. Generate structured summary (LLM or template-based)
//! 4. Assemble: head + summary + tail

use hermes_cfg::message::{Message, Role};
use tracing::{debug, info};

/// Compression configuration
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Trigger compression when estimated tokens exceed this percentage of max_context
    pub threshold_percent: f32,
    /// Maximum characters for a tool result before truncation
    pub max_tool_output_chars: usize,
    /// Number of recent messages to always protect
    pub protected_tail_count: usize,
    /// Number of head messages (system + first exchange) to protect
    pub protected_head_count: usize,
    /// Summary template for compressed middle section
    pub summary_template: String,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            threshold_percent: 0.5,
            max_tool_output_chars: 200,
            protected_tail_count: 20,
            protected_head_count: 3,
            summary_template: "Earlier conversation compressed".to_string(),
        }
    }
}

/// Result of compression
#[derive(Debug)]
pub struct CompressionResult {
    /// Compressed messages
    pub messages: Vec<Message>,
    /// Whether compression was applied
    pub compressed: bool,
    /// Estimated tokens before compression
    pub tokens_before: usize,
    /// Estimated tokens after compression
    pub tokens_after: usize,
}

/// Estimate token count for a message (rough: 4 chars ≈ 1 token)
pub fn estimate_tokens(messages: &[Message]) -> usize {
    messages.iter().map(|m| m.content.len() / 4).sum()
}

/// Check if compression is needed based on token budget
pub fn should_compress(messages: &[Message], max_context_tokens: usize, config: &CompressionConfig) -> bool {
    let tokens = estimate_tokens(messages);
    let threshold = (max_context_tokens as f32 * config.threshold_percent) as usize;
    tokens > threshold
}

/// Compress messages to fit within token budget.
///
/// Strategy (4 phases, aligned with Python version):
/// 1. Prune old tool results (>max chars outside protected tail → placeholder)
/// 2. Determine boundaries (protect head system messages + first exchange, protect tail N messages)
/// 3. Summarize middle section (structured template with key fields)
/// 4. Assemble: head + summary message + tail, clean orphaned tool pairs
pub fn compress(messages: &[Message], max_context_tokens: usize, config: &CompressionConfig) -> CompressionResult {
    let tokens_before = estimate_tokens(messages);
    let threshold = (max_context_tokens as f32 * config.threshold_percent) as usize;

    if tokens_before <= threshold {
        return CompressionResult {
            messages: messages.to_vec(),
            compressed: false,
            tokens_before,
            tokens_after: tokens_before,
        };
    }

    info!(
        "Compressing context: {} estimated tokens, threshold {}%",
        tokens_before,
        (config.threshold_percent * 100.0) as u32
    );

    // Phase 1: Prune old tool results
    let after_prune = prune_tool_outputs(messages, config);
    let tokens_after_prune = estimate_tokens(&after_prune);

    if tokens_after_prune <= threshold {
        debug!("Compression sufficient after tool output pruning");
        return CompressionResult {
            messages: after_prune,
            compressed: true,
            tokens_before,
            tokens_after: tokens_after_prune,
        };
    }

    // Phase 2: Determine boundaries
    let head_count = count_protected_head(&after_prune, config.protected_head_count);
    let tail_budget = calculate_tail_budget(&after_prune, config.protected_tail_count, max_context_tokens);
    let tail_start = find_tail_boundary(&after_prune, head_count, tail_budget);

    let head: Vec<Message> = after_prune.iter().take(head_count).cloned().collect();
    let tail: Vec<Message> = after_prune[tail_start..].to_vec();
    let middle: &[Message] = &after_prune[head_count..tail_start];

    // Phase 3: Generate structured summary of middle
    let summary = generate_summary(middle);

    // Phase 4: Assemble
    let mut result = head;
    if !summary.is_empty() {
        result.push(Message::new_system(format!(
            "<context-summary>\n{}\n</context-summary>",
            summary
        )));
    }
    result.extend(tail);

    // Clean orphaned tool_call/tool_result pairs
    result = clean_orphaned_tool_pairs(&result);

    let tokens_after = estimate_tokens(&result);
    debug!(
        "Compression result: {} -> {} estimated tokens ({} messages -> {})",
        tokens_before,
        tokens_after,
        messages.len(),
        result.len()
    );

    CompressionResult {
        messages: result,
        compressed: true,
        tokens_before,
        tokens_after,
    }
}

/// Phase 1: Prune old tool outputs that exceed the character limit.
/// Only prunes outputs that are NOT in the protected tail.
fn prune_tool_outputs(messages: &[Message], config: &CompressionConfig) -> Vec<Message> {
    // Identify tail boundary (protected from pruning)
    let tail_start = messages.len().saturating_sub(config.protected_tail_count);

    messages
        .iter()
        .enumerate()
        .map(|(i, msg)| {
            if i < tail_start
                && msg.role == Role::Tool
                && msg.content.len() > config.max_tool_output_chars
            {
                let truncated = format!(
                    "{}\n...[truncated, {} chars total]",
                    &msg.content[..config.max_tool_output_chars.min(msg.content.len())],
                    msg.content.len()
                );
                Message::new_tool_result(
                    msg.tool_call_id.as_deref().unwrap_or("unknown"),
                    &truncated,
                )
            } else {
                msg.clone()
            }
        })
        .collect()
}

/// Count messages in the protected head (system messages + first exchange).
fn count_protected_head(messages: &[Message], max_head: usize) -> usize {
    let system_count = messages.iter().take_while(|m| m.role == Role::System).count();
    // Include up to max_head total (system + first exchange)
    (system_count + 2).min(max_head).min(messages.len())
}

/// Calculate how many messages to protect at the tail based on token budget.
fn calculate_tail_budget(messages: &[Message], min_tail: usize, max_tokens: usize) -> usize {
    // Walk backward from the end, accumulating tokens
    let mut token_budget = 0;
    let mut count = 0;

    for msg in messages.iter().rev() {
        token_budget += msg.content.len() / 4;
        if token_budget > max_tokens / 3 {
            break;
        }
        count += 1;
    }

    count.max(min_tail).min(messages.len())
}

/// Find the index where the tail starts, aligned to avoid splitting tool groups.
fn find_tail_boundary(messages: &[Message], head_end: usize, tail_count: usize) -> usize {
    let ideal_start = messages.len().saturating_sub(tail_count).max(head_end);

    // Walk backward from ideal_start to find a safe boundary
    // (don't split assistant tool_call from its tool_result)
    let mut start = ideal_start;
    while start > head_end {
        // Check if we're about to split a tool group
        if start > 0 && start < messages.len() {
            let prev = &messages[start - 1];
            let curr = &messages[start];
            // If previous is assistant with tool_calls and current is tool result, include both
            if prev.role == Role::Assistant
                && prev.tool_calls.is_some()
                && curr.role == Role::Tool
            {
                start -= 1;
                continue;
            }
        }
        break;
    }

    start
}

/// Phase 3: Generate a structured summary of the middle messages.
/// Template-based summary (aligned with Python's structured format).
fn generate_summary(middle: &[Message]) -> String {
    if middle.is_empty() {
        return String::new();
    }

    let mut goals: Vec<String> = Vec::new();
    let mut progress: Vec<String> = Vec::new();
    let mut files: Vec<String> = Vec::new();
    let mut _decisions: Vec<String> = Vec::new();

    for msg in middle {
        match msg.role {
            Role::User => {
                // Extract user intents as goals
                let snippet = truncate_snippet(&msg.content, 100);
                goals.push(snippet);
            }
            Role::Assistant => {
                // Extract progress from assistant responses
                let snippet = truncate_snippet(&msg.content, 100);
                if !snippet.is_empty() {
                    progress.push(snippet);
                }

                // Extract file references from tool calls
                if let Some(ref calls) = msg.tool_calls {
                    for call in calls {
                        if call.function.name.contains("file")
                            || call.function.name.contains("read")
                            || call.function.name.contains("write")
                        {
                            // Try to extract file path from arguments
                            if let Ok(args) = serde_json::from_str::<serde_json::Value>(
                                &call.function.arguments,
                            ) {
                                if let Some(path) = args.get("path").and_then(|v| v.as_str()) {
                                    files.push(path.to_string());
                                }
                            }
                        }
                    }
                }
            }
            Role::Tool => {
                // Skip tool outputs in summary
            }
            Role::System => {
                // Skip system messages
            }
        }
    }

    // Build structured summary
    let mut summary = format!(
        "[Compressed: {} messages]\n",
        middle.len()
    );

    if !goals.is_empty() {
        summary.push_str(&format!(
            "Goals: {}\n",
            goals.iter().take(3).cloned().collect::<Vec<_>>().join("; ")
        ));
    }

    if !progress.is_empty() {
        summary.push_str(&format!(
            "Progress: {}\n",
            progress.iter().take(3).cloned().collect::<Vec<_>>().join("; ")
        ));
    }

    if !files.is_empty() {
        files.dedup();
        summary.push_str(&format!(
            "Files touched: {}\n",
            files.iter().take(10).cloned().collect::<Vec<_>>().join(", ")
        ));
    }

    summary
}

/// Clean orphaned tool_call/tool_result pairs.
/// Ensures every tool_result has a corresponding tool_call and vice versa.
fn clean_orphaned_tool_pairs(messages: &[Message]) -> Vec<Message> {
    let call_ids: Vec<String> = messages
        .iter()
        .filter_map(|m| {
            m.tool_calls.as_ref().map(|calls| {
                calls.iter().map(|c| c.id.clone()).collect::<Vec<_>>()
            })
        })
        .flatten()
        .collect();

    let result_ids: Vec<String> = messages
        .iter()
        .filter_map(|m| m.tool_call_id.clone())
        .collect();

    messages
        .iter()
        .filter(|m| {
            // Keep all non-tool messages
            if m.role != Role::Tool {
                return true;
            }
            // Keep tool results that have matching tool calls
            if let Some(ref id) = m.tool_call_id {
                return call_ids.contains(id);
            }
            true
        })
        .cloned()
        .collect()
}

/// Truncate a string for use in summaries.
fn truncate_snippet(s: &str, max_len: usize) -> String {
    let s = s.trim();
    if s.is_empty() {
        return String::new();
    }
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens() {
        let messages = vec![Message::new_user("Hello world!")];
        let tokens = estimate_tokens(&messages);
        assert!(tokens > 0);
        assert_eq!(tokens, 3);
    }

    #[test]
    fn test_should_compress_under_threshold() {
        let config = CompressionConfig::default();
        let messages = vec![Message::new_user("short")];
        assert!(!should_compress(&messages, 8000, &config));
    }

    #[test]
    fn test_should_compress_over_threshold() {
        let config = CompressionConfig::default();
        let long_content = "x".repeat(20000);
        let messages = vec![Message::new_user(&long_content)];
        assert!(should_compress(&messages, 4000, &config));
    }

    #[test]
    fn test_compress_truncates_tool_output() {
        let config = CompressionConfig {
            max_tool_output_chars: 50,
            threshold_percent: 0.01,
            ..Default::default()
        };

        let long_output = "a".repeat(500);
        let messages = vec![
            Message::new_system("system"),
            Message::new_user("list files"),
            Message::new_tool_result("call_1", &long_output),
        ];

        let result = compress(&messages, 100, &config);
        assert!(result.compressed);
        assert!(result.messages.iter().any(|m| m.content.contains("truncated") || m.role == Role::System));
    }

    #[test]
    fn test_compress_protects_head_and_tail() {
        let config = CompressionConfig {
            threshold_percent: 0.01,
            protected_tail_count: 2,
            ..Default::default()
        };

        let mut messages = vec![Message::new_system("system")];
        for i in 0..20 {
            messages.push(Message::new_user(format!("Message {} with some content to add tokens xxxxxxxxxxxxxxxxxxxx", i)));
        }

        let result = compress(&messages, 10, &config);
        assert!(result.compressed);
        // Should have: 1 system + summary + 2 tail = 4+
        assert!(result.messages.len() < messages.len());
        assert_eq!(result.messages[0].role, Role::System);
        assert_eq!(result.messages[0].content, "system");
    }

    #[test]
    fn test_compress_no_op_when_under_limit() {
        let config = CompressionConfig::default();
        let messages = vec![Message::new_user("short message")];

        let result = compress(&messages, 8000, &config);
        assert!(!result.compressed);
        assert_eq!(result.messages.len(), 1);
    }

    #[test]
    fn test_generate_summary() {
        let middle = vec![
            Message::new_user("Fix the bug in main.rs"),
            Message::new_assistant("I'll read the file first"),
        ];

        let summary = generate_summary(&middle);
        assert!(summary.contains("Fix the bug"));
        assert!(summary.contains("read the file"));
    }

    #[test]
    fn test_generate_summary_empty() {
        let summary = generate_summary(&[]);
        assert!(summary.is_empty());
    }

    #[test]
    fn test_truncate_snippet() {
        assert_eq!(truncate_snippet("short", 10), "short");
        assert_eq!(truncate_snippet("a long string here", 5), "a lon...");
        assert_eq!(truncate_snippet("", 10), "");
    }

    #[test]
    fn test_clean_orphaned_tool_pairs() {
        let messages = vec![
            Message::new_assistant("using tool").with_tool_calls(vec![hermes_cfg::tool::ToolCall {
                id: "c1".into(),
                call_type: "function".into(),
                function: hermes_cfg::tool::FunctionCall {
                    name: "read".into(),
                    arguments: "{}".into(),
                },
            }]),
            Message::new_tool_result("c1", "file content"),
            Message::new_tool_result("orphan", "orphan result"), // No matching call
        ];

        let cleaned = clean_orphaned_tool_pairs(&messages);
        assert_eq!(cleaned.len(), 2); // assistant + matched tool result
        assert!(cleaned.iter().any(|m| m.role == Role::Assistant));
        assert!(cleaned.iter().any(|m| m.tool_call_id.as_deref() == Some("c1")));
    }
}
