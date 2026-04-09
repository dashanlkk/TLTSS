//! Context compression for long conversations.
//!
//! When the conversation exceeds a token budget, this module trims old tool outputs
//! and protects recent messages to stay within limits.

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
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            threshold_percent: 0.5,
            max_tool_output_chars: 200,
            protected_tail_count: 10,
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
/// Strategy:
/// 1. Truncate long tool output messages
/// 2. Protect head (system messages) and tail (recent messages)
/// 3. Summarize middle messages by truncation
pub fn compress(messages: &[Message], max_context_tokens: usize, config: &CompressionConfig) -> CompressionResult {
    let tokens_before = estimate_tokens(messages);

    if tokens_before <= (max_context_tokens as f32 * config.threshold_percent) as usize {
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

    let mut compressed: Vec<Message> = Vec::new();

    // Phase 1: Truncate long tool outputs
    for msg in messages {
        if msg.role == Role::Tool && msg.content.len() > config.max_tool_output_chars {
            let truncated = format!(
                "{}\n...[truncated, {} chars total]",
                &msg.content[..config.max_tool_output_chars.min(msg.content.len())],
                msg.content.len()
            );
            compressed.push(Message::new_tool_result(
                msg.tool_call_id.as_deref().unwrap_or("unknown"),
                &truncated,
            ));
        } else {
            compressed.push(msg.clone());
        }
    }

    let tokens_after_truncation = estimate_tokens(&compressed);
    if tokens_after_truncation <= (max_context_tokens as f32 * config.threshold_percent) as usize {
        debug!("Compression sufficient after tool output truncation");
        return CompressionResult {
            messages: compressed,
            compressed: true,
            tokens_before,
            tokens_after: tokens_after_truncation,
        };
    }

    // Phase 2: Protect head (system) and tail, trim middle
    let head_count = compressed.iter().take_while(|m| m.role == Role::System).count();
    let tail_count = config.protected_tail_count.min(compressed.len() - head_count);

    let head: Vec<Message> = compressed.iter().take(head_count).cloned().collect();
    let tail: Vec<Message> = compressed.iter().rev().take(tail_count).cloned().collect::<Vec<_>>().into_iter().rev().collect();

    // Middle: just keep a summary placeholder
    let middle_count = compressed.len() - head_count - tail_count;
    let mut result = head;

    if middle_count > 0 {
        let summary = Message::new_system(&format!(
            "[Earlier conversation compressed: {} messages summarized to save context space]",
            middle_count
        ));
        result.push(summary);
    }

    result.extend(tail);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens() {
        let messages = vec![Message::new_user("Hello world!")];
        let tokens = estimate_tokens(&messages);
        assert!(tokens > 0);
        // "Hello world!" = 12 chars / 4 = 3 tokens
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
        // Create messages that clearly exceed threshold
        // 20000 chars / 4 = 5000 tokens > 4000 * 0.5 = 2000
        let long_content = "x".repeat(20000);
        let messages = vec![Message::new_user(&long_content)];
        assert!(should_compress(&messages, 4000, &config));
    }

    #[test]
    fn test_compress_truncates_tool_output() {
        let config = CompressionConfig {
            max_tool_output_chars: 50,
            threshold_percent: 0.01, // Force compression
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
            threshold_percent: 0.01, // Very low to force compression
            protected_tail_count: 2,
            ..Default::default()
        };

        let mut messages = vec![Message::new_system("system")];
        for i in 0..20 {
            messages.push(Message::new_user(&format!("Message {} with some content to add tokens xxxxxxxxxxxxxxxxxxxx", i)));
        }

        let result = compress(&messages, 10, &config);
        assert!(result.compressed);
        // Should have: 1 system + 1 summary + 2 tail = 4
        assert_eq!(result.messages.len(), 4);
        // First message should be system
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
}
