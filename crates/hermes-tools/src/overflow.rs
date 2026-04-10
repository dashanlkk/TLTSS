//! Tool result overflow protection.
//!
//! Ported from Python hermes-agent `tools/tool_result_storage.py`:
//! Three-layer defense against context overflow:
//! 1. Per-tool output cap (default: 50K chars)
//! 2. Oversized results replaced with a truncated preview + note
//! 3. Per-turn aggregate budget (default: 200K chars) that truncates
//!    the largest results first

/// Default per-tool output character limit
pub const DEFAULT_PER_TOOL_LIMIT: usize = 50_000;

/// Default per-turn aggregate character budget
pub const DEFAULT_TURN_BUDGET: usize = 200_000;

/// Preview length shown when truncating
const PREVIEW_LENGTH: usize = 500;

/// Result of applying overflow protection to a single tool output.
#[derive(Debug, Clone)]
pub struct ProtectedOutput {
    /// The output (possibly truncated)
    pub content: String,
    /// Whether truncation was applied
    pub was_truncated: bool,
    /// Original length before truncation
    pub original_len: usize,
}

/// Apply per-tool output cap to a single tool result.
///
/// If the output exceeds `limit`, it is replaced with a truncated preview
/// and a note showing the original size.
pub fn cap_single_output(content: &str, limit: usize) -> ProtectedOutput {
    let original_len = content.len();
    if original_len <= limit {
        return ProtectedOutput {
            content: content.to_string(),
            was_truncated: false,
            original_len,
        };
    }

    let preview = if content.len() > PREVIEW_LENGTH {
        format!("{}...", &content[..PREVIEW_LENGTH])
    } else {
        content.to_string()
    };

    ProtectedOutput {
        content: format!(
            "{}\n\n[Output truncated: {} chars total, showing first {} chars]",
            preview,
            original_len,
            PREVIEW_LENGTH.min(original_len),
        ),
        was_truncated: true,
        original_len,
    }
}

/// Apply per-turn aggregate budget across multiple tool outputs.
///
/// If total exceeds `budget`, truncates the largest outputs first
/// until the budget is met. Returns the (possibly modified) outputs.
pub fn apply_turn_budget(
    outputs: &mut [(String, String)], // (tool_name, content)
    budget: usize,
) {
    let total: usize = outputs.iter().map(|(_, c)| c.len()).sum();
    if total <= budget {
        return;
    }

    // Sort indices by content length (descending) — truncate largest first
    let mut indices: Vec<usize> = (0..outputs.len()).collect();
    indices.sort_by(|a, b| outputs[*b].1.len().cmp(&outputs[*a].1.len()));

    let mut remaining_budget = budget;
    let mut overflow = total.saturating_sub(budget);

    for idx in &indices {
        if overflow == 0 {
            break;
        }
        let content_len = outputs[*idx].1.len();
        let over = content_len.saturating_sub(remaining_budget);
        if over > 0 && over < overflow {
            // Can partially truncate this one
            let target_len = content_len.saturating_sub(overflow);
            outputs[*idx].1 = format!(
                "{}...\n\n[Output truncated: original {} chars, showing first {} chars]",
                &outputs[*idx].1[..target_len.min(PREVIEW_LENGTH).min(outputs[*idx].1.len())],
                content_len,
                target_len.min(PREVIEW_LENGTH),
            );
            overflow = 0;
        } else if over >= overflow {
            // Truncate this output to make room
            let target_len = content_len.saturating_sub(overflow);
            if target_len < PREVIEW_LENGTH {
                let preview = if content_len > PREVIEW_LENGTH {
                    &outputs[*idx].1[..PREVIEW_LENGTH]
                } else {
                    &outputs[*idx].1
                };
                outputs[*idx].1 = format!(
                    "{}...\n\n[Output truncated: original {} chars]",
                    preview,
                    content_len,
                );
            } else {
                outputs[*idx].1 = format!(
                    "{}...\n\n[Output truncated: original {} chars, showing first {} chars]",
                    &outputs[*idx].1[..PREVIEW_LENGTH],
                    content_len,
                    PREVIEW_LENGTH,
                );
            }
            overflow = 0;
        }
        remaining_budget = remaining_budget.saturating_sub(outputs[*idx].1.len().min(content_len));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cap_single_output_under_limit() {
        let result = cap_single_output("hello world", 100);
        assert!(!result.was_truncated);
        assert_eq!(result.content, "hello world");
        assert_eq!(result.original_len, 11);
    }

    #[test]
    fn test_cap_single_output_over_limit() {
        let content = "x".repeat(1000);
        let result = cap_single_output(&content, 100);
        assert!(result.was_truncated);
        assert!(result.content.contains("truncated"));
        assert!(result.content.contains("1000 chars"));
        assert!(result.content.len() < 1000);
        assert_eq!(result.original_len, 1000);
    }

    #[test]
    fn test_cap_single_output_empty() {
        let result = cap_single_output("", 100);
        assert!(!result.was_truncated);
        assert_eq!(result.content, "");
    }

    #[test]
    fn test_cap_single_output_exact_limit() {
        let content = "x".repeat(100);
        let result = cap_single_output(&content, 100);
        assert!(!result.was_truncated);
    }

    #[test]
    fn test_apply_turn_budget_under_budget() {
        let mut outputs = vec![
            ("tool_a".to_string(), "short".to_string()),
            ("tool_b".to_string(), "also short".to_string()),
        ];
        apply_turn_budget(&mut outputs, 1000);
        assert_eq!(outputs[0].1, "short");
        assert_eq!(outputs[1].1, "also short");
    }

    #[test]
    fn test_apply_turn_budget_over_budget() {
        let mut outputs = vec![
            ("small".to_string(), "x".repeat(100)),
            ("large".to_string(), "y".repeat(5000)),
        ];
        apply_turn_budget(&mut outputs, 200);
        // The large one should be truncated
        assert!(outputs[1].1.contains("truncated"));
    }

    #[test]
    fn test_apply_turn_budget_empty() {
        let mut outputs: Vec<(String, String)> = vec![];
        apply_turn_budget(&mut outputs, 100);
        assert!(outputs.is_empty());
    }

    #[test]
    fn test_cap_preview_length() {
        let content = "a".repeat(5000);
        let result = cap_single_output(&content, 100);
        // Preview should be at most PREVIEW_LENGTH chars before the suffix
        let before_suffix = result.content.split("\n\n[").next().unwrap();
        assert!(before_suffix.len() <= PREVIEW_LENGTH + 3); // +3 for "..."
    }
}
