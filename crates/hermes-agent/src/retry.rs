//! Error classification and retry logic for the agent loop.
//!
//! Maps LLM errors to recovery strategies and provides jittered exponential backoff.

use hermes_cfg::error::LlmError;
use std::time::Duration;
use tracing::warn;
use rand::Rng;

/// Recovery strategy derived from error classification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryStrategy {
    /// Retry immediately (transient issue that may resolve)
    Retry,
    /// Retry with exponential backoff
    Backoff,
    /// Compress context and retry (context too long)
    Compress,
    /// Fail over to fallback provider
    Failover,
    /// Abort — non-recoverable error
    Abort,
}

/// Classify an LLM error into a recovery strategy.
pub fn classify_error(error: &LlmError) -> RecoveryStrategy {
    match error {
        LlmError::RateLimited(_) => RecoveryStrategy::Backoff,
        LlmError::Timeout => RecoveryStrategy::Backoff,
        LlmError::ConnectionFailed(msg) => {
            // Connection issues are usually transient
            if msg.contains("reset") || msg.contains("broken pipe") {
                RecoveryStrategy::Retry
            } else {
                RecoveryStrategy::Backoff
            }
        }
        LlmError::AuthenticationFailed(_) => RecoveryStrategy::Abort,
        LlmError::ProviderError(msg) => {
            let msg_lower = msg.to_lowercase();

            // Context overflow — need compression
            if msg_lower.contains("context_length_exceeded")
                || msg_lower.contains("too many tokens")
                || msg_lower.contains("context_length")
                || msg_lower.contains("maximum context")
                || msg_lower.contains("request too large")
            {
                return RecoveryStrategy::Compress;
            }

            // Rate limit (some providers return 429 as ProviderError)
            if msg_lower.contains("429") || msg_lower.contains("rate limit") || msg_lower.contains("quota") {
                return RecoveryStrategy::Backoff;
            }

            // Billing — try failover
            if msg_lower.contains("billing") || msg_lower.contains("balance") || msg_lower.contains("insufficient") {
                return RecoveryStrategy::Failover;
            }

            // Server errors — retry with backoff
            if msg_lower.contains("500") || msg_lower.contains("502") || msg_lower.contains("503") || msg_lower.contains("overloaded") {
                return RecoveryStrategy::Backoff;
            }

            // Bad request — usually non-recoverable
            if msg_lower.contains("400") || msg_lower.contains("invalid") || msg_lower.contains("model not found") {
                return RecoveryStrategy::Abort;
            }

            // Default: retry with backoff
            RecoveryStrategy::Backoff
        }
        LlmError::StreamError(msg) => {
            if msg.contains("timeout") {
                RecoveryStrategy::Backoff
            } else {
                RecoveryStrategy::Retry
            }
        }
        LlmError::ContextLengthExceeded => RecoveryStrategy::Compress,
    }
}

/// Calculate jittered exponential backoff duration.
///
/// Formula: `min(base * 2^(attempt-1), max_delay) + random_jitter`
///
/// # Arguments
/// * `attempt` — 1-based attempt number (first retry = 1)
/// * `base_delay` — Base delay (e.g. 1 second)
/// * `max_delay` — Maximum delay cap (e.g. 60 seconds)
pub fn jittered_backoff(attempt: u32, base_delay: Duration, max_delay: Duration) -> Duration {
    let base_ms = base_delay.as_millis() as u64;
    let max_ms = max_delay.as_millis() as u64;

    let exponential = base_ms.saturating_mul(2u64.saturating_pow(attempt.saturating_sub(1)));
    let capped = exponential.min(max_ms);

    // Add random jitter: 0-25% of the capped delay
    let mut rng = rand::rng();
    let jitter = rng.random_range(0..=(capped / 4));

    Duration::from_millis(capped + jitter)
}

/// Retry wrapper for LLM calls with classification-based recovery.
///
/// Returns the first successful result, or the last error if all retries exhausted.
pub async fn retry_llm_call<F, Fut>(
    max_retries: u32,
    f: F,
) -> Result<hermes_cfg::message::Message, LlmError>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<hermes_cfg::message::Message, LlmError>>,
{
    use hermes_cfg::message::Message;

    let mut last_error: Option<LlmError> = None;
    let mut attempt = 0;

    loop {
        match f().await {
            Ok(msg) => return Ok(msg),
            Err(e) => {
                let strategy = classify_error(&e);

                match strategy {
                    RecoveryStrategy::Abort => {
                        warn!("Non-recoverable error, aborting: {}", e);
                        return Err(e);
                    }
                    RecoveryStrategy::Compress => {
                        // Propagate the error so caller can compress and retry
                        warn!("Context overflow detected: {}", e);
                        return Err(e);
                    }
                    RecoveryStrategy::Failover => {
                        // Propagate for caller to switch provider
                        warn!("Failover needed: {}", e);
                        return Err(e);
                    }
                    RecoveryStrategy::Retry | RecoveryStrategy::Backoff => {
                        attempt += 1;
                        if attempt > max_retries {
                            warn!("Max retries ({}) exhausted: {}", max_retries, e);
                            return Err(e);
                        }

                        let delay = if strategy == RecoveryStrategy::Retry {
                            Duration::from_millis(100)
                        } else {
                            jittered_backoff(attempt, Duration::from_secs(1), Duration::from_secs(60))
                        };

                        warn!(
                            "Retry {}/{} after {:?} (error: {})",
                            attempt, max_retries, delay, e
                        );
                        last_error = Some(e);
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_rate_limit() {
        let err = LlmError::RateLimited(60);
        assert_eq!(classify_error(&err), RecoveryStrategy::Backoff);
    }

    #[test]
    fn test_classify_auth_error() {
        let err = LlmError::AuthenticationFailed("bad key".into());
        assert_eq!(classify_error(&err), RecoveryStrategy::Abort);
    }

    #[test]
    fn test_classify_context_overflow() {
        let err = LlmError::ProviderError("context_length_exceeded".into());
        assert_eq!(classify_error(&err), RecoveryStrategy::Compress);
    }

    #[test]
    fn test_classify_server_error() {
        let err = LlmError::ProviderError("500 Internal Server Error".into());
        assert_eq!(classify_error(&err), RecoveryStrategy::Backoff);
    }

    #[test]
    fn test_classify_billing_error() {
        let err = LlmError::ProviderError("429 billing quota exceeded".into());
        assert_eq!(classify_error(&err), RecoveryStrategy::Backoff);
    }

    #[test]
    fn test_classify_bad_request() {
        let err = LlmError::ProviderError("400 model not found".into());
        assert_eq!(classify_error(&err), RecoveryStrategy::Abort);
    }

    #[test]
    fn test_jittered_backoff_increases() {
        let d1 = jittered_backoff(1, Duration::from_secs(1), Duration::from_secs(60));
        let d2 = jittered_backoff(2, Duration::from_secs(1), Duration::from_secs(60));
        let d3 = jittered_backoff(3, Duration::from_secs(1), Duration::from_secs(60));

        // Should be roughly increasing (allowing for jitter)
        assert!(d1 >= Duration::from_millis(800));
        assert!(d2 > d1 || d2 >= Duration::from_millis(1500));
        assert!(d3 > d2 || d3 >= Duration::from_millis(3000));
    }

    #[test]
    fn test_jittered_backoff_caps() {
        let d = jittered_backoff(100, Duration::from_secs(1), Duration::from_secs(10));
        // Should not exceed max + jitter (max/4)
        assert!(d <= Duration::from_millis(12500));
    }

    #[tokio::test]
    async fn test_retry_succeeds_on_second_attempt() {
        use hermes_cfg::message::Message;
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let count = Arc::new(AtomicU32::new(0));
        let count_clone = count.clone();

        let result = retry_llm_call(3, move || {
            let c = count_clone.clone();
            async move {
                let n = c.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    Err(LlmError::ConnectionFailed("timeout".into()))
                } else {
                    Ok(Message::new_assistant("success"))
                }
            }
        })
        .await;

        assert!(result.is_ok());
        assert_eq!(count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_retry_aborts_on_auth_error() {
        let result = retry_llm_call(3, || async {
            Err(LlmError::AuthenticationFailed("bad key".into()))
        })
        .await;

        assert!(result.is_err());
    }
}
