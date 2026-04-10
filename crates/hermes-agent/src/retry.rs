//! Error classification and retry logic for the agent loop.
//!
//! Maps LLM errors to recovery strategies and provides jittered exponential backoff.

use hermes_cfg::error::LlmError;
use std::time::Duration;
use tracing::warn;
use rand::Rng;

/// Recovery strategy derived from error classification
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
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
        _ => RecoveryStrategy::Backoff, // Unknown error variants
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

/// Generic retry wrapper with classification-based recovery.
///
/// Works for any `Result<T, LlmError>` return type — used by both
/// `retry_llm_call` (Message) and `retry_stream_call` (Stream).
async fn retry_generic<F, Fut, T>(max_retries: u32, f: F) -> Result<T, LlmError>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T, LlmError>>,
{
    let mut _last_error: Option<LlmError> = None;
    let mut attempt = 0;

    loop {
        match f().await {
            Ok(val) => return Ok(val),
            Err(e) => {
                let strategy = classify_error(&e);

                match strategy {
                    RecoveryStrategy::Abort => {
                        warn!("Non-recoverable error, aborting: {}", e);
                        return Err(e);
                    }
                    RecoveryStrategy::Compress => {
                        warn!("Context overflow detected: {}", e);
                        return Err(e);
                    }
                    RecoveryStrategy::Failover => {
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
                        _last_error = Some(e);
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }
    }
}

/// Retry wrapper for LLM complete calls returning `Result<Message, LlmError>`.
pub async fn retry_llm_call<F, Fut>(
    max_retries: u32,
    f: F,
) -> Result<hermes_cfg::message::Message, LlmError>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<hermes_cfg::message::Message, LlmError>>,
{
    retry_generic(max_retries, f).await
}

/// Retry wrapper for LLM stream calls returning `Result<Stream, LlmError>`.
///
/// Same classification-based logic as `retry_llm_call`, but for
/// `complete_stream` which returns a stream object instead of a Message.
pub async fn retry_stream_call<F, Fut, S>(
    max_retries: u32,
    f: F,
) -> Result<S, LlmError>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<S, LlmError>>,
{
    retry_generic(max_retries, f).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use hermes_cfg::traits::StreamEvent;
    use std::pin::Pin;

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

    // ── Additional edge case tests ──────────────────────────────

    #[test]
    fn test_classify_timeout() {
        let err = LlmError::Timeout;
        assert_eq!(classify_error(&err), RecoveryStrategy::Backoff);
    }

    #[test]
    fn test_classify_connection_reset() {
        let err = LlmError::ConnectionFailed("connection reset by peer".into());
        assert_eq!(classify_error(&err), RecoveryStrategy::Retry);
    }

    #[test]
    fn test_classify_connection_broken_pipe() {
        let err = LlmError::ConnectionFailed("broken pipe".into());
        assert_eq!(classify_error(&err), RecoveryStrategy::Retry);
    }

    #[test]
    fn test_classify_connection_other() {
        let err = LlmError::ConnectionFailed("refused".into());
        assert_eq!(classify_error(&err), RecoveryStrategy::Backoff);
    }

    #[test]
    fn test_classify_provider_billing_failover() {
        let err = LlmError::ProviderError("billing issue".into());
        assert_eq!(classify_error(&err), RecoveryStrategy::Failover);
    }

    #[test]
    fn test_classify_provider_balance_failover() {
        let err = LlmError::ProviderError("insufficient balance".into());
        assert_eq!(classify_error(&err), RecoveryStrategy::Failover);
    }

    #[test]
    fn test_classify_provider_502() {
        let err = LlmError::ProviderError("502 Bad Gateway".into());
        assert_eq!(classify_error(&err), RecoveryStrategy::Backoff);
    }

    #[test]
    fn test_classify_provider_503() {
        let err = LlmError::ProviderError("503 Service Unavailable".into());
        assert_eq!(classify_error(&err), RecoveryStrategy::Backoff);
    }

    #[test]
    fn test_classify_provider_overloaded() {
        let err = LlmError::ProviderError("server overloaded".into());
        assert_eq!(classify_error(&err), RecoveryStrategy::Backoff);
    }

    #[test]
    fn test_classify_provider_400() {
        let err = LlmError::ProviderError("400 Bad Request".into());
        assert_eq!(classify_error(&err), RecoveryStrategy::Abort);
    }

    #[test]
    fn test_classify_provider_invalid() {
        let err = LlmError::ProviderError("invalid request body".into());
        assert_eq!(classify_error(&err), RecoveryStrategy::Abort);
    }

    #[test]
    fn test_classify_provider_default_backoff() {
        let err = LlmError::ProviderError("unknown transient issue".into());
        assert_eq!(classify_error(&err), RecoveryStrategy::Backoff);
    }

    #[test]
    fn test_classify_stream_error_timeout() {
        let err = LlmError::StreamError("read timeout".into());
        assert_eq!(classify_error(&err), RecoveryStrategy::Backoff);
    }

    #[test]
    fn test_classify_stream_error_other() {
        let err = LlmError::StreamError("connection lost".into());
        assert_eq!(classify_error(&err), RecoveryStrategy::Retry);
    }

    #[test]
    fn test_classify_context_length_exceeded() {
        let err = LlmError::ContextLengthExceeded;
        assert_eq!(classify_error(&err), RecoveryStrategy::Compress);
    }

    #[test]
    fn test_classify_too_many_tokens_variants() {
        assert_eq!(
            classify_error(&LlmError::ProviderError("too many tokens".into())),
            RecoveryStrategy::Compress
        );
        assert_eq!(
            classify_error(&LlmError::ProviderError("context_length".into())),
            RecoveryStrategy::Compress
        );
        assert_eq!(
            classify_error(&LlmError::ProviderError("maximum context exceeded".into())),
            RecoveryStrategy::Compress
        );
        assert_eq!(
            classify_error(&LlmError::ProviderError("request too large".into())),
            RecoveryStrategy::Compress
        );
    }

    #[test]
    fn test_classify_rate_limit_as_429() {
        let err = LlmError::ProviderError("429 Too Many Requests".into());
        assert_eq!(classify_error(&err), RecoveryStrategy::Backoff);
    }

    #[test]
    fn test_classify_rate_limit_text() {
        let err = LlmError::ProviderError("rate limit exceeded".into());
        assert_eq!(classify_error(&err), RecoveryStrategy::Backoff);
    }

    #[test]
    fn test_classify_quota_text() {
        let err = LlmError::ProviderError("quota exceeded".into());
        assert_eq!(classify_error(&err), RecoveryStrategy::Backoff);
    }

    #[tokio::test]
    async fn test_retry_max_retries_exhausted() {
        let result = retry_llm_call(2, || async {
            Err(LlmError::ConnectionFailed("always fails".into()))
        })
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_retry_compress_strategy_returns_error() {
        let result = retry_llm_call(3, || async {
            Err(LlmError::ContextLengthExceeded)
        })
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_retry_failover_strategy_returns_error() {
        let result = retry_llm_call(3, || async {
            Err(LlmError::ProviderError("billing error".into()))
        })
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_retry_stream_call_succeeds() {
        use futures::stream::{self, Stream};

        let result: Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send>>, LlmError> =
            retry_stream_call(3, || async {
                Ok(Box::pin(stream::iter(vec![Ok(StreamEvent::Delta("hi".into()))])) as Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send>>)
            })
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_retry_stream_call_retries() {
        use futures::stream::{self, Stream};
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let count = Arc::new(AtomicU32::new(0));
        let count_clone = count.clone();

        let result: Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send>>, LlmError> =
            retry_stream_call(3, move || {
                let c = count_clone.clone();
                async move {
                    let n = c.fetch_add(1, Ordering::SeqCst);
                    if n == 0 {
                        Err(LlmError::ConnectionFailed("fail".into()))
                    } else {
                        Ok(Box::pin(stream::iter(vec![Ok(StreamEvent::Done)])) as Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send>>)
                    }
                }
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_jittered_backoff_minimum() {
        let d = jittered_backoff(1, Duration::from_secs(1), Duration::from_secs(60));
        assert!(d >= Duration::from_secs(1));
    }
}
