//! Fallback client with smart error-based switching.
//!
//! Routes to the fallback provider only when the error is transient
//! (rate limit, server error, connection). Auth errors and bad requests
//! are not retried on the fallback.

use async_trait::async_trait;
use futures::Stream;
use hermes_cfg::prelude::*;
use hermes_cfg::traits::{LlmClient, StreamEvent};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, warn};

/// Fallback client: switches to backup provider on transient failures.
///
/// Error-based switching logic:
/// - **Switch**: rate limit, server error (5xx), connection failure, timeout
/// - **Don't switch**: auth failure (bad key), bad request (400), context overflow
pub struct FallbackClient {
    primary: Arc<dyn LlmClient>,
    fallback: Arc<dyn LlmClient>,
}

impl FallbackClient {
    pub fn new(primary: Arc<dyn LlmClient>, fallback: Arc<dyn LlmClient>) -> Self {
        Self { primary, fallback }
    }

    /// Whether the error is worth trying on the fallback provider.
    fn should_failover(error: &LlmError) -> bool {
        match error {
            LlmError::RateLimited(_) => true,
            LlmError::Timeout => true,
            LlmError::ConnectionFailed(_) => true,
            LlmError::AuthenticationFailed(_) => false,
            LlmError::ContextLengthExceeded => false,
            LlmError::StreamError(msg) => !msg.contains("400") && !msg.contains("auth"),
            LlmError::ProviderError(msg) => {
                let lower = msg.to_lowercase();
                // Transient: server errors, rate limits
                if lower.contains("429")
                    || lower.contains("500")
                    || lower.contains("502")
                    || lower.contains("503")
                    || lower.contains("overloaded")
                {
                    return true;
                }
                // Permanent: don't switch
                if lower.contains("401")
                    || lower.contains("403")
                    || lower.contains("invalid")
                    || lower.contains("context_length")
                    || lower.contains("model not found")
                {
                    return false;
                }
                // Default: try fallback
                true
            }
            _ => false,
        }
    }
}

#[async_trait]
impl LlmClient for FallbackClient {
    async fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<Message, LlmError> {
        match self.primary.complete(messages, tools).await {
            Ok(msg) => {
                info!("primary provider succeeded");
                Ok(msg)
            }
            Err(e) => {
                if Self::should_failover(&e) {
                    warn!(
                        "primary failed with failover-eligible error ({}), switching to fallback",
                        e
                    );
                    self.fallback.complete(messages, tools).await
                } else {
                    warn!("primary failed with non-failover error ({}), not switching", e);
                    Err(e)
                }
            }
        }
    }

    async fn complete_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send>>, LlmError>
    {
        match self.primary.complete_stream(messages, tools).await {
            Ok(s) => {
                info!("primary stream succeeded");
                Ok(s)
            }
            Err(e) => {
                if Self::should_failover(&e) {
                    warn!(
                        "primary stream failed with failover-eligible error ({}), switching",
                        e
                    );
                    self.fallback.complete_stream(messages, tools).await
                } else {
                    Err(e)
                }
            }
        }
    }

    async fn ping(&self) -> Result<Duration, LlmError> {
        match self.primary.ping().await {
            Ok(d) => Ok(d),
            Err(_) => self.fallback.ping().await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// A minimal mock that always returns a fixed message or error
    struct MockClient {
        response: std::sync::Mutex<Option<Result<Message, LlmError>>>,
        called: AtomicBool,
    }

    impl MockClient {
        fn ok(content: &str) -> Arc<Self> {
            Arc::new(Self {
                response: std::sync::Mutex::new(Some(Ok(Message::new_assistant(content)))),
                called: AtomicBool::new(false),
            })
        }
        fn err(error: LlmError) -> Arc<Self> {
            Arc::new(Self {
                response: std::sync::Mutex::new(Some(Err(error))),
                called: AtomicBool::new(false),
            })
        }
        fn was_called(&self) -> bool {
            self.called.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl LlmClient for MockClient {
        async fn complete(
            &self,
            _messages: &[Message],
            _tools: &[ToolDefinition],
        ) -> Result<Message, LlmError> {
            self.called.store(true, Ordering::SeqCst);
            let mut guard = self.response.lock().unwrap();
            match guard.take() {
                Some(Ok(msg)) => Ok(msg),
                Some(Err(e)) => Err(e),
                None => Err(LlmError::ProviderError("already consumed".into())),
            }
        }
        async fn complete_stream(
            &self,
            _messages: &[Message],
            _tools: &[ToolDefinition],
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send>>, LlmError>
        {
            Err(LlmError::ProviderError("not implemented".into()))
        }
        async fn ping(&self) -> Result<Duration, LlmError> {
            Ok(Duration::from_millis(10))
        }
    }

    #[test]
    fn test_should_failover_rate_limit() {
        assert!(FallbackClient::should_failover(&LlmError::RateLimited(60)));
    }

    #[test]
    fn test_should_failover_server_error() {
        assert!(FallbackClient::should_failover(&LlmError::ProviderError("502 Bad Gateway".into())));
    }

    #[test]
    fn test_should_not_failover_auth() {
        assert!(!FallbackClient::should_failover(&LlmError::AuthenticationFailed("bad key".into())));
    }

    #[test]
    fn test_should_not_failover_context_overflow() {
        assert!(!FallbackClient::should_failover(&LlmError::ContextLengthExceeded));
    }

    #[tokio::test]
    async fn test_fallback_switches_on_rate_limit() {
        let primary = MockClient::err(LlmError::RateLimited(60));
        let fallback = MockClient::ok("fallback response");

        let client = FallbackClient::new(primary.clone(), fallback.clone());
        let result = client.complete(&[], &[]).await.unwrap();

        assert_eq!(result.content, "fallback response");
        assert!(primary.was_called());
        assert!(fallback.was_called());
    }

    #[tokio::test]
    async fn test_fallback_does_not_switch_on_auth() {
        let primary = MockClient::err(LlmError::AuthenticationFailed("bad key".into()));
        let fallback = MockClient::ok("should not reach");

        let client = FallbackClient::new(primary.clone(), fallback.clone());
        let result = client.complete(&[], &[]).await;

        assert!(result.is_err());
        assert!(primary.was_called());
        assert!(!fallback.was_called());
    }

    #[tokio::test]
    async fn test_fallback_uses_primary_on_success() {
        let primary = MockClient::ok("primary response");
        let fallback = MockClient::ok("fallback response");

        let client = FallbackClient::new(primary.clone(), fallback.clone());
        let result = client.complete(&[], &[]).await.unwrap();

        assert_eq!(result.content, "primary response");
        assert!(primary.was_called());
        assert!(!fallback.was_called());
    }
}
