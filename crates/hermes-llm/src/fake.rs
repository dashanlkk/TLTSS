use async_trait::async_trait;
use futures::stream::{self, Stream};
use hermes_cfg::prelude::*;
use hermes_cfg::traits::{LlmClient, StreamEvent};
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// 固定回复的 Mock 客户端
pub struct FakeClient {
    response: String,
    stream_chunks: Vec<String>,
    /// Optional tool calls to include in complete() responses.
    /// Each call is consumed in order; repeats the last one if exhausted.
    tool_calls_sequence: Vec<Vec<hermes_cfg::tool::ToolCall>>,
    /// If true, complete() returns an error on the first N calls (for retry testing).
    fail_count: Arc<AtomicUsize>,
}

impl FakeClient {
    pub fn new(response: impl Into<String>) -> Self {
        let resp = response.into();
        let chunks: Vec<String> = resp.chars().map(|c| c.to_string()).collect();
        Self {
            response: resp,
            stream_chunks: chunks,
            tool_calls_sequence: Vec::new(),
            fail_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn with_chunks(mut self, chunks: Vec<String>) -> Self {
        self.stream_chunks = chunks;
        self
    }

    /// Set tool calls to return. Each element is one round's tool_calls.
    /// The Nth call to complete() returns the Nth element's tool_calls.
    /// After all elements are consumed, returns a plain text response.
    pub fn with_tool_calls_sequence(mut self, sequence: Vec<Vec<hermes_cfg::tool::ToolCall>>) -> Self {
        self.tool_calls_sequence = sequence;
        self
    }

    /// Fail the first N calls to complete() with ConnectionFailed error.
    pub fn with_fail_count(mut self, n: usize) -> Self {
        self.fail_count = Arc::new(AtomicUsize::new(n));
        self
    }
}

#[async_trait]
impl LlmClient for FakeClient {
    async fn complete(
        &self,
        _messages: &[Message],
        _tools: &[ToolDefinition],
    ) -> Result<Message, LlmError> {
        // Simulate failures for retry testing
        let remaining = self.fail_count.load(Ordering::SeqCst);
        if remaining > 0 {
            self.fail_count.fetch_sub(1, Ordering::SeqCst);
            return Err(LlmError::ConnectionFailed("simulated failure".into()));
        }

        let mut msg = Message::new_assistant(&self.response);

        // Check if we have tool_calls for this round
        let call_count = self.tool_calls_sequence.len();
        // Use message count as a proxy for which round we're on
        let round = _messages.iter().filter(|m| m.role == hermes_cfg::message::Role::Tool).count();
        if round < call_count {
            let calls = self.tool_calls_sequence[round].clone();
            msg = msg.with_tool_calls(calls);
        }

        Ok(msg)
    }

    async fn complete_stream(
        &self,
        _messages: &[Message],
        _tools: &[ToolDefinition],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send>>, LlmError> {
        let mut events: Vec<Result<StreamEvent, LlmError>> = self
            .stream_chunks
            .iter()
            .map(|c| Ok(StreamEvent::Delta(c.clone())))
            .collect();

        // Include tool calls in stream if sequence is set
        let call_count = self.tool_calls_sequence.len();
        let round = _messages.iter().filter(|m| m.role == hermes_cfg::message::Role::Tool).count();
        if round < call_count {
            for call in &self.tool_calls_sequence[round] {
                events.push(Ok(StreamEvent::ToolCall {
                    id: call.id.clone(),
                    name: call.function.name.clone(),
                    arguments: call.function.arguments.clone(),
                }));
            }
        }

        events.push(Ok(StreamEvent::Done));
        let stream = stream::iter(events);
        Ok(Box::pin(stream))
    }

    async fn ping(&self) -> Result<Duration, LlmError> {
        Ok(Duration::from_millis(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tool_call(name: &str, args: &str) -> hermes_cfg::tool::ToolCall {
        hermes_cfg::tool::ToolCall {
            id: format!("call_{}", name),
            call_type: "function".to_string(),
            function: hermes_cfg::tool::FunctionCall {
                name: name.to_string(),
                arguments: args.to_string(),
            },
        }
    }

    #[tokio::test]
    async fn test_fake_complete() {
        let client = FakeClient::new("Fake response");
        let msg = client.complete(&[], &[]).await.unwrap();
        assert_eq!(msg.content, "Fake response");
    }

    #[tokio::test]
    async fn test_fake_stream() {
        let client = FakeClient::new("Hi!");
        let mut stream = client.complete_stream(&[], &[]).await.unwrap();
        use futures::StreamExt;
        let mut tokens = Vec::new();
        while let Some(item) = stream.next().await {
            if let Ok(StreamEvent::Delta(t)) = item {
                tokens.push(t);
            }
        }
        assert_eq!(tokens.join(""), "Hi!");
    }

    #[tokio::test]
    async fn test_fake_complete_with_tool_calls() {
        let client = FakeClient::new("Using tool")
            .with_tool_calls_sequence(vec![
                vec![make_tool_call("read_file", r#"{"path":"test.txt"}"#)],
            ]);

        let msg = client.complete(&[], &[]).await.unwrap();
        assert_eq!(msg.content, "Using tool");
        let calls = msg.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read_file");
    }

    #[tokio::test]
    async fn test_fake_stream_with_tool_calls() {
        let client = FakeClient::new("OK")
            .with_tool_calls_sequence(vec![
                vec![make_tool_call("list_dir", r#"{"path":"."}"#)],
            ]);

        let mut stream = client.complete_stream(&[], &[]).await.unwrap();
        use futures::StreamExt;
        let mut tool_calls = Vec::new();
        while let Some(item) = stream.next().await {
            match item {
                Ok(StreamEvent::ToolCall { id, name, arguments }) => {
                    tool_calls.push((id, name, arguments));
                }
                _ => {}
            }
        }
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].1, "list_dir");
    }

    #[tokio::test]
    async fn test_fake_fail_then_succeed() {
        let client = FakeClient::new("Recovered").with_fail_count(2);

        // First 2 calls fail
        let r1 = client.complete(&[], &[]).await;
        assert!(r1.is_err());
        let r2 = client.complete(&[], &[]).await;
        assert!(r2.is_err());

        // Third succeeds
        let r3 = client.complete(&[], &[]).await.unwrap();
        assert_eq!(r3.content, "Recovered");
    }
}
