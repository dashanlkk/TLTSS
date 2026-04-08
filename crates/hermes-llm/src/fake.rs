use async_trait::async_trait;
use futures::stream::{self, Stream};
use hermes_cfg::prelude::*;
use hermes_cfg::traits::{LlmClient, StreamEvent};
use std::pin::Pin;
use std::time::Duration;

/// 固定回复的 Mock 客户端
pub struct FakeClient {
    response: String,
    stream_chunks: Vec<String>,
}

impl FakeClient {
    pub fn new(response: impl Into<String>) -> Self {
        let resp = response.into();
        let chunks: Vec<String> = resp.chars().map(|c| c.to_string()).collect();
        Self {
            response: resp,
            stream_chunks: chunks,
        }
    }

    pub fn with_chunks(mut self, chunks: Vec<String>) -> Self {
        self.stream_chunks = chunks;
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
        Ok(Message::new_assistant(&self.response))
    }

    async fn complete_stream(
        &self,
        _messages: &[Message],
        _tools: &[ToolDefinition],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send>>, LlmError>
    {
        let mut events: Vec<Result<StreamEvent, LlmError>> = self
            .stream_chunks
            .iter()
            .map(|c| Ok(StreamEvent::Delta(c.clone())))
            .collect();
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
}
