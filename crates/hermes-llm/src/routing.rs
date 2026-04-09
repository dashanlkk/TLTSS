//! Smart routing client — selects the best LLM client based on request complexity.
//!
//! Port of Python hermes-agent/agent/smart_model_routing.py
//! Simple queries (short, no code, no URLs, no complex keywords) → light/cheap model.
//! Complex queries (long, tool history, many tools) → heavy/expensive model.
//! Default → standard model.

use async_trait::async_trait;
use futures::Stream;
use hermes_cfg::prelude::*;
use hermes_cfg::traits::{LlmClient, StreamEvent};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info};

/// Complexity keywords that indicate a heavy task
const COMPLEX_KEYWORDS: &[&str] = &[
    "debug", "implement", "refactor", "error", "terminal", "docker",
    "test", "plan", "architect", "design", "fix", "broken", "deploy",
    "security", "audit", "investigate", "analyze", "migrate",
];

/// Smart routing LLM client wrapper.
///
/// Holds multiple LLM clients keyed by tier name ("light", "standard", "heavy").
/// Before each call, analyzes message complexity and routes to the appropriate client.
/// Falls back to the standard client if no specialized client is available.
pub struct RoutingClient {
    clients: HashMap<String, Arc<dyn LlmClient>>,
    standard_client: Arc<dyn LlmClient>,
    light_tier: Option<String>,
    heavy_tier: Option<String>,
    /// Message length threshold for upgrading from light to standard (chars)
    max_simple_chars: usize,
    /// Word count threshold for upgrading from light to standard
    max_simple_words: usize,
}

impl RoutingClient {
    /// Create a new routing client with a standard (default) client.
    pub fn new(standard: Arc<dyn LlmClient>) -> Self {
        let mut clients = HashMap::new();
        clients.insert("standard".to_string(), standard.clone());
        Self {
            clients,
            standard_client: standard,
            light_tier: None,
            heavy_tier: None,
            max_simple_chars: 160,
            max_simple_words: 28,
        }
    }

    /// Add a light (cheap) model client.
    pub fn with_light(mut self, client: Arc<dyn LlmClient>) -> Self {
        self.light_tier = Some("light".to_string());
        self.clients.insert("light".to_string(), client);
        self
    }

    /// Add a heavy (powerful) model client.
    pub fn with_heavy(mut self, client: Arc<dyn LlmClient>) -> Self {
        self.heavy_tier = Some("heavy".to_string());
        self.clients.insert("heavy".to_string(), client);
        self
    }

    /// Set complexity thresholds.
    pub fn with_thresholds(mut self, max_chars: usize, max_words: usize) -> Self {
        self.max_simple_chars = max_chars;
        self.max_simple_words = max_words;
        self
    }

    /// Check if a user message is "simple" (can use light model).
    /// Port of Python's `choose_cheap_model_route()`.
    fn is_simple_query(&self, messages: &[Message]) -> bool {
        // Find the last user message
        let last_user_msg = messages.iter().rev().find(|m| m.role == Role::User);
        let Some(user_msg) = last_user_msg else {
            return false;
        };

        let content = &user_msg.content;

        // Length checks
        if content.len() > self.max_simple_chars {
            return false;
        }

        // Word count check
        let word_count = content.split_whitespace().count();
        if word_count > self.max_simple_words {
            return false;
        }

        // Must be single-line (no more than 1 newline)
        if content.lines().count() > 2 {
            return false;
        }

        // No code fences or backticks
        if content.contains("```") || content.chars().filter(|&c| c == '`').count() >= 2 {
            return false;
        }

        // No URLs
        if content.contains("http://") || content.contains("https://") {
            return false;
        }

        // No complex keywords
        let lower = content.to_lowercase();
        for keyword in COMPLEX_KEYWORDS {
            if lower.contains(keyword) {
                return false;
            }
        }

        true
    }

    /// Check if messages indicate a heavy-complexity task.
    fn is_heavy_query(&self, messages: &[Message], tools: &[ToolDefinition]) -> bool {
        // Has tool call history → complex
        let has_tool_history = messages.iter().any(|m| {
            m.role == Role::Assistant
                && m.tool_calls.as_ref().is_some_and(|c| !c.is_empty())
        });

        // Many tools available → complex
        let many_tools = tools.len() > 6;

        // Long conversation → complex
        let long_conversation = messages.len() > 20;

        // Long total content → complex
        let total_chars: usize = messages.iter().map(|m| m.content.len()).sum();
        let long_content = total_chars > 4000;

        has_tool_history || many_tools || long_conversation || long_content
    }

    /// Select the appropriate client for the given messages + tools.
    fn select_client(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> (&str, &Arc<dyn LlmClient>) {
        // Heavy check first
        if self.heavy_tier.is_some() && self.is_heavy_query(messages, tools) {
            if let Some(client) = self.clients.get("heavy") {
                debug!("Routing to heavy model");
                return ("heavy", client);
            }
        }

        // Simple check
        if self.light_tier.is_some() && self.is_simple_query(messages) {
            if let Some(client) = self.clients.get("light") {
                debug!("Routing to light model");
                return ("light", client);
            }
        }

        // Default: standard
        ("standard", &self.standard_client)
    }
}

#[async_trait]
impl LlmClient for RoutingClient {
    async fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<Message, LlmError> {
        let (tier, client) = self.select_client(messages, tools);
        info!("RoutingClient: using '{}' tier for complete()", tier);
        client.complete(messages, tools).await
    }

    async fn complete_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send>>, LlmError>
    {
        let (tier, client) = self.select_client(messages, tools);
        info!("RoutingClient: using '{}' tier for complete_stream()", tier);
        client.complete_stream(messages, tools).await
    }

    async fn ping(&self) -> Result<Duration, LlmError> {
        self.standard_client.ping().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_query_detection() {
        let routing = RoutingClient::new(Arc::new(crate::FakeClient::new("ok")));
        let messages = vec![Message::new_user("Hello!")];
        assert!(routing.is_simple_query(&messages));
    }

    #[test]
    fn test_simple_query_too_long() {
        let routing = RoutingClient::new(Arc::new(crate::FakeClient::new("ok")));
        let messages = vec![Message::new_user(&"a".repeat(200))];
        assert!(!routing.is_simple_query(&messages));
    }

    #[test]
    fn test_simple_query_has_code() {
        let routing = RoutingClient::new(Arc::new(crate::FakeClient::new("ok")));
        let messages = vec![Message::new_user("Please run ```python code```")];
        assert!(!routing.is_simple_query(&messages));
    }

    #[test]
    fn test_simple_query_has_url() {
        let routing = RoutingClient::new(Arc::new(crate::FakeClient::new("ok")));
        let messages = vec![Message::new_user("Check https://example.com")];
        assert!(!routing.is_simple_query(&messages));
    }

    #[test]
    fn test_simple_query_complex_keyword() {
        let routing = RoutingClient::new(Arc::new(crate::FakeClient::new("ok")));
        let messages = vec![Message::new_user("Debug this error")];
        assert!(!routing.is_simple_query(&messages));
    }

    #[test]
    fn test_heavy_detection_tool_history() {
        let routing = RoutingClient::new(Arc::new(crate::FakeClient::new("ok")));
        let messages = vec![
            Message::new_user("do something"),
            Message::new_assistant("").with_tool_calls(vec![ToolCall {
                id: "t1".into(),
                call_type: "function".into(),
                function: FunctionCall {
                    name: "read".into(),
                    arguments: "{}".into(),
                },
            }]),
        ];
        assert!(routing.is_heavy_query(&messages, &[]));
    }

    #[test]
    fn test_heavy_detection_many_tools() {
        let routing = RoutingClient::new(Arc::new(crate::FakeClient::new("ok")));
        let tools: Vec<ToolDefinition> = (0..8)
            .map(|i| ToolDefinition {
                name: format!("tool_{}", i),
                description: "test".into(),
                parameters: serde_json::json!({}),
            })
            .collect();
        assert!(routing.is_heavy_query(&[], &tools));
    }

    #[tokio::test]
    async fn test_routing_client_routes_simple_to_light() {
        let standard = Arc::new(crate::FakeClient::new("standard response"));
        let light = Arc::new(crate::FakeClient::new("light response"));

        let routing = RoutingClient::new(standard).with_light(light);

        let messages = vec![Message::new_user("Hi")];
        let (tier, _) = routing.select_client(&messages, &[]);
        assert_eq!(tier, "light");
    }

    #[tokio::test]
    async fn test_routing_client_routes_complex_to_standard() {
        let standard = Arc::new(crate::FakeClient::new("standard response"));
        let light = Arc::new(crate::FakeClient::new("light response"));

        let routing = RoutingClient::new(standard).with_light(light);

        let messages = vec![Message::new_user("Please implement a new feature")];
        let (tier, _) = routing.select_client(&messages, &[]);
        assert_eq!(tier, "standard");
    }

    #[tokio::test]
    async fn test_routing_client_complete() {
        let standard = Arc::new(crate::FakeClient::new("standard ok"));
        let routing = RoutingClient::new(standard);

        let messages = vec![Message::new_user("Hello")];
        let result = routing.complete(&messages, &[]).await.unwrap();
        assert_eq!(result.content, "standard ok");
    }
}
