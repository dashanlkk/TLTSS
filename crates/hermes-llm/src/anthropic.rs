//! Anthropic Messages API client.
//!
//! Supports Anthropic-native and third-party Anthropic-compatible endpoints
//! (e.g. ZhipuAI/GLM via `open.bigmodel.cn/api/anthropic`).

use async_trait::async_trait;
use futures::Stream;
use hermes_cfg::prelude::*;
use hermes_cfg::traits::{LlmClient, StreamEvent};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::time::Duration;
use tracing::{debug, error, warn};

// ── Anthropic API types ─────────────────────────────────────────────

/// Anthropic Messages API request
#[derive(Serialize)]
struct MessagesRequest {
    model: String,
    messages: Vec<ApiMessage>,
    /// System prompt: supports plain string or content block array for prompt caching
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<serde_json::Value>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ApiTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

/// Anthropic message format
#[derive(Serialize, Deserialize, Clone, Debug)]
struct ApiMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<serde_json::Value>,
}

/// Anthropic tool definition
#[derive(Serialize, Clone)]
struct ApiTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

/// Anthropic Messages API response
#[derive(Deserialize, Debug)]
struct MessagesResponse {
    content: Vec<ContentBlock>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    usage: Option<ApiUsage>,
}

/// Anthropic content block
#[derive(Deserialize, Debug, Clone)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text {
        text: String,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
    },
}

/// Anthropic usage info
#[derive(Deserialize, Debug)]
struct ApiUsage {
    input_tokens: u32,
    output_tokens: u32,
}

// ── SSE stream types ────────────────────────────────────────────────

/// SSE event from Anthropic streaming API
#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum StreamEventRaw {
    #[serde(rename = "message_start")]
    MessageStart { message: MessageStartData },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        #[serde(flatten)]
        content_block: ContentBlockStartData,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        index: usize,
        delta: ContentBlockDelta,
    },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },
    #[serde(rename = "message_delta")]
    MessageDelta { delta: MessageDeltaData },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "ping")]
    Ping,
}

#[derive(Deserialize, Debug)]
struct MessageStartData {
    id: String,
    model: String,
    #[serde(default)]
    usage: Option<ApiUsage>,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum ContentBlockStartData {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: Option<serde_json::Value>,
    },
    #[serde(rename = "thinking")]
    Thinking { thinking: String },
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum ContentBlockDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta {
        #[serde(default)]
        partial_json: String,
    },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
}

#[derive(Deserialize, Debug)]
struct MessageDeltaData {
    stop_reason: Option<String>,
}

// ── Client implementation ───────────────────────────────────────────

/// Anthropic Messages API client
///
/// Works with both native Anthropic API and third-party Anthropic-compatible
/// endpoints (ZhipuAI, MiniMax, etc.).
pub struct AnthropicClient {
    client: Client,
    base_url: String,
    api_key: String,
    model: String,
    max_tokens: u32,
    temperature: f32,
    /// 启用 prompt caching（system + 最后 3 条消息标记 ephemeral cache）
    prompt_caching: bool,
}

impl AnthropicClient {
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
            api_key: api_key.into(),
            model: model.into(),
            max_tokens: 4096,
            temperature: 0.7,
            prompt_caching: false,
        }
    }

    pub fn with_max_tokens(mut self, max: u32) -> Self {
        self.max_tokens = max;
        self
    }

    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// 启用 prompt caching（"system_and_3" 策略）
    pub fn with_prompt_caching(mut self) -> Self {
        self.prompt_caching = true;
        self
    }

    /// Check if this is a third-party (non-Anthropic) endpoint
    fn is_third_party(&self) -> bool {
        let url = self.base_url.to_lowercase();
        !url.contains("anthropic.com")
    }

    /// Convert Hermes messages to Anthropic format.
    ///
    /// Returns (system_prompt, messages) where system_prompt is extracted
    /// from System-role messages per Anthropic API convention.
    ///
    /// When `prompt_caching` is enabled, system and the last 3 messages get
    /// `cache_control: {"type": "ephemeral"}` markers ("system_and_3" strategy).
    fn convert_messages(messages: &[Message], prompt_caching: bool) -> (Option<serde_json::Value>, Vec<ApiMessage>) {
        let mut system_prompt = String::new();
        let mut api_messages: Vec<ApiMessage> = Vec::new();

        for msg in messages {
            match msg.role {
                Role::System => {
                    if !system_prompt.is_empty() {
                        system_prompt.push('\n');
                    }
                    system_prompt.push_str(&msg.content);
                }
                Role::User => {
                    api_messages.push(ApiMessage {
                        role: "user".to_string(),
                        content: Some(serde_json::Value::String(msg.content.clone())),
                    });
                }
                Role::Assistant => {
                    // Build content blocks for assistant message
                    let mut blocks: Vec<serde_json::Value> = Vec::new();

                    // If there's text content, add a text block
                    if !msg.content.is_empty() {
                        blocks.push(serde_json::json!({
                            "type": "text",
                            "text": msg.content
                        }));
                    }

                    // If there are tool_calls, add tool_use blocks
                    if let Some(calls) = &msg.tool_calls {
                        for call in calls {
                            let arguments: serde_json::Value =
                                serde_json::from_str(&call.function.arguments)
                                    .unwrap_or(serde_json::Value::Object(Default::default()));
                            blocks.push(serde_json::json!({
                                "type": "tool_use",
                                "id": call.id,
                                "name": call.function.name,
                                "input": arguments
                            }));
                        }
                    }

                    if blocks.is_empty() {
                        blocks.push(serde_json::json!({
                            "type": "text",
                            "text": ""
                        }));
                    }

                    api_messages.push(ApiMessage {
                        role: "assistant".to_string(),
                        content: Some(if blocks.len() == 1
                            && blocks[0].get("type").and_then(|t| t.as_str()) == Some("text")
                            && msg.tool_calls.is_none()
                        {
                            blocks.into_iter().next().unwrap()
                        } else {
                            serde_json::Value::Array(blocks)
                        }),
                    });
                }
                Role::Tool => {
                    // Tool result: Anthropic expects content blocks
                    let tool_id = msg.tool_call_id.as_deref().unwrap_or("unknown");
                    api_messages.push(ApiMessage {
                        role: "user".to_string(),
                        content: Some(serde_json::json!([{
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": msg.content
                        }])),
                    });
                }
            }
        }

        // Ensure role alternation: merge consecutive same-role messages
        api_messages = merge_consecutive_roles(api_messages);

        // Build system field — content block array if caching, plain string otherwise
        let system = if system_prompt.is_empty() {
            None
        } else if prompt_caching {
            // System as content block array with cache_control marker
            Some(serde_json::json!([{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"}
            }]))
        } else {
            Some(serde_json::Value::String(system_prompt))
        };

        // Inject cache_control on the last 3 messages when caching is enabled
        if prompt_caching {
            inject_cache_markers(&mut api_messages, 3);
        }

        (system, api_messages)
    }

    /// Convert ToolDefinitions to Anthropic tool format
    fn convert_tools(tools: &[ToolDefinition]) -> Vec<ApiTool> {
        tools
            .iter()
            .map(|t| ApiTool {
                name: t.name.clone(),
                description: t.description.clone(),
                input_schema: t.parameters.clone(),
            })
            .collect()
    }

    /// Parse Anthropic response into Hermes Message
    fn parse_response(&self, resp: MessagesResponse) -> Message {
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut tool_calls = Vec::new();

        for block in resp.content {
            match block {
                ContentBlock::Text { text } => {
                    if !content.is_empty() {
                        content.push('\n');
                    }
                    content.push_str(&text);
                }
                ContentBlock::ToolUse { id, name, input } => {
                    tool_calls.push(ToolCall {
                        id,
                        call_type: "function".to_string(),
                        function: FunctionCall {
                            name,
                            arguments: serde_json::to_string(&input)
                                .unwrap_or_else(|_| "{}".to_string()),
                        },
                    });
                }
                ContentBlock::Thinking { thinking } => {
                    if !reasoning.is_empty() {
                        reasoning.push('\n');
                    }
                    reasoning.push_str(&thinking);
                }
            }
        }

        let mut msg = Message::new_assistant(&content);
        if !tool_calls.is_empty() {
            msg = msg.with_tool_calls(tool_calls);
        }
        if !reasoning.is_empty() {
            msg = msg.with_reasoning(&reasoning);
        }
        msg
    }
}

/// Inject `cache_control: {"type": "ephemeral"}` on the last `n` messages.
///
/// For messages with String content, converts to content block array format.
/// For messages with Object content (single text block), wraps into array.
/// For messages with Array content, appends cache_control to the last block.
fn inject_cache_markers(messages: &mut [ApiMessage], n: usize) {
    let len = messages.len();
    if len == 0 || n == 0 {
        return;
    }

    let start = len.saturating_sub(n);
    for msg in messages.iter_mut().skip(start) {
        let content = match msg.content.take() {
            Some(c) => c,
            None => continue,
        };

        let enriched = match content {
            // Plain string → convert to content block array, add cache_control
            serde_json::Value::String(s) => serde_json::json!([{
                "type": "text",
                "text": s,
                "cache_control": {"type": "ephemeral"}
            }]),
            // Single object block (e.g. assistant {"type":"text","text":"..."}) → wrap into array
            serde_json::Value::Object(mut map) => {
                map.insert(
                    "cache_control".to_string(),
                    serde_json::json!({"type": "ephemeral"}),
                );
                serde_json::Value::Array(vec![serde_json::Value::Object(map)])
            }
            // Array of blocks → add cache_control to the last block
            serde_json::Value::Array(mut blocks) => {
                if let Some(last) = blocks.last_mut() {
                    if let Some(obj) = last.as_object_mut() {
                        obj.insert(
                            "cache_control".to_string(),
                            serde_json::json!({"type": "ephemeral"}),
                        );
                    }
                }
                serde_json::Value::Array(blocks)
            }
            other => other,
        };

        msg.content = Some(enriched);
    }
}

/// Merge consecutive messages with the same role (Anthropic requires alternating roles)
fn merge_consecutive_roles(messages: Vec<ApiMessage>) -> Vec<ApiMessage> {
    if messages.is_empty() {
        return messages;
    }

    let mut merged: Vec<ApiMessage> = Vec::new();
    let mut current = messages[0].clone();

    for msg in messages.into_iter().skip(1) {
        if msg.role == current.role {
            // Merge content
            let current_content = current.content.unwrap_or(serde_json::Value::Null);
            let msg_content = msg.content.unwrap_or(serde_json::Value::Null);

            let merged_content = match (&current_content, &msg_content) {
                (serde_json::Value::String(a), serde_json::Value::String(b)) => {
                    serde_json::Value::String(format!("{}\n{}", a, b))
                }
                (serde_json::Value::Array(a), serde_json::Value::Array(b)) => {
                    let mut combined = a.clone();
                    combined.extend(b.clone());
                    serde_json::Value::Array(combined)
                }
                (serde_json::Value::Array(a), serde_json::Value::String(b)) => {
                    let mut combined = a.clone();
                    combined.push(serde_json::json!({"type": "text", "text": b}));
                    serde_json::Value::Array(combined)
                }
                (serde_json::Value::String(a), serde_json::Value::Array(b)) => {
                    let mut combined = vec![serde_json::json!({"type": "text", "text": a})];
                    combined.extend(b.clone());
                    serde_json::Value::Array(combined)
                }
                _ => msg_content,
            };

            current.content = Some(merged_content);
        } else {
            merged.push(current);
            current = msg;
        }
    }
    merged.push(current);

    merged
}

#[async_trait]
impl LlmClient for AnthropicClient {
    async fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<Message, LlmError> {
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));

        let (system, api_messages) = Self::convert_messages(messages, self.prompt_caching);
        let api_tools = Self::convert_tools(tools);

        let req = MessagesRequest {
            model: self.model.clone(),
            messages: api_messages,
            system,
            max_tokens: self.max_tokens,
            temperature: Some(self.temperature),
            tools: api_tools,
            stream: None,
        };

        debug!("Sending Anthropic request to {} (caching={})", url, self.prompt_caching);

        let resp = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&req)
            .send()
            .await
            .map_err(|e| LlmError::ConnectionFailed(e.to_string()))?;

        let status = resp.status();
        if status == reqwest::StatusCode::UNAUTHORIZED {
            return Err(LlmError::AuthenticationFailed("Invalid API key".into()));
        }
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(LlmError::RateLimited(60));
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            error!("Anthropic error {}: {}", status, body);
            return Err(LlmError::ProviderError(format!("{}: {}", status, body)));
        }

        let chat_resp: MessagesResponse = resp
            .json()
            .await
            .map_err(|e| LlmError::ProviderError(e.to_string()))?;

        debug!(
            "Anthropic response: stop_reason={:?}, blocks={}",
            chat_resp.stop_reason,
            chat_resp.content.len()
        );

        Ok(self.parse_response(chat_resp))
    }

    async fn complete_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send>>, LlmError> {
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));

        let (system, api_messages) = Self::convert_messages(messages, self.prompt_caching);
        let api_tools = Self::convert_tools(tools);

        let req = MessagesRequest {
            model: self.model.clone(),
            messages: api_messages,
            system,
            max_tokens: self.max_tokens,
            temperature: Some(self.temperature),
            tools: api_tools,
            stream: Some(true),
        };

        let resp = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&req)
            .send()
            .await
            .map_err(|e| LlmError::ConnectionFailed(e.to_string()))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::ProviderError(format!(
                "Stream init failed: {} {}",
                status, body
            )));
        }

        // SSE stream parser
        let stream = async_stream::stream! {
            let mut buffer = String::new();
            let mut stream = resp.bytes_stream();
            use futures::StreamExt;

            // Track in-progress tool_use blocks
            let mut pending_tool_calls: std::collections::HashMap<usize, (String, String, String)> =
                std::collections::HashMap::new();

            while let Some(chunk) = stream.next().await {
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => {
                        yield Err(LlmError::StreamError(e.to_string()));
                        break;
                    }
                };
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                // Process complete SSE events (separated by \n\n)
                while let Some(pos) = buffer.find("\n\n") {
                    let event_text = buffer[..pos].to_string();
                    buffer = buffer[pos + 2..].to_string();

                    // Parse SSE event format: "event: type\ndata: json"
                    let mut event_type = String::new();
                    let mut data_str = String::new();

                    for line in event_text.lines() {
                        if let Some(et) = line.strip_prefix("event: ") {
                            event_type = et.trim().to_string();
                        } else if let Some(d) = line.strip_prefix("data: ") {
                            data_str = d.trim().to_string();
                        }
                    }

                    if data_str.is_empty() {
                        continue;
                    }

                    // Dispatch based on event type
                    match event_type.as_str() {
                        "content_block_delta" => {
                            // SSE data is: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"..."}}
                            // We need to extract the inner "delta" field before deserializing
                            if let Ok(data_val) = serde_json::from_str::<serde_json::Value>(&data_str) {
                                if let Some(delta_val) = data_val.get("delta") {
                                    if let Ok(delta) = serde_json::from_value::<ContentBlockDelta>(delta_val.clone()) {
                                        match delta {
                                            ContentBlockDelta::TextDelta { text } => {
                                                if !text.is_empty() {
                                                    yield Ok(StreamEvent::Delta(text));
                                                }
                                            }
                                            ContentBlockDelta::InputJsonDelta { partial_json } => {
                                                debug!("Tool input delta: {}", partial_json);
                                            }
                                            ContentBlockDelta::ThinkingDelta { thinking } => {
                                                if !thinking.is_empty() {
                                                    yield Ok(StreamEvent::Reasoning(thinking));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        "content_block_start" => {
                            // SSE data is: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use",...}}
                            // Tool info is inside "content_block" sub-object
                            if let Ok(start) =
                                serde_json::from_str::<serde_json::Value>(&data_str)
                            {
                                let block = start.get("content_block");
                                if block.and_then(|b| b.get("type")).and_then(|t| t.as_str()) == Some("tool_use") {
                                    let cb = block.unwrap();
                                    let id = cb
                                        .get("id")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    let name = cb
                                        .get("name")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    let index = start
                                        .get("index")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0) as usize;
                                    pending_tool_calls.insert(index, (id, name, String::new()));
                                }
                            }
                        }
                        "content_block_stop" => {
                            // Finalize tool_use block if pending
                            if let Ok(stop) =
                                serde_json::from_str::<serde_json::Value>(&data_str)
                            {
                                let index =
                                    stop.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                                if let Some((id, name, args)) = pending_tool_calls.remove(&index) {
                                    // Collect any accumulated input_json_delta fragments
                                    // For simplicity, we use empty args if not accumulated
                                    let arguments = if args.is_empty() {
                                        "{}".to_string()
                                    } else {
                                        args
                                    };
                                    yield Ok(StreamEvent::ToolCall {
                                        id,
                                        name,
                                        arguments,
                                    });
                                }
                            }
                        }
                        "message_stop" => {
                            yield Ok(StreamEvent::Done);
                            break;
                        }
                        "message_start" | "message_delta" | "ping" => {
                            // Metadata events, no yield needed
                        }
                        _ => {
                            warn!("Unknown Anthropic SSE event: {}", event_type);
                        }
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }

    async fn ping(&self) -> Result<Duration, LlmError> {
        // Anthropic doesn't have a lightweight health endpoint.
        // Use a minimal messages request to test connectivity.
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));
        let start = std::time::Instant::now();

        let resp = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&serde_json::json!({
                "model": self.model,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .map_err(|e| LlmError::ConnectionFailed(e.to_string()))?;

        let _ = resp.status(); // Don't care about the response, just connectivity
        Ok(start.elapsed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_messages_extracts_system() {
        let messages = vec![
            Message::new_system("You are helpful"),
            Message::new_user("Hello"),
        ];
        let (system, api_msgs) = AnthropicClient::convert_messages(&messages, false);
        // Without caching, system is a plain string
        assert_eq!(system.as_ref().and_then(|v| v.as_str()), Some("You are helpful"));
        assert_eq!(api_msgs.len(), 1);
        assert_eq!(api_msgs[0].role, "user");
    }

    #[test]
    fn test_convert_tool_result_as_user_message() {
        let messages = vec![
            Message::new_user("What files are here?"),
            Message::new_assistant("").with_tool_calls(vec![ToolCall {
                id: "tool_1".into(),
                call_type: "function".into(),
                function: FunctionCall {
                    name: "list_dir".into(),
                    arguments: r#"{"path":"."}"#.into(),
                },
            }]),
            Message::new_tool_result("tool_1", "file1.txt\nfile2.txt"),
        ];
        let (_, api_msgs) = AnthropicClient::convert_messages(&messages, false);
        // assistant + tool_result (as user role)
        assert_eq!(api_msgs.len(), 3);
        assert_eq!(api_msgs[0].role, "user");
        assert_eq!(api_msgs[1].role, "assistant");
        assert_eq!(api_msgs[2].role, "user"); // tool_result becomes user
    }

    #[test]
    fn test_parse_response_with_tool_use() {
        let client = AnthropicClient::new("http://localhost", "test-key", "test-model");

        let resp = MessagesResponse {
            content: vec![
                ContentBlock::Text {
                    text: "Let me check.".into(),
                },
                ContentBlock::ToolUse {
                    id: "tool_123".into(),
                    name: "list_dir".into(),
                    input: serde_json::json!({"path": "."}),
                },
            ],
            stop_reason: Some("tool_use".into()),
            usage: Some(ApiUsage {
                input_tokens: 10,
                output_tokens: 20,
            }),
        };

        let msg = client.parse_response(resp);
        assert_eq!(msg.content, "Let me check.");
        assert!(msg.tool_calls.is_some());
        let calls = msg.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "list_dir");
    }

    #[test]
    fn test_parse_response_with_thinking() {
        let client = AnthropicClient::new("http://localhost", "test-key", "test-model");

        let resp = MessagesResponse {
            content: vec![
                ContentBlock::Thinking {
                    thinking: "Let me think about this...".into(),
                },
                ContentBlock::Text {
                    text: "Here is the answer.".into(),
                },
            ],
            stop_reason: Some("end_turn".into()),
            usage: None,
        };

        let msg = client.parse_response(resp);
        assert_eq!(msg.content, "Here is the answer.");
        assert_eq!(
            msg.reasoning.as_deref(),
            Some("Let me think about this...")
        );
    }

    #[test]
    fn test_merge_consecutive_roles() {
        let messages = vec![
            ApiMessage {
                role: "user".into(),
                content: Some(serde_json::Value::String("Hello".into())),
            },
            ApiMessage {
                role: "user".into(),
                content: Some(serde_json::Value::String("World".into())),
            },
        ];
        let merged = merge_consecutive_roles(messages);
        assert_eq!(merged.len(), 1);
        // Should be merged
        let content = merged[0].content.as_ref().unwrap();
        assert!(content.as_str().unwrap().contains("Hello"));
        assert!(content.as_str().unwrap().contains("World"));
    }

    // ── Prompt Caching 测试 ──

    #[test]
    fn test_caching_system_is_block_array() {
        let messages = vec![
            Message::new_system("You are helpful"),
            Message::new_user("Hello"),
        ];
        let (system, _) = AnthropicClient::convert_messages(&messages, true);
        // With caching, system should be an array of content blocks
        let system = system.unwrap();
        assert!(system.is_array());
        let blocks = system.as_array().unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn test_caching_system_without_cache() {
        let messages = vec![
            Message::new_system("You are helpful"),
            Message::new_user("Hello"),
        ];
        let (system, _) = AnthropicClient::convert_messages(&messages, false);
        // Without caching, system is a plain string
        let system = system.unwrap();
        assert!(system.is_string());
        assert_eq!(system.as_str().unwrap(), "You are helpful");
    }

    #[test]
    fn test_caching_marks_last_n_messages() {
        let messages = vec![
            Message::new_system("System"),
            Message::new_user("msg1"),
            Message::new_assistant("reply1"),
            Message::new_user("msg2"),
            Message::new_assistant("reply2"),
            Message::new_user("msg3"),
        ];
        let (_, api_msgs) = AnthropicClient::convert_messages(&messages, true);
        // 5 non-system messages → last 3 should have cache_control
        assert_eq!(api_msgs.len(), 5);

        // First 2 should NOT have cache_control
        for msg in &api_msgs[0..2] {
            let content = msg.content.as_ref().unwrap();
            let has_cache = match content {
                serde_json::Value::Array(blocks) => blocks.iter().any(|b| b.get("cache_control").is_some()),
                serde_json::Value::Object(map) => map.contains_key("cache_control"),
                _ => false,
            };
            assert!(!has_cache, "First 2 messages should not have cache_control");
        }

        // Last 3 should have cache_control (converted to content block arrays)
        for msg in &api_msgs[2..] {
            let content = msg.content.as_ref().unwrap();
            assert!(content.is_array(), "Expected array content for cached message");
            let blocks = content.as_array().unwrap();
            let last_block = blocks.last().unwrap();
            assert_eq!(last_block["cache_control"]["type"], "ephemeral");
        }
    }

    #[test]
    fn test_inject_cache_markers_on_tool_result() {
        let messages = vec![
            Message::new_system("System"),
            Message::new_user("Do something"),
            Message::new_assistant("").with_tool_calls(vec![ToolCall {
                id: "t1".into(),
                call_type: "function".into(),
                function: FunctionCall {
                    name: "read".into(),
                    arguments: "{}".into(),
                },
            }]),
            Message::new_tool_result("t1", "file contents"),
        ];
        let (_, api_msgs) = AnthropicClient::convert_messages(&messages, true);
        // 3 non-system messages, all should be cached (3 <= 3)
        // Last message (tool_result) has array content → last block should have cache_control
        let last = api_msgs.last().unwrap();
        let content = last.content.as_ref().unwrap();
        assert!(content.is_array());
        let blocks = content.as_array().unwrap();
        let last_block = blocks.last().unwrap();
        assert_eq!(last_block["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn test_inject_cache_markers_empty_messages() {
        let mut messages: Vec<ApiMessage> = vec![];
        inject_cache_markers(&mut messages, 3);
        assert!(messages.is_empty());
    }

    #[test]
    fn test_with_prompt_caching_builder() {
        let client = AnthropicClient::new("http://localhost", "key", "model")
            .with_prompt_caching();
        assert!(client.prompt_caching);
    }
}
