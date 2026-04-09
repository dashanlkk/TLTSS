use async_trait::async_trait;
use futures::Stream;
use hermes_cfg::prelude::*;
use hermes_cfg::traits::{LlmClient, StreamEvent};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::time::Duration;
use tracing::{debug, error};

/// OpenAI-compatible API 客户端
pub struct OpenAIClient {
    client: Client,
    base_url: String,
    api_key: String,
    model: String,
    max_tokens: u32,
    temperature: f32,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ApiMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ApiTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Serialize, Deserialize, Clone)]
struct ApiMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ApiToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize, Clone)]
struct ApiTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: ApiFunction,
}

#[derive(Serialize, Clone)]
struct ApiFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Serialize, Deserialize, Clone)]
struct ApiToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: ApiFunctionCall,
}

#[derive(Serialize, Deserialize, Clone)]
struct ApiFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<ApiToolCall>>,
}

impl OpenAIClient {
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

    fn messages_to_api(messages: &[Message]) -> Vec<ApiMessage> {
        messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => "tool",
                };
                // 转换 tool_calls（assistant 消息可能携带工具调用）
                let tool_calls = m.tool_calls.as_ref().map(|calls| {
                    calls
                        .iter()
                        .map(|c| ApiToolCall {
                            id: c.id.clone(),
                            call_type: c.call_type.clone(),
                            function: ApiFunctionCall {
                                name: c.function.name.clone(),
                                arguments: c.function.arguments.clone(),
                            },
                        })
                        .collect()
                });
                ApiMessage {
                    role: role.to_string(),
                    content: m.content.clone(),
                    tool_calls,
                    tool_call_id: m.tool_call_id.clone(),
                }
            })
            .collect()
    }

    fn tools_to_api(tools: &[ToolDefinition]) -> Vec<ApiTool> {
        tools
            .iter()
            .map(|t| ApiTool {
                tool_type: "function".to_string(),
                function: ApiFunction {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: t.parameters.clone(),
                },
            })
            .collect()
    }
}

#[async_trait]
impl LlmClient for OpenAIClient {
    async fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<Message, LlmError> {
        let url = format!("{}/chat/completions", self.base_url);

        let req = ChatRequest {
            model: self.model.clone(),
            messages: Self::messages_to_api(messages),
            tools: Self::tools_to_api(tools),
            max_tokens: Some(self.max_tokens),
            temperature: Some(self.temperature),
            stream: None,
        };

        debug!("Sending request to {}", url);

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&req)
            .send()
            .await
            .map_err(|e| LlmError::ConnectionFailed(e.to_string()))?;

        if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
            return Err(LlmError::AuthenticationFailed("Invalid API key".into()));
        }

        if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(LlmError::RateLimited(60));
        }

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            error!("LLM error {}: {}", status, body);
            return Err(LlmError::ProviderError(format!("{}: {}", status, body)));
        }

        let chat_resp: ChatResponse = resp
            .json()
            .await
            .map_err(|e| LlmError::ProviderError(e.to_string()))?;

        let choice = chat_resp
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| LlmError::ProviderError("No choices returned".into()))?;

        let mut msg = Message::new_assistant(choice.message.content.unwrap_or_default());
        if let Some(api_calls) = choice.message.tool_calls {
            let calls: Vec<ToolCall> = api_calls
                .into_iter()
                .map(|c| ToolCall {
                    id: c.id,
                    call_type: c.call_type,
                    function: FunctionCall {
                        name: c.function.name,
                        arguments: c.function.arguments,
                    },
                })
                .collect();
            msg = msg.with_tool_calls(calls);
        }

        Ok(msg)
    }

    async fn complete_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send>>, LlmError> {
        let url = format!("{}/chat/completions", self.base_url);

        let req = ChatRequest {
            model: self.model.clone(),
            messages: Self::messages_to_api(messages),
            tools: Self::tools_to_api(tools),
            max_tokens: Some(self.max_tokens),
            temperature: Some(self.temperature),
            stream: Some(true),
        };

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&req)
            .send()
            .await
            .map_err(|e| LlmError::ConnectionFailed(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            return Err(LlmError::ProviderError(format!("Stream init failed: {}", status)));
        }

        // SSE 流解析 — 支持 delta.content + delta.tool_calls 增量拼接
        let stream = async_stream::stream! {
            let mut buffer = String::new();
            let mut stream = resp.bytes_stream();
            use futures::StreamExt;

            // Tool call 增量拼接状态：(id, name, arguments)
            let mut pending_tool_calls: Vec<(String, String, String)> = Vec::new();

            while let Some(chunk) = stream.next().await {
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => {
                        yield Err(LlmError::StreamError(e.to_string()));
                        break;
                    }
                };
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(pos) = buffer.find("\n\n") {
                    let line = buffer[..pos].to_string();
                    buffer = buffer[pos + 2..].to_string();

                    if let Some(data) = line.strip_prefix("data: ") {
                        if data == "[DONE]" {
                            // Emit completed tool calls before Done
                            for (id, name, arguments) in pending_tool_calls.drain(..) {
                                yield Ok(StreamEvent::ToolCall {
                                    id,
                                    name,
                                    arguments,
                                });
                            }
                            yield Ok(StreamEvent::Done);
                            break;
                        }
                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data) {
                            let delta = &parsed["choices"][0]["delta"];

                            // Text content
                            if let Some(content) = delta["content"].as_str() {
                                if !content.is_empty() {
                                    yield Ok(StreamEvent::Delta(content.to_string()));
                                }
                            }

                            // Reasoning content (e.g. GLM reasoning_content field)
                            if let Some(reasoning) = delta["reasoning_content"].as_str() {
                                if !reasoning.is_empty() {
                                    yield Ok(StreamEvent::Reasoning(reasoning.to_string()));
                                }
                            }

                            // Tool calls — incremental assembly
                            if let Some(tc_array) = delta["tool_calls"].as_array() {
                                for tc in tc_array {
                                    let idx = tc["index"].as_u64().unwrap_or(0) as usize;

                                    // Extend pending list if needed
                                    if idx >= pending_tool_calls.len() {
                                        pending_tool_calls.resize(idx + 1, (String::new(), String::new(), String::new()));
                                    }

                                    // Update id (only present in first chunk)
                                    if let Some(id) = tc["id"].as_str() {
                                        pending_tool_calls[idx].0 = id.to_string();
                                    }
                                    // Update name (only present in first chunk)
                                    if let Some(name) = tc["function"]["name"].as_str() {
                                        pending_tool_calls[idx].1 = name.to_string();
                                    }
                                    // Append arguments incrementally
                                    if let Some(args) = tc["function"]["arguments"].as_str() {
                                        pending_tool_calls[idx].2.push_str(args);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }

    async fn ping(&self) -> Result<Duration, LlmError> {
        let url = format!("{}/models", self.base_url);
        let start = std::time::Instant::now();
        self.client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .map_err(|e| LlmError::ConnectionFailed(e.to_string()))?;
        Ok(start.elapsed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[test]
    fn test_messages_to_api_roles() {
        let messages = vec![
            Message::new_system("sys"),
            Message::new_user("hello"),
            Message::new_assistant("hi"),
            Message::new_tool_result("c1", "result"),
        ];
        let api = OpenAIClient::messages_to_api(&messages);
        assert_eq!(api[0].role, "system");
        assert_eq!(api[1].role, "user");
        assert_eq!(api[2].role, "assistant");
        assert_eq!(api[3].role, "tool");
    }

    #[test]
    fn test_messages_to_api_tool_calls() {
        let mut msg = Message::new_assistant("let me check");
        msg = msg.with_tool_calls(vec![ToolCall {
            id: "call_1".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "read_file".into(),
                arguments: r#"{"path":"test.rs"}"#.into(),
            },
        }]);
        let api = OpenAIClient::messages_to_api(&[msg]);
        assert!(api[0].tool_calls.is_some());
        let tc = api[0].tool_calls.as_ref().unwrap();
        assert_eq!(tc[0].function.name, "read_file");
    }

    #[test]
    fn test_tools_to_api() {
        let tools = vec![ToolDefinition {
            name: "bash".into(),
            description: "Run command".into(),
            parameters: serde_json::json!({"type": "object"}),
        }];
        let api = OpenAIClient::tools_to_api(&tools);
        assert_eq!(api[0].tool_type, "function");
        assert_eq!(api[0].function.name, "bash");
    }
}
