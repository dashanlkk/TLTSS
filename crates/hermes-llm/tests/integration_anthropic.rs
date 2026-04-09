//! Integration tests for Anthropic client against live API.
//!
//! Run with: ANTHROPIC_API_KEY=... cargo test --package hermes-llm --test integration_anthropic -- --nocapture --ignored

use hermes_cfg::prelude::*;
use hermes_cfg::traits::LlmClient;
use hermes_llm::AnthropicClient;

fn get_client() -> Option<AnthropicClient> {
    let key = std::env::var("ANTHROPIC_API_KEY").ok()?;
    let base = std::env::var("ANTHROPIC_BASE_URL")
        .unwrap_or_else(|_| "https://api.anthropic.com".to_string());
    let model = std::env::var("HERMES_MODEL")
        .unwrap_or_else(|_| "claude-sonnet-4-5".to_string());
    Some(
        AnthropicClient::new(&base, &key, &model)
            .with_max_tokens(200)
            .with_temperature(0.7),
    )
}

#[tokio::test]
#[ignore] // Requires ANTHROPIC_API_KEY
async fn test_anthropic_simple_chat() {
    let client = get_client().expect("Set ANTHROPIC_API_KEY to run this test");

    let messages = vec![Message::new_user("Hello! Who are you? Answer in one sentence.")];
    let result = client.complete(&messages, &[]).await;

    match result {
        Ok(msg) => {
            println!("Response: {}", msg.content);
            assert!(!msg.content.is_empty());
        }
        Err(e) => {
            let err_str = format!("{}", e);
            if err_str.contains("429") || err_str.contains("balance") {
                eprintln!("Skipping: rate/billing: {}", err_str);
            } else {
                panic!("API error: {}", err_str);
            }
        }
    }
}

#[tokio::test]
#[ignore]
async fn test_anthropic_tool_calling() {
    let client = get_client().expect("Set ANTHROPIC_API_KEY to run this test");

    let tools = vec![ToolDefinition {
        name: "get_weather".to_string(),
        description: "Get the current weather for a location".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "location": { "type": "string", "description": "City name" }
            },
            "required": ["location"]
        }),
    }];

    let messages = vec![Message::new_user("What's the weather in Beijing?")];
    let result = client.complete(&messages, &tools).await;

    match result {
        Ok(msg) => {
            println!("Content: {}", msg.content);
            if let Some(calls) = &msg.tool_calls {
                println!("Tool calls: {}", calls.len());
                for c in calls {
                    println!("  {}({})", c.function.name, c.function.arguments);
                }
                assert_eq!(calls[0].function.name, "get_weather");
            } else {
                println!("Model answered directly (no tool call)");
            }
        }
        Err(e) => {
            let err_str = format!("{}", e);
            if err_str.contains("429") || err_str.contains("balance") {
                eprintln!("Skipping: {}", err_str);
            } else {
                panic!("API error: {}", err_str);
            }
        }
    }
}
