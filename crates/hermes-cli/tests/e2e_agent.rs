//! End-to-end test: full agent loop with real API
//!
//! Run with: cargo test -p hermes-cli --test e2e_agent -- --nocapture --ignored

use hermes_agent::{Agent, MemoryStore};
use hermes_agent::agent::AgentConfig;
use hermes_core::config::AppConfig;
use hermes_core::provider::ProviderRegistry;
use std::sync::Arc;

fn get_client() -> Option<Arc<dyn hermes_cfg::traits::LlmClient>> {
    let key = std::env::var("ANTHROPIC_API_KEY")
        .or_else(|_| std::env::var("ANTHROPIC_AUTH_TOKEN"))
        .ok()?;
    if key.is_empty() {
        return None;
    }
    let base = std::env::var("ANTHROPIC_BASE_URL")
        .unwrap_or_else(|_| "https://api.anthropic.com".to_string());
    let model = std::env::var("HERMES_MODEL")
        .or_else(|_| std::env::var("ANTHROPIC_MODEL"))
        .unwrap_or_else(|_| "claude-sonnet-4-5".to_string());

    Some(Arc::new(
        hermes_llm::AnthropicClient::new(&base, &key, &model)
            .with_max_tokens(1024)
            .with_temperature(0.7)
            .with_prompt_caching(),
    ))
}

/// Test basic chat (no tools)
#[tokio::test]
#[ignore]
async fn test_e2e_simple_chat() {
    let llm = get_client().expect("Set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN");
    let registry = Arc::new(hermes_tools::ToolRegistry::new());
    let memory = Arc::new(MemoryStore::new());

    let agent = Agent::new(AgentConfig::default(), llm, registry, memory);
    let reply = agent
        .chat("What is 2+2? Reply with just the number.", hermes_cfg::platform::SessionSource::cli())
        .await
        .expect("chat failed");

    println!("Reply: {}", reply.content);
    assert!(!reply.content.is_empty());
    assert!(reply.content.contains('4'));
}

/// Test streaming chat
#[tokio::test]
#[ignore]
async fn test_e2e_streaming_chat() {
    let llm = get_client().expect("Set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN");
    let registry = Arc::new(hermes_tools::ToolRegistry::new());
    let memory = Arc::new(MemoryStore::new());

    let agent = Agent::new(AgentConfig::default(), llm, registry, memory);
    let mut tokens: Vec<String> = Vec::new();

    let reply = agent
        .chat_stream("Say exactly: Hello World", hermes_cfg::platform::SessionSource::cli(), |token| {
            tokens.push(token.to_string());
        })
        .await
        .expect("stream failed");

    println!("Streamed reply: {}", reply.content);
    println!("Tokens received: {}", tokens.len());
    assert!(!reply.content.is_empty());
    assert!(!tokens.is_empty());
}

/// Test tool calling (read_file)
#[tokio::test]
#[ignore]
async fn test_e2e_tool_call() {
    let llm = get_client().expect("Set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN");
    let registry = Arc::new(hermes_tools::ToolRegistry::new());
    let base_dir = std::env::current_dir().unwrap();

    registry
        .register(Arc::new(hermes_tools::builtin::ReadFileTool::new(&base_dir)))
        .await;

    let memory = Arc::new(MemoryStore::new());
    let agent = Agent::new(AgentConfig::default(), llm, registry, memory);

    let reply = agent
        .chat("Read the file Cargo.toml and tell me the project name.", hermes_cfg::platform::SessionSource::cli())
        .await
        .expect("tool call failed");

    println!("Reply: {}", reply.content);
    assert!(!reply.content.is_empty());
    assert!(
        reply.content.to_lowercase().contains("hermes")
            || reply.content.contains("TLTSS"),
        "Expected agent to mention project name, got: {}",
        reply.content
    );
}

/// Test provider auto-detection from environment (no API call, not ignored)
#[test]
fn test_provider_auto_detect() {
    let config = AppConfig::default();
    let registry = ProviderRegistry::from_app_config(&config);

    let (name, cfg) = registry.default_provider().expect("should detect provider");
    println!("Detected provider: {} (model: {}, base: {:?})", name, cfg.model, cfg.base_url);

    assert_eq!(name, "anthropic");
    assert!(!cfg.api_key.as_ref().unwrap().is_empty());
    assert!(cfg.base_url.is_some());
}
