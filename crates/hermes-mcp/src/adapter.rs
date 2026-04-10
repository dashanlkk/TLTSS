use async_trait::async_trait;
use hermes_cfg::prelude::*;
use hermes_cfg::traits::{ToolContext, ToolHandler};
use tokio::sync::RwLock;
use std::sync::Arc;

use crate::client::McpClient;

/// 将 MCP 远程工具桥接为 ToolHandler
/// 通过 MCP JSON-RPC 调用远程工具，使其可注册到 ToolRegistry
pub struct McpToolAdapter {
    #[allow(dead_code)]
    server_name: String,
    tool_name: String,
    /// Full registered name: mcp_{server}_{tool}
    full_name: String,
    description: String,
    parameters: serde_json::Value,
    client: Arc<RwLock<McpClient>>,
}

impl McpToolAdapter {
    pub fn new(
        server_name: impl Into<String>,
        tool_name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
        client: Arc<RwLock<McpClient>>,
    ) -> Self {
        let server_name = server_name.into();
        let tool_name = tool_name.into();
        let full_name = format!("mcp_{}_{}", server_name, tool_name);
        Self {
            server_name,
            tool_name,
            full_name,
            description: description.into(),
            parameters,
            client,
        }
    }

    /// 从 McpClient 的所有已发现工具创建适配器列表
    pub async fn from_client(client: Arc<RwLock<McpClient>>) -> Vec<Arc<dyn ToolHandler>> {
        let guard = client.read().await;
        let server_name = guard.name().to_string();
        guard.tools().iter().map(|tool| {
            Arc::new(McpToolAdapter {
                server_name: server_name.clone(),
                tool_name: tool.name.clone(),
                full_name: format!("mcp_{}_{}", server_name, tool.name),
                description: tool.description.clone().unwrap_or_default(),
                parameters: tool.input_schema.clone(),
                client: client.clone(),
            }) as Arc<dyn ToolHandler>
        }).collect()
    }
}

#[async_trait]
impl ToolHandler for McpToolAdapter {
    fn name(&self) -> &str {
        &self.full_name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> serde_json::Value {
        self.parameters.clone()
    }

    async fn execute(&self, arguments: &str, _ctx: &ToolContext) -> Result<ToolResult, ToolError> {
        let args: serde_json::Value = serde_json::from_str(arguments)
            .map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

        let client = self.client.read().await;
        let result = client.call_tool(&self.tool_name, args).await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        let content = serde_json::to_string_pretty(&result)
            .unwrap_or_else(|_| result.to_string());

        Ok(ToolResult::success(&self.full_name, content))
    }

    fn to_tool_definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.full_name.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hermes_cfg::traits::ToolHandler;

    /// Spawn the mock MCP server (Python script) and return a connected McpClient.
    async fn connect_mock_server(server_name: &str) -> Arc<RwLock<McpClient>> {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR should be set during tests");
        let script = format!("{}/tests/fixtures/mock_mcp_server.py", manifest_dir);

        let client = McpClient::connect(
            server_name,
            "python",
            &[script],
        )
        .await
        .expect("Failed to connect to mock MCP server");

        Arc::new(RwLock::new(client))
    }

    #[tokio::test]
    async fn test_adapter_new_sets_correct_name_and_description() {
        let client = connect_mock_server("myserver").await;
        let params = serde_json::json!({
            "type": "object",
            "properties": {
                "query": { "type": "string" }
            }
        });

        let adapter = McpToolAdapter::new(
            "myserver",
            "search",
            "Search for something",
            params,
            client,
        );

        assert_eq!(adapter.name(), "mcp_myserver_search");
        assert_eq!(adapter.description(), "Search for something");
    }

    #[tokio::test]
    async fn test_adapter_parameters_schema_roundtrips() {
        let client = connect_mock_server("fs").await;
        let params = serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File path" },
                "mode": { "type": "string", "enum": ["read", "write"] }
            },
            "required": ["path"]
        });

        let adapter = McpToolAdapter::new(
            "fs", "read_file", "Read a file", params.clone(), client,
        );

        let schema = adapter.parameters_schema();
        assert_eq!(schema, params);
        assert!(schema.is_object());
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["path"].is_object());
    }

    #[tokio::test]
    async fn test_adapter_to_tool_definition() {
        let client = connect_mock_server("weather").await;
        let params = serde_json::json!({"type": "object", "properties": {}});

        let adapter = McpToolAdapter::new(
            "weather", "get_forecast", "Get weather forecast", params.clone(), client,
        );

        let def = adapter.to_tool_definition();
        assert_eq!(def.name, "mcp_weather_get_forecast");
        assert_eq!(def.description, "Get weather forecast");
        assert_eq!(def.parameters, params);
    }

    #[tokio::test]
    async fn test_adapter_name_with_hyphens_and_underscores() {
        let client = connect_mock_server("srv").await;
        let adapter = McpToolAdapter::new(
            "my-server_v2",
            "search-index",
            "Search the index",
            serde_json::json!({}),
            client,
        );

        assert_eq!(adapter.name(), "mcp_my-server_v2_search-index");
    }

    #[tokio::test]
    async fn test_adapter_empty_description() {
        let client = connect_mock_server("srv").await;
        let adapter = McpToolAdapter::new(
            "srv", "noop", "", serde_json::json!({}), client,
        );

        assert_eq!(adapter.description(), "");
    }

    #[tokio::test]
    async fn test_from_client_creates_adapters_for_all_tools() {
        let client = connect_mock_server("mock").await;

        // Discover tools first so the client has a populated tool list
        {
            let mut guard = client.write().await;
            guard.discover_tools().await.expect("discover_tools should succeed");
        }

        let adapters = McpToolAdapter::from_client(client.clone()).await;
        assert_eq!(adapters.len(), 2, "mock server should expose 2 tools (echo, add)");

        let names: Vec<&str> = adapters.iter().map(|a| a.name()).collect();
        assert!(names.contains(&"mcp_mock_echo"), "should have echo tool, got: {:?}", names);
        assert!(names.contains(&"mcp_mock_add"), "should have add tool, got: {:?}", names);
    }

    #[tokio::test]
    async fn test_from_client_tool_definitions() {
        let client = connect_mock_server("mock").await;
        {
            let mut guard = client.write().await;
            guard.discover_tools().await.expect("discover_tools should succeed");
        }

        let adapters = McpToolAdapter::from_client(client.clone()).await;
        let defs: Vec<ToolDefinition> = adapters.iter().map(|a| a.to_tool_definition()).collect();

        assert_eq!(defs.len(), 2);
        // Verify the echo tool definition has correct structure
        let echo_def = defs.iter().find(|d| d.name == "mcp_mock_echo").expect("echo tool should exist");
        assert_eq!(echo_def.description, "Echo back the input");
        assert!(echo_def.parameters.is_object());
        assert_eq!(echo_def.parameters["type"], "object");
    }

    #[tokio::test]
    async fn test_adapter_execute_echo_tool() {
        let client = connect_mock_server("mock").await;
        {
            let mut guard = client.write().await;
            guard.discover_tools().await.expect("discover_tools should succeed");
        }

        let adapters = McpToolAdapter::from_client(client.clone()).await;
        let echo = adapters.iter().find(|a| a.name() == "mcp_mock_echo").expect("echo tool");
        let ctx = ToolContext::new("test-session", hermes_cfg::platform::SessionSource::cli());

        let result = echo.execute(r#"{"message": "hello world"}"#, &ctx).await;
        assert!(result.is_ok(), "execute should succeed: {:?}", result);
        let tool_result = result.unwrap();
        assert!(!tool_result.is_error);
        assert!(tool_result.content.contains("hello world"));
    }

    #[tokio::test]
    async fn test_adapter_execute_add_tool() {
        let client = connect_mock_server("mock").await;
        {
            let mut guard = client.write().await;
            guard.discover_tools().await.expect("discover_tools should succeed");
        }

        let adapters = McpToolAdapter::from_client(client.clone()).await;
        let add = adapters.iter().find(|a| a.name() == "mcp_mock_add").expect("add tool");
        let ctx = ToolContext::new("test-session", hermes_cfg::platform::SessionSource::cli());

        let result = add.execute(r#"{"a": 3, "b": 7}"#, &ctx).await;
        assert!(result.is_ok(), "execute should succeed: {:?}", result);
        let tool_result = result.unwrap();
        assert!(!tool_result.is_error);
        assert!(tool_result.content.contains("10"), "3 + 7 = 10, got: {}", tool_result.content);
    }

    #[tokio::test]
    async fn test_adapter_execute_invalid_arguments() {
        let client = connect_mock_server("mock").await;
        {
            let mut guard = client.write().await;
            guard.discover_tools().await.expect("discover_tools should succeed");
        }

        let adapters = McpToolAdapter::from_client(client.clone()).await;
        let echo = adapters.iter().find(|a| a.name() == "mcp_mock_echo").expect("echo tool");
        let ctx = ToolContext::new("test-session", hermes_cfg::platform::SessionSource::cli());

        let result = echo.execute("not valid json {{{", &ctx).await;
        assert!(result.is_err(), "invalid JSON should return an error");
    }
}
