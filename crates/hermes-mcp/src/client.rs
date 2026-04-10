use hermes_cfg::error::McpError;
use hermes_cfg::tool::ToolDefinition;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::transport::StdioTransport;

/// MCP 工具描述（从 server 获取）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolInfo {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

/// MCP 客户端
pub struct McpClient {
    name: String,
    transport: StdioTransport,
    tools: Vec<McpToolInfo>,
}

impl McpClient {
    pub async fn connect(
        name: impl Into<String>,
        command: &str,
        args: &[String],
    ) -> Result<Self, McpError> {
        let transport = StdioTransport::spawn(command, args).await?;
        let client = Self {
            name: name.into(),
            transport,
            tools: Vec::new(),
        };

        // 初始化连接
        let _init = client.transport.send("initialize", Some(serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": { "name": "hermes-rs", "version": "0.1.0" }
        }))).await?;

        // Per MCP spec: client MUST send "initialized" notification after receiving init response
        client.transport.notify("notifications/initialized", None).await?;

        info!("MCP server '{}' connected", client.name);
        Ok(client)
    }

    /// 发现工具列表
    pub async fn discover_tools(&mut self) -> Result<Vec<McpToolInfo>, McpError> {
        let resp = self.transport.send("tools/list", None).await?;
        let result = resp.into_result()?;

        #[derive(Deserialize)]
        struct ToolsList {
            tools: Vec<McpToolInfo>,
        }

        let list: ToolsList = serde_json::from_value(result)
            .map_err(|e| McpError::Transport(e.to_string()))?;

        info!("MCP server '{}' has {} tools", self.name, list.tools.len());
        self.tools = list.tools.clone();
        Ok(list.tools)
    }

    /// 调用工具
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, McpError> {
        let resp = self.transport.send("tools/call", Some(serde_json::json!({
            "name": name,
            "arguments": arguments
        }))).await?;

        resp.into_result()
    }

    /// 获取已发现的工具列表
    pub fn tools(&self) -> &[McpToolInfo] {
        &self.tools
    }

    /// 转换为 ToolDefinition
    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tools.iter().map(|t| ToolDefinition {
            name: format!("mcp_{}_{}", self.name, t.name),
            description: t.description.clone().unwrap_or_default(),
            parameters: t.input_schema.clone(),
        }).collect()
    }

    /// 关闭连接
    pub async fn close(&self) -> Result<(), McpError> {
        self.transport.close().await
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spawn the mock MCP server and return a connected McpClient.
    async fn connect_mock_server(name: &str) -> McpClient {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR should be set during tests");
        let script = format!("{}/tests/fixtures/mock_mcp_server.py", manifest_dir);

        McpClient::connect(name, "python", &[script])
            .await
            .expect("Failed to connect to mock MCP server")
    }

    #[tokio::test]
    async fn test_connect_sets_name() {
        let client = connect_mock_server("my-server").await;
        assert_eq!(client.name(), "my-server");
    }

    #[tokio::test]
    async fn test_connect_name_preserved_exactly() {
        let client = connect_mock_server("test_server_123").await;
        assert_eq!(client.name(), "test_server_123");
    }

    #[tokio::test]
    async fn test_discover_tools_returns_tools() {
        let mut client = connect_mock_server("mock").await;
        let tools = client.discover_tools().await.expect("discover_tools should succeed");

        assert_eq!(tools.len(), 2, "mock server should expose 2 tools");

        let echo = tools.iter().find(|t| t.name == "echo").expect("echo tool should exist");
        assert_eq!(echo.description.as_deref(), Some("Echo back the input"));
        assert!(echo.input_schema.is_object());

        let add = tools.iter().find(|t| t.name == "add").expect("add tool should exist");
        assert_eq!(add.description.as_deref(), Some("Add two numbers"));
    }

    #[tokio::test]
    async fn test_tools_reflects_discovered_tools() {
        let mut client = connect_mock_server("mock").await;

        // Before discover_tools, tools should be empty
        assert!(client.tools().is_empty());

        client.discover_tools().await.expect("discover_tools should succeed");

        // After discover_tools, tools should be populated
        assert_eq!(client.tools().len(), 2);
    }

    #[tokio::test]
    async fn test_tool_definitions_format() {
        let mut client = connect_mock_server("mysrv").await;
        client.discover_tools().await.expect("discover_tools should succeed");

        let defs = client.tool_definitions();
        assert_eq!(defs.len(), 2);

        let echo_def = defs.iter().find(|d| d.name == "mcp_mysrv_echo").expect("echo def");
        assert_eq!(echo_def.description, "Echo back the input");
        assert!(echo_def.parameters.is_object());
        assert_eq!(echo_def.parameters["type"], "object");
    }

    #[tokio::test]
    async fn test_call_tool_echo() {
        let client = connect_mock_server("mock").await;
        let result = client
            .call_tool("echo", serde_json::json!({"message": "hello"}))
            .await
            .expect("call_tool echo should succeed");

        // The mock server returns {"content": [{"type": "text", "text": "hello"}]}
        assert!(result["content"].is_array());
        assert_eq!(result["content"][0]["text"], "hello");
    }

    #[tokio::test]
    async fn test_call_tool_add() {
        let client = connect_mock_server("mock").await;
        let result = client
            .call_tool("add", serde_json::json!({"a": 15, "b": 27}))
            .await
            .expect("call_tool add should succeed");

        assert!(result["content"].is_array());
        assert_eq!(result["content"][0]["text"], "42");
    }

    #[tokio::test]
    async fn test_call_tool_unknown_returns_error() {
        let client = connect_mock_server("mock").await;
        let result = client
            .call_tool("nonexistent_tool", serde_json::json!({}))
            .await;

        assert!(result.is_err(), "calling a nonexistent tool should fail");
    }

    #[tokio::test]
    async fn test_tool_definitions_names_prefixed() {
        let mut client = connect_mock_server("srv").await;
        client.discover_tools().await.unwrap();

        let defs = client.tool_definitions();
        for def in &defs {
            assert!(
                def.name.starts_with("mcp_srv_"),
                "tool definition name should be prefixed: {}",
                def.name
            );
        }
    }
}
