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
        let mut client = Self {
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
