use async_trait::async_trait;
use hermes_cfg::prelude::*;
use hermes_cfg::traits::{ToolContext, ToolHandler};
use tokio::sync::RwLock;
use std::sync::Arc;

use crate::client::McpClient;

/// 将 MCP 远程工具桥接为 ToolHandler
/// 通过 MCP JSON-RPC 调用远程工具，使其可注册到 ToolRegistry
pub struct McpToolAdapter {
    server_name: String,
    tool_name: String,
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
        Self {
            server_name: server_name.into(),
            tool_name: tool_name.into(),
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
        // 懒静态引用：需要在结构体中存储完整名称
        // 这里返回 tool_name，注册时使用 mcp_{server}_{tool} 格式
        &self.tool_name
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

        Ok(ToolResult::success(&self.tool_name, content))
    }

    fn to_tool_definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: format!("mcp_{}_{}", self.server_name, self.tool_name),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
        }
    }
}
