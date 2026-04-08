use hermes_cfg::prelude::*;
use hermes_cfg::traits::{ToolContext, ToolHandler};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// 线程安全的工具注册表
pub struct ToolRegistry {
    tools: RwLock<HashMap<String, Arc<dyn ToolHandler>>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: RwLock::new(HashMap::new()),
        }
    }

    /// 注册工具
    pub async fn register(&self, handler: Arc<dyn ToolHandler>) {
        let name = handler.name().to_string();
        self.tools.write().await.insert(name, handler);
    }

    /// 查找工具
    pub async fn get(&self, name: &str) -> Option<Arc<dyn ToolHandler>> {
        self.tools.read().await.get(name).cloned()
    }

    /// 列出所有已注册工具
    pub async fn list(&self) -> Vec<Arc<dyn ToolHandler>> {
        let tools = self.tools.read().await;
        tools.values().cloned().collect()
    }

    /// 列出所有工具的 OpenAI 定义
    pub async fn tool_definitions(&self) -> Vec<ToolDefinition> {
        let tools = self.tools.read().await;
        tools.values().map(|t| t.to_tool_definition()).collect()
    }

    /// 执行指定工具
    pub async fn execute(
        &self,
        name: &str,
        arguments: &str,
        context: &ToolContext,
    ) -> Result<ToolResult, ToolError> {
        let handler = self
            .get(name)
            .await
            .ok_or_else(|| ToolError::NotFound(name.to_string()))?;
        handler.execute(arguments, context).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyTool;

    #[async_trait::async_trait]
    impl ToolHandler for DummyTool {
        fn name(&self) -> &str { "dummy" }
        fn description(&self) -> &str { "A dummy tool for testing" }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(&self, _args: &str, _ctx: &ToolContext) -> Result<ToolResult, ToolError> {
            Ok(ToolResult::success("dummy", "ok"))
        }
    }

    #[tokio::test]
    async fn test_register_and_find() {
        let registry = ToolRegistry::new();
        registry.register(Arc::new(DummyTool)).await;
        assert!(registry.get("dummy").await.is_some());
        assert!(registry.get("nonexistent").await.is_none());
    }

    #[tokio::test]
    async fn test_list_tools() {
        let registry = ToolRegistry::new();
        registry.register(Arc::new(DummyTool)).await;
        let list = registry.list().await;
        assert_eq!(list.len(), 1);
    }

    #[tokio::test]
    async fn test_execute() {
        let registry = ToolRegistry::new();
        registry.register(Arc::new(DummyTool)).await;
        let ctx = ToolContext::new("s1", SessionSource::cli());
        let result = registry.execute("dummy", "{}", &ctx).await.unwrap();
        assert!(!result.is_error);
    }
}
