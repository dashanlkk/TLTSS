use hermes_cfg::prelude::*;
use hermes_cfg::traits::{ApprovalLevel, ToolContext, ToolHandler};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::approval::ApprovalManager;

/// Approval callback type: takes (tool_name, arguments) → bool (true = approve)
type ApprovalCallback = Box<dyn Fn(&str, &str) -> bool + Send + Sync>;

/// 线程安全的工具注册表
pub struct ToolRegistry {
    tools: RwLock<HashMap<String, Arc<dyn ToolHandler>>>,
    approval: Arc<ApprovalManager>,
    /// Optional synchronous approval callback for CLI integration
    approval_callback: RwLock<Option<Arc<ApprovalCallback>>>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: RwLock::new(HashMap::new()),
            approval: Arc::new(ApprovalManager::new()),
            approval_callback: RwLock::new(None),
        }
    }

    /// 使用自定义审批管理器
    pub fn with_approval(approval: Arc<ApprovalManager>) -> Self {
        Self {
            tools: RwLock::new(HashMap::new()),
            approval,
            approval_callback: RwLock::new(None),
        }
    }

    /// Set a synchronous approval callback (works through Arc).
    /// When a tool requires approval, this callback is invoked with (tool_name, arguments).
    /// Return true to approve, false to reject.
    pub async fn set_approval_callback<F>(&self, callback: F)
    where
        F: Fn(&str, &str) -> bool + Send + Sync + 'static,
    {
        *self.approval_callback.write().await = Some(Arc::new(Box::new(callback)));
    }

    /// 获取审批管理器引用
    pub fn approval_manager(&self) -> &Arc<ApprovalManager> {
        &self.approval
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

    /// 执行指定工具（含审批检查）
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

        // 检查工具级别审批策略
        let level = self.approval.get_level(name).await;
        match level {
            ApprovalLevel::Blocked => {
                return Err(ToolError::PermissionDenied(format!(
                    "Tool '{}' is blocked by approval policy",
                    name
                )));
            }
            ApprovalLevel::RequireApproval => {
                // Try callback first (for CLI integration)
                let cb = self.approval_callback.read().await.clone();
                if let Some(callback) = cb {
                    let approved = callback(name, arguments);
                    if !approved {
                        return Err(ToolError::PermissionDenied(format!(
                            "Tool '{}' execution was rejected by user",
                            name
                        )));
                    }
                    tracing::info!("Tool '{}' approved via callback", name);
                } else {
                    // Fallback: use channel-based approval (for TUI/Gateway)
                    let call_id = uuid::Uuid::new_v4().to_string();
                    let rx = self.approval.request_approval(&call_id).await;
                    tracing::info!(
                        "Tool '{}' requires approval (request_id={})",
                        name,
                        call_id
                    );
                    match rx.await {
                        Ok(true) => {
                            tracing::info!("Tool '{}' approved", name);
                        }
                        Ok(false) => {
                            return Err(ToolError::PermissionDenied(format!(
                                "Tool '{}' execution was rejected",
                                name
                            )));
                        }
                        Err(_) => {
                            return Err(ToolError::PermissionDenied(format!(
                                "Tool '{}' approval channel closed",
                                name
                            )));
                        }
                    }
                }
            }
            ApprovalLevel::AutoApprove => {}
            _ => {}
        }

        // 也检查 context 中的审批级别
        match context.approval_level {
            ApprovalLevel::Blocked => {
                return Err(ToolError::PermissionDenied(format!(
                    "Context blocks tool execution: '{}'",
                    name
                )));
            }
            ApprovalLevel::RequireApproval => {
                // context 级别的审批与工具级别类似
                let call_id = uuid::Uuid::new_v4().to_string();
                let rx = self.approval.request_approval(&call_id).await;
                tracing::info!(
                    "Context requires approval for '{}' (request_id={})",
                    name,
                    call_id
                );
                match rx.await {
                    Ok(true) => {}
                    Ok(false) => {
                        return Err(ToolError::PermissionDenied(format!(
                            "Tool '{}' rejected by context approval",
                            name
                        )));
                    }
                    Err(_) => {
                        return Err(ToolError::PermissionDenied(format!(
                            "Tool '{}' context approval channel closed",
                            name
                        )));
                    }
                }
            }
            ApprovalLevel::AutoApprove => {}
            _ => {}
        }

        let result = handler.execute(arguments, context).await?;

        // 过滤工具输出中的敏感环境变量值
        let filtered_content = hermes_security::env_filter::filter_sensitive(&result.content);
        Ok(ToolResult {
            tool_call_id: result.tool_call_id,
            content: filtered_content,
            is_error: result.is_error,
        })
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

    #[tokio::test]
    async fn test_execute_blocked_tool() {
        let registry = ToolRegistry::new();
        registry.register(Arc::new(DummyTool)).await;
        registry.approval_manager().set_level("dummy", ApprovalLevel::Blocked).await;

        let ctx = ToolContext::new("s1", SessionSource::cli());
        let result = registry.execute("dummy", "{}", &ctx).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("blocked"));
    }

    #[tokio::test]
    async fn test_execute_blocked_context() {
        let registry = ToolRegistry::new();
        registry.register(Arc::new(DummyTool)).await;

        let ctx = ToolContext::new("s1", SessionSource::cli())
            .with_approval(ApprovalLevel::Blocked);
        let result = registry.execute("dummy", "{}", &ctx).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("blocks"));
    }
}
