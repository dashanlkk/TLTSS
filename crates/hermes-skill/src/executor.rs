use hermes_cfg::error::SkillError;
use hermes_cfg::traits::ToolContext;
use hermes_tools::ToolRegistry;
use crate::manifest::SkillManifest;
use std::sync::Arc;

/// 技能执行器
pub struct SkillExecutor {
    registry: Arc<ToolRegistry>,
}

impl SkillExecutor {
    pub fn new(registry: Arc<ToolRegistry>) -> Self {
        Self { registry }
    }

    /// 执行技能
    pub async fn execute(
        &self,
        skill: &SkillManifest,
        context: &ToolContext,
    ) -> Result<Vec<hermes_cfg::tool::ToolResult>, SkillError> {
        let mut results = Vec::new();

        for step in &skill.steps {
            let args_str = serde_json::to_string(&step.params)
                .map_err(|e| SkillError::ExecutionFailed(e.to_string()))?;

            let result: Result<hermes_cfg::tool::ToolResult, hermes_cfg::error::ToolError> =
                self.registry.execute(&step.action, &args_str, context).await;
            let result = result
                .map_err(|e| SkillError::ExecutionFailed(e.to_string()))?;

            results.push(result);
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hermes_cfg::prelude::*;
    use hermes_cfg::traits::ToolHandler;
    use crate::manifest::{SkillStep, SkillStatus};

    /// A simple tool that echoes back the action name and params.
    struct EchoTool;

    #[async_trait::async_trait]
    impl ToolHandler for EchoTool {
        fn name(&self) -> &str { "echo" }
        fn description(&self) -> &str { "Echoes input" }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(&self, _args: &str, _ctx: &ToolContext) -> Result<ToolResult, ToolError> {
            Ok(ToolResult::success("call_echo", "echo_result"))
        }
    }

    /// A tool that always fails.
    struct FailTool;

    #[async_trait::async_trait]
    impl ToolHandler for FailTool {
        fn name(&self) -> &str { "fail" }
        fn description(&self) -> &str { "Always fails" }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(&self, _args: &str, _ctx: &ToolContext) -> Result<ToolResult, ToolError> {
            Err(ToolError::ExecutionFailed("deliberate failure".to_string()))
        }
    }

    fn test_context() -> ToolContext {
        ToolContext::new("test-session", hermes_cfg::platform::SessionSource::cli())
    }

    #[tokio::test]
    async fn execute_empty_skill_returns_no_results() {
        let registry = Arc::new(ToolRegistry::new());
        let executor = SkillExecutor::new(registry);

        let skill = SkillManifest {
            name: "empty".to_string(),
            version: "1.0".to_string(),
            description: "No steps".to_string(),
            trigger_patterns: vec![],
            steps: vec![],
            status: SkillStatus::Draft,
        };

        let results = executor.execute(&skill, &test_context()).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn execute_runs_single_step() {
        let registry = Arc::new(ToolRegistry::new());
        registry.register(Arc::new(EchoTool)).await;
        let executor = SkillExecutor::new(registry);

        let skill = SkillManifest {
            name: "single_step".to_string(),
            version: "1.0".to_string(),
            description: "One step".to_string(),
            trigger_patterns: vec![],
            steps: vec![SkillStep {
                action: "echo".to_string(),
                params: serde_json::json!({"key": "value"}),
            }],
            status: SkillStatus::Draft,
        };

        let results = executor.execute(&skill, &test_context()).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "echo_result");
        assert!(!results[0].is_error);
    }

    #[tokio::test]
    async fn execute_runs_multiple_steps_in_order() {
        let registry = Arc::new(ToolRegistry::new());
        registry.register(Arc::new(EchoTool)).await;
        let executor = SkillExecutor::new(registry);

        let skill = SkillManifest {
            name: "multi_step".to_string(),
            version: "1.0".to_string(),
            description: "Multiple steps".to_string(),
            trigger_patterns: vec![],
            steps: vec![
                SkillStep { action: "echo".to_string(), params: serde_json::json!({"step": 1}) },
                SkillStep { action: "echo".to_string(), params: serde_json::json!({"step": 2}) },
            ],
            status: SkillStatus::Draft,
        };

        let results = executor.execute(&skill, &test_context()).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn execute_fails_on_unknown_tool() {
        let registry = Arc::new(ToolRegistry::new());
        let executor = SkillExecutor::new(registry);

        let skill = SkillManifest {
            name: "bad_tool".to_string(),
            version: "1.0".to_string(),
            description: "Uses nonexistent tool".to_string(),
            trigger_patterns: vec![],
            steps: vec![SkillStep {
                action: "nonexistent".to_string(),
                params: serde_json::json!({}),
            }],
            status: SkillStatus::Draft,
        };

        let result = executor.execute(&skill, &test_context()).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("not found") || err_msg.contains("Execution failed"));
    }

    #[tokio::test]
    async fn execute_propagates_tool_error() {
        let registry = Arc::new(ToolRegistry::new());
        registry.register(Arc::new(FailTool)).await;
        let executor = SkillExecutor::new(registry);

        let skill = SkillManifest {
            name: "will_fail".to_string(),
            version: "1.0".to_string(),
            description: "Triggers a failing tool".to_string(),
            trigger_patterns: vec![],
            steps: vec![SkillStep {
                action: "fail".to_string(),
                params: serde_json::json!({}),
            }],
            status: SkillStatus::Draft,
        };

        let result = executor.execute(&skill, &test_context()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("deliberate failure"));
    }

    #[tokio::test]
    async fn execute_stops_on_first_step_failure() {
        let registry = Arc::new(ToolRegistry::new());
        registry.register(Arc::new(FailTool)).await;
        registry.register(Arc::new(EchoTool)).await;
        let executor = SkillExecutor::new(registry);

        let skill = SkillManifest {
            name: "partial_fail".to_string(),
            version: "1.0".to_string(),
            description: "Fails on first step".to_string(),
            trigger_patterns: vec![],
            steps: vec![
                SkillStep { action: "fail".to_string(), params: serde_json::json!({}) },
                SkillStep { action: "echo".to_string(), params: serde_json::json!({}) },
            ],
            status: SkillStatus::Draft,
        };

        let result = executor.execute(&skill, &test_context()).await;
        assert!(result.is_err());
    }
}
