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
