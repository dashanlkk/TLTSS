//! Delegate task tool — allows the agent to spawn sub-agents for sub-tasks.
//!
//! Port of Python hermes-agent/tools/delegate_tool.py
//! Lives in hermes-agent to avoid circular dependency with hermes-tools.

use async_trait::async_trait;
use hermes_cfg::prelude::*;
use hermes_cfg::traits::{LlmClient, ToolContext, ToolHandler};
use hermes_tools::ToolRegistry;
use std::sync::Arc;
use tracing::{info, warn};

use crate::delegation::{ChildConfig, Delegate, DelegationResult};
use crate::memory::MemoryStore;

/// Tool for delegating tasks to child agents.
///
/// Supports two modes:
/// 1. **Single task**: `goal` parameter — runs one child agent
/// 2. **Batch parallel**: `tasks` array — runs multiple child agents in parallel
pub struct DelegateTaskTool {
    llm: Arc<dyn LlmClient>,
    registry: Arc<ToolRegistry>,
    memory: Arc<MemoryStore>,
    depth: u32,
}

impl DelegateTaskTool {
    pub fn new(
        llm: Arc<dyn LlmClient>,
        registry: Arc<ToolRegistry>,
        memory: Arc<MemoryStore>,
        depth: u32,
    ) -> Self {
        Self { llm, registry, memory, depth }
    }
}

#[async_trait]
impl ToolHandler for DelegateTaskTool {
    fn name(&self) -> &str {
        "delegate_task"
    }

    fn description(&self) -> &str {
        "Delegate a task or batch of tasks to sub-agents. Use for: parallel work, isolated execution, complex multi-step tasks. Supports single task (goal) or batch (tasks array)."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "Single task description (use for one task)"
                },
                "tasks": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Array of task descriptions (use for parallel batch execution)"
                },
                "context": {
                    "type": "string",
                    "description": "Optional context information for the sub-agent(s)"
                },
                "toolsets": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional list of tool names to allow (default: file, terminal, web tools)"
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "Max iterations per sub-agent (default: 50)"
                }
            }
        })
    }

    async fn execute(
        &self,
        args: &str,
        _ctx: &ToolContext,
    ) -> Result<ToolResult, ToolError> {
        let params: serde_json::Value = serde_json::from_str(args)
            .map_err(|e| ToolError::ExecutionFailed(format!("Invalid JSON args: {}", e)))?;

        let delegate = Delegate::new(
            self.llm.clone(),
            self.registry.clone(),
            self.memory.clone(),
            self.depth,
        );

        let max_iterations = params.get("max_iterations")
            .and_then(|v| v.as_u64())
            .unwrap_or(50) as u32;

        let context = params.get("context")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let toolsets: Option<Vec<String>> = params.get("toolsets")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect());

        // Batch mode or single mode
        if let Some(tasks) = params.get("tasks").and_then(|v| v.as_array()) {
            let configs: Vec<ChildConfig> = tasks
                .iter()
                .enumerate()
                .filter_map(|(i, v)| v.as_str().map(|s| ChildConfig {
                    goal: format!("Task {}: {}", i + 1, s),
                    context: context.clone(),
                    toolsets: toolsets.clone(),
                    max_iterations,
                    depth: self.depth + 1,
                }))
                .collect();

            info!("Delegating batch of {} tasks", configs.len());

            match delegate.run_batch(configs).await {
                Ok(results) => {
                    let summary = format_batch_results(&results);
                    Ok(ToolResult::success("delegate_task", summary))
                }
                Err(e) => {
                    warn!("Batch delegation failed: {}", e);
                    Ok(ToolResult::error("delegate_task", &format!("Delegation failed: {}", e)))
                }
            }
        } else if let Some(goal) = params.get("goal").and_then(|v| v.as_str()) {
            let config = ChildConfig {
                goal: goal.to_string(),
                context,
                toolsets,
                max_iterations,
                depth: self.depth + 1,
            };

            info!("Delegating single task: {}", goal);

            match delegate.run_single(config).await {
                Ok(result) => {
                    let summary = format_single_result(&result);
                    Ok(ToolResult::success("delegate_task", summary))
                }
                Err(e) => {
                    warn!("Single delegation failed: {}", e);
                    Ok(ToolResult::error("delegate_task", &format!("Delegation failed: {}", e)))
                }
            }
        } else {
            Ok(ToolResult::error(
                "delegate_task",
                "Must provide either 'goal' (string) or 'tasks' (array of strings)",
            ))
        }
    }
}

fn format_single_result(result: &DelegationResult) -> String {
    let status = if result.success { "SUCCESS" } else { "FAILED" };
    let mut output = format!("## Delegation Result [{}]\n\n**Task**: {}\n", status, result.goal);

    if !result.content.is_empty() {
        output.push_str(&format!("\n**Output**:\n{}\n", result.content));
    }

    if let Some(ref error) = result.error {
        output.push_str(&format!("\n**Error**: {}\n", error));
    }

    output
}

fn format_batch_results(results: &[DelegationResult]) -> String {
    let success_count = results.iter().filter(|r| r.success).count();
    let mut output = format!(
        "## Batch Delegation Results ({}/{})\n\n",
        success_count,
        results.len()
    );

    for result in results {
        let status = if result.success { "OK" } else { "FAIL" };
        output.push_str(&format!(
            "### Task {} [{}]: {}\n",
            result.task_index + 1,
            status,
            result.goal
        ));

        if !result.content.is_empty() {
            let content = if result.content.len() > 1000 {
                format!("{}...[truncated]", &result.content[..1000])
            } else {
                result.content.clone()
            };
            output.push_str(&format!("{}\n", content));
        }

        if let Some(ref error) = result.error {
            output.push_str(&format!("Error: {}\n", error));
        }

        output.push('\n');
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_tool() -> DelegateTaskTool {
        let llm = Arc::new(hermes_llm::FakeClient::new("Child response"));
        let registry = Arc::new(ToolRegistry::new());
        let memory = Arc::new(MemoryStore::new());
        DelegateTaskTool::new(llm, registry, memory, 0)
    }

    #[test]
    fn test_tool_name() {
        let tool = create_tool();
        assert_eq!(tool.name(), "delegate_task");
    }

    #[test]
    fn test_tool_schema() {
        let tool = create_tool();
        let schema = tool.parameters_schema();
        assert!(schema["properties"]["goal"].is_object());
        assert!(schema["properties"]["tasks"].is_object());
    }

    #[tokio::test]
    async fn test_delegate_single_goal() {
        let tool = create_tool();
        let ctx = ToolContext::new("test", SessionSource::cli());
        let result = tool
            .execute(r#"{"goal": "Say hello"}"#, &ctx)
            .await
            .unwrap();

        assert!(result.content.contains("Delegation Result"));
        assert!(result.content.contains("SUCCESS"));
    }

    #[tokio::test]
    async fn test_delegate_batch_tasks() {
        let tool = create_tool();
        let ctx = ToolContext::new("test", SessionSource::cli());
        let result = tool
            .execute(r#"{"tasks": ["Task A", "Task B"]}"#, &ctx)
            .await
            .unwrap();

        assert!(result.content.contains("Batch Delegation Results"));
        assert!(result.content.contains("2/2"));
    }

    #[tokio::test]
    async fn test_delegate_no_params() {
        let tool = create_tool();
        let ctx = ToolContext::new("test", SessionSource::cli());
        let result = tool
            .execute(r#"{}"#, &ctx)
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(result.content.contains("Must provide"));
    }
}
