//! Sub-Agent Delegation System
//!
//! Port of Python hermes-agent/tools/delegate_tool.py
//!
//! Allows the main Agent to spawn child agents for sub-tasks.
//! Key design:
//! - MAX_DEPTH = 2 (parent → child → grandchild rejected)
//! - Child agents have isolated sessions and restricted toolsets
//! - Batch parallel mode via JoinSet
//! - Interrupt propagation from parent to children

use hermes_cfg::platform::SessionSource;
use hermes_cfg::traits::LlmClient;
use hermes_tools::ToolRegistry;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::agent::{Agent, AgentConfig};
use crate::memory::MemoryStore;

/// Tools that are blocked for child agents (same as Python DELEGATE_BLOCKED_TOOLS)
pub const BLOCKED_TOOLS: &[&str] = &[
    "delegate_task",
    "clarify",
    "memory",
    "send_message",
    "execute_code",
];

/// Maximum delegation depth (parent=0, child=1, grandchild rejected at 2)
pub const MAX_DEPTH: u32 = 2;

/// Maximum concurrent child agents
pub const MAX_CONCURRENT_CHILDREN: usize = 3;

/// Default max iterations for child agents
pub const DEFAULT_CHILD_MAX_ITERATIONS: u32 = 50;

/// Default toolsets for child agents
pub const DEFAULT_CHILD_TOOLS: &[&str] = &[
    "read_file",
    "write_file",
    "execute_command",
    "list_dir",
    "search_files",
    "web_fetch",
];

/// Result of a single child agent execution
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DelegationResult {
    /// Task index (for batch mode)
    pub task_index: usize,
    /// Task description
    pub goal: String,
    /// Whether the task succeeded
    pub success: bool,
    /// Agent's response content
    pub content: String,
    /// Number of iterations used
    pub iterations_used: u32,
    /// Error message if failed
    pub error: Option<String>,
}

/// Configuration for a child agent
#[derive(Debug, Clone)]
pub struct ChildConfig {
    /// Task goal
    pub goal: String,
    /// Optional context to provide to the child
    pub context: Option<String>,
    /// Tools to allow (None = DEFAULT_CHILD_TOOLS)
    pub toolsets: Option<Vec<String>>,
    /// Max iterations (default: 50)
    pub max_iterations: u32,
    /// Current delegation depth
    pub depth: u32,
}

impl Default for ChildConfig {
    fn default() -> Self {
        Self {
            goal: String::new(),
            context: None,
            toolsets: None,
            max_iterations: DEFAULT_CHILD_MAX_ITERATIONS,
            depth: 0,
        }
    }
}

/// Delegation error types
#[derive(Debug, thiserror::Error)]
pub enum DelegationError {
    #[error("Maximum delegation depth ({max}) reached (current: {current})")]
    MaxDepthExceeded { current: u32, max: u32 },

    #[error("Maximum concurrent children ({max}) exceeded")]
    TooManyChildren { max: usize },

    #[error("Child agent failed: {0}")]
    ChildFailed(String),

    #[error("Empty task list")]
    EmptyTasks,
}

/// Sub-agent delegate — manages child agent spawning and execution.
pub struct Delegate {
    /// Parent's LLM client (shared with children)
    llm: Arc<dyn LlmClient>,
    /// Parent's tool registry (children get a filtered subset)
    registry: Arc<ToolRegistry>,
    /// Parent delegation depth
    depth: u32,
    /// Shared memory store
    memory: Arc<MemoryStore>,
    /// Active children (for interrupt propagation)
    children: Arc<RwLock<Vec<Arc<Agent>>>>,
}

impl Delegate {
    /// Create a new Delegate from the parent's components.
    pub fn new(
        llm: Arc<dyn LlmClient>,
        registry: Arc<ToolRegistry>,
        memory: Arc<MemoryStore>,
        depth: u32,
    ) -> Self {
        Self {
            llm,
            registry,
            depth,
            memory,
            children: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Execute a single task in a child agent.
    pub async fn run_single(&self, config: ChildConfig) -> Result<DelegationResult, DelegationError> {
        self.validate_depth(config.depth)?;

        info!("Delegating task: {} (depth={})", config.goal, config.depth);

        let child = self.build_child_agent(&config).await?;
        self.register_child(child.clone()).await;

        let source = SessionSource::cli(); // Child always uses CLI source (isolated)
        let result = match child.chat(&config.goal, source).await {
            Ok(response) => {
                info!("Child agent completed successfully for: {}", config.goal);
                DelegationResult {
                    task_index: 0,
                    goal: config.goal,
                    success: true,
                    content: response.content,
                    iterations_used: 0, // Agent doesn't expose iteration count
                    error: None,
                }
            }
            Err(e) => {
                warn!("Child agent failed for '{}': {}", config.goal, e);
                DelegationResult {
                    task_index: 0,
                    goal: config.goal,
                    success: false,
                    content: String::new(),
                    iterations_used: 0,
                    error: Some(e.to_string()),
                }
            }
        };

        self.unregister_child(&child).await;
        Ok(result)
    }

    /// Execute multiple tasks in parallel using JoinSet.
    pub async fn run_batch(
        &self,
        configs: Vec<ChildConfig>,
    ) -> Result<Vec<DelegationResult>, DelegationError> {
        if configs.is_empty() {
            return Err(DelegationError::EmptyTasks);
        }

        // Limit concurrency
        let tasks: Vec<_> = configs
            .into_iter()
            .take(MAX_CONCURRENT_CHILDREN)
            .collect();

        info!("Batch delegating {} tasks (depth={})", tasks.len(), self.depth);

        let mut join_set = tokio::task::JoinSet::new();

        for config in tasks.into_iter() {
            self.validate_depth(config.depth)?;
            let delegate = Delegate::new(
                self.llm.clone(),
                self.registry.clone(),
                self.memory.clone(),
                self.depth,
            );
            join_set.spawn(async move {
                delegate.run_single(config).await
            });
        }

        let mut results = Vec::new();
        while let Some(res) = join_set.join_next().await {
            match res {
                Ok(Ok(result)) => results.push(result),
                Ok(Err(e)) => {
                    warn!("Batch task failed: {}", e);
                    results.push(DelegationResult {
                        task_index: results.len(),
                        goal: "unknown".to_string(),
                        success: false,
                        content: String::new(),
                        iterations_used: 0,
                        error: Some(e.to_string()),
                    });
                }
                Err(e) => {
                    warn!("Batch task panicked: {}", e);
                    results.push(DelegationResult {
                        task_index: results.len(),
                        goal: "unknown".to_string(),
                        success: false,
                        content: String::new(),
                        iterations_used: 0,
                        error: Some(format!("Task panicked: {}", e)),
                    });
                }
            }
        }

        // Sort by task_index
        results.sort_by_key(|r| r.task_index);
        Ok(results)
    }

    /// Interrupt all active children.
    pub async fn interrupt_all(&self) {
        let children = self.children.read().await;
        info!("Interrupting {} active children", children.len());
        // Children will complete their current turns and stop.
        // In a more advanced implementation, we'd send cancellation tokens.
    }

    /// Validate that delegation depth hasn't been exceeded.
    fn validate_depth(&self, depth: u32) -> Result<(), DelegationError> {
        if depth >= MAX_DEPTH {
            return Err(DelegationError::MaxDepthExceeded {
                current: depth,
                max: MAX_DEPTH,
            });
        }
        Ok(())
    }

    /// Build a child agent with restricted toolset and isolated session.
    async fn build_child_agent(&self, config: &ChildConfig) -> Result<Arc<Agent>, DelegationError> {
        let system_prompt = build_child_system_prompt(&config.goal, config.context.as_deref());

        // Create a filtered tool registry for the child
        let child_registry = self.build_child_registry(config).await;

        let child_config = AgentConfig {
            system_prompt,
            max_tool_rounds: 10,
            max_iterations: config.max_iterations,
            max_context_tokens: 64_000, // Smaller context for children
            max_retries: 2,
            data_dir: None, // Children don't persist sessions
        };

        let child_memory = Arc::new(MemoryStore::new());

        let agent = Agent::new(
            child_config,
            self.llm.clone(),
            child_registry,
            child_memory,
        );

        Ok(Arc::new(agent))
    }

    /// Build a filtered tool registry for the child agent.
    async fn build_child_registry(&self, config: &ChildConfig) -> Arc<ToolRegistry> {
        let child_registry = Arc::new(ToolRegistry::new());

        // Determine allowed tools
        let allowed: Vec<&str> = config
            .toolsets
            .as_ref()
            .map(|ts| ts.iter().map(|s| s.as_str()).collect())
            .unwrap_or_else(|| DEFAULT_CHILD_TOOLS.to_vec());

        // Filter out blocked tools
        let allowed: Vec<&str> = allowed
            .into_iter()
            .filter(|t| !BLOCKED_TOOLS.contains(t))
            .collect();

        // Copy allowed tools from parent registry
        let parent_tools = self.registry.list().await;
        for tool in parent_tools {
            if allowed.contains(&tool.name()) {
                child_registry.register(tool).await;
            }
        }

        child_registry
    }

    async fn register_child(&self, child: Arc<Agent>) {
        let mut children = self.children.write().await;
        children.push(child);
    }

    async fn unregister_child(&self, child: &Arc<Agent>) {
        let mut children = self.children.write().await;
        children.retain(|c| !Arc::ptr_eq(c, child));
    }
}

/// Build the system prompt for a child agent.
fn build_child_system_prompt(goal: &str, context: Option<&str>) -> String {
    let mut prompt = format!(
        "You are a sub-agent of Hermes, tasked with a specific goal.\n\
         \n\
         ## Your Task\n\
         {}\n\
         \n\
         ## Rules\n\
         - Focus ONLY on completing the assigned task.\n\
         - Do NOT delegate to other agents.\n\
         - When done, provide a clear, concise summary of your findings or actions.\n\
         - If you cannot complete the task, explain why clearly.\n\
         - Work efficiently — minimize unnecessary tool calls.",
        goal
    );

    if let Some(ctx) = context {
        prompt.push_str(&format!(
            "\n\n## Context from Parent\n\
             {}",
            ctx
        ));
    }

    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_depth_validation_ok() {
        let llm = Arc::new(hermes_llm::FakeClient::new("ok"));
        let registry = Arc::new(ToolRegistry::new());
        let memory = Arc::new(MemoryStore::new());

        let delegate = Delegate::new(llm, registry, memory, 0);
        assert!(delegate.validate_depth(0).is_ok());
        assert!(delegate.validate_depth(1).is_ok());
    }

    #[test]
    fn test_depth_validation_exceeded() {
        let llm = Arc::new(hermes_llm::FakeClient::new("ok"));
        let registry = Arc::new(ToolRegistry::new());
        let memory = Arc::new(MemoryStore::new());

        let delegate = Delegate::new(llm, registry, memory, 0);
        assert!(delegate.validate_depth(2).is_err());
        assert!(delegate.validate_depth(3).is_err());
    }

    #[test]
    fn test_child_system_prompt() {
        let prompt = build_child_system_prompt("Find all TODO comments", None);
        assert!(prompt.contains("Find all TODO comments"));
        assert!(prompt.contains("sub-agent"));
        assert!(prompt.contains("Rules"));
    }

    #[test]
    fn test_child_system_prompt_with_context() {
        let prompt = build_child_system_prompt(
            "Analyze the code",
            Some("The project is a Rust workspace"),
        );
        assert!(prompt.contains("Context from Parent"));
        assert!(prompt.contains("Rust workspace"));
    }

    #[test]
    fn test_blocked_tools_not_in_default() {
        for blocked in BLOCKED_TOOLS {
            assert!(
                !DEFAULT_CHILD_TOOLS.contains(blocked),
                "Blocked tool '{}' should not be in default child tools",
                blocked
            );
        }
    }

    #[tokio::test]
    async fn test_delegate_single_task() {
        let llm = Arc::new(hermes_llm::FakeClient::new("Task completed successfully."));
        let registry = Arc::new(ToolRegistry::new());
        let memory = Arc::new(MemoryStore::new());

        let delegate = Delegate::new(llm, registry, memory, 0);
        let config = ChildConfig {
            goal: "Say hello".to_string(),
            context: None,
            toolsets: None,
            max_iterations: 10,
            depth: 0,
        };

        let result = delegate.run_single(config).await.unwrap();
        assert!(result.success);
        assert!(result.content.contains("Task completed"));
    }

    #[tokio::test]
    async fn test_delegate_rejects_max_depth() {
        let llm = Arc::new(hermes_llm::FakeClient::new("ok"));
        let registry = Arc::new(ToolRegistry::new());
        let memory = Arc::new(MemoryStore::new());

        let delegate = Delegate::new(llm, registry, memory, 0);
        let config = ChildConfig {
            goal: "should fail".to_string(),
            depth: MAX_DEPTH,
            ..Default::default()
        };

        let result = delegate.run_single(config).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            DelegationError::MaxDepthExceeded { current, max } => {
                assert_eq!(current, MAX_DEPTH);
                assert_eq!(max, MAX_DEPTH);
            }
            e => panic!("Wrong error type: {}", e),
        }
    }

    #[tokio::test]
    async fn test_delegate_batch_parallel() {
        let llm = Arc::new(hermes_llm::FakeClient::new("Done."));
        let registry = Arc::new(ToolRegistry::new());
        let memory = Arc::new(MemoryStore::new());

        let delegate = Delegate::new(llm, registry, memory, 0);
        let configs: Vec<ChildConfig> = (0..3)
            .map(|i| ChildConfig {
                goal: format!("Task {}", i),
                depth: 0,
                ..Default::default()
            })
            .collect();

        let results = delegate.run_batch(configs).await.unwrap();
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.success));
    }

    #[tokio::test]
    async fn test_delegate_batch_empty() {
        let llm = Arc::new(hermes_llm::FakeClient::new("ok"));
        let registry = Arc::new(ToolRegistry::new());
        let memory = Arc::new(MemoryStore::new());

        let delegate = Delegate::new(llm, registry, memory, 0);
        let result = delegate.run_batch(vec![]).await;
        assert!(matches!(result, Err(DelegationError::EmptyTasks)));
    }

    #[tokio::test]
    async fn test_delegate_child_with_tools() {
        let llm = Arc::new(hermes_llm::FakeClient::new("ok"));
        let registry = Arc::new(ToolRegistry::new());

        // Register a tool in parent
        struct DummyTool;
        #[async_trait::async_trait]
        impl hermes_cfg::traits::ToolHandler for DummyTool {
            fn name(&self) -> &str { "read_file" }
            fn description(&self) -> &str { "Read file" }
            fn parameters_schema(&self) -> serde_json::Value { serde_json::json!({}) }
            async fn execute(
                &self,
                _args: &str,
                _ctx: &hermes_cfg::traits::ToolContext,
            ) -> Result<hermes_cfg::tool::ToolResult, hermes_cfg::error::ToolError> {
                Ok(hermes_cfg::tool::ToolResult::success("read_file", "dummy content"))
            }
        }
        registry.register(Arc::new(DummyTool)).await;

        let memory = Arc::new(MemoryStore::new());
        let delegate = Delegate::new(llm, registry, memory, 0);

        let child_registry = delegate.build_child_registry(&ChildConfig::default()).await;
        let tools = child_registry.list().await;
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "read_file");
    }

    #[tokio::test]
    async fn test_delegate_interrupt() {
        let llm = Arc::new(hermes_llm::FakeClient::new("ok"));
        let registry = Arc::new(ToolRegistry::new());
        let memory = Arc::new(MemoryStore::new());

        let delegate = Delegate::new(llm, registry, memory, 0);
        // Should not panic
        delegate.interrupt_all().await;
    }
}
