//! Hermes Tools — 工具注册与内置工具

pub mod registry;
pub mod builtin;
pub mod builtin_ext;
pub mod approval;
pub mod coerce;
pub mod parallel;
pub mod destructive;
pub mod overflow;

pub use registry::ToolRegistry;
pub use approval::ApprovalManager;
pub use parallel::{should_parallelize, ToolCallInfo, MAX_TOOL_WORKERS};
pub use destructive::is_destructive_command;
pub use overflow::{cap_single_output, apply_turn_budget, DEFAULT_PER_TOOL_LIMIT, DEFAULT_TURN_BUDGET, ProtectedOutput};

// Re-export extended built-in tools for ergonomic access
pub use builtin_ext::{
    WebSearchTool, WebExtractTool, SessionSearchTool,
    SkillsListTool, SkillViewTool, VisionAnalyzeTool, PatchTool,
};
