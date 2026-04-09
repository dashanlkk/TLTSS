use serde::{Deserialize, Serialize};

/// 工具审批权限级别
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalLevel {
    #[default]
    AutoApprove,
    RequireApproval,
    Blocked,
}

impl ApprovalLevel {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s {
            "require_approval" => Self::RequireApproval,
            "blocked" => Self::Blocked,
            _ => Self::AutoApprove,
        }
    }
}
