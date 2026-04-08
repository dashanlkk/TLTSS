use serde::{Deserialize, Serialize};

/// 工具审批权限级别
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalLevel {
    AutoApprove,
    RequireApproval,
    Blocked,
}

impl Default for ApprovalLevel {
    fn default() -> Self {
        Self::AutoApprove
    }
}

impl ApprovalLevel {
    pub fn from_str(s: &str) -> Self {
        match s {
            "require_approval" => Self::RequireApproval,
            "blocked" => Self::Blocked,
            _ => Self::AutoApprove,
        }
    }
}
