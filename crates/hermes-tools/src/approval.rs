use hermes_cfg::traits::ApprovalLevel;
use tokio::sync::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// 工具审批管理器
pub struct ApprovalManager {
    pending: Arc<RwLock<HashMap<String, tokio::sync::oneshot::Sender<bool>>>>,
    levels: Arc<RwLock<HashMap<String, ApprovalLevel>>>,
}

impl ApprovalManager {
    pub fn new() -> Self {
        Self {
            pending: Arc::new(RwLock::new(HashMap::new())),
            levels: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// 设置工具的审批级别
    pub async fn set_level(&self, tool_name: &str, level: ApprovalLevel) {
        self.levels.write().await.insert(tool_name.to_string(), level);
    }

    /// 获取工具的审批级别
    pub async fn get_level(&self, tool_name: &str) -> ApprovalLevel {
        self.levels
            .read()
            .await
            .get(tool_name)
            .cloned()
            .unwrap_or(ApprovalLevel::AutoApprove)
    }

    /// 请求审批，返回 receiver 等待结果
    pub async fn request_approval(
        &self,
        tool_call_id: &str,
    ) -> tokio::sync::oneshot::Receiver<bool> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.pending.write().await.insert(tool_call_id.to_string(), tx);
        rx
    }

    /// 批准请求
    pub async fn approve(&self, tool_call_id: &str) -> bool {
        if let Some(tx) = self.pending.write().await.remove(tool_call_id) {
            tx.send(true).is_ok()
        } else {
            false
        }
    }

    /// 拒绝请求
    pub async fn reject(&self, tool_call_id: &str) -> bool {
        if let Some(tx) = self.pending.write().await.remove(tool_call_id) {
            tx.send(false).is_ok()
        } else {
            false
        }
    }

    /// 列出待审批请求
    pub async fn pending_ids(&self) -> Vec<String> {
        self.pending.read().await.keys().cloned().collect()
    }
}
