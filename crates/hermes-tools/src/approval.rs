use hermes_cfg::traits::ApprovalLevel;
use tokio::sync::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// 工具审批管理器
pub struct ApprovalManager {
    pending: Arc<RwLock<HashMap<String, tokio::sync::oneshot::Sender<bool>>>>,
    levels: Arc<RwLock<HashMap<String, ApprovalLevel>>>,
}

impl Default for ApprovalManager {
    fn default() -> Self {
        Self::new()
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_set_and_get_level() {
        let mgr = ApprovalManager::new();
        assert!(matches!(mgr.get_level("tool_a").await, ApprovalLevel::AutoApprove));
        mgr.set_level("tool_a", ApprovalLevel::RequireApproval).await;
        assert!(matches!(mgr.get_level("tool_a").await, ApprovalLevel::RequireApproval));
    }

    #[tokio::test]
    async fn test_approve_flow() {
        let mgr = ApprovalManager::new();
        let rx = mgr.request_approval("call_1").await;
        assert!(mgr.pending_ids().await.contains(&"call_1".to_string()));
        mgr.approve("call_1").await;
        assert!(rx.await.unwrap());
        assert!(mgr.pending_ids().await.is_empty());
    }

    #[tokio::test]
    async fn test_reject_flow() {
        let mgr = ApprovalManager::new();
        let rx = mgr.request_approval("call_2").await;
        mgr.reject("call_2").await;
        assert!(!rx.await.unwrap());
    }

    #[tokio::test]
    async fn test_approve_nonexistent() {
        let mgr = ApprovalManager::new();
        assert!(!mgr.approve("nonexistent").await);
    }
}
