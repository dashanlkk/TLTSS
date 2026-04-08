use hermes_cfg::traits::TerminalBackend;
use std::sync::Arc;

use crate::backend::LocalBackend;

/// 根据配置创建对应的终端后端
pub fn create_backend(
    backend_type: &str,
    working_dir: &std::path::Path,
) -> Arc<dyn TerminalBackend> {
    match backend_type {
        "local" => Arc::new(LocalBackend::new(working_dir.to_path_buf())),
        // Docker 后端留作后续实现
        _ => Arc::new(LocalBackend::new(working_dir.to_path_buf())),
    }
}
