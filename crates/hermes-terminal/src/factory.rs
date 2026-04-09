use hermes_cfg::traits::TerminalBackend;
use std::sync::Arc;

use crate::backend::LocalBackend;
use crate::docker::DockerBackend;

/// 根据配置创建对应的终端后端
pub fn create_backend(
    backend_type: &str,
    working_dir: &std::path::Path,
) -> Arc<dyn TerminalBackend> {
    match backend_type {
        "docker" => {
            Arc::new(DockerBackend::new("alpine:latest", working_dir.to_path_buf()))
        }
        _ => Arc::new(LocalBackend::new(working_dir.to_path_buf())),
    }
}

/// 创建 Docker 后端并指定镜像
pub fn create_docker_backend(
    image: &str,
    working_dir: &std::path::Path,
) -> Arc<dyn TerminalBackend> {
    Arc::new(DockerBackend::new(image, working_dir.to_path_buf()))
}
