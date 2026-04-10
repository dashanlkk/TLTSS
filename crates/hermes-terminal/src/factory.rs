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

/// 根据配置创建终端后端（带可选 Docker 镜像）
pub fn create_backend_with_config(
    backend_type: &str,
    docker_image: Option<&str>,
    working_dir: &std::path::Path,
) -> Arc<dyn TerminalBackend> {
    match backend_type {
        "docker" => {
            let image = docker_image.unwrap_or("alpine:latest");
            Arc::new(DockerBackend::new(image, working_dir.to_path_buf()))
        }
        _ => Arc::new(LocalBackend::new(working_dir.to_path_buf())),
    }
}
