use async_trait::async_trait;
use hermes_cfg::traits::{TerminalBackend, TerminalOutput};
use hermes_cfg::error::TerminalError;
use std::path::PathBuf;
use std::time::Duration;
use tracing::{debug, warn};

/// Docker 容器终端后端 — 通过 docker CLI 在容器中执行命令
pub struct DockerBackend {
    image: String,
    working_dir: PathBuf,
    default_timeout: Duration,
}

impl DockerBackend {
    pub fn new(image: impl Into<String>, working_dir: PathBuf) -> Self {
        Self {
            image: image.into(),
            working_dir,
            default_timeout: Duration::from_secs(30),
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = timeout;
        self
    }
}

#[async_trait]
impl TerminalBackend for DockerBackend {
    async fn execute(
        &self,
        command: &str,
        timeout: Option<Duration>,
    ) -> Result<TerminalOutput, TerminalError> {
        let timeout = timeout.unwrap_or(self.default_timeout);
        debug!("Docker exec (image={}): {}", self.image, command);

        let output = tokio::time::timeout(
            timeout,
            tokio::process::Command::new("docker")
                .args([
                    "run",
                    "--rm",
                    "-w",
                    self.working_dir.to_str().unwrap_or("/"),
                    &self.image,
                    "sh",
                    "-c",
                    command,
                ])
                .output(),
        )
        .await
        .map_err(|_| TerminalError::Timeout(timeout.as_secs()))?
        .map_err(|e| TerminalError::ExecutionFailed(e.to_string()))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(-1);

        if !output.status.success() {
            warn!(
                "Docker command failed (exit={}): {}",
                exit_code,
                stderr.trim()
            );
        }

        Ok(TerminalOutput {
            stdout,
            stderr,
            exit_code,
        })
    }

    async fn close(&self) -> Result<(), TerminalError> {
        Ok(())
    }

    fn backend_name(&self) -> &str {
        "docker"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_docker_backend_name() {
        let backend = DockerBackend::new("alpine:latest", PathBuf::from("/workspace"));
        assert_eq!(backend.backend_name(), "docker");
    }
}
