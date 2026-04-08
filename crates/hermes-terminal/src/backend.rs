use async_trait::async_trait;
use hermes_cfg::prelude::*;
use hermes_cfg::traits::{TerminalBackend, TerminalOutput};
use std::time::Duration;
use tokio::process::Command;

/// 本地命令执行后端
pub struct LocalBackend {
    working_dir: std::path::PathBuf,
}

impl LocalBackend {
    pub fn new(working_dir: impl Into<std::path::PathBuf>) -> Self {
        Self {
            working_dir: working_dir.into(),
        }
    }
}

#[async_trait]
impl TerminalBackend for LocalBackend {
    async fn execute(
        &self,
        command: &str,
        timeout: Option<Duration>,
    ) -> Result<TerminalOutput, TerminalError> {
        let timeout = timeout.unwrap_or(Duration::from_secs(30));

        let result = tokio::time::timeout(
            timeout,
            Command::new("sh")
                .arg("-c")
                .arg(command)
                .current_dir(&self.working_dir)
                .output(),
        )
        .await
        .map_err(|_| TerminalError::Timeout(timeout.as_secs()))?
        .map_err(|e| TerminalError::ExecutionFailed(e.to_string()))?;

        let output = TerminalOutput {
            stdout: String::from_utf8_lossy(&result.stdout).to_string(),
            stderr: String::from_utf8_lossy(&result.stderr).to_string(),
            exit_code: result.status.code().unwrap_or(-1),
        };

        Ok(output)
    }

    async fn close(&self) -> Result<(), TerminalError> {
        Ok(())
    }

    fn backend_name(&self) -> &str {
        "local"
    }
}
