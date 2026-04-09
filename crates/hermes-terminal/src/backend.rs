use async_trait::async_trait;
use hermes_cfg::prelude::*;
use hermes_cfg::traits::{TerminalBackend, TerminalOutput};
use std::time::Duration;
use tokio::process::Command;
use tracing::debug;

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

    /// 构建跨平台命令（Windows 用 cmd /C，Unix 用 sh -c）
    fn build_command(&self, command: &str) -> Command {
        if cfg!(target_os = "windows") {
            let mut cmd = Command::new("cmd");
            cmd.arg("/C").arg(command).current_dir(&self.working_dir);
            cmd
        } else {
            let mut cmd = Command::new("sh");
            cmd.arg("-c").arg(command).current_dir(&self.working_dir);
            cmd
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
        let timeout_duration = timeout.unwrap_or(Duration::from_secs(30));
        debug!("Executing command: {} (timeout: {}s)", command, timeout_duration.as_secs());

        let result = tokio::time::timeout(
            timeout_duration,
            self.build_command(command).output(),
        )
        .await
        .map_err(|_| TerminalError::Timeout(timeout_duration.as_secs()))?
        .map_err(|e| TerminalError::ExecutionFailed(e.to_string()))?;

        let output = TerminalOutput {
            stdout: String::from_utf8_lossy(&result.stdout).to_string(),
            stderr: String::from_utf8_lossy(&result.stderr).to_string(),
            exit_code: result.status.code().unwrap_or(-1),
        };

        debug!("Command exit_code: {}", output.exit_code);
        Ok(output)
    }

    async fn close(&self) -> Result<(), TerminalError> {
        Ok(())
    }

    fn backend_name(&self) -> &str {
        "local"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_execute_echo() {
        let backend = LocalBackend::new(std::env::temp_dir());
        let result = backend.execute("echo hello", None).await;
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.stdout.trim().contains("hello"));
    }

    #[tokio::test]
    async fn test_execute_timeout() {
        let backend = LocalBackend::new(std::env::temp_dir());
        let result = backend.execute(
            if cfg!(target_os = "windows") { "ping -n 10 127.0.0.1" } else { "sleep 10" },
            Some(Duration::from_millis(100)),
        ).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            TerminalError::Timeout(_) => {} // expected
            other => panic!("Expected Timeout, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_execute_failing_command() {
        let backend = LocalBackend::new(std::env::temp_dir());
        let result = backend.execute(
            if cfg!(target_os = "windows") { "exit /b 1" } else { "false" },
            None,
        ).await;
        assert!(result.is_ok());
        assert_ne!(result.unwrap().exit_code, 0);
    }

    #[tokio::test]
    async fn test_backend_name() {
        let backend = LocalBackend::new(std::env::temp_dir());
        assert_eq!(backend.backend_name(), "local");
    }
}
