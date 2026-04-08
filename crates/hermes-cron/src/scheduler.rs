use chrono::Utc;
use hermes_cfg::error::CronError;
use hermes_cfg::traits::TerminalBackend;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info};

use crate::job::{CronJob, JobPayload, JobRun, RunStatus};

/// 定时任务调度器
pub struct Scheduler {
    jobs: Arc<RwLock<HashMap<String, CronJob>>>,
    runs: Arc<RwLock<Vec<JobRun>>>,
    terminal: Arc<dyn TerminalBackend>,
}

impl Scheduler {
    pub fn new(terminal: Arc<dyn TerminalBackend>) -> Self {
        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
            runs: Arc::new(RwLock::new(Vec::new())),
            terminal,
        }
    }

    /// 添加任务
    pub async fn add_job(&self, job: CronJob) -> Result<String, CronError> {
        // 验证 cron 表达式
        let schedule_result: Result<cron::Schedule, cron::error::Error> = job.cron_expression.parse();
        let _schedule = schedule_result
            .map_err(|e| CronError::InvalidExpression(e.to_string()))?;

        let id = job.id.clone();
        info!("Added cron job: {} ({})", job.name, job.cron_expression);
        self.jobs.write().await.insert(id.clone(), job);
        Ok(id)
    }

    /// 删除任务
    pub async fn remove_job(&self, id: &str) -> Result<(), CronError> {
        self.jobs.write().await.remove(id);
        info!("Removed cron job: {}", id);
        Ok(())
    }

    /// 启用/禁用任务
    pub async fn toggle_job(&self, id: &str, enabled: bool) -> Result<(), CronError> {
        let mut jobs = self.jobs.write().await;
        if let Some(job) = jobs.get_mut(id) {
            job.enabled = enabled;
            info!("Job {} {}", id, if enabled { "enabled" } else { "disabled" });
        }
        Ok(())
    }

    /// 手动触发任务
    pub async fn run_now(&self, id: &str) -> Result<JobRun, CronError> {
        let job = self.jobs.read().await.get(id).cloned()
            .ok_or_else(|| CronError::NotFound(id.to_string()))?;

        self.execute_job(&job).await
    }

    /// 执行任务并记录
    async fn execute_job(&self, job: &CronJob) -> Result<JobRun, CronError> {
        let mut run = JobRun::new(&job.id);
        run.status = RunStatus::Running;
        run.started_at = Some(Utc::now());

        let result = match &job.payload {
            JobPayload::Command { command } => {
                self.terminal.execute(command, None).await
                    .map_err(|e| CronError::ExecutionFailed(e.to_string()))
            }
            JobPayload::Message { text } => {
                // 消息类型由 Agent 主循环处理，此处仅记录
                Ok(hermes_cfg::traits::TerminalOutput::success(text.clone()))
            }
        };

        match result {
            Ok(output) => {
                run.status = RunStatus::Completed;
                run.output = Some(output.stdout);
            }
            Err(e) => {
                run.status = RunStatus::Failed;
                run.error = Some(e.to_string());
            }
        }
        run.finished_at = Some(Utc::now());

        self.runs.write().await.push(run.clone());
        Ok(run)
    }

    /// 列出所有任务
    pub async fn list_jobs(&self) -> Vec<CronJob> {
        self.jobs.read().await.values().cloned().collect()
    }

    /// 列出执行记录
    pub async fn list_runs(&self) -> Vec<JobRun> {
        self.runs.read().await.clone()
    }

    /// 获取任务
    pub async fn get_job(&self, id: &str) -> Option<CronJob> {
        self.jobs.read().await.get(id).cloned()
    }
}
