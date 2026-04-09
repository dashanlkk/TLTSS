use chrono::Utc;
use cron::Schedule;
use hermes_cfg::error::CronError;
use hermes_cfg::traits::TerminalBackend;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::job::{CronJob, JobPayload, JobRun, RunStatus};

/// 定时任务调度器
pub struct Scheduler {
    jobs: Arc<RwLock<HashMap<String, CronJob>>>,
    runs: Arc<RwLock<Vec<JobRun>>>,
    terminal: Arc<dyn TerminalBackend>,
    data_dir: PathBuf,
    running: Arc<RwLock<bool>>,
}

impl Scheduler {
    pub fn new(terminal: Arc<dyn TerminalBackend>) -> Self {
        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
            runs: Arc::new(RwLock::new(Vec::new())),
            terminal,
            data_dir: PathBuf::from(".hermes/cron"),
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// 设置数据持久化目录
    pub fn with_data_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.data_dir = dir.into();
        self
    }

    /// 添加任务
    pub async fn add_job(&self, job: CronJob) -> Result<String, CronError> {
        // 验证 cron 表达式
        let schedule_result: Result<Schedule, cron::error::Error> = job.cron_expression.parse();
        let _schedule = schedule_result
            .map_err(|e| CronError::InvalidExpression(e.to_string()))?;

        let id = job.id.clone();
        info!("Added cron job: {} ({})", job.name, job.cron_expression);
        self.jobs.write().await.insert(id.clone(), job);
        self.persist().await?;
        Ok(id)
    }

    /// 删除任务
    pub async fn remove_job(&self, id: &str) -> Result<(), CronError> {
        self.jobs.write().await.remove(id);
        info!("Removed cron job: {}", id);
        self.persist().await?;
        Ok(())
    }

    /// 启用/禁用任务
    pub async fn toggle_job(&self, id: &str, enabled: bool) -> Result<(), CronError> {
        let mut jobs = self.jobs.write().await;
        if let Some(job) = jobs.get_mut(id) {
            job.enabled = enabled;
            info!("Job {} {}", id, if enabled { "enabled" } else { "disabled" });
        }
        drop(jobs);
        self.persist().await?;
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
                // 消息类型记录为执行成功，具体路由由 Agent 层处理
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

    /// 持久化任务到 JSON 文件
    async fn persist(&self) -> Result<(), CronError> {
        let dir = &self.data_dir;
        tokio::fs::create_dir_all(dir).await
            .map_err(|e| CronError::ExecutionFailed(e.to_string()))?;

        let jobs = self.jobs.read().await;
        let json = serde_json::to_string_pretty(&*jobs)
            .map_err(|e| CronError::ExecutionFailed(e.to_string()))?;

        let path = dir.join("jobs.json");
        tokio::fs::write(&path, json).await
            .map_err(|e| CronError::ExecutionFailed(e.to_string()))?;

        Ok(())
    }

    /// 从文件加载持久化的任务
    pub async fn load_from_dir(&self) -> Result<(), CronError> {
        let path = self.data_dir.join("jobs.json");
        if !path.exists() {
            return Ok(());
        }

        let content = tokio::fs::read_to_string(&path).await
            .map_err(|e| CronError::ExecutionFailed(e.to_string()))?;

        let jobs: HashMap<String, CronJob> = serde_json::from_str(&content)
            .map_err(|e| CronError::ExecutionFailed(e.to_string()))?;

        let count = jobs.len();
        *self.jobs.write().await = jobs;
        info!("Loaded {} cron jobs from {}", count, path.display());
        Ok(())
    }

    /// 启动后台调度循环
    pub async fn start(&self) {
        let mut running = self.running.write().await;
        if *running {
            warn!("Scheduler already running");
            return;
        }
        *running = true;
        drop(running);

        let jobs = self.jobs.clone();
        let runs = self.runs.clone();
        let terminal = self.terminal.clone();
        let is_running = self.running.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(10));
            let mut last_tick = chrono::Utc::now();

            loop {
                interval.tick().await;

                {
                    let r = is_running.read().await;
                    if !*r {
                        break;
                    }
                }

                let now = chrono::Utc::now();
                let job_list: Vec<CronJob> = {
                    let j = jobs.read().await;
                    j.values().filter(|j| j.enabled).cloned().collect()
                };

                for job in job_list {
                    // 检查 cron 表达式是否匹配当前时间窗口
                    if let Ok(schedule) = job.cron_expression.parse::<Schedule>() {
                        // 查找 last_tick 到 now 之间是否有匹配的触发时间
                        let should_run = schedule
                            .after(&last_tick)
                            .take_while(|t| *t <= now)
                            .count()
                            > 0;

                        if should_run {
                            let mut run = JobRun::new(&job.id);
                            run.status = RunStatus::Running;
                            run.started_at = Some(Utc::now());

                            let result = match &job.payload {
                                JobPayload::Command { command } => {
                                    terminal.execute(command, None).await
                                        .map_err(|e| CronError::ExecutionFailed(e.to_string()))
                                }
                                JobPayload::Message { text } => {
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
                            runs.write().await.push(run);

                            // 更新 last_run_at
                            if let Some(j) = jobs.write().await.get_mut(&job.id) {
                                j.last_run_at = Some(now);
                            }
                        }
                    }
                }

                last_tick = now;
            }
        });

        info!("Cron scheduler started");
    }

    /// 停止调度循环
    pub async fn stop(&self) {
        *self.running.write().await = false;
        info!("Cron scheduler stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hermes_terminal::backend::LocalBackend;

    #[tokio::test]
    async fn test_add_and_list_job() {
        let terminal = Arc::new(LocalBackend::new(std::env::temp_dir()));
        let scheduler = Scheduler::new(terminal);

        let job = CronJob::new("test", "0 * * * * *", JobPayload::Command {
            command: "echo hello".into(),
        });
        let _id = scheduler.add_job(job).await.unwrap();

        let jobs = scheduler.list_jobs().await;
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].name, "test");
    }

    #[tokio::test]
    async fn test_invalid_cron_expression() {
        let terminal = Arc::new(LocalBackend::new(std::env::temp_dir()));
        let scheduler = Scheduler::new(terminal);

        let job = CronJob::new("bad", "invalid", JobPayload::Command {
            command: "echo".into(),
        });
        let result = scheduler.add_job(job).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_run_now_command() {
        let terminal = Arc::new(LocalBackend::new(std::env::temp_dir()));
        let scheduler = Scheduler::new(terminal);

        let job = CronJob::new("echo_test", "0 * * * * *", JobPayload::Command {
            command: if cfg!(target_os = "windows") { "echo test123" } else { "echo test123" }.into(),
        });
        let id = scheduler.add_job(job).await.unwrap();
        let run = scheduler.run_now(&id).await.unwrap();

        assert_eq!(run.status, RunStatus::Completed);
        assert!(run.output.unwrap().contains("test123"));
    }

    #[tokio::test]
    async fn test_toggle_job() {
        let terminal = Arc::new(LocalBackend::new(std::env::temp_dir()));
        let scheduler = Scheduler::new(terminal);

        let job = CronJob::new("toggle", "0 * * * * *", JobPayload::Command {
            command: "echo".into(),
        });
        let id = scheduler.add_job(job).await.unwrap();

        scheduler.toggle_job(&id, false).await.unwrap();
        let job = scheduler.get_job(&id).await.unwrap();
        assert!(!job.enabled);

        scheduler.toggle_job(&id, true).await.unwrap();
        let job = scheduler.get_job(&id).await.unwrap();
        assert!(job.enabled);
    }

    #[tokio::test]
    async fn test_remove_job() {
        let terminal = Arc::new(LocalBackend::new(std::env::temp_dir()));
        let scheduler = Scheduler::new(terminal);

        let job = CronJob::new("remove_me", "0 * * * * *", JobPayload::Command {
            command: "echo".into(),
        });
        let id = scheduler.add_job(job).await.unwrap();
        assert_eq!(scheduler.list_jobs().await.len(), 1);

        scheduler.remove_job(&id).await.unwrap();
        assert_eq!(scheduler.list_jobs().await.len(), 0);
    }

    #[tokio::test]
    async fn test_message_payload() {
        let terminal = Arc::new(LocalBackend::new(std::env::temp_dir()));
        let scheduler = Scheduler::new(terminal);

        let job = CronJob::new("msg", "0 * * * * *", JobPayload::Message {
            text: "Hello from cron".into(),
        });
        let id = scheduler.add_job(job).await.unwrap();
        let run = scheduler.run_now(&id).await.unwrap();

        assert_eq!(run.status, RunStatus::Completed);
        assert_eq!(run.output.unwrap(), "Hello from cron");
    }

    #[tokio::test]
    async fn test_persist_and_load() {
        let dir = std::env::temp_dir().join("hermes_cron_test_persist");
        let _ = std::fs::remove_dir_all(&dir);

        let terminal = Arc::new(LocalBackend::new(std::env::temp_dir()));

        // 写入
        {
            let scheduler = Scheduler::new(terminal.clone()).with_data_dir(&dir);
            let job = CronJob::new("persist_test", "0 * * * * *", JobPayload::Command {
                command: "echo persisted".into(),
            });
            scheduler.add_job(job).await.unwrap();
        }

        // 读取
        {
            let scheduler = Scheduler::new(terminal).with_data_dir(&dir);
            scheduler.load_from_dir().await.unwrap();
            let jobs = scheduler.list_jobs().await;
            assert_eq!(jobs.len(), 1);
            assert_eq!(jobs[0].name, "persist_test");
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_list_runs() {
        let terminal = Arc::new(LocalBackend::new(std::env::temp_dir()));
        let scheduler = Scheduler::new(terminal);

        let job = CronJob::new("runs_test", "0 * * * * *", JobPayload::Command {
            command: "echo runs".into(),
        });
        let id = scheduler.add_job(job).await.unwrap();

        scheduler.run_now(&id).await.unwrap();
        scheduler.run_now(&id).await.unwrap();

        let runs = scheduler.list_runs().await;
        assert_eq!(runs.len(), 2);
        assert!(runs.iter().all(|r| r.status == RunStatus::Completed));
    }
}
