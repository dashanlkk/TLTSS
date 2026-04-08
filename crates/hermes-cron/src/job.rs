use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// 定时任务
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CronJob {
    pub id: String,
    pub name: String,
    pub cron_expression: String,
    pub payload: JobPayload,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub max_retries: u32,
    pub created_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_run_at: Option<DateTime<Utc>>,
}

/// 任务负载
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JobPayload {
    /// 执行 shell 命令
    Command { command: String },
    /// 发送消息给 Agent
    Message { text: String },
}

/// 任务执行记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRun {
    pub id: String,
    pub job_id: String,
    pub status: RunStatus,
    pub scheduled_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub finished_at: Option<DateTime<Utc>>,
    pub output: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

impl CronJob {
    pub fn new(
        name: impl Into<String>,
        cron_expression: impl Into<String>,
        payload: JobPayload,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.into(),
            cron_expression: cron_expression.into(),
            payload,
            enabled: true,
            max_retries: 3,
            created_at: Utc::now(),
            last_run_at: None,
        }
    }
}

impl JobRun {
    pub fn new(job_id: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            job_id: job_id.into(),
            status: RunStatus::Pending,
            scheduled_at: Utc::now(),
            started_at: None,
            finished_at: None,
            output: None,
            error: None,
        }
    }
}
