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
#[non_exhaustive]
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
#[non_exhaustive]
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

#[cfg(test)]
mod tests {
    use super::*;

    // ---- CronJob::new() ----

    #[test]
    fn cron_job_new_sets_defaults() {
        let job = CronJob::new("backup", "0 3 * * * *", JobPayload::Command {
            command: "cp -r /data /backup".into(),
        });

        assert!(!job.id.is_empty(), "id should be a non-empty UUID string");
        assert!(uuid::Uuid::parse_str(&job.id).is_ok(), "id should be valid UUID v4");
        assert_eq!(job.name, "backup");
        assert_eq!(job.cron_expression, "0 3 * * * *");
        assert!(job.enabled, "new jobs should be enabled by default");
        assert_eq!(job.max_retries, 3);
        assert!(job.last_run_at.is_none(), "new jobs should have no last_run_at");
        assert!(job.created_at <= Utc::now());
    }

    #[test]
    fn cron_job_new_with_message_payload() {
        let job = CronJob::new("notify", "0 9 * * 1-5", JobPayload::Message {
            text: "Standup time".into(),
        });

        assert_eq!(job.name, "notify");
        assert!(matches!(job.payload, JobPayload::Message { ref text } if text == "Standup time"));
    }

    #[test]
    fn cron_job_new_accepts_string_references() {
        // Verifies impl Into<String> works with &str
        let job = CronJob::new("name", "expr", JobPayload::Command {
            command: "echo".into(),
        });
        assert_eq!(job.name, "name");
        assert_eq!(job.cron_expression, "expr");
    }

    // ---- JobRun::new() ----

    #[test]
    fn job_run_new_sets_defaults() {
        let run = JobRun::new("job-001");

        assert!(!run.id.is_empty());
        assert!(uuid::Uuid::parse_str(&run.id).is_ok(), "id should be valid UUID v4");
        assert_eq!(run.job_id, "job-001");
        assert_eq!(run.status, RunStatus::Pending);
        assert!(run.started_at.is_none());
        assert!(run.finished_at.is_none());
        assert!(run.output.is_none());
        assert!(run.error.is_none());
        assert!(run.scheduled_at <= Utc::now());
    }

    #[test]
    fn job_run_new_accepts_string() {
        let run = JobRun::new(String::from("job-002"));
        assert_eq!(run.job_id, "job-002");
    }

    // ---- CronJob serialization round-trip ----

    #[test]
    fn cron_job_roundtrip_json() {
        let original = CronJob::new("test", "0 * * * * *", JobPayload::Command {
            command: "echo hello".into(),
        });

        let json = serde_json::to_string(&original).expect("serialization should succeed");
        let restored: CronJob = serde_json::from_str(&json).expect("deserialization should succeed");

        assert_eq!(restored.id, original.id);
        assert_eq!(restored.name, original.name);
        assert_eq!(restored.cron_expression, original.cron_expression);
        assert_eq!(restored.enabled, original.enabled);
        assert_eq!(restored.max_retries, original.max_retries);
        assert_eq!(restored.last_run_at, original.last_run_at);
    }

    #[test]
    fn cron_job_with_message_payload_roundtrip() {
        let original = CronJob::new("msg", "0 0 * * *", JobPayload::Message {
            text: "daily report".into(),
        });

        let json = serde_json::to_string(&original).unwrap();
        let restored: CronJob = serde_json::from_str(&json).unwrap();

        assert!(matches!(restored.payload, JobPayload::Message { ref text } if text == "daily report"));
    }

    #[test]
    fn cron_job_skips_none_last_run_at() {
        let job = CronJob::new("test", "0 * * * * *", JobPayload::Command {
            command: "true".into(),
        });

        let json = serde_json::to_string(&job).unwrap();
        assert!(!json.contains("last_run_at"), "None last_run_at should be skipped");
    }

    // ---- JobRun serialization round-trip ----

    #[test]
    fn job_run_roundtrip_json() {
        let mut run = JobRun::new("job-abc");
        run.status = RunStatus::Running;
        run.started_at = Some(Utc::now());
        run.output = Some("partial output".into());

        let json = serde_json::to_string(&run).unwrap();
        let restored: JobRun = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.id, run.id);
        assert_eq!(restored.job_id, "job-abc");
        assert_eq!(restored.status, RunStatus::Running);
        assert!(restored.started_at.is_some());
        assert_eq!(restored.output.unwrap(), "partial output");
        assert!(restored.finished_at.is_none());
        assert!(restored.error.is_none());
    }

    #[test]
    fn job_run_completed_roundtrip() {
        let mut run = JobRun::new("job-done");
        run.status = RunStatus::Completed;
        run.started_at = Some(Utc::now());
        run.finished_at = Some(Utc::now());
        run.output = Some("done".into());

        let json = serde_json::to_string(&run).unwrap();
        let restored: JobRun = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.status, RunStatus::Completed);
        assert!(restored.finished_at.is_some());
    }

    // ---- JobPayload serialization ----

    #[test]
    fn job_payload_command_serialization() {
        let payload = JobPayload::Command { command: "ls -la".into() };
        let json = serde_json::to_string(&payload).unwrap();

        assert!(json.contains("Command"));
        assert!(json.contains("ls -la"));

        let restored: JobPayload = serde_json::from_str(&json).unwrap();
        assert!(matches!(restored, JobPayload::Command { ref command } if command == "ls -la"));
    }

    #[test]
    fn job_payload_message_serialization() {
        let payload = JobPayload::Message { text: "hello world".into() };
        let json = serde_json::to_string(&payload).unwrap();

        assert!(json.contains("Message"));
        assert!(json.contains("hello world"));

        let restored: JobPayload = serde_json::from_str(&json).unwrap();
        assert!(matches!(restored, JobPayload::Message { ref text } if text == "hello world"));
    }

    // ---- RunStatus serialization ----

    #[test]
    fn run_status_serialize_snake_case() {
        assert_eq!(
            serde_json::to_string(&RunStatus::Pending).unwrap(),
            "\"pending\""
        );
        assert_eq!(
            serde_json::to_string(&RunStatus::Running).unwrap(),
            "\"running\""
        );
        assert_eq!(
            serde_json::to_string(&RunStatus::Completed).unwrap(),
            "\"completed\""
        );
        assert_eq!(
            serde_json::to_string(&RunStatus::Failed).unwrap(),
            "\"failed\""
        );
    }

    #[test]
    fn run_status_deserialize_snake_case() {
        let status: RunStatus = serde_json::from_str("\"pending\"").unwrap();
        assert_eq!(status, RunStatus::Pending);

        let status: RunStatus = serde_json::from_str("\"running\"").unwrap();
        assert_eq!(status, RunStatus::Running);

        let status: RunStatus = serde_json::from_str("\"completed\"").unwrap();
        assert_eq!(status, RunStatus::Completed);

        let status: RunStatus = serde_json::from_str("\"failed\"").unwrap();
        assert_eq!(status, RunStatus::Failed);
    }

    #[test]
    fn run_status_equality() {
        assert_eq!(RunStatus::Pending, RunStatus::Pending);
        assert_ne!(RunStatus::Pending, RunStatus::Running);
        assert_ne!(RunStatus::Completed, RunStatus::Failed);
    }

    #[test]
    fn run_status_roundtrip_preserves_value() {
        for status in [RunStatus::Pending, RunStatus::Running, RunStatus::Completed, RunStatus::Failed] {
            let json = serde_json::to_string(&status).unwrap();
            let back: RunStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, status, "roundtrip failed for {:?}", status);
        }
    }
}
