use clap::{Parser, Subcommand};
use hermes_agent::Agent;
use hermes_core::config::AppConfig;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "hermes", version, about = "Hermes Agent — Rust Edition")]
struct Cli {
    /// Config file path
    #[arg(long, global = true)]
    config: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start interactive chat
    Chat {
        /// Enable TUI mode
        #[arg(long)]
        tui: bool,
        /// Use streaming mode
        #[arg(long)]
        stream: bool,
    },
    /// Show current configuration
    Config {
        #[command(subcommand)]
        action: ConfigCommands,
    },
    /// List registered tools
    Tools,
    /// Manage skills
    Skills {
        #[command(subcommand)]
        action: SkillCommands,
    },
    /// Manage cron jobs
    Cron {
        #[command(subcommand)]
        action: CronCommands,
    },
    /// Security audit
    Security {
        #[command(subcommand)]
        action: SecurityCommands,
    },
}

#[derive(Subcommand)]
enum ConfigCommands {
    /// Show current config
    Show,
    /// Validate config file
    Check,
}

#[derive(Subcommand)]
enum SkillCommands {
    /// List all skills
    List,
    /// Show skill details
    Show { name: String },
}

#[derive(Subcommand)]
enum CronCommands {
    /// List all jobs
    List,
    /// Run a job immediately
    Run { id: String },
}

#[derive(Subcommand)]
enum SecurityCommands {
    /// Audit security status
    Audit {
        /// Path to check
        #[arg(long)]
        path: Option<String>,
        /// Prompt to scan
        #[arg(long)]
        prompt: Option<String>,
    },
}

// ── 配置发现 ──────────────────────────────────────────────────────

/// 按优先级查找配置文件：--config > ./hermes.yaml > ~/.hermes/config.yaml
fn discover_config(cli_path: Option<&str>) -> Option<PathBuf> {
    // 1. CLI 指定路径
    if let Some(p) = cli_path {
        let path = PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
        eprintln!("Warning: specified config not found: {}", p);
    }

    // 2. 当前目录
    let local = PathBuf::from("hermes.yaml");
    if local.exists() {
        return Some(local);
    }

    // 3. 用户主目录
    if let Some(home) = dirs::home_dir() {
        let global = home.join(".hermes").join("config.yaml");
        if global.exists() {
            return Some(global);
        }
    }

    None
}

/// 加载配置：文件 + 环境变量覆盖
fn load_config(cli_path: Option<&str>) -> AppConfig {
    match discover_config(cli_path) {
        Some(path) => {
            tracing::info!("Loading config from {}", path.display());
            match AppConfig::from_file(&path) {
                Ok(cfg) => cfg,
                Err(e) => {
                    eprintln!("Warning: config parse error (using defaults): {}", e);
                    AppConfig::default()
                }
            }
        }
        None => {
            tracing::info!("No config file found, using defaults");
            AppConfig::default()
        }
    }
}

// ── 组件初始化 ────────────────────────────────────────────────────

/// 创建 LLM 客户端：有 API key 则 OpenAI，否则 FakeClient
fn create_llm_client(config: &AppConfig) -> Arc<dyn hermes_cfg::traits::LlmClient> {
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    let model = std::env::var("HERMES_MODEL")
        .unwrap_or_else(|_| config.model.default.clone());
    let base_url = std::env::var("OPENAI_BASE_URL")
        .ok()
        .or_else(|| config.model.base_url.clone())
        .unwrap_or_else(|| "https://api.openai.com/v1".to_string());

    if api_key.is_empty() {
        tracing::info!("No OPENAI_API_KEY set, using FakeClient");
        Arc::new(hermes_llm::FakeClient::new(
            "Hello! I'm Hermes (fake mode). Set OPENAI_API_KEY for real LLM.",
        ))
    } else {
        tracing::info!("Using OpenAI client (model: {})", model);
        Arc::new(
            hermes_llm::OpenAIClient::new(&base_url, &api_key, &model)
                .with_max_tokens(config.model.max_tokens)
                .with_temperature(config.model.temperature),
        )
    }
}

/// 创建工具注册表并注册内置工具
async fn create_tool_registry(
    config: &AppConfig,
    terminal: Arc<dyn hermes_cfg::traits::TerminalBackend>,
) -> Arc<hermes_tools::ToolRegistry> {
    let registry = Arc::new(hermes_tools::ToolRegistry::new());
    let base_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    // 注册内置工具
    registry
        .register(Arc::new(hermes_tools::builtin::ReadFileTool::new(&base_dir)))
        .await;
    registry
        .register(Arc::new(hermes_tools::builtin::WriteFileTool::new(&base_dir)))
        .await;
    registry
        .register(Arc::new(hermes_tools::builtin::ExecuteCommandTool::new(terminal.clone())))
        .await;
    registry
        .register(Arc::new(hermes_tools::builtin::ListDirTool::new(&base_dir)))
        .await;

    // 注册 MCP 工具
    for server_cfg in &config.mcp.servers {
        tracing::info!("Connecting MCP server: {}", server_cfg.name);
        match hermes_mcp::client::McpClient::connect(
            &server_cfg.name,
            &server_cfg.command,
            &server_cfg.args,
        )
        .await
        {
            Ok(client) => {
                let client = Arc::new(tokio::sync::RwLock::new(client));
                let adapters = hermes_mcp::adapter::McpToolAdapter::from_client(client).await;
                for adapter in adapters {
                    let name = adapter.name().to_string();
                    registry.register(adapter).await;
                    tracing::info!("  Registered MCP tool: {}", name);
                }
            }
            Err(e) => {
                tracing::warn!("Failed to connect MCP server '{}': {}", server_cfg.name, e);
            }
        }
    }

    registry
}

/// 创建终端后端
fn create_terminal(config: &AppConfig) -> Arc<dyn hermes_cfg::traits::TerminalBackend> {
    let work_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    hermes_terminal::factory::create_backend(&config.terminal.backend, &work_dir)
}

// ── 主入口 ────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("hermes=debug")
        .init();

    // 加载 .env
    hermes_core::env::load_env();

    let cli = Cli::parse();
    let config = load_config(cli.config.as_deref());

    match cli.command {
        Commands::Chat { tui, stream } => {
            run_chat(&config, tui, stream).await?;
        }
        Commands::Config { action } => match action {
            ConfigCommands::Show => show_config(&config)?,
            ConfigCommands::Check => check_config(&config)?,
        },
        Commands::Tools => {
            list_tools(&config).await?;
        }
        Commands::Skills { action } => match action {
            SkillCommands::List => list_skills()?,
            SkillCommands::Show { name } => show_skill(&name)?,
        },
        Commands::Cron { action } => match action {
            CronCommands::List => list_cron_jobs().await?,
            CronCommands::Run { id } => run_cron_job(&id).await?,
        },
        Commands::Security { action } => match action {
            SecurityCommands::Audit { path, prompt } => {
                run_security_audit(path, prompt)?;
            }
        },
    }

    Ok(())
}

// ── Chat 命令 ─────────────────────────────────────────────────────

async fn run_chat(config: &AppConfig, tui: bool, _stream: bool) -> anyhow::Result<()> {
    use hermes_agent::{Agent, MemoryStore};
    use hermes_agent::agent::AgentConfig;

    let terminal = create_terminal(config);
    let llm = create_llm_client(config);
    let registry = create_tool_registry(config, terminal).await;
    let memory = Arc::new(MemoryStore::new());

    // 加载技能
    let skills = load_skills();
    let skill_manifests = if !skills.is_empty() {
        tracing::info!("Loaded {} skills", skills.len());
        skills
    } else {
        Vec::new()
    };

    let agent_config = AgentConfig {
        system_prompt: format!(
            "You are Hermes, an intelligent AI assistant. You can use tools to help users.\n\
             Available skills: {}",
            if skill_manifests.is_empty() {
                "none".to_string()
            } else {
                skill_manifests
                    .iter()
                    .map(|s| format!("{} ({})", s.name, s.description))
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        ),
        ..AgentConfig::default()
    };

    let agent = Agent::new(agent_config, llm, registry, memory);

    if tui {
        run_tui_chat(agent).await
    } else {
        run_reedline_chat(agent).await
    }
}

/// TUI 模式：使用 ratatui 界面
async fn run_tui_chat(agent: Agent) -> anyhow::Result<()> {
    let mut app = hermes_ui::TuiApp::new();
    app.add_message("System", "Welcome to Hermes! Type a message and press Enter.");

    let agent_ref = Arc::new(tokio::sync::Mutex::new(agent));
    hermes_ui::render::run_tui(&mut app, |input| {
        let agent = agent_ref.clone();
        let input = input.to_string();
        let rt = tokio::runtime::Handle::current();
        rt.block_on(async {
            let agent = agent.lock().await;
            agent
                .chat(&input, hermes_cfg::platform::SessionSource::cli())
                .await
                .ok()
                .map(|msg| msg.content)
        })
    })?;

    Ok(())
}

/// Reedline 模式：带历史和自动补全的交互式 CLI
async fn run_reedline_chat(agent: Agent) -> anyhow::Result<()> {
    use reedline::{DefaultPrompt, Reedline, Signal};

    println!("Hermes Agent v0.1.0");
    println!("Type 'exit' to quit.\n");

    let mut line_editor = Reedline::create();
    let prompt = DefaultPrompt::default();

    loop {
        let sig = line_editor.read_line(&prompt);
        match sig {
            Ok(Signal::Success(buffer)) => {
                let trimmed = buffer.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if trimmed == "exit" || trimmed == "quit" {
                    println!("Goodbye!");
                    break;
                }

                match agent
                    .chat(trimmed, hermes_cfg::platform::SessionSource::cli())
                    .await
                {
                    Ok(msg) => println!("{}\n", msg.content),
                    Err(e) => eprintln!("Error: {}\n", e),
                }
            }
            Ok(Signal::CtrlC) => {
                println!("\nGoodbye!");
                break;
            }
            Ok(Signal::CtrlD) => {
                println!("\nGoodbye!");
                break;
            }
            Err(e) => {
                eprintln!("Input error: {}", e);
                break;
            }
        }
    }

    Ok(())
}

// ── Config 命令 ───────────────────────────────────────────────────

fn show_config(config: &AppConfig) -> anyhow::Result<()> {
    println!("{}", serde_json::to_string_pretty(config)?);
    Ok(())
}

fn check_config(config: &AppConfig) -> anyhow::Result<()> {
    let mut issues: Vec<String> = Vec::new();
    let mut ok_count = 0;

    // Model
    if config.model.default.is_empty() {
        issues.push("model.default is empty".to_string());
    } else {
        ok_count += 1;
    }
    if config.model.max_tokens == 0 {
        issues.push("model.max_tokens is 0".to_string());
    } else {
        ok_count += 1;
    }
    if config.model.temperature <= 0.0 || config.model.temperature > 2.0 {
        issues.push(format!("model.temperature ({}) out of range (0, 2]", config.model.temperature));
    } else {
        ok_count += 1;
    }

    // Terminal
    if config.terminal.timeout_secs == 0 {
        issues.push("terminal.timeout_secs is 0".to_string());
    } else {
        ok_count += 1;
    }

    // MCP servers
    for server in &config.mcp.servers {
        if server.command.is_empty() {
            issues.push(format!("mcp server '{}' has empty command", server.name));
        } else {
            ok_count += 1;
        }
    }

    // API key
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    if api_key.is_empty() {
        issues.push("OPENAI_API_KEY not set (chat will use FakeClient)".to_string());
    } else {
        ok_count += 1;
    }

    println!("Config validation:");
    println!("  OK: {} checks", ok_count);
    if !issues.is_empty() {
        println!("  Issues:");
        for issue in &issues {
            println!("    - {}", issue);
        }
    } else {
        println!("  No issues found.");
    }

    Ok(())
}

// ── Tools 命令 ────────────────────────────────────────────────────

async fn list_tools(config: &AppConfig) -> anyhow::Result<()> {
    let terminal = create_terminal(config);
    let registry = create_tool_registry(config, terminal).await;
    let tools = registry.list().await;

    if tools.is_empty() {
        println!("No tools registered.");
    } else {
        println!("Registered tools ({}):", tools.len());
        for tool in &tools {
            println!("  - {:20} {}", tool.name(), tool.description());
        }
    }

    Ok(())
}

// ── Skills 命令 ───────────────────────────────────────────────────

fn skills_dir() -> PathBuf {
    // 优先当前目录，其次用户主目录
    let local = PathBuf::from("skills");
    if local.exists() {
        return local;
    }
    if let Some(home) = dirs::home_dir() {
        let global = home.join(".hermes").join("skills");
        if global.exists() {
            return global;
        }
    }
    // 返回默认路径（即使不存在）
    PathBuf::from("skills")
}

fn load_skills() -> Vec<hermes_skill::manifest::SkillManifest> {
    let store = hermes_skill::store::SkillStore::new(skills_dir());
    store.load_all()
}

fn list_skills() -> anyhow::Result<()> {
    let skills = load_skills();
    if skills.is_empty() {
        println!("No skills found.");
        println!("Place .yaml files in ./skills/ or ~/.hermes/skills/");
    } else {
        println!("Skills ({}):", skills.len());
        for skill in &skills {
            let status = match skill.status {
                hermes_skill::manifest::SkillStatus::Draft => "[draft]",
                hermes_skill::manifest::SkillStatus::Published => "[published]",
            };
            println!(
                "  - {:20} {} {}  triggers: {}",
                skill.name,
                status,
                skill.description,
                skill.trigger_patterns.join(", ")
            );
        }
    }
    Ok(())
}

fn show_skill(name: &str) -> anyhow::Result<()> {
    let store = hermes_skill::store::SkillStore::new(skills_dir());
    match store.find(name) {
        Some(manifest) => {
            println!("{}", serde_json::to_string_pretty(&manifest)?);
        }
        None => {
            println!("Skill '{}' not found.", name);
        }
    }
    Ok(())
}

// ── Cron 命令 ─────────────────────────────────────────────────────

fn cron_data_dir() -> PathBuf {
    if let Some(home) = dirs::home_dir() {
        home.join(".hermes").join("cron")
    } else {
        PathBuf::from(".hermes").join("cron")
    }
}

async fn create_scheduler() -> hermes_cron::scheduler::Scheduler {
    let terminal = create_terminal(&AppConfig::default());
    let scheduler = hermes_cron::scheduler::Scheduler::new(terminal).with_data_dir(cron_data_dir());
    let _ = scheduler.load_from_dir().await;
    scheduler
}

async fn list_cron_jobs() -> anyhow::Result<()> {
    let scheduler = create_scheduler().await;
    let jobs = scheduler.list_jobs().await;

    if jobs.is_empty() {
        println!("No cron jobs.");
    } else {
        println!("Cron jobs ({}):", jobs.len());
        for job in &jobs {
            let status = if job.enabled { "enabled" } else { "disabled" };
            let last_run = job
                .last_run_at
                .map(|t| t.format("%Y-%m-%d %H:%M").to_string())
                .unwrap_or_else(|| "never".to_string());
            println!(
                "  - {:36} {:15} {:20} last: {}",
                job.id, job.name, status, last_run
            );
        }
    }
    Ok(())
}

async fn run_cron_job(id: &str) -> anyhow::Result<()> {
    let scheduler = create_scheduler().await;
    match scheduler.run_now(id).await {
        Ok(run) => {
            println!("Job {} executed:", id);
            println!("  Status: {:?}", run.status);
            if let Some(output) = run.output {
                println!("  Output: {}", output);
            }
            if let Some(error) = run.error {
                println!("  Error: {}", error);
            }
        }
        Err(e) => {
            println!("Failed to run job '{}': {}", id, e);
        }
    }
    Ok(())
}

// ── Security 命令 ─────────────────────────────────────────────────

fn run_security_audit(path: Option<String>, prompt: Option<String>) -> anyhow::Result<()> {
    if let Some(ref p) = path {
        let base = std::path::PathBuf::from(".");
        match hermes_security::validate_path(&base, p) {
            Ok(resolved) => println!("[OK] Path resolved: {}", resolved.display()),
            Err(e) => println!("[DENIED] {}", e),
        }
    }
    if let Some(ref p) = prompt {
        match hermes_security::scan_prompt(p) {
            hermes_security::prompt::ScanResult::Safe => println!("[OK] Prompt is safe"),
            hermes_security::prompt::ScanResult::Suspicious { matched_pattern } => {
                println!(
                    "[SUSPICIOUS] Prompt injection pattern detected: {}",
                    matched_pattern
                );
            }
        }
    }
    if path.is_none() && prompt.is_none() {
        println!("Usage: hermes security audit --path <path> --prompt <text>");
        println!("  --path   Validate a file path for traversal attacks");
        println!("  --prompt Scan text for prompt injection patterns");
    }
    Ok(())
}
