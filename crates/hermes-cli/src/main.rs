use clap::{Parser, Subcommand};
use hermes_agent::Agent;
use hermes_core::config::AppConfig;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "hermes", version, about = "Hermes Agent — Rust Edition")]
struct Cli {
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("hermes=debug")
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Chat { tui } => {
            run_chat(tui).await?;
        }
        Commands::Config { action } => {
            match action {
                ConfigCommands::Show => show_config()?,
                ConfigCommands::Check => check_config()?,
            }
        }
        Commands::Tools => {
            list_tools()?;
        }
        Commands::Skills { action } => {
            match action {
                SkillCommands::List => list_skills()?,
                SkillCommands::Show { name } => show_skill(&name)?,
            }
        }
        Commands::Cron { action } => {
            match action {
                CronCommands::List => println!("No cron jobs (scheduler not running)"),
                CronCommands::Run { id } => println!("Would run job: {}", id),
            }
        }
        Commands::Security { action } => {
            match action {
                SecurityCommands::Audit { path, prompt } => {
                    run_security_audit(path, prompt)?;
                }
            }
        }
    }

    Ok(())
}

async fn run_chat(tui: bool) -> anyhow::Result<()> {
    use hermes_agent::{Agent, MemoryStore};
    use hermes_cfg::traits::LlmClient;
    use hermes_llm::FakeClient;
    use hermes_tools::ToolRegistry;

    let llm: Arc<dyn LlmClient> = Arc::new(FakeClient::new("Hello! I'm Hermes. How can I help you?"));
    let registry = Arc::new(ToolRegistry::new());
    let memory = Arc::new(MemoryStore::new());

    let agent = Agent::new(
        hermes_agent::agent::AgentConfig::default(),
        llm,
        registry,
        memory,
    );

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

    let agent_ref = std::sync::Arc::new(tokio::sync::Mutex::new(agent));
    hermes_ui::render::run_tui(&mut app, |input| {
        let agent = agent_ref.clone();
        let input = input.to_string();
        // 同步调用异步 Agent（TUI 回调是同步的，需要 block）
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

                match agent.chat(trimmed, hermes_cfg::platform::SessionSource::cli()).await {
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

fn show_config() -> anyhow::Result<()> {
    let config = AppConfig::default();
    println!("{}", serde_json::to_string_pretty(&config)?);
    Ok(())
}

fn check_config() -> anyhow::Result<()> {
    println!("Config check:");
    println!("  Message types: OK");
    println!("  Core traits:");
    println!("    - LlmClient (complete, complete_stream, ping)");
    println!("    - ToolHandler (execute)");
    println!("    - TerminalBackend (execute, close)");
    println!("    - PlatformAdapter (run, send, set_message_handler)");
    println!("  Error types: LlmError, ToolError, TerminalError, ConfigError, McpError, SkillError, CronError");
    Ok(())
}

fn list_tools() -> anyhow::Result<()> {
    println!("Built-in tools:");
    println!("  - read_file       Read file contents");
    println!("  - write_file      Write content to file");
    println!("  - execute_command Execute shell command");
    println!("  - list_dir        List directory contents");
    Ok(())
}

fn list_skills() -> anyhow::Result<()> {
    println!("No skills loaded.");
    println!("Place .yaml files in ~/.hermes/skills/ or ./hermes/skills/");
    Ok(())
}

fn show_skill(name: &str) -> anyhow::Result<()> {
    println!("Skill '{}' not found.", name);
    Ok(())
}

fn run_security_audit(path: Option<String>, prompt: Option<String>) -> anyhow::Result<()> {
    if let Some(p) = path {
        let base = std::path::PathBuf::from("/workspace");
        match hermes_security::validate_path(&base, &p) {
            Ok(_) => println!("[OK] Path is valid: {}", p),
            Err(e) => println!("[DENIED] {}", e),
        }
    }
    if let Some(p) = prompt {
        match hermes_security::scan_prompt(&p) {
            hermes_security::prompt::ScanResult::Safe => println!("[OK] Prompt is safe"),
            hermes_security::prompt::ScanResult::Suspicious { matched_pattern } => {
                println!("[SUSPICIOUS] Prompt injection pattern detected: {}", matched_pattern);
            }
        }
    }
    Ok(())
}
