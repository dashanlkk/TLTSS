/// TUI 应用状态
pub struct TuiApp {
    pub messages: Vec<(String, String)>, // (role, content)
    pub input: String,
    pub status: String,
    pub running: bool,
}

impl Default for TuiApp {
    fn default() -> Self {
        Self::new()
    }
}

impl TuiApp {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            input: String::new(),
            status: "Ready".to_string(),
            running: true,
        }
    }

    pub fn add_message(&mut self, role: &str, content: &str) {
        self.messages.push((role.to_string(), content.to_string()));
    }

    pub fn submit(&mut self) -> String {
        let input = self.input.clone();
        self.input.clear();
        self.status = "Processing...".to_string();
        input
    }
}
