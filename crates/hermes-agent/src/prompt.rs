use hermes_cfg::message::Message;
use hermes_skill::SkillManifest;

use crate::context_files::ContextFile;

/// Prompt 组装器：将系统提示、记忆、技能、对话历史组装为 messages 列表
pub struct PromptBuilder {
    system_prompt: String,
    memories: Vec<String>,
    skills: Vec<SkillManifest>,
    context_files: Vec<ContextFile>,
    memory_context: String,
    max_context_messages: usize,
}

impl PromptBuilder {
    pub fn new(system_prompt: impl Into<String>) -> Self {
        Self {
            system_prompt: system_prompt.into(),
            memories: Vec::new(),
            skills: Vec::new(),
            context_files: Vec::new(),
            memory_context: String::new(),
            max_context_messages: 50,
        }
    }

    pub fn with_memories(mut self, memories: Vec<String>) -> Self {
        self.memories = memories;
        self
    }

    pub fn with_skills(mut self, skills: Vec<SkillManifest>) -> Self {
        self.skills = skills;
        self
    }

    pub fn with_context_files(mut self, files: Vec<ContextFile>) -> Self {
        self.context_files = files;
        self
    }

    pub fn with_memory_context(mut self, ctx: String) -> Self {
        self.memory_context = ctx;
        self
    }

    pub fn with_max_context(mut self, max: usize) -> Self {
        self.max_context_messages = max;
        self
    }

    /// 组装最终 messages 列表
    pub fn build(&self, history: &[Message]) -> Vec<Message> {
        let mut messages = Vec::new();

        // 系统提示
        let mut system_content = self.system_prompt.clone();

        // 注入上下文文件 (.hermes.md / AGENTS.md / CLAUDE.md / .cursorrules)
        if !self.context_files.is_empty() {
            let block = crate::context_files::format_context_block(&self.context_files);
            if !block.is_empty() {
                system_content.push_str("\n\n");
                system_content.push_str(&block);
            }
        }

        // 注入 MemoryManager 持久化记忆 (MEMORY.md / USER.md)
        if !self.memory_context.is_empty() {
            system_content.push_str("\n\n");
            system_content.push_str(&self.memory_context);
        }

        // 注入 TF-IDF 搜索记忆
        if !self.memories.is_empty() {
            system_content.push_str("\n\n## Memories\n");
            for mem in &self.memories {
                system_content.push_str(&format!("- {}\n", mem));
            }
        }

        // 注入可用技能
        if !self.skills.is_empty() {
            system_content.push_str("\n\n## Available Skills\n");
            for skill in &self.skills {
                system_content.push_str(&format!(
                    "- **{}**: {} (triggers: {})\n",
                    skill.name,
                    skill.description,
                    skill.trigger_patterns.join(", ")
                ));
            }
        }

        messages.push(Message::new_system(&system_content));

        // 对话历史（截断到最大窗口）
        let start = if history.len() > self.max_context_messages {
            history.len() - self.max_context_messages
        } else {
            0
        };
        for msg in &history[start..] {
            messages.push(msg.clone());
        }

        messages
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hermes_cfg::message::Role;

    #[test]
    fn test_prompt_build() {
        let builder = PromptBuilder::new("You are Hermes.");
        let history = vec![Message::new_user("Hello")];
        let messages = builder.build(&history);

        assert_eq!(messages[0].role, Role::System);
        assert!(messages[0].content.contains("You are Hermes."));
        assert_eq!(messages[1].role, Role::User);
    }

    #[test]
    fn test_memory_injection() {
        let builder = PromptBuilder::new("You are helpful.")
            .with_memories(vec!["User prefers Chinese".to_string()]);

        let messages = builder.build(&[]);
        assert!(messages[0].content.contains("User prefers Chinese"));
    }
}
