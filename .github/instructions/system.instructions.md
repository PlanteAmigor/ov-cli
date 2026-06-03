---
description: Describe when these instructions should be loaded by the agent based on task context
# applyTo: 'Describe when these instructions should be loaded by the agent based on task context' # when provided, instructions will automatically be added to the request context when the pattern matches an attached file
---

<!-- Tip: Use /create-instructions in chat to generate content with agent assistance -->

Provide project context and coding guidelines that AI should follow when generating code, answering questions, or reviewing changes.

## Command Execution

- When running terminal commands, **do not use `tail`, `head`, `less`, `more`, or any pager that truncates/hides output**.
- Always display the **full, untruncated output** of commands.
- If a command would produce very long output, still show all of it unless explicitly instructed otherwise by the user.
- Avoid automatically piping to `tail` or redirecting to files that would hide output from the user.