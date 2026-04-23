# Reusable Prompts

Project-agnostic prompts for the two-file system used across software,
data science, and research projects.

## The two-file system

- **`session_context.md`** - durable rationale, constraints, architecture,
  and decisions. Long-lived. Answers "why".
- **`plan.md`** - concise execution plan for the current task. Short-lived.
  Answers "what next".

Execution does not need a separate prompt file. Once `plan.md` exists,
you can simply ask the LLM to read and execute it.

## Files

| File | Purpose |
|---|---|
| [`GETTING_STARTED.md`](GETTING_STARTED.md) | Start here for workflow and examples |
| [`session_context.md`](session_context.md) | CREATE and UPDATE prompts for durable project context |
| [`plan.md`](plan.md) | CREATE and UPDATE prompt for a concise execution plan |

## Quick start

1. Read [`GETTING_STARTED.md`](GETTING_STARTED.md) once.
2. For a new project:
   - Use [`session_context.md`](session_context.md) CREATE.
   - Use [`plan.md`](plan.md) to create the first plan.
3. To execute work:
   - Ask the LLM to read the plan and execute it step by step.
   - Example: `Please read dds/plan.md and execute it step by step.`

## Conventions

- Rationale lives in `session_context.md`; actions live in `plan.md`.
- When a step completes, only flip its Status cell.
- If scope changes materially, create a new plan instead of rewriting the old one.
- Use absolute dates.
- For DS/ML work, record dataset, split, seed, metric, and acceptance threshold.
