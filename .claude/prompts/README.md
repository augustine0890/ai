# Reusable Prompts

Project-agnostic prompts for maintaining the two-file context system used
across software, data science, and research projects.

## The two-file system

- **`session_context.md`** — durable rationale, decisions, design notes.
  Append-only, long-lived, answers "why".
- **`plan.md`** — actionable checklist for the current task with
  verification and rollback. Short-lived, execution-focused, answers
  "what/how/verify".

Each prompt handles both CREATE (file does not exist) and UPDATE (file
exists) via explicit mode detection.

## Files

| File | Purpose |
|---|---|
| [`GETTING_STARTED.md`](GETTING_STARTED.md) | **Start here** — walkthrough and examples for new projects |
| [`session_context.md`](session_context.md) | Prompt to create/update a durable context document (cumulative) |
| [`plan.md`](plan.md) | Prompt to create/update an execution plan (ephemeral per task) |
| [`execute_plan.md`](execute_plan.md) | Prompt to read an existing plan and implement it step-by-step |

## Quick start

1. Read [`GETTING_STARTED.md`](GETTING_STARTED.md) once to understand the
   workflow.
2. For each new project:
   - Open [`session_context.md`](session_context.md) and copy the prompt.
   - Replace `<TARGET_PATH>` with your project folder.
   - Paste into Claude with your project summary.
   - Then repeat with [`plan.md`](plan.md).

## Full usage

1. Open the prompt file (either `session_context.md` or `plan.md`).
2. Copy the fenced code block between the triple backticks.
3. Replace `<TARGET_PATH>` with the directory where the doc should live
   (e.g., `dds/docs`, `experiments/exp-042`, `research/plans`).
4. Paste into a Claude session along with any task-specific context.

## Conventions reinforced by the prompts

- Rationale lives in `session_context.md`, actions live in `plan.md` —
  never duplicate.
- Completed plan steps only flip their Status cell; descriptions stay
  intact for audit trail.
- Dates are absolute (YYYY-MM-DD), never relative.
- Research/DS plans must name dataset, split, seed, metric, and
  acceptance criterion for any training/evaluation step.
