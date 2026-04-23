# Prompt: Create or Update `plan.md`

Reusable prompt for maintaining a concise execution plan across software,
data science, and research projects. Replace `<TARGET_PATH>` with the path
where `plan.md` should live.

---

## Prompt

```text
You are creating or updating `<TARGET_PATH>/plan.md`, the execution plan
for the current task. This file is short-lived and execution-focused. It
should tell a future human or LLM what to do next and how to verify it.

Keep rationale in `session_context.md`, not here. Link to it when useful.

## Mode

Check whether `<TARGET_PATH>/plan.md` exists.
- If not, use CREATE mode.
- If yes, use UPDATE mode.
State the mode before writing.

## Pre-work

1. Read `session_context.md` if it exists.
2. Read the files, notebooks, configs, or docs the plan will touch.
3. If scope is unclear, ask one clarifying question instead of guessing.
4. If updating, read the existing plan in full first.

## File shape

Use this structure. Omit sections that are truly unnecessary.

    # Plan: <short task title>

    One short paragraph: what this plan accomplishes and, if useful,
    "See `session_context.md` Section N for rationale."

    ---

    ## 1. Scope
    In scope:
      - concrete outputs or files
    Out of scope:
      - what this plan will not touch

    ## 2. Prerequisites   (optional)
    Only if access, approvals, data, or decisions block execution.

    ## 3. Execution Steps
    Table: # | Step | Status
      - each row is one discrete, verifiable action
      - Status values: `Todo` | `In progress` | `Done` | `Blocked` | `Skipped`
      - name real files, commands, artifacts, or outputs where possible

    ## 4. Verification
    Exact checks that prove the work is done.

    ## 5. Rollback   (optional)
    Only if the work needs a real undo path.

    ## 6. Notes   (optional)
    Only short executor-facing notes that do not belong above.

## Rules

- Keep the file compact. If it gets large, split the work into a new plan.
- Do not put rationale, TODO lists, or open questions in the step table.
- Every step must be independently verifiable.
- If a step trains or evaluates a model, name dataset, split, seed, metric,
  and acceptance threshold.
- Use absolute dates.

## UPDATE hygiene

- If a step is completed, only flip its Status cell to `Done`.
- Do not rewrite old step descriptions after the fact.
- If the next action changes, append a new step instead of mutating history.
- If scope shifts materially, stop and write a new plan.
- Move durable insights to `session_context.md`, not `plan.md`.

## Output

1. The edited `plan.md`.
2. A short summary of what changed and the next action.
3. Call out blockers or scope drift explicitly.

## Do not

- Do not implement the plan from this prompt.
- Do not rewrite `session_context.md`.
- Do not invent steps when the next action is unknown.
```
