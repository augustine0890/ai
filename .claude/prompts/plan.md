# Prompt: Create or Update `plan.md`

Reusable prompt for maintaining an execution plan across software, data
science, and research projects. Paste this into any session; replace
`<TARGET_PATH>` with the path where `plan.md` should live
(e.g., `dds/plan.md`, `experiments/exp-042/plan.md`, `research/plans/ablation.md`).

---

## Prompt

```
You are creating or updating `<TARGET_PATH>/plan.md`, the execution plan
and status tracker for the current work. This file is short-lived and
execution-focused — it tells a future LLM or human exactly what to do
next and how to verify each step.

Durable rationale must live in `session_context.md`, NOT here. `plan.md`
should link to the relevant section of `session_context.md` instead of
duplicating the reasoning.

## Mode detection

First, check whether `<TARGET_PATH>/plan.md` exists.

- If it does NOT exist → CREATE mode.
- If it DOES exist → UPDATE mode.

State which mode you are in before writing.

## Pre-work (both modes)

1. Read `session_context.md` if it exists, so the plan can reference the
   right section and will not re-derive rationale.
2. Read the files, notebooks, or configs that the plan will touch, so
   each step can name real paths, real functions, and real commands.
3. If the user's request is ambiguous (scope unclear, boundary between
   in-scope and out-of-scope fuzzy), STOP and ask before writing. One
   clarifying round is cheaper than a wrong plan.
4. If UPDATE: read the existing plan in full and identify:
   - which steps are Done (do not rewrite their description, only update
     the Status cell)
   - which steps are blocked and why
   - whether the remaining work still fits in one plan or should be split

## File structure

Use exactly these sections, in order. Omit sections that are truly
irrelevant — most plans use all of them.

    # Plan: <short task title>

    One-paragraph intro: what this plan accomplishes, plus a pointer such
    as "See `session_context.md` §N for rationale". For research/DS
    plans, also state the hypothesis or question being tested in one
    line.

    ---

    ## 1. Scope

    In scope:
      - concrete, file-level or artifact-level bullets

    Out of scope:
      - what this plan will NOT touch, including things the reader might
        assume are included

    ## 2. Prerequisites   (only if the plan has unresolved blockers)
    Numbered list of decisions, access, data, or approvals needed before
    step 1 can start. If this list is non-empty, do not proceed to Execution.

    ## 3. Current Issues   (only if this is a cleanup/refactor/debug plan)
    Table: # | Issue | Evidence (file path, line, log excerpt, metric).

    ## 4. Target State     (only if the plan changes structure or defines
    a deliverable shape)
    A code block, schema, or bullet list showing the target layout, API,
    model interface, experiment config, or paper outline.

    ## 5. Execution Steps
    Markdown table with columns: # | Step | Status.
      - Each step is a discrete, verifiable action:
          software: one mkdir / one move / one edit / one command / one
                    focused code change
          data sci:  one data pull / one feature / one model variant /
                    one ablation / one eval run
          research:  one literature check / one experiment / one figure /
                    one draft section
      - Steps must be ordered so each can be done without the next.
      - Status values: `Todo` | `In progress` | `Done` | `Blocked` | `Skipped`.
      - Avoid vague steps ("refactor X", "improve model"). Break them into
        named sub-actions with measurable completion criteria.
      - When a step produces an artifact (a file, a figure, a metric),
        name the artifact in the step.

    ## 6. Verification
    Exact commands, grep patterns, notebook cells, or metrics that prove
    the plan worked.
      - Software: copy-pasteable shell, prints OK/FAIL where possible.
      - Data science: the exact evaluation cell, metric name, and
        acceptance threshold (e.g., "val AUC ≥ 0.82 on seed 42").
      - Research: the figure or table that the reader should regenerate
        and what it should show.

    ## 7. Risks and Mitigations
    Table: Risk | Mitigation. Only real risks — not paranoia. For DS/ML,
    always include at least: data leakage, train/eval distribution shift,
    reproducibility (seed, environment).

    ## 8. Rollback
    Exact commands or instructions to undo:
      - software: `git checkout --`, `git revert <sha>`, config reversal
      - data science: how to restore prior model artifact, prior feature
        table, prior config
      - research: how to revert draft changes (git) and which experiment
        run(s) to discard

    ## 9. Notes   (optional)
    Anything the executor needs that does not fit above. No narrative.

## Content rules (both modes)

- Plans describe actions, not rationale. Rationale → `session_context.md`.
- No narrative paragraphs inside the step table. Every step fits one row.
- No open questions inside Execution. Unknowns go to §2 Prerequisites and
  block the plan until resolved.
- Keep the whole file under ~200 lines. If it is bigger, the scope is
  too broad — propose splitting into sibling plans (`plan-a.md`,
  `plan-b.md`) and ask the user.
- Every step must be independently verifiable. If you cannot write a
  check for it in §6, rewrite the step.
- For data science / research: a step that trains or evaluates a model
  MUST name: dataset + split, seed, metric, acceptance criterion. A step
  without these is not executable.
- Use absolute dates, never relative ones.

## Hygiene on UPDATE

- When a step completes, the ONLY change is its Status cell flipping to
  `Done`. Do not rewrite the step description to match what actually
  happened — that destroys the audit trail.
- If a step needs to change, leave it `Done`/`Skipped`, then add a new
  numbered step after it. Plans grow forward, not sideways.
- If the plan's scope shifts materially, stop and write a NEW plan. Do
  not silently mutate the old one.
- Move durable insights learned during execution into
  `session_context.md` as a new section — do not let them drown in
  `plan.md`.

## Output

1. The edited file contents.
2. A 2-sentence summary: what changed and the next action the executor
   should take.
3. If you detected scope drift or a broken Done-step, call it out
   explicitly before ending.

## Do not

- Do not implement any step. Write-only edit to `plan.md`.
- Do not rewrite `session_context.md` from this prompt.
- Do not fabricate steps. If you do not know the next action, ask the
  user — an honest question beats a fake plan.
```
