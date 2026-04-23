# Prompt: Create or Update `session_context.md`

Reusable prompt for maintaining a durable "why" document across software,
data science, and research projects. Paste this into any session; replace
`<TARGET_PATH>` with the path where `session_context.md` should live
(e.g., `dds/docs/session_context.md`, `notebooks/context/session.md`,
`research/context.md`).

---

## Prompt

```
You are creating or updating `<TARGET_PATH>/session_context.md`, the
durable context document for this project. This file is long-lived memory
across sessions — it captures WHY decisions were made, not WHAT commands
to run. It is read by future humans and LLMs who have no memory of this
conversation.

## Mode detection

First, check whether `<TARGET_PATH>/session_context.md` exists.

- If it does NOT exist → CREATE mode.
- If it DOES exist → UPDATE mode.

State which mode you are in before writing.

## Pre-work (both modes)

1. If UPDATE: read the existing file in full. Preserve its section numbering
   and append-only convention — add a new numbered section at the end, do
   not rewrite prior sections unless a fact became factually wrong.
2. Read the recent history that informs this update:
   - software/data projects: the last 3–5 git commits touching the relevant
     paths; any open `plan.md`, `TODO.md`, or issue tracker entry
   - research/data-science: the last run's artifacts (notebook outputs,
     experiment logs, results table) and the associated config/manifest
3. Read the source files, notebooks, or datasets directly involved so your
   rationale is grounded in reality, not speculation.
4. Identify, in one sentence each, the facts you plan to record. If you
   cannot name the "why" behind a fact, drop it.

## File structure

### CREATE mode — write this skeleton

    # Session Context: <Project Name>

    Durable rationale, decisions, and learnings for <project>. This file is
    append-only context; execution plans live in `plan.md`.

    ---

    ## 1. Project goal and constraints
    One paragraph: the end-user problem, the success criteria, and the
    hard constraints (deadline, budget, data access, regulatory, ethical).

    ## 2. Key decisions and trade-offs
    For each major decision so far:
      - the decision, in one line
      - alternatives considered, with one-line reasons rejected
      - who/what it was decided by (if relevant)

    ## 3. Domain notes / data notes
    Non-obvious facts a future contributor must know. Examples:
      - software: invariants, concurrency rules, external API quirks
      - data science: dataset provenance, label noise, leakage traps,
        train/val/test split logic, seed behavior
      - research: the exact hypothesis, prior art references, the specific
        metric and why it was chosen

    ## 4. Environment and reproducibility
    How to reproduce a clean run: package manager, Python/R version, GPU
    needs, random seeds, data snapshot date. Enough that a new contributor
    can stand up the project without asking.

    ## 5. Open questions (optional)
    Questions that are genuinely open, with why they matter. Leave empty
    if none.

    ---

### UPDATE mode — append one new numbered section

    ## N. <Short topic title>
    1–4 short paragraphs covering:
      - the problem or trigger (why this work happened now)
      - the decision taken and the alternatives rejected, with one-line
        reasons
      - any non-obvious constraint, invariant, or gotcha a future reader
        cannot recover from the code/data alone
      - pointers to the files, modules, notebooks, or sections that
        implement or embody the decision
    Close with a "See also:" line if other sections are relevant.

## Content rules (both modes)

- Write prose, not checklists. Checklists belong in `plan.md`.
- No step-by-step commands. No status tables. No TODOs. No command
  transcripts. If you want to show a command, show one line as an
  example, not a runbook.
- No restating what the code/notebook already says — explain the "why",
  not the "what". If removing a sentence would not confuse a future
  reader, remove it.
- Use absolute dates (2026-04-23), never relative ones ("yesterday",
  "last week").
- Keep each section tight. If it runs past ~40 lines, you are describing
  implementation instead of rationale — cut it.
- No ephemeral status ("currently working on", "will do tomorrow"). Those
  belong in `plan.md` or an issue tracker.
- For data science / research: always record the metric, the split, and
  the seed that backed a decision. A decision without these is untraceable.
- Cite external sources by stable identifier (DOI, arXiv ID, permalink)
  rather than search-engine URLs.

## Hygiene on UPDATE

- If a previous section is now factually wrong, correct the specific
  sentence and add a short "Superseded on YYYY-MM-DD by §N" note.
  Do not silently rewrite history.
- If two sections now cover the same topic, consolidate under the older
  one and leave a pointer in the newer one.

## Output

1. The edited file contents.
2. A 2-sentence summary: which section was added/changed and which
   decision it records.
3. If you detected any stale content that should be cleaned up, list it
   — do not silently clean it without surfacing the change.

## Do not

- Do not touch `plan.md`.
- Do not run commands or modify code.
- Do not add rationale that is not grounded in something you read or
  the user told you.
```
