# Prompt: Create or Update `session_context.md`

Reusable prompts for maintaining a durable "why" document across software,
data science, and research projects. Use the CREATE prompt when the file
does not exist yet. Use the UPDATE prompt when the file already exists and
needs to capture a new decision, constraint, or learning.

Replace `<TARGET_PATH>` with the path where `session_context.md` should
live (for example `dds/docs/session_context.md` or `research/context.md`).

---

## Prompt: CREATE `session_context.md`

Use this when `<TARGET_PATH>/session_context.md` does not exist yet.

```text
You are creating `<TARGET_PATH>/session_context.md`, the durable context
document for this project. This file is long-lived memory across sessions -
it captures WHY decisions were made, not WHAT commands to run. It is read
by future humans and LLMs who have no memory of this conversation. Your job
is to make the project legible to a capable but cold-start reader.

State explicitly that you are in CREATE mode before writing.

## Objective

Create a context file that gives a future contributor enough grounding to
understand:
- what problem the project exists to solve
- why the project is shaped the way it is
- what architectural and product constraints already exist
- what assumptions are safe vs unsafe
- which decisions are settled, which are provisional, and which are open

The file must be detailed enough that another LLM (Claude, Gemini, Codex)
can read only this document plus the codebase and immediately operate with
minimal confusion.

## Pre-work

Before writing, read and synthesize as much of the following as exists:

1. Project-defining artifacts:
   - README, design docs, RFCs, ADRs, PRDs, issue epics, architecture notes
   - any existing `plan.md`, `TODO.md`, milestone doc, or implementation brief
2. Source of truth in the repository:
   - the main entrypoints
   - the core modules or packages
   - configs, schemas, manifests, environment examples
   - tests that reveal intended behavior
3. Operational or data context:
   - datasets, notebooks, experiment logs, result summaries
   - deployment config, infra notes, CI files, observability docs
4. User-provided context from this conversation:
   - stated goals
   - hard constraints
   - success criteria
   - known problems, risks, or non-goals

If some information is missing, record the uncertainty explicitly instead of
inventing detail.

## What to extract before writing

Identify, in one sentence each:
- the business or research goal
- the primary user or stakeholder
- the core system boundary
- the main architectural shape
- the most important constraints
- the highest-risk unknowns
- the decisions that appear already locked in

If you cannot explain why a fact matters to a future reader, omit it.

## File structure

Write the file using this skeleton. Expand each section with enough detail
to be genuinely useful; do not leave it as a thin outline.

    # Session Context: <Project Name>

    Durable rationale, decisions, and learnings for <project>. This file is
    append-only context; execution plans live in `plan.md`.

    ---

    ## 1. Project background
    Explain the origin of the project and the surrounding context. Include:
      - the problem statement in plain language
      - the business, product, research, or operational motivation
      - the current stage of the project
      - the primary stakeholders, users, or downstream consumers
      - any historical context needed to understand why this effort exists now

    ## 2. Project goal and success criteria
    Describe the intended outcome in concrete terms. Include:
      - the desired end state
      - what success looks like
      - measurable acceptance criteria if known
      - non-goals and out-of-scope areas if they shape decisions

    ## 3. Constraints and fixed requirements
    Record the hard boundaries future contributors must respect. Include:
      - deadlines, budget, headcount, compute, latency, throughput, scale
      - regulatory, legal, privacy, security, and compliance constraints
      - compatibility requirements, integration contracts, platform limits
      - operational realities such as support model, hosting environment,
        data windows, or external SLAs

    ## 4. System architecture and project shape
    Describe the project at a high level. Include:
      - major components, services, modules, pipelines, or notebooks
      - the responsibility of each major part
      - key data flows or request flows
      - important external systems or dependencies
      - interfaces, contracts, or boundaries that matter

    ## 5. Key decisions and trade-offs
    For each major decision so far, record:
      - the decision
      - the alternatives considered
      - why the chosen path won
      - why the rejected paths lost
      - who or what drove the decision, if relevant

    ## 6. Domain notes / product notes / data notes
    Record the non-obvious facts a future contributor must know.

    ## 7. Environment and reproducibility
    Explain what is needed to reproduce a clean run or safe development
    environment.

    ## 8. Risks, failure modes, and operational gotchas
    Capture the ways this project can go wrong and what future contributors
    should be careful not to break.

    ## 9. Open questions
    Record genuinely open questions, each with why it matters and what would
    close it.

    ## 10. References and evidence
    Point to the artifacts that justify the context in this file.

## Content rules

- Write prose, not checklists. Checklists belong in `plan.md`.
- No step-by-step commands, status tables, or TODO lists.
- Explain the why, not the code listing.
- Use absolute dates.
- If something is inferred, say that it is inferred.
- For data science or research, record the metric, split, and seed behind
  material decisions.
- Cite stable external references where relevant.

## Output

1. The full `session_context.md` contents.
2. A concise summary of the major themes captured.
3. A short list of missing facts that would materially improve the file.

## Do not

- Do not touch `plan.md`.
- Do not run commands or modify code.
- Do not invent rationale that is not grounded in the available material.
- Do not collapse uncertainty into confident prose.
```

---

## Prompt: UPDATE `session_context.md`

Use this when `<TARGET_PATH>/session_context.md` already exists.

```text
You are updating `<TARGET_PATH>/session_context.md`, the durable context
document for this project. It stores long-lived rationale, not execution
steps.

State explicitly that you are in UPDATE mode before writing.

## Workflow

1. Read the existing file in full.
2. Read only the files, results, or notes needed to understand the new change.
3. Add one new numbered section at the end unless you are correcting a
   factually wrong line.

## Write one new section

    ## N. <Short topic title>

Write 1-3 short paragraphs covering:
  - what changed and why it mattered
  - the decision taken and the main alternative rejected
  - any non-obvious constraint, invariant, or gotcha
  - pointers to the relevant files, modules, plans, or prior sections

Add a `See also:` line if it helps.

## Rules

- Write prose, not checklists.
- No commands, status tables, TODOs, or temporary progress notes.
- Explain the why, not the code diff.
- Use absolute dates.
- If a prior section is wrong, correct the specific line and add
  `Superseded on YYYY-MM-DD by Section N`.
- If the information is operational noise rather than durable rationale,
  leave it out.

## Output

1. The edited file.
2. A short summary of the section added or corrected.
3. Any stale content that should be cleaned up later.

## Do not

- Do not touch `plan.md`.
- Do not run commands or modify code.
- Do not invent rationale that is not grounded in the source material or
  the user's input.
```
