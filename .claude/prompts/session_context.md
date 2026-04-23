# Prompt: Create or Update `session_context.md`

Reusable prompts for maintaining a durable "why" document across software,
data science, and research projects. Use the CREATE prompt when the file
does not exist yet. Use the UPDATE prompt when the file already exists and
needs to capture a new decision, constraint, or learning.

Replace `<TARGET_PATH>` with the path where `session_context.md` should
live (e.g., `dds/docs/session_context.md`, `notebooks/context/session.md`,
`research/context.md`).

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
      - the current stage of the project (new build, migration, stabilization,
        exploratory research, rewrite, etc.)
      - the primary stakeholders, users, or downstream consumers
      - any historical context needed to understand why this effort exists now

    ## 2. Project goal and success criteria
    Describe the intended outcome in concrete terms. Include:
      - the desired end state
      - what "success" looks like
      - measurable acceptance criteria if known
      - non-goals and out-of-scope areas if they shape decisions

    ## 3. Constraints and fixed requirements
    Record the hard boundaries that future contributors must respect. Include:
      - deadlines, budget, headcount, compute, latency, throughput, scale
      - regulatory, legal, privacy, security, and compliance constraints
      - compatibility requirements, integration contracts, platform limits
      - operational realities such as maintenance burden, support model,
        hosting environment, data availability windows, or external SLAs

    ## 4. System architecture and project shape
    Describe how the project is structured at a high level. Include:
      - major components, services, modules, pipelines, or notebooks
      - the responsibility of each major part
      - key data flows or request flows
      - important external systems or dependencies
      - interfaces, contracts, or boundaries that matter
    This section should explain the shape of the system, not list every file.

    ## 5. Key decisions and trade-offs
    For each major decision so far, record:
      - the decision
      - the alternatives considered
      - why the chosen path won
      - why the rejected paths lost
      - who or what drove the decision, if relevant
    Prefer a small number of substantive decision paragraphs over shallow bullets.

    ## 6. Domain notes / product notes / data notes
    Record the non-obvious facts a future contributor must know. Examples:
      - software: invariants, concurrency rules, idempotency assumptions,
        auth model, API quirks, failure semantics
      - data science: dataset provenance, label noise, leakage traps,
        split strategy, target definition, seed behavior
      - research: exact hypothesis, baseline choice, prior art, metric logic,
        evaluation caveats
      - product: user workflow assumptions, business rules, edge-case behavior,
        backward-compatibility expectations

    ## 7. Environment and reproducibility
    Explain what is needed to reproduce a clean run or safe development
    environment. Include:
      - package manager and language/runtime versions
      - required services, credentials, or local tooling
      - data snapshot dates or dataset versions
      - hardware needs such as CPU, memory, GPU, disk
      - seeds, config files, and reproducibility assumptions
    Enough detail that a new contributor can stand up the project without
    asking basic setup questions.

    ## 8. Risks, failure modes, and operational gotchas
    Capture the ways this project can go wrong. Include:
      - known technical risks
      - failure modes or fragility points
      - quality traps, silent corruption risks, leakage risks, race conditions,
        flaky dependencies, or vendor quirks
      - what future contributors should be careful not to break

    ## 9. Open questions
    Record genuinely open questions, each with:
      - why it is unresolved
      - why it matters
      - what evidence or decision would close it
    Leave the section present even if it is short.

    ## 10. References and evidence
    Point to the artifacts that justify the context in this file. Include:
      - key files, directories, modules, notebooks, or configs
      - stable identifiers for external references (DOI, arXiv ID, permalink)
      - internal docs or issue links if available
    This is not a bibliography dump; include only sources that shaped decisions.

## Content rules

- Write prose, not checklists. Checklists belong in `plan.md`.
- No step-by-step commands. No status tables. No TODOs. No command
  transcripts. If you want to show a command, show one line as an
  example, not a runbook.
- No restating what the code or notebook already says - explain the "why",
  not the "what". If removing a sentence would not confuse a future
  reader, remove it.
- Use absolute dates (`2026-04-23`), never relative ones ("yesterday",
  "last week").
- Prefer explicit assumptions over vague wording. If something is inferred,
  say that it is inferred.
- Keep implementation details subordinate to rationale. The reader should
  finish the file understanding the system's intent and constraints first.
- For data science or research: always record the metric, the split, and
  the seed that backed a decision. A decision without these is untraceable.
- Cite external sources by stable identifier (DOI, arXiv ID, permalink)
  rather than search-engine URLs.
- Keep the writing dense with signal. More detail is good; padding is not.

## Output

1. The full `session_context.md` contents.
2. A concise summary of the major themes captured.
3. A short list of any missing facts that would materially improve the file.

## Do not

- Do not touch `plan.md`.
- Do not run commands or modify code.
- Do not add rationale that is not grounded in something you read or
  the user told you.
- Do not collapse important uncertainty into confident prose.
```

---

## Prompt: UPDATE `session_context.md`

Use this when `<TARGET_PATH>/session_context.md` already exists.

```text
You are updating `<TARGET_PATH>/session_context.md`, the durable context
document for this project. This file is long-lived memory across sessions -
it captures WHY decisions were made, not WHAT commands to run. It is read
by future humans and LLMs who have no memory of this conversation.

State explicitly that you are in UPDATE mode before writing.

## Objective

Append the new rationale, decision, constraint, or learning in a way that
preserves the history of the project. The file is append-only by default.
Future readers should be able to see what changed, why it changed, and what
older context is still valid.

## Pre-work

1. Read the existing file in full.
2. Preserve its section numbering and append-only convention - add a new
   numbered section at the end unless you are fixing a factually wrong line.
3. Read the recent history that informs this update:
   - software or data projects: the last 3-5 git commits touching relevant
     paths; any open `plan.md`, `TODO.md`, issue, or design note
   - research or data-science projects: the latest run artifacts, notebook
     outputs, experiment logs, result tables, and associated configs
4. Read the source files, notebooks, or datasets directly involved so your
   rationale is grounded in reality, not speculation.
5. Identify, in one sentence each, the facts you plan to record. If you
   cannot name the "why" behind a fact, drop it.

## Update shape

Append one new numbered section with a short, specific title:

    ## N. <Short topic title>

Write 1-4 short paragraphs covering:
  - the problem or trigger: why this work happened now
  - the decision taken and the alternatives rejected, with reasons
  - any non-obvious constraint, invariant, or gotcha a future reader
    cannot recover from the code or data alone
  - pointers to the files, modules, notebooks, plans, or prior sections
    that embody the decision

Close with a `See also:` line if other sections are relevant.

## Content rules

- Write prose, not checklists. Checklists belong in `plan.md`.
- No step-by-step commands. No status tables. No TODOs. No command
  transcripts. If you want to show a command, show one line as an
  example, not a runbook.
- No restating what the code or notebook already says - explain the "why",
  not the "what". If removing a sentence would not confuse a future
  reader, remove it.
- Use absolute dates (`2026-04-23`), never relative ones.
- No ephemeral status ("currently working on", "will do tomorrow"). Those
  belong in `plan.md` or an issue tracker.
- For data science or research: always record the metric, the split, and
  the seed that backed a decision. A decision without these is untraceable.
- Cite external sources by stable identifier (DOI, arXiv ID, permalink)
  rather than search-engine URLs.

## Hygiene on UPDATE

- If a previous section is now factually wrong, correct only the specific
  sentence and add a short `Superseded on YYYY-MM-DD by Section N` note.
  Do not silently rewrite history.
- If two sections now cover the same topic, consolidate under the older
  one and leave a pointer in the newer one.
- If the new information is operational noise rather than durable rationale,
  do not put it in `session_context.md`.

## Output

1. The edited file contents.
2. A 2-sentence summary: which section was added or changed and which
   decision it records.
3. If you detected stale content that should be cleaned up, list it - do
   not silently clean it without surfacing the change.

## Do not

- Do not touch `plan.md`.
- Do not run commands or modify code.
- Do not add rationale that is not grounded in something you read or
  the user told you.
```
