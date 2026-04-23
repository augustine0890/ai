# Getting Started: Using the Two-File System in a New Project

This guide walks through setting up `session_context.md` and `plan.md` at
the beginning of a new project, using real examples.

---

## Overview: The Workflow

When you start a new project or major task, follow this sequence:

1. **Day 1 -> Create `session_context.md`** - capture the problem, constraints,
   and decisions made so far
2. **Day 1 -> Create `plan.md`** - lay out the execution steps and
   verification strategy
3. **Daily -> Update `plan.md`** - flip step status as you work
4. **When decisions shift -> Update `session_context.md`** - append new
   sections explaining why

At any point, a new person (or future-you) can read these two files and
jump in.

---

## Step 1: Create `session_context.md` first

### When to use it

- You have just agreed on a project with a user or team.
- You have read the codebase, the data, the problem statement, or prior
  work.
- You know the constraints: deadline, compute budget, regulatory, etc.
- You want future LLM sessions to understand the project without relying
  on chat memory.

### How to use the prompt

1. Go to `.claude/prompts/session_context.md` and use the **CREATE**
   prompt, not the UPDATE prompt.
2. Replace `<TARGET_PATH>` with your new project folder. Examples:
   - New ML experiment: `experiments/exp-2026-april-churn`
   - New data pipeline: `pipelines/user_events_etl`
   - New feature: `src/features/recommendation-v2`
   - New research direction: `research/federated-learning`

3. Paste into a fresh Claude session along with:
   - A plain-language summary of the problem and why it matters now
   - Links to any existing docs, datasets, prior art, RFCs, PRDs, or
     architecture notes
   - Constraints (deadline, budget, compute, regulatory, security,
     compatibility, staffing)
   - The intended users, stakeholders, or downstream consumers
   - Any known architectural direction, system boundaries, integrations,
     and non-goals
   - Existing code paths, configs, notebooks, schemas, or tests that seem
     central to the project

4. Ask Claude to read the repository artifacts first and then produce a
   high-signal `session_context.md` that explains not only the goal, but
   also the background, system shape, trade-offs, invariants, risks, and
   open questions.

5. Claude will reply with three things:
   - The new `session_context.md` file
   - A summary of which major themes and decisions it recorded
   - A short list of missing facts that would make the context stronger

6. **Save the file immediately** in your project folder.

### What good CREATE input looks like

The CREATE prompt works best when you provide enough project background
for Claude to infer the real system shape rather than produce a thin
template. Useful inputs include:

- Why the project exists now, not just what it does
- What came before it: prior system, failed experiment, manual workflow,
  customer pain, or research gap
- Which architectural choices are already constrained vs still open
- Which external systems, vendors, datasets, or APIs define the boundaries
- What must not be broken: compatibility promises, compliance rules,
  latency budgets, reproducibility expectations, data contracts
- What success will be judged on by humans, not just by code

If you give only a one-line task description, you will usually get a
technically correct but strategically shallow `session_context.md`.

### Example: new data science experiment

```text
I'm starting a new churn prediction experiment. We have 6 weeks, GPU
access, and 50M user events from 2025-01-01 onwards (labeled by
customer churn/retain in 2025-04-01). Prior art: https://arxiv.org/abs/2405.12345.
Legal says we must not use postal code or age in features.
The model will support a nightly retention-ops workflow used by the CRM
team, so explain any latency, interpretability, and batch-processing
constraints that matter.

Please create session_context.md at experiments/churn-v3/session_context.md
explaining the project background, goals, system shape, constraints, and
decisions so far.

[paste the CREATE prompt from .claude/prompts/session_context.md]
```

Claude produces:

```text
# Session Context: Churn Prediction Experiment v3

## 1. Project goal and constraints
Predict customer churn 30 days forward using behavioral event logs,
serving a nightly batch job. Deadline: 2026-06-01. Legal constraint:
no postal code or age features.

## 2. Key decisions and trade-offs
- Chose 30-day lookahead over 7-day because nightly batch window gives
  operational lead time. Alternative: 7-day for higher accuracy, rejected
  due to insufficient operational lead.
- Using event sequences (RNN/Transformer) over tabular aggregate because
  prior work shows sequence models outperform hand-crafted aggregates by
  ~3 AUC points on this dataset family.
- Split: 70% train (Jan-Feb), 15% val (Mar), 15% test (Apr). Temporal
  split enforces no lookahead leakage.

## 3. Data notes
[details about dataset provenance, label noise, seed behavior]

## 4. Environment and reproducibility
[Python version, PyTorch version, seed, GPU type needed]

## 5. Open questions
- Can we use customer tier or segment as a feature given legal guidance?
  (awaiting legal review by 2026-05-01)
```

This abbreviated example shows the shape, not the ideal depth. In a real
project, the CREATE prompt should usually produce more detail than this,
especially for project background, architecture, constraints, and risks.

You save this to `experiments/churn-v3/session_context.md` and move on.

Even in this small example, the better version of `session_context.md`
would usually also capture:

- why retention operations need a 30-day horizon instead of a shorter one
- what downstream teams will do with the predictions
- which dataset boundaries create leakage risk
- why the chosen evaluation metric is acceptable for the business decision
- what system or reporting constraints shaped the modeling choice

---

## Step 2: Create `plan.md` for each new task

### When to use it

- You have just completed the prior task (or plan expired).
- A new feature, bug fix, or investigation is ready to start.
- You have identified the major steps needed for this specific task.
- You are ready to start working and need a checklist.

### Key insight: `plan.md` is ephemeral

**`plan.md` is NOT cumulative.** Each new task gets a new plan file:

- `plan-setup.md` (for initial project setup)
- `plan-baseline.md` (for baseline model training)
- `plan-optimization.md` (for hyperparameter tuning)
- `plan-deployment.md` (for deploying to production)

When a plan is complete, leave it in git for audit trail, but do not update
it. Create a fresh `plan-<next-task>.md` for the new work.

Rationale: `plan.md` captures intent at a moment in time. Reusing or heavily
updating it obscures what you actually did vs. what you planned to do.

### How to use the prompt

1. Go to `.claude/prompts/plan.md` and copy the fenced code block.
2. Replace `<TARGET_PATH>` with the same folder as step 1. Example:
   `experiments/churn-v3/plan-baseline.md` (not `plan.md`)

3. Paste into a fresh Claude session (or the same one) along with:
   - The task or spike to execute
   - Any known blockers or prerequisites
   - The success criteria
   - A reference to `session_context.md` if it exists

4. Claude will reply with:
   - The new `plan-<task>.md` file with executable steps
   - A summary and the next action

5. **Save the file immediately** in the same folder.

### Example: continuing the churn experiment

```text
I just created experiments/churn-v3/session_context.md. Now I need a plan
to build the model. Here's what I need to do:
- Pull and clean event data
- Engineer temporal features
- Try a baseline (logistic regression on aggregates)
- Try an RNN model
- Evaluate both on the held-out test set
- Document findings

We have 3 weeks and one GPU. No data access blockers.

Please create plan.md at experiments/churn-v3/plan.md with discrete,
verifiable steps.

[paste the prompt from .claude/prompts/plan.md]
```

Claude produces:

~~~~text
# Plan: Churn Prediction v3 Baseline and RNN

Build and compare a logistic regression baseline and an LSTM model for
churn prediction. See session_context.md Section 2 for trade-off rationale.

## 1. Scope
In scope:
- Pull event data (Jan-Apr 2025) from warehouse
- Aggregate features for logistic regression baseline
- Build LSTM model with temporal event sequences
- Evaluate both on 2025-04-01 test split
- Log metrics to WandB

Out of scope:
- Hyperparameter sweeps (use sensible defaults only)
- Feature engineering beyond domain-obvious ones (avoid dredging)
- Deploying to production

## 5. Execution Steps
| # | Step | Status |
|---|---|---|
| 5.1 | Pull event data for 2025-01-01 to 2025-04-01; verify 50M rows | Todo |
| 5.2 | Build data loader that pads sequences to 100 events | Todo |
| 5.3 | Train logistic regression on aggregate features (10 epochs) | Todo |
| 5.4 | Evaluate logistic regression on test split; log AUC to WandB | Todo |
| 5.5 | Build LSTM(embed -> 64 -> 32 -> sigmoid); train for 20 epochs | Todo |
| 5.6 | Evaluate LSTM on test split; log AUC to WandB | Todo |
| 5.7 | Create comparison table: logistic AUC vs LSTM AUC | Todo |
| 5.8 | Write findings to experiments/churn-v3/RESULTS.md | Todo |

## 6. Verification
```bash
cd experiments/churn-v3
# Step 5.1 check: data loaded
python -c "import data; print(len(data.load_events()))"  # expect ~50M

# Step 5.4 check: logistic baseline metrics
grep "val_auc" wandb_logs.json | head -1  # expect >= 0.60

# Step 5.6 check: LSTM metrics
grep "lstm_auc" wandb_logs.json | head -1  # expect >= 0.70

# Step 5.7 check: comparison table exists
test -f RESULTS.md && grep "| Logistic" RESULTS.md
```

## 7. Risks and mitigations
[data leakage, reproducibility, compute]
~~~~

You save this to `experiments/churn-v3/plan.md`.

---

## Step 3: Execute the plan and update status daily

### As you work

- Each time you complete a step, **flip only its Status cell** from `Todo`
  to `Done`. Do NOT rewrite the step description.
- If a step gets blocked, change it to `Blocked` and log the reason.
- If you discover a new step mid-execution, **do not insert it into the
  middle** - add it at the end and number sequentially.

### Example: after finishing step 5.1

```text
| 5.1 | Pull event data for 2025-01-01 to 2025-04-01; verify 50M rows | Done |
```

No other change. The step description stays intact so future reviewers see
what was planned vs. what actually happened.

### When to call Claude during execution

Once `plan.md` exists, you do not need a separate execution prompt file.
Just ask Claude to read the plan and execute it step by step.

- Ask Claude to execute the plan as written.
- If a step fails: ask Claude to debug. Do not modify the plan - it
  records intent.
- If scope shifts materially: stop, ask Claude to create a new plan in a
  new file (`plan-<next-task>.md`), and decide with the user.
- If you learn something important: pause execution, update
  `session_context.md` with the learning (append a new section), then
  resume the plan.

---

## Step 4: Update `session_context.md` when decisions shift

### When to use it

- You discovered something during execution that changes the approach
  (e.g., a data quality issue, a performance plateau).
- You made a trade-off decision mid-project (e.g., switching from RNN to
  Transformer).
- You want to record a lesson learned for the next run.

### How to use the prompt

Paste the **UPDATE** prompt this time, referencing the **existing**
`session_context.md`:

```text
I'm updating experiments/churn-v3/session_context.md. During step 5.5,
I discovered that event sequences longer than 50 events have constant
features (no variance) - they are just noise. I decided to truncate to
50 events instead of 100. This reduces model size and training time by
40% with no AUC loss.

Please append a new section explaining this trade-off and why truncation
was chosen.

[paste the UPDATE prompt from .claude/prompts/session_context.md]
```

Claude appends:

```text
## 6. Event sequence truncation (learned 2026-04-23)
During LSTM training (plan step 5.5), discovered that event sequences
> 50 events show no further signal - features plateau. Decided to truncate
to 50 to reduce model size (40% smaller) and training time (from 2h to
1.2h per epoch) with no AUC degradation. See RESULTS.md comparison.

See also: Section 1 goal (nightly batch job with latency constraints).
```

---

## Directory layout at project start

After steps 1-2, your project looks like this:

```text
experiments/churn-v3/
|- session_context.md         # Why you are doing this (cumulative)
|- plan-baseline.md           # Current task (ephemeral, replaced per task)
|- plan-optimization.md       # Next task (will be created later)
|- data/                      # (created during work)
|- models/                    # (created during work)
|- notebooks/                 # (created during work)
`- RESULTS.md                 # (created during work)
```

- `session_context.md` stays and grows throughout the project.
- Each `plan-<task>.md` covers one task and stays in git as a historical
  record after completion.
- A new person can read `session_context.md` in 5 minutes to understand
  the full context, then check the latest `plan-*.md` to see what is
  currently in flight.

---

## Step 5: Execute the plan directly

Once you have `plan.md`, ask Claude to execute it directly:

```text
Please read experiments/churn-v3/plan-baseline.md and execute it step by
step. Follow the plan order, run the verification for each completed
step, update only the Status cells, and pause if anything is blocked or
ambiguous.
```

This is enough. A separate execution prompt usually adds token cost
without adding much value once the plan itself is clear.

---

## Quick reference: when to use which prompt

| Situation | File | Prompt |
|---|---|---|
| Starting a new project | `session_context.md` | Paste CREATE prompt from `session_context.md` |
| New task ready to build | `plan-<task>.md` | Paste `plan.md` prompt |
| Ready to implement steps | `plan-<task>.md` | Ask Claude to read and execute the plan |
| Completed a step | `plan-<task>.md` | Flip Status to `Done` (no paste needed) |
| Step failed; need help | - | Ask Claude to debug (do not modify plan) |
| Discovered a constraint | `session_context.md` | Paste UPDATE prompt from `session_context.md` |
| Task complete, new task starting | `plan-<next-task>.md` | Create new plan file (e.g., `plan-optimization.md`) |
| Learned something important | `session_context.md` | Paste UPDATE prompt to append new section |

---

## Tips

1. **Keep them dense, not thin.** A strong `session_context.md` can be long
   if the project is complex. Cut repetition and low-signal implementation
   detail, not important background, rationale, architecture, or constraints.
2. **Use absolute dates.** `2026-04-23`, never `yesterday` or `last week`.
3. **Link between them.** In `plan.md`, say "See `session_context.md` Section 2
   for the trade-off rationale". In `session_context.md`, say "See
   `plan.md` Section 5 for execution steps".
4. **Commit them.** These files are project artifacts - treat them like
   code: `git add`, `git commit`, `git push`.
5. **Share them.** When asking a colleague or Claude for help, paste both
   files so they have full context without asking 10 follow-up questions.
6. **Front-load context.** The CREATE prompt is only as good as the raw
   project background you provide. Give architecture, stakeholders,
   interfaces, constraints, and risks up front.
7. **Use UPDATE for durable learnings only.** If something is temporary
   status, leave it in `plan.md`. If it changes the reasoning, record it in
   `session_context.md`.

---

## Example: one-liner to start

```text
I'm starting a new project: [1-sentence goal]. Here are the constraints:
[list]. Please create session_context.md and plan.md at my/project/path.

[paste the CREATE prompt from session_context.md and the plan prompt,
 filling in <TARGET_PATH>]
```

Done. You have durable context and an execution roadmap.
