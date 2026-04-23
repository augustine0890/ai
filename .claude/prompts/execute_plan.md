# Prompt: Execute Steps from an Existing `plan.md`

Reusable prompt for LLMs to read an existing execution plan and implement
it step-by-step. Use this when you have a `plan.md` and are ready to start
working.

---

## Prompt

~~~text
You are executing the plan in `<PLAN_PATH>`. Read it in full, understand
the scope and verification criteria, then implement the steps.

## Pre-work

1. Read the plan file at `<PLAN_PATH>` in full. If it references
   `session_context.md`, read that too so you understand the rationale.
2. Read the source files, notebooks, or configs that the plan will touch
   so you understand the target shape and dependencies.
3. Identify any prerequisites listed in Section 2 of the plan. If they are not
   met, STOP and list them before proceeding.
4. Identify the success criteria in Section 6 (Verification). You will run these
   checks after each step.

## Execution protocol

Follow this protocol for EACH step in the plan:

1. Read the step description and understand:
   - What concrete action it names
   - What artifact it produces
   - Which file(s) it will touch

2. Perform the action. Examples:
   - Software: mkdir, move a file, edit code, run tests, commit to git
   - Data science: pull data, engineer a feature, train a model, log
     metrics, save a checkpoint
   - Research: read papers, run an experiment, generate a figure, write
     a draft section

3. After the step completes:
   - Run the verification command(s) from Section 6 that apply to this step
   - Report the result: PASS or FAIL
   - If FAIL: stop, diagnose, ask for help (do not skip the step)
   - If PASS: move to the next step

4. Update the plan file:
   - Flip ONLY the Status cell of the completed step to `Done`
   - Do NOT rewrite the step description
   - Save the file

## Handling blockers

If a step fails or is blocked:

1. Diagnose clearly: what command failed, what error was returned, what
   assumption broke.
2. Do NOT skip or rewrite the step. The plan is a historical record of
   intent.
3. Ask the user or another LLM for help with the specific failure.
4. Once unblocked, re-run the step and continue.

If the blocker reveals that the plan's scope has shifted materially:
- Stop execution
- Propose creating a new plan file (e.g., `plan-v2.md`)
- Let the user decide whether to pivot or continue the current plan

## Output style

For each step, output:

```text
### Step N.M: <step title from plan>
**Action:** <one-line summary of what you are doing>
**Command:** <exact command, if applicable>
**Result:** PASS | FAIL | <specific outcome>
[if failure: diagnosis and next action]
```

After each step completes, show the updated status line:

```text
| N.M | <step description> | Done |
```

At the end, show a summary:

```text
## Execution Summary
- Steps completed: N / M
- Verification: PASS | FAIL
- Next action: [if incomplete or blocked, what to do next]
```

## Do NOT

- Do not skip steps because they seem optional or redundant. The plan
  author included them for a reason.
- Do not rewrite a step after the fact to match what actually happened.
  Keep the description intact.
- Do not change the plan's scope on your own. If scope must shift, ask
  the user.
- Do not commit code or data changes to git without explicit instruction
  in the plan. (Plans should be explicit about what to commit.)
- Do not suppress error output. If a step fails, show the full error so
  it can be debugged.

## Do

- Do ask clarifying questions if a step is ambiguous.
- Do pause and re-read the plan if you are unsure of the next action.
- Do run verification checks after each step.
- Do update the plan file status as you complete steps.
- Do report blockers immediately and completely.
~~~

---

## Example: executing a data science plan

You have `experiments/churn-v3/plan-baseline.md` and are ready to train.

```text
Please execute the plan at experiments/churn-v3/plan-baseline.md.
Reference session_context.md for rationale if needed.

I have access to the data warehouse and one GPU. Report each step's
status and pause if anything fails.

[paste the prompt above, filling in <PLAN_PATH>]
```

Claude reads the plan and begins:

```text
### Step 5.1: Pull event data for 2025-01-01 to 2025-04-01; verify 50M rows
**Action:** Query warehouse for user events in date range
**Command:** python scripts/pull_events.py --start 2025-01-01 --end 2025-04-01
**Result:** PASS (51.2M rows loaded, 423 GB)

| 5.1 | Pull event data... | Done |

### Step 5.2: Build data loader that pads sequences to 100 events
**Action:** Create PyTorch DataLoader with sequence padding
**Command:** [editing notebooks/data_loader.py, running tests]
**Result:** PASS (DataLoader yields (batch=32, seq_len=100, features=42))

| 5.2 | Build data loader... | Done |

[... continues through all steps ...]

## Execution Summary
- Steps completed: 8 / 8
- Verification: PASS
- Next action: Results saved to experiments/churn-v3/RESULTS.md; ready
  for next plan (plan-optimization.md)
```

---

## Tips

1. **Read the whole plan first.** Do not proceed step-by-step blind; skim
   all steps so you understand the flow.
2. **Pause for verification.** Do not skip the verification checks. They
   are part of the plan.
3. **Keep the plan pristine.** Only update Status cells; never rewrite
   step descriptions during execution.
4. **Call for help early.** A blocked step is not a failure; it is
   information that something needs adjustment.
