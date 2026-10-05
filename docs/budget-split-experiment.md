# Budget split: one agent with B, or N agents with B/N?

An experiment harness for one question: given a fixed token budget B, does an
agent do better spending it alone, or splitting it across N sub-agents that get
B/N each?

Both arms run on `claw_budget`, a sibling of `ale_claw` that adds a token
envelope. `ale_claw` itself is untouched, so its benchmark results do not move.

## Run it

```bash
uv run python -m ale_run run budget_exp.yaml --dry-run
uv run python -m ale_run run budget_exp.yaml -v
```

The experiment lists both arms in `agents:`, which the runner executes
independently over every task. That is what we want here: they are conditions,
not collaborators.

| Arm | Config | `n_subagents` | Delegation |
|---|---|---|---|
| A, single | `configs/agents/claw_budget_single.yaml` | 0 | removed |
| B, split | `configs/agents/claw_budget_split.yaml` | N (default 4) | `delegate_general` |

The two presets must differ in exactly those keys. If you change the model,
`max_turns`, thinking level or image retention, change it in both, or the run
measures more than one thing.

## What B covers

B is a ceiling on `input + output` tokens for the whole run unit: orchestrator
turns, every sub-agent turn, and every helper call (compaction, memory flush,
vision). The multi-agent arm therefore pays for its own coordination overhead
out of the same envelope rather than getting it free.

A sub-agent's cap is `min(B/N, remaining)`, recomputed at spawn time. It is a
view on the shared envelope, not an allocation, so N sub-agents cannot
collectively promise more than is left, and an underspending sub-agent returns
its slack to the pool.

Accounting happens at the litellm call level rather than in the agent loops,
because the harness's own helper path (`harness/model/helper_runtime.py`)
returns text and tool calls only and discards `response.usage`. Counting in the
loops would therefore miss compaction and vision spend, and miss more of it in
the arm that runs N compaction pipelines instead of one.

## Reading a run

Three artifacts, in increasing detail:

| File | Contents |
|---|---|
| `run.json` | `usage` totals, reconciled from the ledger |
| `budget_summary.json` | per-agent spend, caps, stop reasons, overshoot, enforcement events |
| `budget_ledger.jsonl` | one line per model call, written as it happens |

The ledger is the authoritative record and is the only one that survives a
wall-clock cancellation mid-run. It also keeps cache and cost fields per call,
so a finished experiment can be re-scored on cost without re-running anything.

Trajectory steps carry `extra["agent_id"]`, so per-agent behaviour can be read
back out of `trajectory.json`. Sub-agent turns are flat steps rather than nested
trajectories, because `finalize()` sums only the top-level steps: nesting would
hide sub-agent work from `final_metrics` and therefore from `run.json`.

## Before trusting a number

- **Check `stop_reason` first.** Only runs that ended on `budget_exhausted` are
  evidence about budget allocation. A run that ended on `max_steps`, on the wall
  clock, or on a `done` signal with budget to spare answers a different
  question. Report the breakdown, do not average over it.
- **Overshoot is bounded, not zero.** Usage is only known once a call returns,
  so the last call of each agent can cross the line. `budget_summary.json`
  reports `overshoot_tokens`; decide a tolerance up front and discard runs above
  it rather than after seeing which arm they favour.
- **Raw `input + output` structurally penalises the single arm.** One long
  conversation re-sends its history every turn, so its input tokens are roughly
  the integral of context length, while fresh sub-agents start short. This is a
  real property of the comparison rather than a bug, but it should be stated
  alongside any result. Prompt caching changes the cost picture by a large
  factor without changing raw token counts.
- **Watch the `unattributed` bucket.** Calls that hop to a worker thread (the
  vision tool does) start from an empty context and cannot be attributed to a
  sub-budget. They are still charged to the envelope. If that bucket is a large
  fraction of a run, per-agent numbers from that run are weak.
- **N is a spawn ceiling, not a target.** The orchestrator decides how many
  sub-agents to use and what to delegate. A run in arm B that delegated nothing
  is a legitimate outcome and should be reported, not dropped.

## Self-served models

A model litellm does not recognise gets the default 200000-token context
window, which is silently wrong for most self-served endpoints and fails late:
the agent never compacts and calls start getting rejected mid-run. There is no
config field for it, so export the real value before the run:

```bash
export CONTEXT_WINDOW_OVERRIDE=131072   # vLLM's --max-model-len
```

The model string must also start with `openrouter/`. The harness registers its
vendored loop for `openrouter/.*` only, and any string containing "openai"
infers the Responses API, which vLLM does not serve.

## Limits

`delegate_gui` is disabled in both arms. It is a second delegation axis that N
does not bound, and leaving it on would let arm B spawn uncounted agents.

Sub-agents get a hard `max_tokens` per call. The stock sub-agent session sets
none, which would leave per-call overshoot unbounded.

The harness is a fork of three `ale_claw` files (`config.py`, `deployer.py`,
`transcript_to_trajectory.py`). Upstream fixes to those do not reach it
automatically.
