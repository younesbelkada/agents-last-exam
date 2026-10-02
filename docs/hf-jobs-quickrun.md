# Fast ALE subset on HF Jobs

Run a handful of ALE tasks end to end on Hugging Face infrastructure, with a
model and harness you choose, and push the run logs to a bucket you own. Use
this to validate a harness or compare models in under an hour, not to produce a
full benchmark number.

Everything is driven by one script:

```bash
scripts/hf_quickrun.sh --help
```

## Topology

Two layers of HF Jobs:

- **Sandbox jobs**: one per task, started by the `hfsandbox` provider from
  `agentslastexam/ale-ubuntu22-docker`. The agent and the task run in there.
- **Orchestrator**: the `ale_run` process that creates those jobs, grades the
  output, and writes logs. It runs on your machine by default, or as its own
  `cpu-basic` job with `--submit`. With `--submit` you need nothing locally but
  the `hf` CLI.

Only the Linux subset runs here (`cpu-free-ubuntu` tasks, same coverage as the
local Docker provider). The script rejects anything else before it costs you a
job.

## One-time setup

1. An HF token with the `jobs` scope:

   ```bash
   hf auth login
   hf auth whoami
   ```

2. Task data. The image ships without task inputs or references, so download
   the gated archive (request access first) and upload it to a private bucket:

   ```bash
   scripts/fetch_task_data.sh task-data
   hf buckets create <namespace>/ale-task-data --private
   hf buckets sync task-data hf://buckets/<namespace>/ale-task-data
   ```

3. A bucket for results. `hf buckets create` makes a **public** bucket unless
   you pass `--private`, and run logs contain full agent trajectories and task
   output, so keep it private:

   ```bash
   hf buckets create <namespace>/ale-results --private
   hf buckets info <namespace>/ale-results        # "private": true
   ```

   An existing public bucket can be closed with
   `hf buckets settings <namespace>/ale-results --private`.

4. Export the two buckets so you can stop typing them:

   ```bash
   export ALE_DATA_BUCKET=hf://buckets/<namespace>/ale-task-data
   export ALE_RESULTS_BUCKET=hf://buckets/<namespace>/ale-results
   ```

5. Local runs need the repo installed (`uv sync --all-packages`) and
   `secret/.env` to carry `HF_TOKEN=...`. The `--submit` path does both inside
   the job.

## Run it

Smoke test first, one demo task, no task data needed:

```bash
scripts/hf_quickrun.sh --tasks demo/hello
```

Then a real subset, six tasks spread across six domains:

```bash
scripts/hf_quickrun.sh \
  --harness qwen_code_hf --model Qwen/Qwen3.5-9B \
  --tasks 6 --wall-time 1800 --concurrency 6
```

Same thing with nothing running locally:

```bash
scripts/hf_quickrun.sh --harness qwen_code_hf --model Qwen/Qwen3.5-9B \
  --tasks 6 --submit
```

Add `--dry-run` to any of these to see the task selection, the generated
configs, and the run matrix (or the job command, with `--submit`) without
starting anything.

## Choosing the model and harness

`--harness` names a preset in `configs/agents/`; `--model` replaces that
preset's `model:` line. Three presets route through the HF Inference Providers
router and authenticate with your `HF_TOKEN`, so any model id the router serves
works:

| `--harness` | CLI under test | Default model |
|---|---|---|
| `openhands_cli_hf` | OpenHands CLI | `zai-org/GLM-5.3` |
| `qwen_code_hf` | Qwen Code | `Qwen/Qwen3.5-9B` |
| `zcode_hf` | ZCode (Z.ai) | `Qwen/Qwen3.5-9B` |

Other presets in `configs/agents/` work too, but they bring their own provider
and need that provider's key in `secret/.env` (for example `claude_code.yaml`
with `ANTHROPIC_API_KEY`).

## Choosing the subset

`--tasks` takes three forms:

- a number: sample that many tasks from the pool, interleaved by domain, so
  `--tasks 6` gives six different domains rather than six finance tasks;
- a `.txt` path: any list under `selected_tasks/`, or your own;
- a comma-separated list: `--tasks demo/hello,legal/legal_dr_fees_01`.

`--from` sets the pool for the numeric form. It defaults to
`selected_tasks/docker_support.txt` (99 Linux tasks), which is the right pool
for this provider.

## What makes a run fast

| Knob | Effect |
|---|---|
| `--tasks N` | Fewer units. The main lever. |
| `--wall-time S` | Caps the agent per task. Task cards allow 7200s; 1800s is usually enough to see whether a harness works at all, and anything stopped at the cap is recorded as `timeout`. |
| `--concurrency N` | Sandbox jobs in flight, so wall-clock is roughly `ceil(N_tasks / concurrency)` task slots. Each job is a separate `--flavor` machine. |
| `--flavor` | Sandbox hardware. Default `cpu-upgrade` (8 vCPU / 32 GB). See `hf jobs hardware`. |

The script fixes `max_attempts: 1` (failures are reported, not retried) and
`cleanup_mode: delete` (jobs are cancelled as soon as a unit finishes).

Unavoidable overhead: a cold pull of the ~40 GB sandbox image takes 10 to 15
minutes per job, before the agent starts. Short runs are dominated by it.

## Results

Logs are written under `.logs/ale/<agent>/<model>/<task>/v<i>/<timestamp>/`:

| File | Contents |
|---|---|
| `run.json` | Status, score, duration, sandbox metadata |
| `eval_result.json` | `eval_status`, `score`, eval duration |
| `trajectory.json` | Unified ATIF trajectory |
| `events.jsonl` | Append-only phase trace, the authoritative record |

With `--results-bucket` (or `ALE_RESULTS_BUCKET`) the whole tree is synced when
the run ends:

```bash
hf buckets sync .logs/ale hf://buckets/<namespace>/ale-results
```

The destination is only as private as the bucket you name, so before anything
starts the script reads its visibility and refuses to run against a public
bucket. If it cannot read the visibility (no token, bucket not created yet) it
warns and continues, and the sync at the end will fail on a missing bucket.

The script also prints the per-unit status and score table before syncing.
Generated configs for each invocation are kept under
`.logs/quickrun/<run-name>/` so a run can be reproduced or edited by hand:

```bash
uv run python -m ale_run run .logs/quickrun/<run-name>/experiment.yaml -v
```

## Grading keys

Tasks in the Linux pool grade deterministically, with one exception:
`business_finance/pe_screening_memo_1` scores part of its rubric with an OpenAI
judge and raises without a key. To grade it, put the key in
`secret/eval_time/openai.env`:

```bash
echo "OPENAI_API_KEY=sk-..." > secret/eval_time/openai.env
```

With `--submit`, an `OPENAI_API_KEY` present in your shell is forwarded to the
orchestrator job as a secret and written to that file there.

## Billing to an organization

`--namespace <org>` bills both the orchestrator job and every sandbox job to
that org instead of the token owner.
