# Fast ALE subset on HF Jobs

Run a handful of ALE tasks end to end on Hugging Face infrastructure, with a
model and harness you choose, and push the run logs to a bucket you own. Use
this to validate a harness or compare models in under an hour, not to produce a
full benchmark number.

Everything is driven by one script:

```bash
scripts/hf_quickrun.py --help
```

There are four scripts in total, and each shell script has a Python port that
takes the same flags. The ports are the ones that get new features (a model
matrix, percentage subsets, the score report); the shell versions stay for
single-model runs.

| Script | Shell equivalent | Purpose |
|---|---|---|
| `scripts/hf_quickrun.py` | `hf_quickrun.sh` | Run a subset against one or more models |
| `scripts/fetch_task_subset.py` | `fetch_task_subset.sh` | Stage a few tasks' data into a bucket |
| `scripts/model_size_sweep.py` | (none) | A controlled model-size comparison |
| `scripts/serve_model_job.py` | (none) | Serve a model the router does not offer |

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

   For subset runs, skip the 48 GB archive and fetch only the tasks you need.
   `scripts/fetch_task_subset.py` pulls `input/` + `software/` from the open
   dataset and `reference/` from the gated one, merges them, restores the
   native filenames and symlinks, and syncs the result:

   ```bash
   scripts/fetch_task_subset.py \
     --tasks .logs/quickrun/<run-name>/tasks.txt \
     --bucket hf://buckets/<namespace>/ale-task-data
   ```

   Reference access is what makes a run scoreable. Without it, pass
   `--no-reference` and expect every reference-based grader to score 0.

   The bucket root must hold the domain directories themselves, because
   staging copies from `<mount>/<domain>/<task>/<variant>/input`:

   ```bash
   hf buckets ls <namespace>/ale-task-data
   # business_finance/  computing_math/  education_info/ ...
   ```

   If the root instead holds a single wrapper directory (the archive extracted
   one level deeper), you do not need to re-upload. Point at the subdirectory,
   since the volume grammar accepts a prefix:

   ```bash
   --data-bucket hf://buckets/<namespace>/ale-task-data/<subdir>
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

5. Local runs need the repo installed (`uv sync --all-packages`).

## Credentials

`hf auth login` is enough. The token reaches three places:

- **Creating sandbox jobs.** The provider calls `get_token()`, which reads
  `HF_TOKEN` or the stored login.
- **The agent's model calls.** The HF-router presets set
  `api_key: ${env:HF_TOKEN}`, and that substitution resolves only from the
  process environment, not from the stored login. The script bridges the gap by
  exporting `hf auth token` when `HF_TOKEN` is unset, and refuses to start if
  neither is available.
- **A `--submit` job.** Passed as `hf jobs run --secrets HF_TOKEN`, whose value
  is read from your environment (extended with the stored login), never from
  the command line. The job writes it to `secret/.env` on the clone.

To use an explicit token instead, either `export HF_TOKEN=hf_...` or put
`HF_TOKEN=hf_...` in `secret/.env`. Note that `secret/.env` is loaded with
override, so a blank `HF_TOKEN=` line there silently beats your shell export.
The script stops with an error rather than letting the agent run with an empty
key.

## Run it

Smoke test first, one demo task, no task data needed:

```bash
scripts/hf_quickrun.py --tasks demo/hello
```

Then a real subset, six tasks spread across six domains:

```bash
scripts/hf_quickrun.py \
  --harness qwen_code_hf --model Qwen/Qwen3.5-9B \
  --tasks 6 --wall-time 1800 --concurrency 6
```

Same thing with nothing running locally:

```bash
scripts/hf_quickrun.py --harness qwen_code_hf --model Qwen/Qwen3.5-9B \
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

`--tasks` takes four forms:

- a number: sample that many tasks from the pool, interleaved by domain, so
  `--tasks 6` gives six different domains rather than six finance tasks;
- a percentage of the pool: `--tasks 10%` is 10 of the 99 Linux tasks, rounded
  up (Python scripts only);
- a `.txt` path: any list under `selected_tasks/`, or your own;
- a comma-separated list: `--tasks demo/hello,legal/legal_dr_fees_01`.

`--from` sets the pool for the numeric and percentage forms. It defaults to
`selected_tasks/docker_support.txt` (99 Linux tasks), which is the right pool
for this provider.

Selection is deterministic, not random: the pool is interleaved by domain and
the first N are taken. The same `--tasks` value against the same pool always
picks the same tasks, which is what lets two runs be compared.

## Comparing models

Pass `--model` more than once and the run becomes a model matrix: every model
runs the same tasks inside one experiment, with one orchestrator, one task
selection and one results sync.

```bash
scripts/hf_quickrun.py --harness qwen_code_hf \
  --model Qwen/Qwen3-8B --model Qwen/Qwen3-32B --tasks 6
```

`scripts/model_size_sweep.py` is that, preconfigured for a size ladder. It
holds everything but `model:` fixed, records the resolved plan under
`.logs/sweep/<run-name>/`, and can stage the subset's task data first:

```bash
scripts/model_size_sweep.py \
  --fetch-data \
  --data-bucket hf://buckets/<ns>/ale-task-data \
  --results-bucket hf://buckets/<ns>/ale-results-qwen \
  --namespace <ns> --submit \
  --repo https://github.com/<you>/agents-last-exam --ref <branch>
```

The default ladder is the dense Qwen3 sizes (4B, 8B, 14B, 32B) plus
Qwen3.8-27B, all served by the HF Inference Providers router. Override it with
repeated `--model` flags. The default subset is `10%`, so the default sweep is
5 models times 10 tasks, or 50 sandbox jobs.

Results land in `.logs/ale/<run-name>/<harness>/<model-slug>/<task>/v0/<ts>/`.
Because `auto_resume` is scoped to that output root, re-invoking with the same
`--run-name` re-runs only the cells that have no completed run; without
`--run-name` each invocation starts a fresh timestamped root.

Resume counts a `timeout` as done, alongside `completed`, so a cell killed by
`--wall-time` is skipped on the next invocation. Re-running one takes
`--disable-resume`, which re-runs every *selected* unit, so narrow the
selection to what you actually want repeated:

```bash
scripts/model_size_sweep.py --run-name <same-name> --harness <same-harness> \
  --model Qwen/Qwen3.8-27B \
  --tasks computing_math/branch_bound_atsp,physical_sciences/adapt_vqe_molecular_energy \
  --wall-time 7200 --disable-resume ...
```

The re-run writes a new timestamped directory next to the old one. Nothing is
overwritten, and the report reads the most recent result per (model, task), so
the matrix picks up the new score on its own. Keep `--wall-time` at or below
the `vm.timeout` in each selected task card, since the experiment-wide value
overrides it in both directions.

A local run prints a task by model score matrix when it ends (`--no-report`
suppresses it). After a `--submit` run, pull the logs and render the same table
from them:

```bash
hf buckets sync hf://buckets/<ns>/ale-results-qwen .logs/ale
scripts/model_size_sweep.py --report-only .logs/ale/<run-name>
```

```
task                                           Qwen3-4B-Instruct-2507  Qwen3-8B  Qwen3-14B  Qwen3-32B
business_finance/american_option_pricing_ls                      0.00      0.25       0.50       1.00
computing_math/branch_bound_atsp                                    F      0.00       0.50       0.75
-----------------------------------------------------------------------------------------------------
mean (scored)                                                    0.00      0.12       0.50       0.88
scored / total                                                    1/2       2/2        2/2        2/2
```

Two caveats when reading the result. A small model that scores 0 everywhere
may be failing to use the absolute input paths rather than failing the task
(see the `--prompt-suffix` note below), and the sweep's default suffix exists
for exactly that reason. And a model that is not served by the router fails
every unit at the first API call, which shows up as `F` across its column.

Before committing to a ladder, check that the router actually serves each
model. The catalog is the authority, not the Hub:

```bash
curl -s https://router.huggingface.co/v1/models \
  -H "Authorization: Bearer $(hf auth token)" | jq -r '.data[].id' | grep Qwen
```

A model repo can exist on the Hub with no provider serving it. For those, see
the next section.

## Serving a model yourself

`scripts/serve_model_job.py` runs vLLM on a GPU Job and exposes its port
through the HF Jobs proxy, so a model no Inference Provider offers still gets
an OpenAI-compatible endpoint at `https://<job-id>--8000.hf.jobs/v1`. Access
needs `Authorization: Bearer <HF token>`, which is exactly the bearer the
presets already send as `api_key`, so a generated preset needs no other change
and works both locally and from inside a task sandbox.

```bash
scripts/serve_model_job.py --model Qwen/Qwen3.5-4B
```

It picks the cheapest GPU flavor that fits the bf16 weights (override with
`--flavor`), waits for vLLM to answer `/health`, writes
`configs/agents/<from-preset>_served_<model-slug>.yaml`, and cancels the job on
Ctrl-C. The harness is in the filename, and so in the agent `id`, so two
harnesses serving the same model keep separate output branches. T4
flavors are never auto-selected: compute capability 7.5 has no bf16 and vLLM
refuses the checkpoint rather than downcasting.

The default image is `vllm/vllm-openai:nightly`, not `:latest`. A model the
router does not serve is usually one whose architecture landed after the last
stable vLLM, so stable would fail to load it. Pin `:latest` with `--image` when
you know the architecture is supported.

`--tool-call-parser` is model-specific and worth checking against the model
card: Qwen3.5 asks for `qwen3_coder`, other families differ.

`--max-model-len` defaults to 0, meaning the model's own maximum. Resist
lowering it to save memory. A context shorter than what the harness sends makes
vLLM reject every call with `400 Bad Request`, which surfaces in the server log
as nothing but a status line while the whole run fails; a KV cache that does
not fit, by contrast, fails loudly at startup. Hybrid-attention models are
cheaper than they look here: Qwen3.5-2B runs full attention on only 6 of 24
layers with 2 KV heads, so its full 262144-token context costs about 3.2 GB per
sequence.

Arguments forwarded with `--vllm-arg` need the `=` form, since argparse would
otherwise read a leading dash as the next flag:

```bash
--vllm-arg=--language-model-only    # skip the vision tower, more KV cache
```

The preset carries its own `id:`, so a self-hosted run never lands in the same
output branch as the same model served by the router.

To use one in a sweep, start it detached, run against the preset it wrote, then
stop it. One vLLM job serves one model, so self-hosted models are swept one at
a time; a shared `--run-name` merges them into a single output root and a
single report, because `auto_resume` skips the cells already done:

```bash
scripts/serve_model_job.py --model Qwen/Qwen3.5-2B --detach
scripts/model_size_sweep.py --run-name qwen-ladder \
  --harness served_qwen-qwen3-5-2b --model Qwen/Qwen3.5-2B --no-report ...
scripts/serve_model_job.py --stop <job-id>

# repeat per served model, then render the combined matrix
scripts/model_size_sweep.py --report-only .logs/ale/qwen-ladder
```

### Walking away from a run

Jobs are server-side. Closing your laptop never cancels one, which is the
answer you want for `--submit` and the answer you do not want otherwise.

With `--submit`, everything runs on HF: the orchestrator job, the sandboxes it
creates and the final bucket sync. Add `--detach` so the launching command
returns once the job exists instead of blocking on its log stream; without it
you only lose the stream, not the run. Reattach with `hf jobs logs -f <id>`.

Without `--submit`, the orchestrator is the process on your laptop, and it is
the only thing that cancels a sandbox when its unit finishes. If it is
suspended or killed, every sandbox it started keeps running and billing until
`--job-timeout` expires, with nothing driving them. Self-hosted models force
this mode, because the generated preset holds a `base_url` that only exists
after the server job starts. So either keep the machine awake:

```bash
caffeinate -is ./scripts/model_size_sweep.py ...
```

or lower the ceiling on what an orphan can cost, with `--job-timeout 4h`
instead of the 24h default.

### Job labels

Every job ALE creates is labelled, which is what makes a multi-model run
legible in `hf jobs ps` and lets you cancel one model's jobs without touching
another's.

| Label | On | Value |
|---|---|---|
| `ale` | all | `orchestrator`, `sandbox` or `model-server` |
| `ale_model` | all | the model id, e.g. `Qwen_Qwen3.5-9B` |
| `ale_harness` | orchestrator, sandbox | e.g. `pi_cli` |
| `ale_task` | sandbox | e.g. `legal_legal_dr_fees_01` |

Label values must match `^[a-zA-Z0-9._-]*$`, so `/` becomes `_` and the value
is truncated at 60 characters. An orchestrator running a model matrix carries
no `ale_model`, since one job covers several; it is named `ale-<run-name>`
instead.

```bash
hf jobs ps --label ale=sandbox                        # everything in flight
hf jobs ps --label ale_model=Qwen_Qwen3.8-27B         # just the 27B rung
hf jobs ps --label ale=sandbox --label ale_harness=pi_cli
```

To sweep up strays:

```bash
hf jobs ps --label ale=sandbox -q      | xargs -I{} hf jobs cancel {}
hf jobs ps --label ale=model-server -q | xargs -I{} hf jobs cancel {}
```

Two more things to weigh. A served model is billed for the entire
job lifetime, including the image pull and the weight download, so a detached
job with no later `--stop` keeps charging until `--timeout` expires. And a
ladder that mixes self-hosted vLLM with router-served models is no longer a
clean size comparison: sampling defaults, quantization, context length and
tool-parser behaviour all differ between your vLLM and whatever the provider
runs. Self-host every rung, or none.

## What makes a run fast

| Knob | Effect |
|---|---|
| `--tasks N` | Fewer units. The main lever. |
| `--wall-time S` | Caps the agent per task. Task cards allow 7200s; 1800s is usually enough to see whether a harness works at all, and anything stopped at the cap is recorded as `timeout`. |
| `--concurrency N` | Sandbox jobs in flight, so wall-clock is roughly `ceil(N_tasks / concurrency)` task slots. Each job is a separate `--flavor` machine. |
| `--flavor` | Sandbox hardware. Default `cpu-upgrade` (8 vCPU / 32 GB). See `hf jobs hardware`. |
| `--start-timeout S` | How long a sandbox may take to boot, image pull included. Default 3600. Does not speed anything up, but too low a value burns the whole run on pull timeouts. |

The script fixes `max_attempts: 1` (failures are reported, not retried) and
`cleanup_mode: delete` (jobs are cancelled as soon as a unit finishes).

Unavoidable overhead: every sandbox job pulls the ~40 GB image before the agent
starts. The shipped config estimates 10 to 15 minutes, but a measured run with
4 units in flight took about 28 minutes, so `--start-timeout` defaults to 3600s
here rather than the provider's 1800s. A unit whose sandbox does not come up in
that window fails with:

```
RuntimeError: hfsandbox: job <id> not RUNNING after <n>s
  (image pull of agentslastexam/ale-ubuntu22-docker:latest still in progress?)
```

If you hit that, raise `--start-timeout` and lower `--concurrency`: each unit
is a separate node doing its own cold pull, so more units in flight means more
simultaneous pulls, not a shared one.

A different failure looks similar in the sandbox log but is not a timeout. If
the pod boots, serves one `GET /status` and one `POST /cmd`, then shuts down a
few seconds later, the sandbox was fine and staging failed on the first call it
makes:

```
RuntimeError: task_data_source=mounted: expected input/ at
  '/mnt/ale-task-data/<domain>/<task>/<variant>' inside the sandbox, not found.
  Is the task-data volume mounted at '/mnt/ale-task-data'?
```

The unit then fails and the cleanup cancels the job, which is the shutdown you
see. Check the bucket layout as described in the setup section. The script
preflights it before launching anything.

A third case looks like missing data but is not. If the run reaches the agent
and the trajectory fills with `File not found`, staging already succeeded (it
would have failed the unit otherwise). Inputs live at an absolute path built
from the image's `task_data_root`:

```
/media/user/data/agenthle/<domain>/<task>/<variant>/input/...
```

If the agent is reading under its own work dir (`/home/user/.ale/<agent>/<run_id>/`)
instead, it is resolving relative paths against its cwd rather than using the
task directory it was given. Confirm with:

```bash
grep -o '/media/user/data/agenthle[^"]*' trajectory.json | head -3
```

If the instruction carries that path, the agent ignored it. Smaller models do
this often. Try a larger model, or ground it with `--prompt-suffix`, which is
appended verbatim to every task prompt.

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
