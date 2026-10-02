#!/usr/bin/env bash
# Run a small ALE subset on Hugging Face Jobs: pick a harness + model, sample a
# few tasks, cap the per-task wall clock, then sync the run logs to your bucket.
#
# Task sandboxes always run as HF Jobs (provider: hfsandbox). The orchestrator
# runs wherever you invoke this: your machine by default, or inside a cpu-basic
# HF Job with --submit (then nothing local is needed but the `hf` CLI).
#
# One-time setup (details: docs/hf-jobs-quickrun.md):
#   hf auth login                                   # token needs the `jobs` scope
#   scripts/fetch_task_data.sh task-data            # gated dataset
#   hf buckets create <ns>/ale-task-data --private
#   hf buckets sync task-data hf://buckets/<ns>/ale-task-data
#   hf buckets create <ns>/ale-results --private
#
# Usage:
#   scripts/hf_quickrun.sh [options]
#
#   --harness NAME       preset under configs/agents/ (default: openhands_cli_hf)
#   --model ID           model id, overrides the preset's `model:`
#   --tasks N|FILE|CSV   subset size (sampled across domains), a .txt list,
#                        or an explicit comma-separated list (default: 6)
#   --from FILE          pool to sample from (default: selected_tasks/docker_support.txt)
#   --wall-time S        per-task agent cap in seconds (default: 1800)
#   --concurrency N      sandbox jobs in flight (default: 4)
#   --flavor F           sandbox hardware, `hf jobs hardware` (default: cpu-upgrade)
#   --start-timeout S    seconds to wait for a sandbox to boot, image pull
#                        included (default: 3600)
#   --data-bucket URI    hf://buckets/<ns>/<bucket> holding the task data; omit
#                        only for a demo/ smoke run (baked_in_sandbox)
#   --results-bucket URI hf://buckets/<ns>/<bucket> to sync .logs/ale into
#   --namespace NS       bill jobs to an org instead of the token owner
#   --submit             run the orchestrator itself as an HF Job
#   --repo URL/--ref REF clone source used by --submit (default: upstream main)
#   --dry-run            print the plan (run matrix, or the job command) and stop
#
# ALE_DATA_BUCKET / ALE_RESULTS_BUCKET supply the bucket defaults.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

HARNESS="openhands_cli_hf"
MODEL=""
TASKS="6"
FROM="selected_tasks/docker_support.txt"
WALL_TIME=1800
CONCURRENCY=4
FLAVOR="cpu-upgrade"
START_TIMEOUT=3600
DATA_BUCKET="${ALE_DATA_BUCKET:-}"
RESULTS_BUCKET="${ALE_RESULTS_BUCKET:-}"
NAMESPACE=""
SUBMIT=0
DRY_RUN=0
GIT_REPO="https://github.com/rdi-berkeley/agents-last-exam.git"
GIT_REF="main"
ORCH_TIMEOUT="24h"

die() { echo "ERROR: $*" >&2; exit 1; }

while (( $# )); do
  case "$1" in
    --harness)        HARNESS="$2"; shift 2 ;;
    --model)          MODEL="$2"; shift 2 ;;
    --tasks)          TASKS="$2"; shift 2 ;;
    --from)           FROM="$2"; shift 2 ;;
    --wall-time)      WALL_TIME="$2"; shift 2 ;;
    --concurrency)    CONCURRENCY="$2"; shift 2 ;;
    --flavor)         FLAVOR="$2"; shift 2 ;;
    --start-timeout)  START_TIMEOUT="$2"; shift 2 ;;
    --data-bucket)    DATA_BUCKET="$2"; shift 2 ;;
    --results-bucket) RESULTS_BUCKET="$2"; shift 2 ;;
    --namespace)      NAMESPACE="$2"; shift 2 ;;
    --submit)         SUBMIT=1; shift ;;
    --repo)           GIT_REPO="$2"; shift 2 ;;
    --ref)            GIT_REF="$2"; shift 2 ;;
    --dry-run)        DRY_RUN=1; shift ;;
    -h|--help)        sed -n '2,35p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)                die "unknown flag: $1 (try --help)" ;;
  esac
done

AGENT_SRC="configs/agents/${HARNESS}.yaml"
[[ -f "$AGENT_SRC" ]] || die "no harness preset at $AGENT_SRC"
[[ -z "$DATA_BUCKET" || "$DATA_BUCKET" == hf://* ]] \
  || die "--data-bucket must be an hf:// URI, got: $DATA_BUCKET"
[[ -z "$RESULTS_BUCKET" || "$RESULTS_BUCKET" == hf://* ]] \
  || die "--results-bucket must be an hf:// URI, got: $RESULTS_BUCKET"

HF=(hf)
command -v hf >/dev/null 2>&1 || HF=(uv run hf)

# The provider reads the token through huggingface_hub's get_token(), which
# honors `hf auth login`, but the agent presets reference ${env:HF_TOKEN},
# which resolves only from the process environment. Bridge the two so a plain
# `hf auth login` is enough.
if [[ -z "${HF_TOKEN:-}" ]]; then
  token="$("${HF[@]}" auth token 2>/dev/null | head -1 || true)"
  [[ -n "$token" ]] && export HF_TOKEN="$token"
fi
[[ -n "${HF_TOKEN:-}" ]] \
  || die "no HF token. Run \`hf auth login\`, or export HF_TOKEN=hf_..."

# `secret_file:` is loaded with override=True, so a blank HF_TOKEN there wins
# over the environment and hands the agent an empty API key.
if grep -qE '^[[:space:]]*(export[[:space:]]+)?HF_TOKEN=[[:space:]]*$' secret/.env 2>/dev/null; then
  die "secret/.env sets an empty HF_TOKEN, which overrides your shell. Fill it in or drop the line."
fi

# `hf buckets create` defaults to PUBLIC, and run logs carry full agent
# trajectories plus task output. Refuse a world-readable destination here
# rather than discover it after the sync.
if [[ -n "$RESULTS_BUCKET" ]]; then
  bucket_id="$(cut -d/ -f1,2 <<<"${RESULTS_BUCKET#hf://buckets/}")"
  case "$("${HF[@]}" buckets info "$bucket_id" 2>/dev/null || true)" in
    *'"private": true'*|*'"private":true'*) ;;
    *'"private": false'*|*'"private":false'*)
      die "results bucket ${bucket_id} is PUBLIC. Fix it with:
       hf buckets settings ${bucket_id} --private" ;;
    *)
      echo ">> WARNING: could not read ${bucket_id} visibility. Check it with:" >&2
      echo ">>          hf buckets info ${bucket_id}" >&2 ;;
  esac
fi

RUN_NAME="quickrun_${HARNESS}_$(date +%Y%m%d-%H%M%S)"
RUN_DIR=".logs/quickrun/${RUN_NAME}"
mkdir -p "$RUN_DIR"

# ---- task selection -------------------------------------------------------
# A sampled subset is interleaved by domain (round-robin over the first path
# segment) so a 6-task run still spans 6 domains instead of 6 variants of one.
if [[ "$TASKS" =~ ^[0-9]+$ ]]; then
  [[ -f "$FROM" ]] || die "task pool not found: $FROM"
  sed 's/#.*//' "$FROM" | tr -d '[:blank:]' | grep . | awk -F/ '
      { if (!($1 in n)) order[++k] = $1
        q[$1, ++n[$1]] = $0
        if (n[$1] > deep) deep = n[$1] }
      END { for (i = 1; i <= deep; i++)
              for (j = 1; j <= k; j++)
                if (i <= n[order[j]]) print q[order[j], i] }
    ' > "$RUN_DIR/pool.txt" || true
  head -n "$TASKS" "$RUN_DIR/pool.txt" > "$RUN_DIR/tasks.txt"
  rm -f "$RUN_DIR/pool.txt"
elif [[ -f "$TASKS" ]]; then
  { sed 's/#.*//' "$TASKS" | tr -d '[:blank:]' | grep . > "$RUN_DIR/tasks.txt"; } || true
else
  { tr ',' '\n' <<<"$TASKS" | tr -d '[:blank:]' | grep . > "$RUN_DIR/tasks.txt"; } || true
fi
[[ -s "$RUN_DIR/tasks.txt" ]] || die "empty task selection from --tasks $TASKS"
# hfsandbox boots ale-ubuntu22-docker, so only cpu-free-ubuntu tasks can run;
# reject the rest here instead of after a 10-minute image pull.
while read -r t; do
  card="tasks/$t/task_card.json"
  [[ -f "$card" ]] || die "no such task: $t"
  grep -q '"cpu-free-ubuntu"' "$card" \
    || die "$t is not a cpu-free-ubuntu task — hfsandbox runs the Linux subset only"
done < "$RUN_DIR/tasks.txt"

# ---- generated configs ----------------------------------------------------
if [[ -n "$MODEL" ]]; then
  grep -q '^model:' "$AGENT_SRC" || die "$AGENT_SRC has no top-level \`model:\` to override"
  sed "s|^model:.*|model: ${MODEL}|" "$AGENT_SRC" > "$RUN_DIR/agent.yaml"
else
  cp "$AGENT_SRC" "$RUN_DIR/agent.yaml"
fi

{
  echo "provider: hfsandbox"
  echo "image: ale-ubuntu22-docker"
  echo "flavor: ${FLAVOR}"
  echo "transport: job"
  echo "job_timeout: 24h"
  # A cold pull of the ~40 GB sandbox image has been measured at ~28 min, so
  # the provider default of 1800 can expire while the pull is still running.
  echo "start_timeout: ${START_TIMEOUT}"
  if [[ -n "$NAMESPACE" ]]; then echo "namespace: ${NAMESPACE}"; fi
  if [[ -n "$DATA_BUCKET" ]]; then
    echo "volumes:"
    echo "  - ${DATA_BUCKET}:/mnt/ale-task-data:ro"
    echo "task_data_source: mounted:/mnt/ale-task-data"
  else
    echo "task_data_source: baked_in_sandbox"
  fi
  echo "output_path: local"
} > "$RUN_DIR/environment.yaml"

# max_attempts: 1 — a fast run reports failures instead of paying for retries.
cat > "$RUN_DIR/experiment.yaml" <<YAML
name: ${RUN_NAME}
secret_file: ${REPO_ROOT}/secret/.env
agents:
  - ${REPO_ROOT}/${RUN_DIR}/agent.yaml
environment: ${REPO_ROOT}/${RUN_DIR}/environment.yaml
tasks: ${REPO_ROOT}/${RUN_DIR}/tasks.txt
output:
  root: .logs/ale
concurrency: ${CONCURRENCY}
wall_time_s: ${WALL_TIME}
auto_resume: true
max_attempts: 1
cleanup_mode: delete
YAML

# Staging copies from <mount>/<domain>/<task>/<variant>/input, so the bucket
# root must hold domain dirs. A tarball that extracted into a wrapper dir is
# the usual reason it does not, and the sandbox only finds out after booting.
if [[ -n "$DATA_BUCKET" ]]; then
  listing="$("${HF[@]}" buckets ls "${DATA_BUCKET#hf://buckets/}" 2>/dev/null || true)"
  if [[ -n "$listing" ]]; then
    missing=()
    while read -r domain; do
      grep -qE "(^|[[:space:]/])${domain}(/|[[:space:]]|$)" <<<"$listing" || missing+=("$domain")
    done < <(cut -d/ -f1 "$RUN_DIR/tasks.txt" | sort -u)
    if (( ${#missing[@]} )); then
      echo "ERROR: ${DATA_BUCKET} has no ${missing[*]} directory at its root." >&2
      echo "       Staging reads <bucket>/<domain>/<task>/<variant>/input. Root currently holds:" >&2
      echo "$listing" | head -10 | sed 's/^/         /' >&2
      die "point --data-bucket at the subdirectory holding the domain dirs, e.g.
       --data-bucket ${DATA_BUCKET}/<subdir>"
    fi
  else
    echo ">> WARNING: could not list ${DATA_BUCKET}; skipping the layout check" >&2
  fi
fi

echo ">> harness:     ${HARNESS}$([[ -n "$MODEL" ]] && echo " (model ${MODEL})")"
echo ">> tasks:       $(wc -l < "$RUN_DIR/tasks.txt" | tr -d ' ') selected"
sed 's/^/                 /' "$RUN_DIR/tasks.txt"
echo ">> sandboxes:   ${FLAVOR}, ${CONCURRENCY} in flight, ${WALL_TIME}s per task"
echo ">> boot budget: ${START_TIMEOUT}s per sandbox (cold image pull runs ~30 min)"
echo ">> task data:   ${DATA_BUCKET:-baked_in_sandbox (demo/ tasks only — real tasks need --data-bucket)}"
echo ">> configs:     ${RUN_DIR}/"

# ---- submit the orchestrator as an HF Job ---------------------------------
if (( SUBMIT )); then
  remote_flags=(
    --harness "$HARNESS"
    --tasks "$(paste -sd, "$RUN_DIR/tasks.txt")"
    --wall-time "$WALL_TIME" --concurrency "$CONCURRENCY" --flavor "$FLAVOR"
    --start-timeout "$START_TIMEOUT"
  )
  [[ -n "$DATA_BUCKET" ]] && remote_flags+=(--data-bucket "$DATA_BUCKET")
  [[ -n "$MODEL" ]] && remote_flags+=(--model "$MODEL")
  [[ -n "$RESULTS_BUCKET" ]] && remote_flags+=(--results-bucket "$RESULTS_BUCKET")
  [[ -n "$NAMESPACE" ]] && remote_flags+=(--namespace "$NAMESPACE")

  secrets=(--secrets HF_TOKEN)
  [[ -n "${OPENAI_API_KEY:-}" ]] && secrets+=(--secrets OPENAI_API_KEY)
  ns_flag=()
  [[ -n "$NAMESPACE" ]] && ns_flag=(--namespace "$NAMESPACE")

  job_script="set -euo pipefail
apt-get update -qq && apt-get install -y -qq git >/dev/null
git clone --depth 1 --branch ${GIT_REF} ${GIT_REPO} /work
cd /work
mkdir -p secret secret/eval_time
printf 'HF_TOKEN=%s\\n' \"\$HF_TOKEN\" > secret/.env
if [ -n \"\${OPENAI_API_KEY:-}\" ]; then
  printf 'OPENAI_API_KEY=%s\\n' \"\$OPENAI_API_KEY\" > secret/eval_time/openai.env
fi
uv sync --all-packages
exec scripts/hf_quickrun.sh $(printf '%q ' "${remote_flags[@]}")"

  if (( DRY_RUN )); then
    echo ">> would submit (orchestrator job, cpu-basic):"
    echo "$job_script" | sed 's/^/                 /'
    exit 0
  fi
  exec hf jobs run --flavor cpu-basic --timeout "$ORCH_TIMEOUT" \
    "${secrets[@]}" "${ns_flag[@]}" \
    ghcr.io/astral-sh/uv:python3.12-bookworm \
    bash -c "$job_script"
fi

# ---- run here -------------------------------------------------------------
ALE=(uv run python -m ale_run)
command -v uv >/dev/null 2>&1 || ALE=(python3 -m ale_run)

if (( DRY_RUN )); then
  exec "${ALE[@]}" run "$RUN_DIR/experiment.yaml" --dry-run
fi

status=0
"${ALE[@]}" run "$RUN_DIR/experiment.yaml" -v || status=$?

if [[ -n "$RESULTS_BUCKET" ]]; then
  echo ">> syncing .logs/ale -> ${RESULTS_BUCKET}"
  "${HF[@]}" buckets sync .logs/ale "$RESULTS_BUCKET"
fi

exit "$status"
