#!/usr/bin/env bash
# Fetch input + reference data for a handful of tasks and push it to your
# task-data bucket, instead of moving the 48 GB archive around.
#
# Pulls from the two browsable datasets, merging them into one tree:
#   agents-last-exam-data        input/ + software/   (open)
#   agents-last-exam-reference   reference/           (gated, needs approval)
# The result is <domain>/<task>/<variant>/{input,software,reference}, which is
# the layout `task_data_source: mounted:` expects.
#
# Usage:
#   scripts/fetch_task_subset.sh --tasks <FILE|CSV> [options]
#
#   --tasks FILE|CSV   task list file (e.g. a run's tasks.txt) or
#                      comma-separated task paths
#   --dest DIR         download dir (default: task-data-subset)
#   --bucket URI       hf://buckets/<ns>/<bucket> to sync into when done
#   --no-reference     skip the gated repo (runs will not be scoreable)
#   --dry-run          print the commands without running them
#
# Then point a run at it:
#   scripts/hf_quickrun.sh --data-bucket <URI> --tasks <same list> ...

if [ -z "${BASH_VERSION:-}" ] || [ -n "${POSIXLY_CORRECT:-}" ]; then
  unset POSIXLY_CORRECT
  exec bash "$0" "$@"
fi

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

INPUT_REPO="agents-last-exam/agents-last-exam-data"
REF_REPO="agents-last-exam/agents-last-exam-reference"
TASKS=""
DEST="task-data-subset"
BUCKET=""
WITH_REF=1
DRY_RUN=0

die() { echo "ERROR: $*" >&2; exit 1; }
run() { if (( DRY_RUN )); then echo "   $*"; else "$@"; fi }

while (( $# )); do
  case "$1" in
    --tasks)        TASKS="$2"; shift 2 ;;
    --dest)         DEST="$2"; shift 2 ;;
    --bucket)       BUCKET="$2"; shift 2 ;;
    --no-reference) WITH_REF=0; shift ;;
    --dry-run)      DRY_RUN=1; shift ;;
    -h|--help)      sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)              die "unknown flag: $1 (try --help)" ;;
  esac
done

[[ -n "$TASKS" ]] || die "--tasks <FILE|CSV> is required"

HF=(hf)
command -v hf >/dev/null 2>&1 || HF=(uv run hf)

# No mapfile: macOS ships bash 3.2 and the re-exec guard lands on it.
if [[ -f "$TASKS" ]]; then
  list="$(sed 's/#.*//' "$TASKS" | tr -d '[:blank:]' | grep . || true)"
else
  list="$(tr ',' '\n' <<<"$TASKS" | tr -d '[:blank:]' | grep . || true)"
fi
SELECTED=()
while IFS= read -r line; do
  [[ -n "$line" ]] && SELECTED+=("$line")
done <<<"$list"
(( ${#SELECTED[@]} )) || die "no tasks parsed from --tasks $TASKS"

includes=()
for t in "${SELECTED[@]}"; do
  [[ -f "tasks/$t/task_card.json" ]] || die "no such task: $t"
  includes+=(--include "tasks/$t/*")
done

echo ">> ${#SELECTED[@]} task(s) -> ${DEST}/"
printf '     %s\n' "${SELECTED[@]}"

echo ">> input + software from ${INPUT_REPO}"
run "${HF[@]}" download "$INPUT_REPO" --repo-type dataset "${includes[@]}" --local-dir "$DEST"

if (( WITH_REF )); then
  echo ">> reference from ${REF_REPO} (gated)"
  run "${HF[@]}" download "$REF_REPO" --repo-type dataset "${includes[@]}" --local-dir "$DEST"
else
  echo ">> skipping reference: runs will stage no ground truth and score 0"
fi

# Some native filenames cannot be stored on the Hub and ship in a side archive.
# It covers only two tasks, so pull the 328 MB tar only when one is selected.
echo ">> checking path-compatibility manifest"
manifest="$DEST/_transport/v1.1/path-compatibility.json"
run "${HF[@]}" download "$INPUT_REPO" --repo-type dataset \
  --include "_transport/v1.1/path-compatibility.json" --local-dir "$DEST"
if [[ -f "$manifest" ]]; then
  affected=()
  for t in "${SELECTED[@]}"; do
    grep -q "\"tasks/$t/" "$manifest" && affected+=("$t")
  done
  if (( ${#affected[@]} )); then
    echo ">> restoring native filenames for: ${affected[*]}"
    run "${HF[@]}" download "$INPUT_REPO" --repo-type dataset \
      --include "_transport/v1.1/path-compatibility.tar" --local-dir "$DEST"
    run tar -xf "$DEST/_transport/v1.1/path-compatibility.tar" -C "$DEST"
  else
    echo ">> none of the selected tasks need it"
  fi
elif (( DRY_RUN )); then
  echo ">> (dry-run: manifest not fetched, so the affected-task check is skipped)"
fi

echo ">> restoring native symlinks (10 KB)"
run "${HF[@]}" download "$INPUT_REPO" --repo-type dataset \
  --include "_transport/candidate-postvalidation-20260918-r06/tcga-public-symlinks.tar" \
  --local-dir "$DEST"
symlinks="$DEST/_transport/candidate-postvalidation-20260918-r06/tcga-public-symlinks.tar"
[[ -f "$symlinks" ]] && run tar -xf "$symlinks" -C "$DEST/tasks"

if (( ! DRY_RUN )); then
  missing=()
  for t in "${SELECTED[@]}"; do
    compgen -G "$DEST/tasks/$t/*/input" >/dev/null || missing+=("$t:input")
    if (( WITH_REF )); then
      compgen -G "$DEST/tasks/$t/*/reference" >/dev/null || missing+=("$t:reference")
    fi
  done
  if (( ${#missing[@]} )); then
    echo ">> WARNING: not downloaded: ${missing[*]}" >&2
    echo ">>          a missing reference/ means that task cannot be scored." >&2
  fi
fi

if [[ -n "$BUCKET" ]]; then
  # reference/ is gated ground truth. Never push it to a public bucket.
  bucket_id="$(cut -d/ -f1,2 <<<"${BUCKET#hf://buckets/}")"
  case "$("${HF[@]}" buckets info "$bucket_id" 2>/dev/null || true)" in
    *'"private": true'*|*'"private":true'*) ;;
    *'"private": false'*|*'"private":false'*)
      die "${bucket_id} is PUBLIC and this tree holds gated reference data. Fix it with:
       hf buckets settings ${bucket_id} --private" ;;
    *)
      echo ">> WARNING: could not read ${bucket_id} visibility. Confirm it is private" >&2
      echo ">>          before syncing gated reference data: hf buckets info ${bucket_id}" >&2 ;;
  esac
  echo ">> syncing ${DEST}/tasks -> ${BUCKET}"
  run "${HF[@]}" buckets sync "$DEST/tasks" "$BUCKET"
  echo ">> run with:  --data-bucket ${BUCKET}"
else
  echo ">> no --bucket given; sync it yourself with:"
  echo "     hf buckets sync ${DEST}/tasks hf://buckets/<ns>/<bucket>"
fi
