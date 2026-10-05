#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pyyaml>=6"]
# ///
"""Controlled model-size sweep: one task subset, one harness, N model sizes.

Everything except `model:` is held fixed. The subset is resolved once and the
same explicit task list is handed to every model inside a single experiment, so
the runs share a task selection, a wall-clock cap, a sandbox flavor and a
prompt suffix. Results land in
`.logs/ale/<run-name>/<harness>/<model-slug>/<task>/v0/<timestamp>/`.

`auto_resume` is scoped to that output root, so re-invoking with the same
--run-name re-runs only the (model, task) cells that have no completed run.
Without --run-name each invocation gets a fresh timestamped root and starts
over.

Typical use, 10% of the Linux pool against the default ladder:

    scripts/model_size_sweep.py \\
      --data-bucket hf://buckets/<ns>/ale-task-data \\
      --results-bucket hf://buckets/<ns>/ale-results-qwen \\
      --namespace <ns> --submit \\
      --repo https://github.com/<you>/agents-last-exam --ref <branch>

Add --fetch-data on the first run to stage that subset's task data into the
data bucket. A local run prints a task x model score matrix when it ends; after
a --submit run, pull the logs and render the same table with --report-only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import hf_quickrun
from ale_hf_common import (
    DEFAULT_POOL,
    REPO_ROOT,
    ScriptError,
    info,
    require_linux_subset,
    select_tasks,
)
from fetch_task_subset import fetch_subset

# Dense Qwen3 ladder (4B/8B/14B/32B) plus Qwen3.8-27B. All five are served by
# the HF Inference Providers router, which is what `qwen_code_hf` routes to.
MODEL_LADDER = (
    # "Qwen/Qwen3.5-2B",
    # "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.8-27B",
)

# Smaller models in the ladder resolve the staged input paths against their own
# work dir instead of using the absolute path they were given, and then report
# a task as impossible. Grounding them keeps the comparison about capability.
PROMPT_SUFFIX = (
    "All paths in this prompt are absolute. Do not resolve them against your "
    "working directory. Read and write files at the exact paths given."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--harness", default="qwen_code_hf",
                        help="preset under configs/agents/ (default: %(default)s)")
    parser.add_argument("--agent", action="append", default=[], metavar="NAME",
                        help="preset under configs/agents/ used verbatim; repeat to run "
                             "several arms (e.g. a prompt ablation) in one experiment. "
                             "Each needs its own `id:`. Overrides --harness/--model")
    parser.add_argument("--model", action="append", default=[], metavar="ID",
                        help="replace the default ladder; repeat per model")
    parser.add_argument("--tasks", default="10%", metavar="N|N%|FILE|CSV",
                        help="subset of the pool to run against every model (default: %(default)s)")
    parser.add_argument("--from", dest="pool", default=DEFAULT_POOL, metavar="FILE",
                        help="pool the subset is drawn from (default: %(default)s)")
    parser.add_argument("--wall-time", type=int, default=3600, metavar="S",
                        help="per-task agent cap in seconds (default: %(default)s)")
    parser.add_argument("--concurrency", type=int, default=6, metavar="N",
                        help="sandbox jobs in flight (default: %(default)s)")
    parser.add_argument("--flavor", default="cpu-upgrade",
                        help="sandbox hardware (default: %(default)s)")
    parser.add_argument("--start-timeout", type=int, default=3600, metavar="S",
                        help="seconds to wait for a sandbox to boot (default: %(default)s)")
    parser.add_argument("--prompt-suffix", default=PROMPT_SUFFIX, metavar="TXT",
                        help="appended to every task prompt; pass '' to disable")
    parser.add_argument("--data-bucket", default=os.environ.get("ALE_DATA_BUCKET", ""),
                        metavar="URI", help="task-data bucket. Default: $ALE_DATA_BUCKET")
    parser.add_argument("--results-bucket", default=os.environ.get("ALE_RESULTS_BUCKET", ""),
                        metavar="URI", help="results bucket. Default: $ALE_RESULTS_BUCKET")
    parser.add_argument("--namespace", default="", metavar="NS",
                        help="bill jobs to an org instead of the token owner")
    parser.add_argument("--fetch-data", action="store_true",
                        help="stage this subset's task data into --data-bucket before running")
    parser.add_argument("--no-reference", dest="with_reference", action="store_false",
                        help="with --fetch-data, skip the gated reference repo")
    parser.add_argument("--job-timeout", default=hf_quickrun.SANDBOX_JOB_TIMEOUT, metavar="D",
                        help="hard cap on each sandbox job, which is what bounds the bill "
                             "if the orchestrator dies without cancelling its sandboxes "
                             "(default: %(default)s)")
    parser.add_argument("--disable-resume", action="store_true",
                        help="re-run every selected cell, even one that already has a "
                             "completed or timeout result. Resume counts a timeout as "
                             "done, so this is what re-runs a cell that hit --wall-time")
    parser.add_argument("--submit", action="store_true",
                        help="run the orchestrator as an HF Job instead of locally")
    parser.add_argument("--detach", action="store_true",
                        help="with --submit, return as soon as the orchestrator job is "
                             "created instead of streaming its logs")
    parser.add_argument("--repo", dest="git_repo", default=hf_quickrun.UPSTREAM_REPO,
                        metavar="URL", help="clone source used by --submit")
    parser.add_argument("--ref", dest="git_ref", default="main", metavar="REF",
                        help="clone ref used by --submit (default: %(default)s)")
    parser.add_argument("--run-name", default="", metavar="NAME",
                        help="name the run; reuse it to resume one, since auto_resume is "
                             "scoped to .logs/ale/<run-name>")
    parser.add_argument("--no-report", dest="report", action="store_false",
                        help="skip the score matrix a local run prints when it ends")
    parser.add_argument("--report-only", metavar="DIR", default="",
                        help="skip the run; print the score matrix from an existing log tree")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and stop")
    return parser.parse_args(argv)


def record_plan(
    run_name: str, models: list[str], tasks: list[str], args: argparse.Namespace
) -> Path:
    """Write the sweep's fixed parameters so the run can be reproduced or audited."""
    sweep_dir = REPO_ROOT / ".logs" / "sweep" / run_name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "tasks.txt").write_text("\n".join(tasks) + "\n", encoding="utf-8")
    (sweep_dir / "plan.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "harness": args.harness,
                "models": models,
                "tasks": tasks,
                "task_spec": args.tasks,
                "pool": args.pool,
                "wall_time_s": args.wall_time,
                "concurrency": args.concurrency,
                "flavor": args.flavor,
                "prompt_suffix": args.prompt_suffix,
                "data_bucket": args.data_bucket,
                "results_bucket": args.results_bucket,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return sweep_dir


def load_scores(log_root: Path) -> dict[tuple[tuple[str, str], str], tuple[str, float | None]]:
    """Latest (status, score) per ((agent id, model), task) from a `.logs/ale` tree.

    Keyed on what run.json records rather than on the directory slugs. The agent
    id is part of the key because an ablation holds the model fixed and varies
    the agent config, so keying on the model alone would merge the arms.
    """
    latest: dict[tuple[tuple[str, str], str], tuple[str, str, float | None]] = {}
    for run_json in log_root.rglob("run.json"):
        try:
            payload = json.loads(run_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        agent = payload.get("agent") or {}
        arm = (agent.get("id") or "?", agent.get("model") or "?")
        task = ((payload.get("task") or {}).get("path") or "?").removeprefix("tasks/")
        stamp = payload.get("timestamp_utc") or run_json.parent.name
        key = (arm, task)
        if key not in latest or stamp >= latest[key][0]:
            latest[key] = (stamp, payload.get("status") or "?", payload.get("score"))
    return {key: (status, score) for key, (_, status, score) in latest.items()}


def column_labels(arms: list[tuple[str, str]]) -> list[str]:
    """Shortest labels that still tell the arms apart.

    One agent across several models is a model sweep, so label by model; one
    model across several agents is an ablation, so label by agent id.
    """
    by_model = [m.split("/")[-1] for _, m in arms]
    by_agent = [a for a, _ in arms]
    if len({a for a, _ in arms}) == 1:
        return by_model
    if len({m for _, m in arms}) == 1:
        return by_agent
    return [f"{a}/{m}" for a, m in zip(by_agent, by_model, strict=True)]


def print_report(log_root: Path, models: list[str], tasks: list[str]) -> None:
    """Print a task x arm score matrix, model sweeps left to right in ladder order."""
    scores = load_scores(log_root)
    if not scores:
        info(f"no run.json found under {log_root}; nothing to report")
        return

    def ladder_rank(arm: tuple[str, str]) -> tuple[int, str, str]:
        agent, model = arm
        rank = MODEL_LADDER.index(model) if model in MODEL_LADDER else len(MODEL_LADDER)
        return (rank, model, agent)

    found = sorted({arm for arm, _ in scores}, key=ladder_rank)
    row_arms = [arm for arm in found if not models or arm[1] in models]
    row_tasks = tasks or sorted({t for _, t in scores})
    columns = column_labels(row_arms)
    task_width = max((len(t) for t in [*row_tasks, "scored / total"]), default=4) + 2
    widths = [len(c) + 2 for c in columns]

    def render(label: str, cells: list[str]) -> str:
        return (
            label.ljust(task_width)
            + "".join(cell.rjust(w) for cell, w in zip(cells, widths, strict=True))
        ).rstrip()

    scored: dict[tuple[str, str], list[float]] = defaultdict(list)
    rows = []
    for task in row_tasks:
        cells = []
        for arm in row_arms:
            status, score = scores.get((arm, task), ("", None))
            if score is None:
                cells.append({"": "-", "failed": "F", "timeout": "T"}.get(status, status[:8]))
            else:
                scored[arm].append(float(score))
                cells.append(f"{float(score):.2f}")
        rows.append(render(task, cells))

    print()
    print(f"score matrix from {log_root}")
    print("cells: score, or - (no run) / F (failed) / T (timeout)")
    print(render("task", columns))
    print("\n".join(rows))
    print("-" * (task_width + sum(widths)))
    print(render("mean (scored)", [
        f"{sum(scored[a]) / len(scored[a]):.2f}" if scored[a] else "-" for a in row_arms
    ]))
    print(render("scored / total", [
        f"{len(scored[a])}/{len(row_tasks)}" for a in row_arms
    ]))


def sweep(args: argparse.Namespace) -> int:
    models = [] if args.agent else (list(args.model) or list(MODEL_LADDER))
    tasks = select_tasks(args.tasks, pool=args.pool)
    require_linux_subset(tasks)

    run_name = args.run_name or f"sweep_{args.harness}_{time.strftime('%Y%m%d-%H%M%S')}"
    sweep_dir = record_plan(run_name, models, tasks, args)

    pool = Path(args.pool)
    info(f"sweep:       {run_name}")
    info(f"subset:      {args.tasks} of "
         f"{pool.relative_to(REPO_ROOT) if pool.is_relative_to(REPO_ROOT) else pool}")
    info(f"plan:        {sweep_dir.relative_to(REPO_ROOT)}/")

    if args.fetch_data:
        if not args.data_bucket:
            raise ScriptError("--fetch-data needs --data-bucket to sync into")
        fetch_subset(
            tasks,
            bucket=args.data_bucket,
            with_reference=args.with_reference,
            dry_run=args.dry_run,
        )

    settings = hf_quickrun.Settings(
        harness=args.harness,
        agents=tuple(args.agent),
        models=tuple(models),
        tasks=",".join(tasks),
        pool=args.pool,
        wall_time=args.wall_time,
        concurrency=args.concurrency,
        flavor=args.flavor,
        start_timeout=args.start_timeout,
        job_timeout=args.job_timeout,
        prompt_suffix=args.prompt_suffix,
        data_bucket=args.data_bucket,
        results_bucket=args.results_bucket,
        namespace=args.namespace,
        disable_resume=args.disable_resume,
        submit=args.submit,
        detach=args.detach,
        git_repo=args.git_repo,
        git_ref=args.git_ref,
        dry_run=args.dry_run,
        run_name=run_name,
    )
    status = hf_quickrun.quickrun(settings)
    log_root = REPO_ROOT / ".logs" / "ale" / run_name

    if args.report and not args.submit and not args.dry_run:
        print_report(log_root, models, tasks)
    elif args.submit:
        info("submitted. Once it finishes, pull the results and report with:")
        print(f"     hf buckets sync {args.results_bucket or '<results-bucket>'} .logs/ale")
        print(f"     scripts/model_size_sweep.py --report-only .logs/ale/{run_name}")
    return status


def main() -> int:
    args = parse_args()
    # Relative paths are the caller's, but everything below runs from the repo root.
    report_only = Path(args.report_only).resolve() if args.report_only else None
    if (pool := Path(args.pool)).is_file():
        args.pool = str(pool.resolve())
    os.chdir(REPO_ROOT)
    try:
        if report_only is not None:
            print_report(report_only, [], [])
            return 0
        return sweep(args)
    except ScriptError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())