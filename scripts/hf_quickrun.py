#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pyyaml>=6"]
# ///
"""Run an ALE subset on Hugging Face Jobs: pick a harness and one or more
models, sample a few tasks, cap the per-task wall clock, then sync the run logs
to your bucket.

Task sandboxes always run as HF Jobs (provider: hfsandbox). The orchestrator
runs wherever you invoke this: your machine by default, or inside a cpu-basic
HF Job with --submit (then nothing local is needed but the `hf` CLI).

Passing --model more than once turns the run into a model matrix: every model
runs the same tasks inside one experiment, and the output tree separates them
by model slug. That is the supported way to run a controlled comparison; see
`model_size_sweep.py` for a ready-made one.

One-time setup (details: docs/hf-jobs-quickrun.md):
    hf auth login                                   # token needs the `jobs` scope
    scripts/fetch_task_subset.py --tasks <list> --bucket hf://buckets/<ns>/ale-task-data
    hf buckets create <ns>/ale-results --private

ALE_DATA_BUCKET / ALE_RESULTS_BUCKET supply the bucket defaults.

This is the Python port of `hf_quickrun.sh`, which stays in place and behaves
the same for a single model.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from ale_hf_common import (
    DEFAULT_POOL,
    REPO_ROOT,
    ScriptError,
    bucket_id,
    capture,
    ensure_hf_token,
    hf_cli,
    info,
    reject_blank_token_in_secret_env,
    require_bucket_uri,
    require_linux_subset,
    require_private_bucket,
    select_tasks,
    warn,
)

UPSTREAM_REPO = "https://github.com/rdi-berkeley/agents-last-exam.git"
ORCHESTRATOR_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm"
ORCHESTRATOR_TIMEOUT = "24h"
SANDBOX_IMAGE = "ale-ubuntu22-docker"
SANDBOX_JOB_TIMEOUT = "24h"


@dataclass
class Settings:
    """One quickrun invocation. `parse_args` builds it from the command line."""

    harness: str = "openhands_cli_hf"
    models: tuple[str, ...] = ()
    tasks: str = "6"
    pool: str = DEFAULT_POOL
    wall_time: int = 1800
    concurrency: int = 4
    flavor: str = "cpu-upgrade"
    start_timeout: int = 3600
    prompt_suffix: str = ""
    data_bucket: str = field(default_factory=lambda: os.environ.get("ALE_DATA_BUCKET", ""))
    results_bucket: str = field(default_factory=lambda: os.environ.get("ALE_RESULTS_BUCKET", ""))
    namespace: str = ""
    submit: bool = False
    git_repo: str = UPSTREAM_REPO
    git_ref: str = "main"
    dry_run: bool = False
    run_name: str = ""


def parse_args(argv: list[str] | None = None) -> Settings:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--harness", default=Settings.harness,
                        help="preset under configs/agents/ (default: %(default)s)")
    parser.add_argument("--model", action="append", default=[], metavar="ID",
                        help="model id overriding the preset's `model:`; repeat (or pass a "
                             "comma-separated list) to run a model matrix")
    parser.add_argument("--tasks", default=Settings.tasks, metavar="N|N%|FILE|CSV",
                        help="subset size, a percentage of the pool, a .txt list, or an "
                             "explicit comma-separated list (default: %(default)s)")
    parser.add_argument("--from", dest="pool", default=DEFAULT_POOL, metavar="FILE",
                        help="pool to sample from (default: %(default)s)")
    parser.add_argument("--wall-time", type=int, default=Settings.wall_time, metavar="S",
                        help="per-task agent cap in seconds (default: %(default)s)")
    parser.add_argument("--concurrency", type=int, default=Settings.concurrency, metavar="N",
                        help="sandbox jobs in flight (default: %(default)s)")
    parser.add_argument("--flavor", default=Settings.flavor,
                        help="sandbox hardware, `hf jobs hardware` (default: %(default)s)")
    parser.add_argument("--start-timeout", type=int, default=Settings.start_timeout, metavar="S",
                        help="seconds to wait for a sandbox to boot, image pull included "
                             "(default: %(default)s)")
    parser.add_argument("--prompt-suffix", default="", metavar="TXT",
                        help="appended to every task prompt; use it to ground an agent that "
                             "invents paths instead of using the given one")
    parser.add_argument("--data-bucket", default=os.environ.get("ALE_DATA_BUCKET", ""),
                        metavar="URI",
                        help="hf://buckets/<ns>/<bucket> holding the task data; omit only for "
                             "a demo/ smoke run (baked_in_sandbox). Default: $ALE_DATA_BUCKET")
    parser.add_argument("--results-bucket", default=os.environ.get("ALE_RESULTS_BUCKET", ""),
                        metavar="URI",
                        help="hf://buckets/<ns>/<bucket> to sync .logs/ale into. "
                             "Default: $ALE_RESULTS_BUCKET")
    parser.add_argument("--namespace", default="", metavar="NS",
                        help="bill jobs to an org instead of the token owner")
    parser.add_argument("--submit", action="store_true",
                        help="run the orchestrator itself as an HF Job")
    parser.add_argument("--repo", dest="git_repo", default=UPSTREAM_REPO, metavar="URL",
                        help="clone source used by --submit (default: upstream)")
    parser.add_argument("--ref", dest="git_ref", default="main", metavar="REF",
                        help="clone ref used by --submit (default: %(default)s)")
    parser.add_argument("--run-name", default="", metavar="NAME",
                        help="name the run, its .logs/quickrun/ config dir and its "
                             ".logs/ale/ output root; reuse a name to resume that run")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan (run matrix, or the job command) and stop")
    args = parser.parse_args(argv)

    models = tuple(m.strip() for spec in args.model for m in spec.split(",") if m.strip())
    # Relative paths are the caller's, but `quickrun` runs from the repo root.
    for attr in ("tasks", "pool"):
        if (given := Path(getattr(args, attr))).is_file():
            setattr(args, attr, str(given.resolve()))
    return Settings(
        harness=args.harness,
        models=models,
        tasks=args.tasks,
        pool=args.pool,
        wall_time=args.wall_time,
        concurrency=args.concurrency,
        flavor=args.flavor,
        start_timeout=args.start_timeout,
        prompt_suffix=args.prompt_suffix,
        data_bucket=args.data_bucket,
        results_bucket=args.results_bucket,
        namespace=args.namespace,
        submit=args.submit,
        git_repo=args.git_repo,
        git_ref=args.git_ref,
        dry_run=args.dry_run,
        run_name=args.run_name,
    )


def model_slug(model: str) -> str:
    """Filename-safe model id, matching run_writer.slug_model for the ids we use."""
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-") or "model"


def write_agent_configs(settings: Settings, run_dir: Path) -> list[Path]:
    """One agent yaml per model (or a copy of the preset when no model is given)."""
    preset = REPO_ROOT / "configs" / "agents" / f"{settings.harness}.yaml"
    if not preset.is_file():
        raise ScriptError(f"no harness preset at {preset.relative_to(REPO_ROOT)}")
    raw = yaml.safe_load(preset.read_text(encoding="utf-8")) or {}

    if not settings.models:
        target = run_dir / "agent.yaml"
        target.write_text(preset.read_text(encoding="utf-8"), encoding="utf-8")
        return [target]

    if "model" not in raw:
        raise ScriptError(f"{preset.relative_to(REPO_ROOT)} has no top-level `model:` to override")
    written = []
    for model in settings.models:
        target = run_dir / f"agent-{model_slug(model)}.yaml"
        target.write_text(
            yaml.safe_dump({**raw, "model": model}, sort_keys=False), encoding="utf-8"
        )
        written.append(target)
    return written


def write_environment_config(settings: Settings, run_dir: Path) -> Path:
    env: dict[str, object] = {
        "provider": "hfsandbox",
        "image": SANDBOX_IMAGE,
        "flavor": settings.flavor,
        "transport": "job",
        "job_timeout": SANDBOX_JOB_TIMEOUT,
        # A cold pull of the ~40 GB sandbox image has been measured at ~28 min, so
        # the provider default of 1800 can expire while the pull is still running.
        "start_timeout": settings.start_timeout,
    }
    if settings.namespace:
        env["namespace"] = settings.namespace
    if settings.data_bucket:
        env["volumes"] = [f"{settings.data_bucket}:/mnt/ale-task-data:ro"]
        env["task_data_source"] = "mounted:/mnt/ale-task-data"
    else:
        env["task_data_source"] = "baked_in_sandbox"
    env["output_path"] = "local"

    target = run_dir / "environment.yaml"
    target.write_text(yaml.safe_dump(env, sort_keys=False), encoding="utf-8")
    return target


def write_experiment_config(
    settings: Settings,
    run_dir: Path,
    run_name: str,
    agent_configs: list[Path],
    environment_config: Path,
    tasks_file: Path,
) -> Path:
    # max_attempts: 1 - a fast run reports failures instead of paying for retries.
    experiment: dict[str, object] = {
        "name": run_name,
        "secret_file": str(REPO_ROOT / "secret" / ".env"),
        "agents": [str(p) for p in agent_configs],
        "environment": str(environment_config),
        "tasks": str(tasks_file),
        "output": {"root": ".logs/ale"},
        "concurrency": settings.concurrency,
        "wall_time_s": settings.wall_time,
        "auto_resume": True,
        "max_attempts": 1,
        "cleanup_mode": "delete",
    }
    if settings.prompt_suffix:
        experiment["prompt_suffix"] = settings.prompt_suffix

    target = run_dir / "experiment.yaml"
    target.write_text(yaml.safe_dump(experiment, sort_keys=False), encoding="utf-8")
    return target


def preflight_data_bucket(settings: Settings, tasks: list[str]) -> None:
    """Staging copies from `<mount>/<domain>/<task>/<variant>/input`, so the bucket
    root must hold domain dirs. A tarball that extracted into a wrapper dir is the
    usual reason it does not, and the sandbox only finds out after booting.
    """
    if settings.data_bucket.startswith("hf://datasets/"):
        warn("mounting a dataset repo directly. The open input dataset ships")
        warn("        input/ and software/ but no reference/, so tasks will run and")
        warn("        then fail to score. Use it to smoke-test a harness, not to rank.")
        return
    if not settings.data_bucket:
        return

    listing = capture([*hf_cli(), "buckets", "ls", bucket_id(settings.data_bucket)])
    if not listing:
        warn(f"could not list {settings.data_bucket}; skipping the layout check")
        return
    missing = [
        domain
        for domain in sorted({t.split("/", 1)[0] for t in tasks})
        if not re.search(rf"(^|[\s/]){re.escape(domain)}(/|\s|$)", listing, re.MULTILINE)
    ]
    if missing:
        head = "\n".join(f"         {line}" for line in listing.splitlines()[:10])
        raise ScriptError(
            f"{settings.data_bucket} has no {' '.join(missing)} directory at its root.\n"
            f"       Staging reads <bucket>/<domain>/<task>/<variant>/input. "
            f"Root currently holds:\n{head}\n"
            f"       --data-bucket must be the TASK-DATA bucket, not the results bucket.\n"
            f"       If this is the right bucket but the archive extracted one level "
            f"deeper,\n       point at that subdir: "
            f"--data-bucket {settings.data_bucket}/<subdir>"
        )


def submit_orchestrator(settings: Settings, tasks: list[str]) -> int:
    """Run this same script inside a cpu-basic HF Job against a fresh clone."""
    flags = [
        "--harness", settings.harness,
        "--tasks", ",".join(tasks),
        "--wall-time", str(settings.wall_time),
        "--concurrency", str(settings.concurrency),
        "--flavor", settings.flavor,
        "--start-timeout", str(settings.start_timeout),
    ]
    for model in settings.models:
        flags += ["--model", model]
    for flag, value in (
        ("--data-bucket", settings.data_bucket),
        ("--prompt-suffix", settings.prompt_suffix),
        ("--results-bucket", settings.results_bucket),
        ("--namespace", settings.namespace),
        ("--run-name", settings.run_name),
    ):
        if value:
            flags += [flag, value]

    job_script = f"""set -euo pipefail
apt-get update -qq && apt-get install -y -qq git >/dev/null
git clone --depth 1 --branch {settings.git_ref} {settings.git_repo} /work
cd /work
mkdir -p secret secret/eval_time
printf 'HF_TOKEN=%s\\n' "$HF_TOKEN" > secret/.env
if [ -n "${{OPENAI_API_KEY:-}}" ]; then
  printf 'OPENAI_API_KEY=%s\\n' "$OPENAI_API_KEY" > secret/eval_time/openai.env
fi
uv sync --all-packages
exec uv run python scripts/hf_quickrun.py {shlex.join(flags)}"""

    if settings.dry_run:
        info("would submit (orchestrator job, cpu-basic):")
        print("\n".join(f"                 {line}" for line in job_script.splitlines()))
        return 0

    argv = ["hf", "jobs", "run", "--flavor", "cpu-basic", "--timeout", ORCHESTRATOR_TIMEOUT,
            "--secrets", "HF_TOKEN"]
    if os.environ.get("OPENAI_API_KEY"):
        argv += ["--secrets", "OPENAI_API_KEY"]
    if settings.namespace:
        argv += ["--namespace", settings.namespace]
    argv += [ORCHESTRATOR_IMAGE, "bash", "-c", job_script]
    return subprocess.run(argv, check=False).returncode


def quickrun(settings: Settings) -> int:
    """Generate the configs and execute the run. Returns the process exit code."""
    os.chdir(REPO_ROOT)
    require_bucket_uri(settings.data_bucket, "--data-bucket")
    require_bucket_uri(settings.results_bucket, "--results-bucket")
    ensure_hf_token()
    reject_blank_token_in_secret_env()
    if settings.results_bucket:
        require_private_bucket(settings.results_bucket, holds="full agent trajectories")

    run_name = settings.run_name or f"quickrun_{settings.harness}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = REPO_ROOT / ".logs" / "quickrun" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tasks = select_tasks(settings.tasks, pool=settings.pool)
    require_linux_subset(tasks)
    tasks_file = run_dir / "tasks.txt"
    tasks_file.write_text("\n".join(tasks) + "\n", encoding="utf-8")

    agent_configs = write_agent_configs(settings, run_dir)
    environment_config = write_environment_config(settings, run_dir)
    experiment_config = write_experiment_config(
        settings, run_dir, run_name, agent_configs, environment_config, tasks_file
    )
    preflight_data_bucket(settings, tasks)

    models = settings.models or ("<preset default>",)
    info(f"harness:     {settings.harness}")
    info(f"models:      {len(models)} - {', '.join(models)}")
    info(f"tasks:       {len(tasks)} selected")
    for task in tasks:
        print(f"                 {task}", flush=True)
    info(f"units:       {len(models) * len(tasks)} (models x tasks)")
    info(f"sandboxes:   {settings.flavor}, {settings.concurrency} in flight, "
         f"{settings.wall_time}s per task")
    info(f"boot budget: {settings.start_timeout}s per sandbox (cold image pull runs ~30 min)")
    info(f"task data:   {settings.data_bucket or 'baked_in_sandbox (demo/ tasks only)'}")
    info(f"configs:     {run_dir.relative_to(REPO_ROOT)}/")
    info(f"results:     .logs/ale/{run_name}/")

    if settings.submit:
        return submit_orchestrator(settings, tasks)

    ale = ["uv", "run", "python", "-m", "ale_run"] if shutil.which("uv") \
        else [sys.executable, "-m", "ale_run"]
    if settings.dry_run:
        return subprocess.run(
            [*ale, "run", str(experiment_config), "--dry-run"], check=False
        ).returncode

    status = subprocess.run([*ale, "run", str(experiment_config), "-v"], check=False).returncode
    if settings.results_bucket:
        info(f"syncing .logs/ale -> {settings.results_bucket}")
        subprocess.run(
            [*hf_cli(), "buckets", "sync", ".logs/ale", settings.results_bucket], check=False
        )
    return status


def main() -> int:
    try:
        return quickrun(parse_args())
    except ScriptError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
