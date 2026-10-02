"""Shared helpers for the HF Jobs scripts.

`hf_quickrun.py`, `fetch_task_subset.py` and `model_size_sweep.py` all resolve
the `hf` CLI, parse task lists, validate task cards and guard bucket
visibility, so those live here instead of in three copies.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# hfsandbox boots ale-ubuntu22-docker, so only this snapshot tag can run there.
LINUX_SNAPSHOT = "cpu-free-ubuntu"

# 99 Linux tasks: the right pool for the HF Jobs provider.
DEFAULT_POOL = "selected_tasks/docker_support.txt"


class ScriptError(RuntimeError):
    """Fatal, user-facing error. `main()` prints it and exits 1."""


def info(message: str) -> None:
    # Flushed, so progress stays interleaved with the subprocesses we hand stdout to.
    print(f">> {message}", flush=True)


def warn(message: str) -> None:
    print(f">> WARNING: {message}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- subprocess

def hf_cli() -> list[str]:
    """The `hf` command, falling back to the uv-managed one."""
    return ["hf"] if shutil.which("hf") else ["uv", "run", "hf"]


def run(argv: list[str], *, dry_run: bool = False) -> None:
    """Run argv, raising ScriptError on a non-zero exit."""
    if dry_run:
        print(f"   {shlex.join(argv)}")
        return
    try:
        subprocess.run(argv, check=True)
    except subprocess.CalledProcessError as e:
        raise ScriptError(f"command failed ({e.returncode}): {shlex.join(argv)}") from e


def capture(argv: list[str]) -> str:
    """stdout of argv, or "" if it fails. For probing the hf CLI."""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout if proc.returncode == 0 else ""


# --------------------------------------------------------------------- tasks

def parse_task_list(spec: str) -> list[str]:
    """Task paths from a list file (`#` comments allowed) or a comma-separated string."""
    source = Path(spec)
    lines = (
        source.read_text(encoding="utf-8").splitlines()
        if source.is_file()
        else spec.split(",")
    )
    return [entry for line in lines if (entry := line.split("#", 1)[0].strip())]


def interleave_by_domain(tasks: list[str]) -> list[str]:
    """Round-robin over the first path segment, so a head-N slice spans N domains."""
    by_domain: dict[str, list[str]] = {}
    for task in tasks:
        by_domain.setdefault(task.split("/", 1)[0], []).append(task)
    queues = list(by_domain.values())
    deepest = max((len(q) for q in queues), default=0)
    return [q[i] for i in range(deepest) for q in queues if i < len(q)]


def select_tasks(spec: str, *, pool: str = DEFAULT_POOL) -> list[str]:
    """Resolve a `--tasks` value: a count, a percentage of the pool, a list file, or a CSV list.

    A count or a percentage takes the head of the domain-interleaved pool, so
    the subset spans as many domains as it has tasks and the same spec always
    picks the same tasks. A percentage rounds up and never yields fewer than 1.
    """
    size = re.fullmatch(r"(\d+)|(\d+(?:\.\d+)?)%", spec.strip())
    if size is None:
        selected = parse_task_list(spec)
        if not selected:
            raise ScriptError(f"empty task selection from --tasks {spec}")
        return selected

    if not Path(pool).is_file():
        raise ScriptError(f"task pool not found: {pool}")
    candidates = interleave_by_domain(parse_task_list(pool))
    if not candidates:
        raise ScriptError(f"task pool is empty: {pool}")
    count, percent = size.groups()
    wanted = int(count) if count else max(1, math.ceil(len(candidates) * float(percent) / 100))
    if wanted > len(candidates):
        raise ScriptError(f"asked for {wanted} tasks but {pool} holds only {len(candidates)}")
    return candidates[:wanted]


def task_card(task_path: str) -> dict[str, Any]:
    card = REPO_ROOT / "tasks" / task_path / "task_card.json"
    if not card.is_file():
        raise ScriptError(f"no such task: {task_path}")
    return json.loads(card.read_text(encoding="utf-8"))


def require_linux_subset(tasks: list[str]) -> None:
    """Reject anything hfsandbox cannot boot, before it costs a 40 GB image pull."""
    for task in tasks:
        card = task_card(task)
        snapshot = card.get("vm", {}).get("snapshot") or card.get("snapshot")
        if snapshot != LINUX_SNAPSHOT:
            raise ScriptError(
                f"{task} has snapshot {snapshot!r} - hfsandbox runs the "
                f"{LINUX_SNAPSHOT} subset only"
            )


# ------------------------------------------------------------------- buckets

_PRIVATE_FIELD = re.compile(r'"private"\s*:\s*(true|false)')


def bucket_id(uri: str) -> str:
    """`<ns>/<bucket>` from an `hf://buckets/<ns>/<bucket>[/prefix]` URI."""
    return "/".join(uri.removeprefix("hf://buckets/").split("/")[:2])


def require_bucket_uri(uri: str, flag: str) -> None:
    if uri and not uri.startswith("hf://"):
        raise ScriptError(f"{flag} must be an hf:// URI, got: {uri}")


def require_private_bucket(uri: str, *, holds: str) -> None:
    """Refuse a world-readable destination; warn when visibility cannot be read.

    `hf buckets create` defaults to PUBLIC, and both run trajectories and gated
    reference data are things you only discover you leaked after the sync.
    """
    name = bucket_id(uri)
    field = _PRIVATE_FIELD.search(capture([*hf_cli(), "buckets", "info", name]))
    if field is None:
        warn(f"could not read {name} visibility. Confirm it is private before syncing {holds}:")
        warn(f"         hf buckets info {name}")
    elif field.group(1) == "false":
        raise ScriptError(
            f"{name} is PUBLIC and this sync carries {holds}. Fix it with:\n"
            f"       hf buckets settings {name} --private"
        )


# --------------------------------------------------------------- credentials

def ensure_hf_token() -> str:
    """Return HF_TOKEN, seeding it from the stored login, and export it.

    The provider reads the token through huggingface_hub's `get_token()`, which
    honors `hf auth login`, but the agent presets reference `${env:HF_TOKEN}`,
    which resolves only from the process environment. Bridge the two so a plain
    `hf auth login` is enough.
    """
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        lines = capture([*hf_cli(), "auth", "token"]).strip().splitlines()
        token = lines[0].strip() if lines else ""
    if not token:
        raise ScriptError("no HF token. Run `hf auth login`, or export HF_TOKEN=hf_...")
    os.environ["HF_TOKEN"] = token
    return token


def reject_blank_token_in_secret_env() -> None:
    """`secret_file:` is loaded with override=True, so a blank HF_TOKEN there wins."""
    env_file = REPO_ROOT / "secret" / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if re.fullmatch(r"\s*(?:export\s+)?HF_TOKEN=\s*", line):
            raise ScriptError(
                "secret/.env sets an empty HF_TOKEN, which overrides your shell. "
                "Fill it in or drop the line."
            )
