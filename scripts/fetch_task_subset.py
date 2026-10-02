#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Fetch input + reference data for a handful of tasks and push it to your
task-data bucket, instead of moving the 48 GB archive around.

Pulls from the two browsable datasets, merging them into one tree:
    agents-last-exam-data        input/ + software/   (open)
    agents-last-exam-reference   reference/           (gated, needs approval)
The result is <domain>/<task>/<variant>/{input,software,reference}, which is
the layout `task_data_source: mounted:` expects.

Then point a run at it:
    scripts/hf_quickrun.py --data-bucket <URI> --tasks <same list> ...

This is the Python port of `fetch_task_subset.sh`, which stays in place.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ale_hf_common import (
    REPO_ROOT,
    ScriptError,
    hf_cli,
    info,
    parse_task_list,
    require_private_bucket,
    run,
    task_card,
    warn,
)

INPUT_REPO = "agents-last-exam/agents-last-exam-data"
REFERENCE_REPO = "agents-last-exam/agents-last-exam-reference"

# Some native filenames cannot be stored on the Hub and ship in a side archive.
COMPAT_MANIFEST = "_transport/v1.1/path-compatibility.json"
COMPAT_ARCHIVE = "_transport/v1.1/path-compatibility.tar"
SYMLINK_ARCHIVE = "_transport/candidate-postvalidation-20260918-r06/tcga-public-symlinks.tar"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tasks", required=True, metavar="FILE|CSV",
                        help="task list file (e.g. a run's tasks.txt) or comma-separated paths")
    parser.add_argument("--dest", default="task-data-subset", metavar="DIR",
                        help="download dir (default: %(default)s)")
    parser.add_argument("--bucket", default="", metavar="URI",
                        help="hf://buckets/<ns>/<bucket> to sync into when done")
    parser.add_argument("--no-reference", dest="with_reference", action="store_false",
                        help="skip the gated repo (runs will not be scoreable)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the commands without running them")
    return parser.parse_args(argv)


def fetch_subset(
    tasks: list[str],
    *,
    dest: str = "task-data-subset",
    bucket: str = "",
    with_reference: bool = True,
    dry_run: bool = False,
) -> Path:
    """Download the per-task data for `tasks` into `dest`, optionally syncing to `bucket`.

    Returns the directory that holds the domain tree (`<dest>/tasks`), which is
    what a `--data-bucket` must be pointed at.
    """
    hf = hf_cli()
    dest_dir = Path(dest)
    for task in tasks:
        task_card(task)  # raises ScriptError on an unknown task path
    includes = [arg for task in tasks for arg in ("--include", f"tasks/{task}/*")]
    download = [*hf, "download", INPUT_REPO, "--repo-type", "dataset"]

    info(f"{len(tasks)} task(s) -> {dest_dir}/")
    for task in tasks:
        print(f"     {task}")

    info(f"input + software from {INPUT_REPO}")
    run([*download, *includes, "--local-dir", dest], dry_run=dry_run)

    if with_reference:
        info(f"reference from {REFERENCE_REPO} (gated)")
        run([*hf, "download", REFERENCE_REPO, "--repo-type", "dataset",
             *includes, "--local-dir", dest], dry_run=dry_run)
    else:
        info("skipping reference: runs will stage no ground truth and score 0")

    fetch_path_compatibility(tasks, dest_dir, download, dry_run=dry_run)

    info("restoring native symlinks (10 KB)")
    run([*download, "--include", SYMLINK_ARCHIVE, "--local-dir", dest], dry_run=dry_run)
    symlinks = dest_dir / SYMLINK_ARCHIVE
    if symlinks.is_file():
        run(["tar", "-xf", str(symlinks), "-C", str(dest_dir / "tasks")], dry_run=dry_run)

    if not dry_run:
        warn_on_missing(tasks, dest_dir, with_reference=with_reference)

    if bucket:
        # reference/ is gated ground truth. Never push it to a public bucket.
        require_private_bucket(bucket, holds="gated reference data")
        info(f"syncing {dest_dir}/tasks -> {bucket}")
        run([*hf, "buckets", "sync", str(dest_dir / "tasks"), bucket], dry_run=dry_run)
        info(f"run with:  --data-bucket {bucket}")
    else:
        info("no --bucket given; sync it yourself with:")
        print(f"     hf buckets sync {dest_dir}/tasks hf://buckets/<ns>/<bucket>")
    return dest_dir / "tasks"


def fetch_path_compatibility(
    tasks: list[str], dest_dir: Path, download: list[str], *, dry_run: bool
) -> None:
    """Pull the 328 MB native-filename archive, but only when a selected task needs it."""
    info("checking path-compatibility manifest")
    run([*download, "--include", COMPAT_MANIFEST, "--local-dir", str(dest_dir)], dry_run=dry_run)
    manifest = dest_dir / COMPAT_MANIFEST
    if not manifest.is_file():
        if dry_run:
            info("(dry-run: manifest not fetched, so the affected-task check is skipped)")
        return

    blob = manifest.read_text(encoding="utf-8")
    affected = [task for task in tasks if f'"tasks/{task}/' in blob]
    if not affected:
        info("none of the selected tasks need it")
        return
    info(f"restoring native filenames for: {' '.join(affected)}")
    run([*download, "--include", COMPAT_ARCHIVE, "--local-dir", str(dest_dir)], dry_run=dry_run)
    run(["tar", "-xf", str(dest_dir / COMPAT_ARCHIVE), "-C", str(dest_dir)], dry_run=dry_run)


def warn_on_missing(tasks: list[str], dest_dir: Path, *, with_reference: bool) -> None:
    wanted = ["input", "reference"] if with_reference else ["input"]
    missing = [
        f"{task}:{kind}"
        for task in tasks
        for kind in wanted
        if not any((dest_dir / "tasks" / task).glob(f"*/{kind}"))
    ]
    if missing:
        warn(f"not downloaded: {' '.join(missing)}")
        warn("         a missing reference/ means that task cannot be scored.")


def main() -> int:
    args = parse_args()
    # Relative paths are the caller's, but everything below runs from the repo root.
    if (given := Path(args.tasks)).is_file():
        args.tasks = str(given.resolve())
    os.chdir(REPO_ROOT)
    try:
        tasks = parse_task_list(args.tasks)
        if not tasks:
            raise ScriptError(f"no tasks parsed from --tasks {args.tasks}")
        fetch_subset(
            tasks,
            dest=args.dest,
            bucket=args.bucket,
            with_reference=args.with_reference,
            dry_run=args.dry_run,
        )
    except ScriptError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
