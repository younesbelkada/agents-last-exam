#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface-hub>=1.0", "pyyaml>=6", "requests>=2.28"]
# ///
"""Serve a model with vLLM on a Hugging Face Job and point an ALE harness at it.

Use this for models the Inference Providers router does not serve. The job
exposes vLLM's port through the HF Jobs proxy, so the OpenAI-compatible API is
reachable at `https://<job_id>--<port>.hf.jobs/v1` with an
`Authorization: Bearer <HF token>` header. That is the same bearer the agent
presets already send as `api_key: ${env:HF_TOKEN}`, so a generated preset works
with no other change, from your machine or from inside a task sandbox.

    scripts/serve_model_job.py --model Qwen/Qwen3.5-4B

That blocks until Ctrl-C and cancels the job on the way out. For a sweep, start
it detached, run the sweep against the preset it wrote, then stop it:

    scripts/serve_model_job.py --model Qwen/Qwen3.5-4B --detach
    scripts/model_size_sweep.py --harness qwen_code_served --model Qwen/Qwen3.5-4B ...
    scripts/serve_model_job.py --stop <job-id>

One server can back several harnesses: repeat --from-preset and each gets its
own preset, same base_url, distinct agent id.

    scripts/serve_model_job.py --model Qwen/Qwen3.5-4B --detach \\
      --from-preset qwen_code_hf --from-preset pi_cli_hf

The GPU flavor is chosen from the model's parameter count unless --flavor says
otherwise. A served model is billed for the whole time the job runs, including
the image pull and the weight download, so --detach without a later --stop
keeps charging until the job timeout expires.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests
import yaml
from ale_hf_common import REPO_ROOT, ScriptError, ensure_hf_token, info, job_label, warn
from hf_quickrun import model_slug
from huggingface_hub import HfApi

# Nightly, not :latest. The models worth serving here are the ones no Inference
# Provider offers yet, which is usually because their architecture landed after
# the last stable vLLM. Qwen3.5 (`Qwen3_5ForConditionalGeneration`) is one such:
# its model card requires vLLM from main. Pin :latest with --image when a stable
# release is known to support the architecture.
VLLM_IMAGE = "vllm/vllm-openai:nightly"

# (flavor, VRAM GB, $/hour), cheapest first. The T4 flavors are deliberately
# absent: compute capability 7.5 has no bf16, and vLLM refuses a bf16
# checkpoint below 8.0 rather than silently downcasting.
GPU_FLAVORS = (
    ("l4x1", 24, 0.80),
    ("a10g-small", 24, 1.00),
    ("a10g-large", 24, 1.50),
    ("l40sx1", 48, 1.80),
    ("a100-large", 80, 2.50),
    ("rtx-pro-6000", 96, 2.75),
    ("a10g-largex2", 48, 3.00),
    ("l4x4", 96, 3.80),
    ("h200", 141, 5.00),
    ("l40sx4", 192, 8.30),
    ("h200x2", 282, 10.00),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", metavar="ID", help="HF model repo to serve")
    parser.add_argument("--stop", metavar="JOB_ID", default="",
                        help="cancel a job started earlier and exit")
    parser.add_argument("--flavor", default="",
                        help="GPU flavor; default is picked from the parameter count")
    parser.add_argument("--port", type=int, default=8000, help="vLLM port (default: %(default)s)")
    parser.add_argument("--max-model-len", type=int, default=0, metavar="N",
                        help="vLLM context window. Default 0 leaves it at the model's "
                             "own maximum: capping it below what the harness sends "
                             "turns every request into a 400 at run time, whereas a KV "
                             "cache that will not fit fails loudly at startup")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, metavar="F",
                        help="vLLM GPU memory fraction (default: %(default)s)")
    parser.add_argument("--tool-call-parser", default="qwen3_coder",
                        help="vLLM tool-call parser; agentic harnesses need one, and the "
                             "right value is model-specific (Qwen3.5 asks for "
                             "qwen3_coder). Pass '' to disable (default: %(default)s)")
    parser.add_argument("--image", default=VLLM_IMAGE,
                        help=f"serving image (default: {VLLM_IMAGE})")
    parser.add_argument("--vllm-arg", action="append", default=[], metavar="ARG",
                        help="extra argument passed through to vLLM; repeat as needed")
    parser.add_argument("--timeout", default="6h",
                        help="hard cap on the job (default: %(default)s)")
    parser.add_argument("--ready-timeout", type=int, default=1800, metavar="S",
                        help="seconds to wait for the image pull, weight download and "
                             "model load (default: %(default)s)")
    parser.add_argument("--namespace", default="", metavar="NS",
                        help="bill the job to an org instead of the token owner")
    parser.add_argument("--from-preset", action="append", default=[], metavar="NAME",
                        help="preset under configs/agents/ to copy harness and config from; "
                             "repeat to point several harnesses at the one server "
                             "(default: qwen_code_hf)")
    parser.add_argument("--preset", default="", metavar="PATH",
                        help="agent preset to write; needs a single --from-preset "
                             "(default: configs/agents/<from-preset>_served_<model-slug>.yaml)")
    parser.add_argument("--no-preset", dest="write_preset", action="store_false",
                        help="do not write an agent preset")
    parser.add_argument("--detach", action="store_true",
                        help="leave the job running and exit instead of blocking")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the job spec and stop")
    args = parser.parse_args(argv)
    args.from_preset = args.from_preset or ["qwen_code_hf"]
    if args.preset and len(args.from_preset) > 1:
        parser.error("--preset names one file; pass a single --from-preset or drop it")
    return args


def pick_flavor(model: str, api: HfApi) -> tuple[str, float]:
    """Cheapest GPU flavor whose VRAM fits the bf16 weights plus working memory."""
    safetensors = api.model_info(model).safetensors
    if safetensors is None or not safetensors.total:
        raise ScriptError(
            f"{model} publishes no safetensors parameter count, so the flavor cannot "
            f"be sized automatically. Pass --flavor explicitly."
        )
    # bf16 weights, plus ~30% for the KV cache and activations, plus a 4 GB floor
    # for the CUDA context and vLLM itself.
    needed = safetensors.total * 2 / 1e9 * 1.3 + 4
    for flavor, vram, price in GPU_FLAVORS:
        if vram >= needed:
            info(f"model:       {model} ({safetensors.total / 1e9:.1f}B params, "
                 f"~{needed:.0f} GB needed)")
            info(f"flavor:      {flavor} ({vram} GB, ${price:.2f}/hour)")
            return flavor, price
    raise ScriptError(
        f"{model} needs ~{needed:.0f} GB, more than any single listed flavor. "
        f"Pass --flavor with a multi-GPU option and --vllm-arg --tensor-parallel-size=N."
    )


def vllm_command(args: argparse.Namespace) -> list[str]:
    """Full container command. HF Jobs sends `command` as the entrypoint override,
    so the image's own ENTRYPOINT does not apply and vLLM is invoked explicitly.
    """
    command = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--host", "0.0.0.0",
        "--port", str(args.port),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    ]
    if args.max_model_len:
        command += ["--max-model-len", str(args.max_model_len)]
    if args.tool_call_parser:
        command += ["--enable-auto-tool-choice", "--tool-call-parser", args.tool_call_parser]
    return command + list(args.vllm_arg)


def wait_until_serving(api: HfApi, job_id: str, owner: str, args: argparse.Namespace) -> str:
    """Block until vLLM answers /health, returning the proxy base URL."""
    token = os.environ["HF_TOKEN"]
    headers = {"Authorization": f"Bearer {token}"}
    deadline = time.monotonic() + args.ready_timeout
    url = ""
    while time.monotonic() < deadline:
        status = api.inspect_job(job_id=job_id, namespace=owner).status
        if status.stage in ("ERROR", "COMPLETED", "CANCELED", "DELETED"):
            raise ScriptError(f"job {job_id} ended during startup: {status}")
        if status.stage == "RUNNING" and status.expose_urls:
            url = status.expose_urls[0].rstrip("/")
            break
        time.sleep(5)
    if not url:
        raise ScriptError(f"job {job_id} did not reach RUNNING in {args.ready_timeout}s")

    info(f"job running, waiting for vLLM to load weights at {url}")
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{url}/health", headers=headers, timeout=15).status_code == 200:
                return url
        except requests.RequestException:
            pass
        status = api.inspect_job(job_id=job_id, namespace=owner).status
        if status.stage in ("ERROR", "COMPLETED", "CANCELED", "DELETED"):
            raise ScriptError(
                f"job {job_id} died while loading the model: {status}\n"
                f"       logs: hf jobs logs {job_id}"
            )
        time.sleep(10)
    raise ScriptError(
        f"vLLM in {job_id} did not answer /health within {args.ready_timeout}s.\n"
        f"       Check `hf jobs logs {job_id}`, then cancel it with "
        f"`scripts/serve_model_job.py --stop {job_id}`"
    )


def preset_path(args: argparse.Namespace, from_preset: str) -> Path:
    # The harness is part of the name, and so of the agent id: two harnesses
    # serving the same model must not share an output branch.
    name = args.preset or f"configs/agents/{from_preset}_served_{model_slug(args.model)}.yaml"
    target = Path(name)
    return target if target.is_absolute() else REPO_ROOT / target


def source_preset(name: str) -> Path:
    """The configs/agents/ preset a served preset is copied from."""
    source = REPO_ROOT / "configs" / "agents" / f"{name}.yaml"
    if not source.is_file():
        raise ScriptError(f"no preset at {source.relative_to(REPO_ROOT)}")
    return source


def write_preset(args: argparse.Namespace, from_preset: str, base_url: str) -> Path:
    """Copy a router preset, repointing it at the served model."""
    source = source_preset(from_preset)
    target = preset_path(args, from_preset)

    preset = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    preset["model"] = args.model
    # A distinct agent id keeps a self-hosted run in its own branch of the output
    # tree, so it is never confused with the same model served by the router.
    preset["id"] = target.stem
    config = dict(preset.get("config") or {})
    config["base_url"] = f"{base_url}/v1"
    # The jobs proxy wants `Authorization: Bearer <HF token>`, which is exactly what
    # the harness sends as its API key, and vLLM started without --api-key ignores it.
    config["api_key"] = "${env:HF_TOKEN}"
    preset["config"] = config

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(preset, sort_keys=False), encoding="utf-8")
    return target


def serve(args: argparse.Namespace) -> int:
    api = HfApi()
    flavor, price = (args.flavor, 0.0) if args.flavor else pick_flavor(args.model, api)
    command = vllm_command(args)
    # A misspelled --from-preset must not surface after the GPU job is already billing.
    if args.write_preset:
        for from_preset in args.from_preset:
            source_preset(from_preset)

    if args.dry_run:
        info(f"would run {args.image} on {flavor}, exposing :{args.port}")
        print(f"     {' '.join(command)}")
        return 0

    job = api.run_job(
        image=args.image,
        command=command,
        secrets={"HF_TOKEN": os.environ["HF_TOKEN"]},
        flavor=flavor,
        timeout=args.timeout,
        expose=[args.port],
        labels={"ale": "model-server", "ale_model": job_label(args.model)},
        namespace=args.namespace or None,
    )
    job_id, owner = job.id, job.owner.name
    info(f"job:         {job_id} (https://huggingface.co/jobs/{owner}/{job_id})")
    if price:
        info(f"cost:        ~${price:.2f}/hour until it is stopped")

    try:
        base_url = wait_until_serving(api, job_id, owner, args)
    except Exception:
        api.cancel_job(job_id=job_id, namespace=owner)
        info(f"cancelled {job_id}")
        raise

    info(f"serving:     {base_url}/v1")
    if args.write_preset:
        for from_preset in args.from_preset:
            preset = write_preset(args, from_preset, base_url)
            info(f"preset:      {preset.relative_to(REPO_ROOT)}")
            info(f"run with:    --harness {preset.stem} --model {args.model}")

    if args.detach:
        info(f"detached. Stop it with: scripts/serve_model_job.py --stop {job_id}")
        return 0

    info("Ctrl-C to stop the job.")
    try:
        while True:
            time.sleep(30)
            stage = api.inspect_job(job_id=job_id, namespace=owner).status.stage
            if stage != "RUNNING":
                warn(f"job {job_id} is no longer running ({stage})")
                return 1
    except KeyboardInterrupt:
        print()
    finally:
        api.cancel_job(job_id=job_id, namespace=owner)
        info(f"cancelled {job_id}")
    return 0


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    try:
        ensure_hf_token()
        if args.stop:
            HfApi().cancel_job(job_id=args.stop, namespace=args.namespace or None)
            info(f"cancelled {args.stop}")
            return 0
        if not args.model:
            raise ScriptError("--model is required (or --stop <job-id>)")
        return serve(args)
    except ScriptError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
