"""Parse OpenClaw transcripts into Trajectory steps, sub-agents included.

Fork of ``ale_run/agents/ale_claw/transcript_to_trajectory.py``. Self-contained
on purpose: importing the ale_claw module runs that package's ``__init__``,
which pulls in the whole vendored harness, and this parser is the regression
surface for the experiment's headline number.

Three differences from the original, each fixing something that would make the
budget-split comparison wrong:

* **Sub-agent transcripts are discovered.** They live at
  ``openclaw_sessions/<task>/subagents/<run_id>/transcript.jsonl``, one level
  deeper than the original's globs reach, so sub-agent turns were missing from
  the trajectory entirely.
* **Sub-agent usage is de-cumulated.** ``subagent_session.py`` writes a running
  total into every assistant message, so reusing the main-transcript path would
  grow per-step input tokens monotonically and sum them quadratically.
* **The call ledger wins.** ``budget_ledger.jsonl`` is written by the meter and
  is the only record that includes helper calls (compaction, memory flush,
  vision), which ``helper_runtime.py`` discards. When present it replaces the
  legacy aggregate; the two are never added together.

Every step carries ``extra["agent_id"]`` so per-agent behaviour can be read back
out of a trajectory.


OpenClaw (the harness in :mod:`ale.agents.ale_claw.harness`) writes per-run
artifacts under ``<work_dir>/openclaw_sessions/<task_id>/``:

  ├── transcript.jsonl      — append-only JSONL: session header, message entries,
  │                            compaction entries
  └── state.json            — running totals (token usage, step count,
                              compaction count, model)

And per-turn API payloads under
``<work_dir>/trajectories/<traj_id>/turn_NNN/<NNNN>_api_result.json``
(LiteLLM-shape OpenAI response — preserves cache token breakdown).

This module parses those into ALE Steps via :class:`TrajectoryBuilder`.
**No InteractionLog / InteractionStep intermediate** — we emit ATIF Steps
directly. Logic largely ported from agenthle's
``orchestration/agents/ale_claw/transcript_log.py``; key differences:

  - No agenthle deps — only stdlib + ``ale.agents.trajectory``.
  - Drop the ``InteractionLog.save(...)`` step. Caller's :meth:`collect`
    drives the trajectory builder directly.
  - Per-turn assistant messages collapse into ONE :class:`Step`
    (text → ``message``, thinking → ``reasoning``, function_calls →
    ``tool_calls``).
  - Per-message usage → :class:`StepMetrics` on that step.
  - Aggregated totals (state.json + per-turn cache) land in
    ``builder.trajectory.extra["ale_claw"]["usage"]`` for downstream
    consumers that want exact accounting (the default
    ``builder.finalize`` sum lacks cache_read/cache_write since the
    transcript itself doesn't carry them).
"""
from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Any

from ale_run.base_interface import (
    ContentPart,
    Observation,
    StepMetrics,
    ToolCall,
    ToolResult,
    TrajectoryBuilder,
)

logger = logging.getLogger(__name__)

# Subdirs the harness writes under our work_dir. Mirror the agenthle layout
# so transcript shape is unchanged.
_SESSIONS_SUBDIR = "openclaw_sessions"
_LEDGER_FILE = "budget_ledger.jsonl"
_EXTRA_KEY = "claw_budget"
ORCHESTRATOR = "orchestrator"
_TRAJECTORIES_SUBDIR = "trajectories"


# =============================================================================
# Public entry point
# =============================================================================

def parse_transcripts_into(work_dir: Path, builder: TrajectoryBuilder) -> None:
    """Walk OpenClaw artifacts under ``work_dir`` → emit Steps into ``builder``.

    Idempotent on partial / missing data: emits a single ``system`` step
    when no transcript is found, otherwise appends one Step per assistant
    turn + one per tool reply, then writes aggregate usage to
    ``builder.trajectory.extra["ale_claw"]["usage"]``.
    """
    sessions_root = work_dir / _SESSIONS_SUBDIR
    main = sorted(sessions_root.glob("*/transcript.jsonl")) if sessions_root.is_dir() else []
    subs = sorted(sessions_root.glob("*/subagents/*/transcript.jsonl")) if sessions_root.is_dir() else []
    if not main and not subs:
        builder.add_step(
            source="system",
            message="claw-budget: no transcript at "
                    f"{sessions_root}/*/transcript.jsonl",
            extra={"reason": "no_transcript", "expected_root": str(sessions_root)},
        )
        return

    for path in main:
        _parse_one_transcript(path, builder, agent_id=ORCHESTRATOR, decumulate=False)
    # Sub-agent turns land after the orchestrator's rather than interleaved by
    # timestamp: the transcripts carry no common clock we can trust across
    # concurrently running sessions, and the ledger is what carries ordering
    # for analysis anyway.
    for path in subs:
        _parse_one_transcript(
            path, builder, agent_id=path.parent.name, decumulate=True,
        )

    aggregated = _aggregate_usage(work_dir)
    ledger = _ledger_totals(work_dir)
    extra = builder.trajectory.extra.setdefault(_EXTRA_KEY, {})
    extra["usage"] = aggregated
    if main:
        extra["raw_transcript"] = str(main[0])
    if ledger:
        extra["ledger"] = ledger
    _reconcile_final_metrics(builder, aggregated, ledger)


def _reconcile_final_metrics(
    builder: TrajectoryBuilder,
    aggregated: dict[str, Any],
    ledger: dict[str, Any] | None = None,
) -> None:
    """Feed the authoritative aggregate into ``final_metrics`` via the builder.

    The default per-step ``StepMetrics`` sum drops the prompt-cache split (the
    transcript never carries it) and the final/helper turns (they bypass the
    transcript writer), so ``final_metrics`` under-reports tokens, cache, and
    cost. ``aggregated`` (from :func:`_aggregate_usage`: state.json tokens +
    per-turn api_result cache/cost) is complete, so prefer it. No-op when there
    is no authoritative input-token total (degraded run) — keeps the per-step
    sum rather than zeroing it out.
    """
    # The ledger is a superset of the legacy aggregate (it alone sees helper
    # and sub-agent calls), so it replaces it. Adding them would double count.
    if ledger and ledger.get("input_tokens", 0) > 0:
        builder.override_final_metrics(
            total_input_tokens=ledger["input_tokens"],
            total_output_tokens=ledger["output_tokens"],
            total_cache_read_tokens=ledger.get("cache_read_tokens", 0),
            total_cache_creation_tokens=ledger.get("cache_write_tokens", 0),
            total_cost_usd=ledger.get("cost_usd") or None,
        )
        return
    if aggregated.get("overall_input_tokens", 0) <= 0:
        return
    builder.override_final_metrics(
        total_input_tokens=aggregated.get("overall_input_tokens"),
        total_output_tokens=aggregated.get("output_tokens"),
        total_cache_read_tokens=aggregated.get("cache_read_input_tokens", 0),
        total_cache_creation_tokens=aggregated.get("cache_write_input_tokens", 0),
        # Only override cost when the aggregate has it; else keep per-step sum.
        total_cost_usd=aggregated.get("total_cost_usd"),
    )


# =============================================================================
# Transcript walker
# =============================================================================

def _parse_one_transcript(
    path: Path,
    builder: TrajectoryBuilder,
    *,
    agent_id: str = ORCHESTRATOR,
    decumulate: bool = False,
) -> None:
    """Walk one ``transcript.jsonl`` file → append Steps to builder.

    ``decumulate`` turns running totals into per-step deltas, which sub-agent
    transcripts need and the orchestrator's must not have.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:                      # noqa: BLE001
        logger.warning("ale-claw: failed to read %s: %s", path, exc)
        return

    prev: dict[str, float] | None = {"input": 0, "output": 0, "cost": 0.0} if decumulate else None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("type") != "message":
            continue
        _consume_message(entry, builder, agent_id=agent_id, prev=prev)


def _consume_message(
    entry: dict[str, Any],
    builder: TrajectoryBuilder,
    *,
    agent_id: str = ORCHESTRATOR,
    prev: dict[str, float] | None = None,
) -> None:
    """One transcript ``{"type":"message", ...}`` entry → one Step.

    Assistant messages become one ``agent`` Step that may carry text
    (``message``), thinking (``reasoning``), and tool_calls (joined
    function_call blocks). Tool messages become one ``environment`` Step
    with all tool_results under ``observation.results``.
    """
    msg = entry.get("message") or {}
    role = msg.get("role")
    content = msg.get("content")
    if not isinstance(content, list):
        return
    usage = msg.get("usage") or {}
    stop_reason = msg.get("stopReason")

    if role == "assistant":
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                t = block.get("text") or ""
                if t:
                    text_parts.append(t)
            elif btype == "thinking":
                # Upstream uses {"type":"thinking","thinking":"..."} — falls
                # back to "content" when emitted via newer SDK code paths.
                t = block.get("thinking") or block.get("content") or ""
                if t:
                    reasoning_parts.append(t)
            elif btype == "function_call":
                raw_args = block.get("arguments") or ""
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except (json.JSONDecodeError, TypeError):
                    args = {"_raw": raw_args}
                if not isinstance(args, dict):
                    args = {"value": args}
                tool_calls.append(ToolCall(
                    id=block.get("id") or "",
                    name=block.get("name") or "",
                    arguments=args,
                ))

        # Skip empty messages (no text / no thinking / no tool_call) to keep
        # the trajectory tight.
        if not (text_parts or reasoning_parts or tool_calls):
            return

        builder.add_step(
            source="agent",
            message="\n".join(text_parts) if text_parts else None,
            reasoning="\n".join(reasoning_parts) if reasoning_parts else None,
            tool_calls=tool_calls,
            metrics=_metrics_from_message_usage(usage, prev),
            extra=(
                {"agent_id": agent_id, "stop_reason": stop_reason}
                if stop_reason else {"agent_id": agent_id}
            ),
        )
        return

    if role == "tool":
        results: list[ToolResult] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_result":
                continue
            raw = block.get("content")
            text = _normalize_tool_result_content(raw)
            results.append(ToolResult(
                tool_call_id=block.get("tool_use_id") or "",
                content=[ContentPart(type="text", text=text)],
                is_error=bool(block.get("is_error", False)),
            ))
        if results:
            builder.add_step(
                source="environment",
                observation=Observation(results=results),
                extra={"agent_id": agent_id},
            )
        return

    # Other roles (user re-injects from compaction etc.) — log to extra and skip.
    builder.trajectory.extra.setdefault(_EXTRA_KEY, {}).setdefault(
        "skipped_messages", []
    ).append({"role": role, "content_blocks": len(content)})


def _metrics_from_message_usage(
    usage: dict[str, Any], prev: dict[str, float] | None = None
) -> StepMetrics | None:
    """OpenClaw transcript usage → :class:`StepMetrics`.

    OpenClaw's per-message usage is ``{"input": N, "output": N, "total": N,
    "cost": F}`` (note the unsuffixed key names). cache_read/cache_write are
    NOT in the transcript — they live in the per-turn ``api_result.json``
    and are surfaced via :func:`_aggregate_usage` into trajectory.extra.
    """
    if not usage:
        return None
    in_t = usage.get("input")
    out_t = usage.get("output")
    cost = usage.get("cost")
    if in_t is None and out_t is None and cost is None:
        return None
    if prev is not None:
        # Sub-agent transcripts carry running totals, so the delta is the step.
        cur_in, cur_out = int(in_t or 0), int(out_t or 0)
        cur_cost = float(cost or 0.0)
        in_t, out_t = cur_in - int(prev["input"]), cur_out - int(prev["output"])
        cost = cur_cost - prev["cost"] if cost is not None else None
        prev["input"], prev["output"], prev["cost"] = cur_in, cur_out, cur_cost
    return StepMetrics(
        input_tokens=int(in_t) if in_t is not None else None,
        output_tokens=int(out_t) if out_t is not None else None,
        cost_usd=float(cost) if cost is not None else None,
    )


# =============================================================================
# Tool-result content normalization
# =============================================================================

def _normalize_tool_result_content(raw: Any) -> str:
    """Best-effort convert OpenClaw's tool_result content to JSON text.

    OpenClaw stores tool_result content as ``str(dict)`` (Python repr) rather
    than ``json.dumps(dict)``, e.g. ``"{'success': True, ...}"`` instead of
    ``{"success": true, ...}``. Re-serialize as JSON for downstream
    consumers (matches the InteractionLog the agenthle wrapper used to
    write).

    Pass-through (no rewrite) when:
    - input is not a str (already structured — caller json.dumps it)
    - input doesn't look like a Python repr (treat as plain text)
    - literal_eval fails (treat as plain text — covers shell output,
      stack traces, etc.)
    """
    if raw is None:
        return ""
    if not isinstance(raw, str):
        return json.dumps(raw, ensure_ascii=False)
    s = raw.strip()
    if not (s.startswith("{") or s.startswith("[")):
        return raw
    try:
        parsed = ast.literal_eval(s)
    except (ValueError, SyntaxError, MemoryError):
        return raw
    if not isinstance(parsed, (dict, list)):
        return raw
    return json.dumps(parsed, ensure_ascii=False, indent=2)


# =============================================================================
# Aggregate usage (state.json + per-turn api_result.json)
# =============================================================================

def _aggregate_usage(work_dir: Path) -> dict[str, Any]:
    """Sum total tokens (state.json) + cache breakdown (per-turn api_result.json).

    state.json is the in-memory session_mgr accumulator and is incremented for
    EVERY yielded step out of ``agent.run()``, including helper / compaction /
    VLM calls that bypass the transcript message writer. Cache aggregation
    walks every per-turn ``api_result.json`` so it captures those same
    helper calls' cache breakdown — disjoint partition only balances when
    overall_input_tokens is sourced from state.json (not transcript).
    """
    state_in, state_out = _aggregate_state_json_tokens(work_dir)
    cache_read, cache_write, api_cost = _aggregate_api_result_usage(work_dir)
    msg_in, msg_out, msg_cost = _aggregate_message_usage(work_dir)

    in_t = state_in or msg_in
    out_t = state_out or msg_out
    uncached_in = max(in_t - cache_read - cache_write, 0)
    # Per-call api_result cost is authoritative — it includes the final/helper
    # turns the transcript-message cost (msg_cost) drops. Fall back to msg_cost
    # only when no api_result dumps are present.
    cost = api_cost or msg_cost

    out: dict[str, Any] = {
        "uncached_input_tokens": uncached_in,
        "output_tokens": out_t,
        "overall_input_tokens": in_t,
    }
    if cost > 0:
        out["total_cost_usd"] = round(cost, 6)
    if cache_read > 0:
        out["cache_read_input_tokens"] = cache_read
    if cache_write > 0:
        out["cache_write_input_tokens"] = cache_write
    return out


def _aggregate_state_json_tokens(work_dir: Path) -> tuple[int, int]:
    """Read main-agent input/output token totals from each session's ``state.json``."""
    sessions_root = work_dir / _SESSIONS_SUBDIR
    if not sessions_root.is_dir():
        return 0, 0
    in_t = out_t = 0
    for path in sorted(sessions_root.glob("*/state.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        tt = data.get("total_tokens") or {}
        in_t += int(tt.get("input_tokens", 0) or 0)
        out_t += int(tt.get("output_tokens", 0) or 0)
    return in_t, out_t


def _aggregate_api_result_usage(work_dir: Path) -> tuple[int, int, float]:
    """Sum cache split + cost from per-turn API result dumps.

    Returns ``(cache_read, cache_write, cost)``. Unlike the transcript-message
    usage, these dumps cover EVERY provider call — including the final "done"
    turn and helper/compaction/VLM calls that bypass the transcript writer — so
    the cost here is the authoritative total. Usage lives under ``result.usage``
    (NOT top-level ``usage``).
    """
    trajectories_root = work_dir / _TRAJECTORIES_SUBDIR
    if not trajectories_root.is_dir():
        return 0, 0, 0.0
    cache_read = cache_write = 0
    cost = 0.0
    for path in trajectories_root.glob("*/turn_*/[0-9]*_api_result.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        usage = (data.get("result") or {}).get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        cache_read += int(details.get("cached_tokens") or 0)
        cache_write += int(details.get("cache_write_tokens") or 0)
        cost += float(usage.get("cost") or 0.0)
    return cache_read, cache_write, cost


def _aggregate_message_usage(work_dir: Path) -> tuple[int, int, float]:
    """Fallback: sum (input, output, cost) once per assistant message."""
    sessions_root = work_dir / _SESSIONS_SUBDIR
    if not sessions_root.is_dir():
        return 0, 0, 0.0
    in_t = out_t = 0
    cost = 0.0
    for path in sorted(sessions_root.glob("*/transcript.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") != "message":
                continue
            msg = entry.get("message") or {}
            if msg.get("role") != "assistant":
                continue
            usage = msg.get("usage") or {}
            in_t += int(usage.get("input") or 0)
            out_t += int(usage.get("output") or 0)
            c = usage.get("cost")
            if c:
                cost += float(c)
    return in_t, out_t, cost


# =============================================================================
# Call ledger (written per call by the budget meter)
# =============================================================================

def _ledger_totals(work_dir: Path) -> dict[str, Any]:
    """Sum ``budget_ledger.jsonl`` into totals plus a per-agent breakdown.

    This is the only record that covers helper calls, so when it exists it is
    the authoritative account of what the run spent.
    """
    path = work_dir / _LEDGER_FILE
    if not path.is_file():
        return {}
    totals = {
        "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
        "cache_write_tokens": 0, "cost_usd": 0.0, "calls": 0,
    }
    per_agent: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("claw-budget: cannot read ledger %s: %s", path, exc)
        return {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        agent = str(row.get("agent_id") or "unattributed")
        bucket = per_agent.setdefault(
            agent,
            {"calls": 0, "input_tokens": 0, "output_tokens": 0,
             "cache_read_tokens": 0, "cache_write_tokens": 0, "cost_usd": 0.0},
        )
        bucket["calls"] += 1
        totals["calls"] += 1
        for key in ("input_tokens", "output_tokens", "cache_read_tokens",
                    "cache_write_tokens"):
            val = int(row.get(key) or 0)
            bucket[key] += val
            totals[key] += val
        cost = float(row.get("cost_usd") or 0.0)
        bucket["cost_usd"] += cost
        totals["cost_usd"] += cost
    totals["cost_usd"] = round(totals["cost_usd"], 6)
    totals["per_agent"] = per_agent
    totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
    return totals
