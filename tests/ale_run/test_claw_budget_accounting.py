"""Accounting regressions for the budget-split harness.

The experiment compares one agent holding B against N sub-agents holding B/N,
so a token the run spends but does not report is not a cosmetic bug: it lands
disproportionately on the multi-agent arm, which runs N extra compaction
pipelines. These tests pin the three ways that could happen.

Pure stdlib + pydantic, like ``test_ale_claw_cost_accounting.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

from ale_run.agents.claw_budget.transcript_to_trajectory import parse_transcripts_into
from ale_run.base_interface import TrajectoryBuilder


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _assistant(text: str, usage: dict) -> dict:
    return {
        "type": "message",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "usage": usage,
        },
    }


def _ledger_row(agent: str, i: int, o: int, **kw) -> dict:
    return {"run_id": "run-1", "agent_id": agent, "input_tokens": i,
            "output_tokens": o, "cache_read_tokens": kw.get("cr", 0),
            "cache_write_tokens": kw.get("cw", 0), "cost_usd": kw.get("cost", 0.0)}


def _build(tmp_path: Path, *, with_ledger: bool = True) -> Path:
    """A run with one orchestrator session and one delegated sub-agent."""
    wd = tmp_path / "claw-budget"
    session = wd / "openclaw_sessions" / "sess0"

    # Orchestrator: per-message usage is per-turn, not cumulative.
    _jsonl(session / "transcript.jsonl", [
        _assistant("plan", {"input": 100, "output": 10, "total": 110, "cost": 0.001}),
        _assistant("merge", {"input": 150, "output": 20, "total": 170, "cost": 0.002}),
    ])
    (session / "state.json").write_text(
        json.dumps({"total_tokens": {"input_tokens": 250, "output_tokens": 30}}),
        encoding="utf-8",
    )

    # Sub-agent: running totals, written one level deeper than the old globs saw.
    _jsonl(session / "subagents" / "sub-abc" / "transcript.jsonl", [
        _assistant("sub step 1", {"input": 40, "output": 5, "total": 45, "cost": 0.0004}),
        _assistant("sub step 2", {"input": 90, "output": 12, "total": 102, "cost": 0.0009}),
    ])

    if with_ledger:
        # Includes a compaction call the harness itself records nowhere.
        _jsonl(wd / "budget_ledger.jsonl", [
            _ledger_row("orchestrator", 100, 10, cost=0.001),
            _ledger_row("orchestrator", 150, 20, cost=0.002),
            _ledger_row("sub-abc", 40, 5, cost=0.0004),
            _ledger_row("sub-abc", 50, 7, cost=0.0005),
            _ledger_row("unattributed", 70, 9, cr=30, cost=0.0007),
        ])
    return wd


def _parse(wd: Path):
    builder = TrajectoryBuilder(
        agent_name="claw-budget", model="test/model",
        task_path="demo/hello", variant_index=0, instruction="do the thing",
    )
    parse_transcripts_into(wd, builder)
    return builder.finalize(reward=1.0, status="completed")


def test_subagent_turns_reach_the_trajectory(tmp_path):
    """The regression: sub-agent transcripts sit a level below where the
    original parser looked, so their turns were absent entirely."""
    traj = _parse(_build(tmp_path))
    agents = {s.extra.get("agent_id") for s in traj.steps if s.extra}
    assert "sub-abc" in agents
    assert "orchestrator" in agents


def test_every_step_is_attributed(tmp_path):
    traj = _parse(_build(tmp_path))
    # Every step here comes from a transcript; the lifecycle seeds the
    # instruction step separately, outside this parser.
    assert all(s.extra.get("agent_id") for s in traj.steps)


def test_subagent_metrics_are_deltas_not_running_totals(tmp_path):
    """Sub-agent transcripts store cumulative usage. Summed naively, per-step
    input tokens would climb and the total would be quadratic."""
    traj = _parse(_build(tmp_path))
    sub = [s for s in traj.steps if (s.extra or {}).get("agent_id") == "sub-abc"
           and s.metrics and s.metrics.input_tokens is not None]
    assert [s.metrics.input_tokens for s in sub] == [40, 50]   # not [40, 90]
    assert [s.metrics.output_tokens for s in sub] == [5, 7]    # not [5, 12]


def test_ledger_is_authoritative_and_not_added_to_the_aggregate(tmp_path):
    """state.json says 250/30; the ledger says 410/51 because it also sees the
    sub-agent and the unattributed helper call. The ledger must win outright."""
    traj = _parse(_build(tmp_path))
    m = traj.final_metrics
    assert m.total_input_tokens == 410
    assert m.total_output_tokens == 51
    assert m.total_cache_read_tokens == 30


def test_falls_back_to_the_legacy_aggregate_without_a_ledger(tmp_path):
    """A run cancelled before the meter wrote anything still reports what the
    harness itself recorded, rather than zeroing out."""
    traj = _parse(_build(tmp_path, with_ledger=False))
    assert traj.final_metrics.total_input_tokens == 250
    assert traj.final_metrics.total_output_tokens == 30


def test_ledger_totals_are_conserved_per_agent(tmp_path):
    wd = _build(tmp_path)
    traj = _parse(wd)
    ledger = traj.extra["claw_budget"]["ledger"]
    per_agent = ledger["per_agent"]
    assert per_agent["sub-abc"]["input_tokens"] == 90
    assert per_agent["unattributed"]["calls"] == 1
    assert sum(a["input_tokens"] for a in per_agent.values()) == ledger["input_tokens"]
    assert ledger["total_tokens"] == traj.final_metrics.total_input_tokens + traj.final_metrics.total_output_tokens


def test_subagent_trajectories_stays_empty(tmp_path):
    """Sub-agent work is flat steps by design: finalize() sums only self.steps,
    so nesting would hide it from final_metrics and from run.json usage."""
    traj = _parse(_build(tmp_path))
    assert traj.subagent_trajectories == []


def test_missing_transcripts_degrade_to_one_system_step(tmp_path):
    traj = _parse(tmp_path / "empty")
    assert len(traj.steps) == 1
    assert traj.steps[-1].source == "system"
    assert traj.steps[-1].extra["reason"] == "no_transcript"


def test_subagent_only_run_is_still_parsed(tmp_path):
    """If the orchestrator transcript is missing but a sub-agent ran, its spend
    must not vanish."""
    wd = tmp_path / "wd"
    _jsonl(wd / "openclaw_sessions" / "s0" / "subagents" / "sub-x" / "transcript.jsonl",
           [_assistant("only sub", {"input": 10, "output": 2, "total": 12})])
    traj = _parse(wd)
    assert any((s.extra or {}).get("agent_id") == "sub-x" for s in traj.steps)
