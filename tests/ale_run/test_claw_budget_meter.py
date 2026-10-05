"""Tests for the call-level token meter.

The meter is the only thing that charges the envelope, so these tests pin its
two jobs: attribute every call to the right agent, and never lose a call. The
helper-call case matters most - compaction and vision spend is invisible to the
harness's own counters, and it is the multi-agent arm that makes the most of it.

Exercises ``MeterCore`` directly, so no litellm install is required.
"""
from __future__ import annotations

import json

import pytest

from ale_run.agents.claw_budget.budget.envelope import ORCHESTRATOR, UNATTRIBUTED, TokenEnvelope
from ale_run.agents.claw_budget.budget.meter import (
    METER,
    MeterCore,
    call_metadata,
    clear_identity,
    set_identity,
)


@pytest.fixture(autouse=True)
def _isolate_identity():
    """pytest runs these in one context, so an identity set by one test would
    otherwise bleed into the next."""
    clear_identity()
    yield
    clear_identity()


class _Usage:
    def __init__(self, i, o, cached=0):
        self.prompt_tokens = i
        self.completion_tokens = o
        self.prompt_tokens_details = {"cached_tokens": cached}


class _Response:
    def __init__(self, i, o, cached=0, cost=0.0, model="test/model"):
        self.usage = _Usage(i, o, cached)
        self.model = model
        self._hidden_params = {"response_cost": cost}


def _setup(tmp_path, total=10_000, n=2):
    meter = MeterCore()
    env = TokenEnvelope(total, n_subagents=n)
    sink = meter.register("run-1", env, tmp_path / "budget_ledger.jsonl")
    return meter, env, sink


def _ledger(tmp_path):
    path = tmp_path / "budget_ledger.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_metadata_attribution(tmp_path):
    meter, env, _ = _setup(tmp_path)
    kwargs = {"litellm_params": {"metadata": call_metadata("run-1", "sub-7")}}
    meter.record(kwargs=kwargs, response=_Response(100, 25))
    assert env.spent == 125
    assert [a.agent_id for a in env.agents()] == ["sub-7"]


def test_contextvar_attribution_when_metadata_absent(tmp_path):
    """Helper calls build their own kwargs and carry no metadata, so the
    contextvar is what keeps compaction spend attributable."""
    meter, env, _ = _setup(tmp_path)
    set_identity("run-1", "sub-3")
    meter.record(kwargs={"model": "m"}, response=_Response(40, 10))
    assert env.spent == 50
    assert [a.agent_id for a in env.agents()] == ["sub-3"]


def test_unattributed_calls_are_still_charged(tmp_path):
    """A call from a worker thread has an empty context. It must not escape the
    envelope just because we cannot name its owner."""
    meter, env, _ = _setup(tmp_path)
    meter.record(kwargs={"litellm_params": {"metadata": {"ale_run_id": "run-1"}}},
                 response=_Response(30, 5))
    assert env.spent == 35
    assert [a.agent_id for a in env.agents()] == [UNATTRIBUTED]
    assert _ledger(tmp_path)[0]["attributed"] is False


def test_reservation_is_settled_by_the_meter(tmp_path):
    """Admission reserves an estimate; the meter reconciles to the truth, so
    slack returns to the pool rather than staying committed."""
    meter, env, sink = _setup(tmp_path)
    res = env.admit("sub-1", 500, cap=env.sub_cap())
    sink.park("sub-1", res)
    assert env.remaining == 9_500

    meter.record(kwargs={"litellm_params": {"metadata": call_metadata("run-1", "sub-1")}},
                 response=_Response(80, 20))
    assert env.spent == 100
    assert env.remaining == 9_900
    assert res.settled


def test_calls_from_other_runs_are_ignored(tmp_path):
    """Several episodes share this process, so the meter must not cross-charge."""
    meter, env, _ = _setup(tmp_path)
    out = meter.record(
        kwargs={"litellm_params": {"metadata": call_metadata("some-other-run", "x")}},
        response=_Response(999, 999),
    )
    assert out is None
    assert env.spent == 0


def test_ledger_is_written_per_call(tmp_path):
    meter, _env, _ = _setup(tmp_path)
    for i in range(3):
        meter.record(
            kwargs={"litellm_params": {"metadata": call_metadata("run-1", f"sub-{i}")}},
            response=_Response(10, 2, cached=4, cost=0.001),
        )
    rows = _ledger(tmp_path)
    assert len(rows) == 3
    assert rows[-1]["spent_total"] == 36
    assert rows[0]["cache_read_tokens"] == 4
    assert rows[0]["cost_usd"] == 0.001


def test_failed_calls_are_recorded(tmp_path):
    """litellm retries internally; a failed attempt can still have burned
    prompt tokens, and it must appear in the ledger either way."""
    meter, env, _ = _setup(tmp_path)
    meter.record(kwargs={"litellm_params": {"metadata": call_metadata("run-1", ORCHESTRATOR)}},
                 response=_Response(10, 0), error="failure")
    assert _ledger(tmp_path)[0]["error"] == "failure"
    assert env.spent == 10


def test_conservation_across_mixed_traffic(tmp_path):
    meter, env, _sink = _setup(tmp_path)
    md = lambda a: {"litellm_params": {"metadata": call_metadata("run-1", a)}}  # noqa: E731
    meter.record(kwargs=md(ORCHESTRATOR), response=_Response(200, 50))
    meter.record(kwargs=md("sub-1"), response=_Response(100, 30))
    set_identity("run-1", "sub-1")
    meter.record(kwargs={}, response=_Response(60, 10))        # sub-agent compaction
    meter.record(kwargs={"litellm_params": {"metadata": {"ale_run_id": "run-1"}}},
                 response=_Response(20, 5))                     # threaded vision call

    summary = env.summary()
    assert summary["spent_total"] == 475
    assert sum(a["total_tokens"] for a in summary["per_agent"]) == 475
    assert sum(r["input_tokens"] + r["output_tokens"] for r in _ledger(tmp_path)) == 475


def test_module_level_meter_is_a_meter_core():
    assert isinstance(METER, MeterCore)


def test_install_is_refcounted(monkeypatch):
    """Run units share one process and one litellm callback list. If the first
    unit to finish unhooked the meter, every other in-flight unit would stop
    being charged without anything looking wrong."""
    from ale_run.agents.claw_budget.budget import meter as m

    sentinel = object()
    monkeypatch.setattr(m, "_do_install", lambda: (m._installed.append(sentinel), sentinel)[1])
    monkeypatch.setattr(m, "_installed", [])
    monkeypatch.setattr(m, "_refcount", 0)

    m.install()
    m.install()                      # a second concurrent run unit
    assert m._installed == [sentinel]

    m.uninstall()                    # the first unit finishes
    assert m._installed == [sentinel], "meter must survive until the last unit"

    m.uninstall()                    # the last one
    assert m._installed == []
