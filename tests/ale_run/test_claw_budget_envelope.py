"""Unit tests for the shared token envelope behind the budget-split experiment.

The experiment compares one agent holding ``B`` against ``N`` sub-agents holding
``B/N`` each, so these tests pin the two properties that make that comparison
honest: the envelope is never promised more than it holds, and a sub-agent's cap
is a view on the remaining pool rather than a standing allocation.

Pure stdlib: no harness, no cua-agent, no network.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from ale_run.agents.claw_budget.budget.envelope import ORCHESTRATOR, TokenEnvelope


def test_reserve_settle_round_trip_returns_slack():
    env = TokenEnvelope(1000)
    res = env.admit(ORCHESTRATOR, 300)
    assert res is not None
    # Reservation is committed immediately, so it is not re-offerable...
    assert env.remaining == 700
    # ...but spend only counts what actually happened.
    env.settle(res, input_tokens=40, output_tokens=10)
    assert env.spent == 50
    assert env.remaining == 950          # the unused 250 returned to the pool


def test_release_returns_an_unused_reservation():
    env = TokenEnvelope(100)
    res = env.admit(ORCHESTRATOR, 80)
    assert env.remaining == 20
    env.release(res)
    assert env.remaining == 100
    assert env.spent == 0


def test_concurrent_admissions_cannot_oversubscribe():
    """K threads race for a budget that affords 3 reservations; exactly 3 win.

    Without reserve-then-settle every caller reads the same ``remaining`` and
    all K pass, which is how the multi-agent arm would silently overspend.
    """
    env = TokenEnvelope(300)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: env.admit(f"sub-{i}", 100), range(8)))
    assert sum(r is not None for r in results) == 3
    assert env.remaining == 0


def test_sub_cap_shrinks_as_the_envelope_drains():
    """cap_i = min(B/N, remaining), so sub-agents cannot collectively promise
    more than is left after the orchestrator has spent."""
    env = TokenEnvelope(1000, n_subagents=4)
    assert env.sub_cap() == 250                      # B/N while the pool is full

    res = env.admit(ORCHESTRATOR, 0)
    env.settle(res, agent_id=ORCHESTRATOR, input_tokens=800, output_tokens=0)
    assert env.remaining == 200
    assert env.sub_cap() == 200                      # clamped by what is left


def test_sub_cap_is_zero_without_subagents():
    assert TokenEnvelope(1000, n_subagents=0).sub_cap() == 0


def test_agent_cap_denies_once_its_share_is_used():
    env = TokenEnvelope(1000, n_subagents=4)
    cap = env.sub_cap()
    res = env.admit("sub-1", 100, cap=cap)
    env.settle(res, input_tokens=200, output_tokens=40)   # 240 of a 250 cap
    assert env.admit("sub-1", 100, cap=cap) is None       # 240 + 100 > 250
    assert env.admit("sub-2", 100, cap=cap) is not None   # a different agent is fine


def test_unadmitted_calls_are_still_charged():
    """Helper calls (compaction, memory flush, vision) never pass admission but
    must still come out of the envelope, or the arms are not comparable."""
    env = TokenEnvelope(1000)
    env.settle(None, input_tokens=100, output_tokens=20)
    assert env.spent == 120
    assert any(a.agent_id == "unattributed" for a in env.agents())


def test_overshoot_is_reported_not_hidden():
    env = TokenEnvelope(100)
    res = env.admit(ORCHESTRATOR, 50)
    env.settle(res, input_tokens=90, output_tokens=40)    # the call overran
    assert env.spent == 130
    assert env.overshoot == 30
    assert env.remaining == 0
    assert env.exhausted


def test_summary_conserves_tokens():
    env = TokenEnvelope(1000, n_subagents=2)
    env.settle(env.admit(ORCHESTRATOR, 10), input_tokens=100, output_tokens=25, role="orchestrator")
    env.settle(env.admit("sub-1", 10), input_tokens=60, output_tokens=15)
    env.settle(None, input_tokens=5, output_tokens=5)     # helper call
    s = env.summary()
    per_agent = sum(a["total_tokens"] for a in s["per_agent"])
    assert per_agent == s["spent_total"] == 210
    assert s["budget_total_tokens"] == 1000
    assert s["overshoot_tokens"] == 0


def test_rejects_nonsense_construction():
    with pytest.raises(ValueError):
        TokenEnvelope(0)
    with pytest.raises(ValueError):
        TokenEnvelope(100, n_subagents=-1)
