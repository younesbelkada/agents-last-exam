"""Token envelope: one shared spend ceiling for a whole run unit.

The experiment this serves compares one agent holding a budget ``B`` against an
orchestrator that delegates to ``N`` sub-agents. For that comparison to mean
anything, every token the run spends has to come out of the same envelope:
orchestrator turns, sub-agent turns, and the helper calls (compaction, memory
flush, vision) that each of them makes.

A sub-budget is therefore a *view* on the envelope, never an allocation. Its cap
is ``min(B/N, remaining)``, so N sub-agents cannot collectively promise more
than the envelope holds, and slack from a cheap sub-agent returns to the pool
automatically because only spend is tracked.

Admission is reserve-then-settle: a caller reserves its own estimate before the
request leaves, and the meter reconciles to the real usage when the response
lands. Without the reservation step, K concurrent sub-agents all read the same
``remaining`` and all pass a check the envelope can only honour once.

Locking is :class:`threading.Lock`, not ``asyncio.Lock``: some model calls run on
a worker thread with its own event loop (the vision tool does this), so the
state has to be safe from any thread. Every operation here is O(1), so holding
it never blocks a loop meaningfully.
"""
from __future__ import annotations

import dataclasses
import threading
from typing import Any

ORCHESTRATOR = "orchestrator"
UNATTRIBUTED = "unattributed"


@dataclasses.dataclass
class AgentSpend:
    """Per-agent running totals, in tokens."""

    agent_id: str
    role: str = "subagent"
    model: str = ""
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    cap: int | None = None
    stop_reason: str | None = None

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["total_tokens"] = self.total
        return d


@dataclasses.dataclass
class Reservation:
    """An admitted, not-yet-settled request."""

    agent_id: str
    amount: int
    settled: bool = False


class BudgetExhausted(RuntimeError):
    """Raised only by callers that prefer an exception to a falsy admission."""


class TokenEnvelope:
    """Shared ``input + output`` token ceiling for one run unit."""

    def __init__(self, total_tokens: int, *, n_subagents: int = 0) -> None:
        if total_tokens <= 0:
            raise ValueError(f"total_tokens must be positive, got {total_tokens}")
        if n_subagents < 0:
            raise ValueError(f"n_subagents must be >= 0, got {n_subagents}")
        self._total = int(total_tokens)
        self._n = int(n_subagents)
        self._settled = 0
        self._reserved = 0
        self._agents: dict[str, AgentSpend] = {}
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []

    # ---- introspection ----------------------------------------------------

    @property
    def total(self) -> int:
        return self._total

    @property
    def spent(self) -> int:
        """Settled tokens only. This is the number the experiment reports."""
        with self._lock:
            return self._settled

    @property
    def committed(self) -> int:
        """Settled plus outstanding reservations: what admission must respect."""
        with self._lock:
            return self._settled + self._reserved

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(self._total - self._settled - self._reserved, 0)

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    @property
    def overshoot(self) -> int:
        """Tokens spent beyond the ceiling. Non-zero is expected and bounded:
        usage is only known once a call returns, so the last call of each agent
        can cross the line."""
        with self._lock:
            return max(self._settled - self._total, 0)

    def sub_cap(self) -> int:
        """Per-sub-agent ceiling: ``min(B/N, remaining)``."""
        if self._n <= 0:
            return 0
        with self._lock:
            share = self._total // self._n
            left = max(self._total - self._settled - self._reserved, 0)
        return min(share, left)

    def agent(self, agent_id: str) -> AgentSpend:
        with self._lock:
            return self._agents.setdefault(agent_id, AgentSpend(agent_id=agent_id))

    def agents(self) -> list[AgentSpend]:
        with self._lock:
            return list(self._agents.values())

    # ---- admission --------------------------------------------------------

    def admit(self, agent_id: str, estimate: int, *, cap: int | None = None) -> Reservation | None:
        """Reserve ``estimate`` tokens for ``agent_id``, or return None.

        ``cap`` bounds this agent's cumulative spend (its ``B/N`` share). The
        envelope ceiling always applies on top.
        """
        estimate = max(int(estimate), 0)
        with self._lock:
            spend = self._agents.setdefault(agent_id, AgentSpend(agent_id=agent_id))
            if cap is not None:
                spend.cap = cap
                if spend.total + estimate > cap:
                    self._events.append(
                        {"event": "denied", "agent_id": agent_id, "reason": "sub_cap",
                         "estimate": estimate, "agent_spent": spend.total, "cap": cap}
                    )
                    return None
            if self._settled + self._reserved + estimate > self._total:
                self._events.append(
                    {"event": "denied", "agent_id": agent_id, "reason": "envelope",
                     "estimate": estimate, "remaining": max(self._total - self._settled - self._reserved, 0)}
                )
                return None
            self._reserved += estimate
            return Reservation(agent_id=agent_id, amount=estimate)

    def settle(
        self,
        reservation: Reservation | None,
        *,
        agent_id: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        cost_usd: float = 0.0,
        model: str = "",
        role: str | None = None,
    ) -> None:
        """Reconcile a reservation against real usage.

        ``reservation`` may be None for calls that were never admitted (helper
        calls, or anything the meter saw but admission did not). Those are still
        charged: the envelope must account for every token the run spent.
        """
        aid = agent_id or (reservation.agent_id if reservation else UNATTRIBUTED)
        with self._lock:
            if reservation is not None and not reservation.settled:
                reservation.settled = True
                self._reserved = max(self._reserved - reservation.amount, 0)
            spend = self._agents.setdefault(aid, AgentSpend(agent_id=aid))
            if role:
                spend.role = role
            if model:
                spend.model = model
            spend.calls += 1
            spend.input_tokens += int(input_tokens)
            spend.output_tokens += int(output_tokens)
            spend.cache_read_tokens += int(cache_read_tokens)
            spend.cache_write_tokens += int(cache_write_tokens)
            spend.cost_usd += float(cost_usd)
            self._settled += int(input_tokens) + int(output_tokens)

    def release(self, reservation: Reservation | None) -> None:
        """Drop a reservation whose call never happened (error before send)."""
        if reservation is None or reservation.settled:
            return
        with self._lock:
            reservation.settled = True
            self._reserved = max(self._reserved - reservation.amount, 0)

    # ---- reporting --------------------------------------------------------

    def note(self, event: str, **fields: Any) -> None:
        with self._lock:
            self._events.append({"event": event, **fields})

    def set_stop_reason(self, agent_id: str, reason: str) -> None:
        with self._lock:
            self._agents.setdefault(agent_id, AgentSpend(agent_id=agent_id)).stop_reason = reason

    def summary(self) -> dict[str, Any]:
        with self._lock:
            agents = [a.to_dict() for a in self._agents.values()]
            settled, total = self._settled, self._total
        return {
            "budget_total_tokens": total,
            "enforced_metric": "input_output",
            "spent_total": settled,
            "overshoot_tokens": max(settled - total, 0),
            "n_subagents": self._n,
            "per_agent": agents,
            "events": list(self._events),
        }
