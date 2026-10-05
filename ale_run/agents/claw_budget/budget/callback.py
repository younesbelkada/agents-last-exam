"""Orchestrator-side budget gate.

Stops the main agent loop by returning ``False`` from ``on_run_continue``, which
``agent_loop.py:231`` already honours (the GUI sub-agent's ``SteerInboxCallback``
is the existing precedent). Raising instead would be wrong three ways: the run
would be mapped to ``status="failed"`` by the deployer's except arm, the
``_on_run_end`` callbacks would never flush, and the fire-and-forget sub-agent
tasks would keep spending after we declared the budget gone.

The first turn is always allowed. ``agent_loop.py`` assigns ``loop_kwargs``
inside the while body (:244) but reads it after the loop (:355), so refusing the
very first iteration raises ``UnboundLocalError``. The deployer checks
affordability before entering the loop at all, so this guard only covers the
pathological case of a budget too small for a single turn.
"""
from __future__ import annotations

import logging
from typing import Any

from agent.callbacks.base import AsyncCallbackHandler

from .envelope import ORCHESTRATOR, TokenEnvelope

logger = logging.getLogger(__name__)


class BudgetCallback(AsyncCallbackHandler):
    """Refuse further orchestrator turns once the envelope cannot fund one."""

    def __init__(
        self,
        envelope: TokenEnvelope,
        *,
        min_turn_tokens: int = 2_000,
        agent_id: str = ORCHESTRATOR,
    ) -> None:
        self._envelope = envelope
        self._min_turn = max(int(min_turn_tokens), 1)
        self._agent_id = agent_id
        self._turns = 0
        self.stopped_for_budget = False

    async def on_run_continue(
        self,
        kwargs: dict[str, Any],
        old_items: list[dict[str, Any]],
        new_items: list[dict[str, Any]],
    ) -> bool:
        self._turns += 1
        if self._turns == 1:
            return True
        remaining = self._envelope.remaining
        if remaining < self._min_turn:
            self.stopped_for_budget = True
            self._envelope.set_stop_reason(self._agent_id, "budget_exhausted")
            self._envelope.note(
                "orchestrator_stopped",
                agent_id=self._agent_id,
                turn=self._turns,
                remaining=remaining,
                min_turn_tokens=self._min_turn,
            )
            logger.info(
                "claw_budget: stopping orchestrator at turn %d (%d tokens left, "
                "need %d)", self._turns, remaining, self._min_turn,
            )
            return False
        return True
