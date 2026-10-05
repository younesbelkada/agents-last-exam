"""Budget-aware general sub-agent.

Subclasses rather than patches ``GeneralSubagentSession``: ``ale_claw`` is a
scored benchmark harness, and its results must not move because of an
experiment that sits beside it. Nothing under ``ale_run/agents/ale_claw/`` is
modified.

Two things the base session does not do, both of which would let the split arm
outspend its envelope:

* it sets no ``max_tokens`` (``subagent_session.py:354-359``), so a single reply
  is unbounded and no pre-call check can bound the overshoot;
* it carries no call metadata, so the meter could only attribute its calls by
  contextvar.

Source: ``ale_run/agents/ale_claw/harness/subagent/subagent_session.py`` and
``subagent_general.py`` at the commit this fork was taken from.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from ale_run.agents.ale_claw.harness.context.token_estimation import estimate_messages_tokens
from ale_run.agents.ale_claw.harness.subagent.subagent_registry import (
    SubagentRegistry,
    SubagentUsage,
)
from ale_run.agents.ale_claw.harness.subagent.subagent_session import (
    DEFAULT_MAX_STEPS,
    GeneralSubagentSession,
)

from .envelope import BudgetExhausted, TokenEnvelope
from .meter import call_metadata, set_identity

logger = logging.getLogger(__name__)


class BudgetedGeneralSubagentSession(GeneralSubagentSession):
    """A general sub-agent that must buy each call from the shared envelope."""

    def __init__(
        self,
        *,
        envelope: TokenEnvelope,
        sink: Any,
        run_id_for_budget: str,
        cap: int,
        max_output_tokens: int,
        meter_run_id: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._envelope = envelope
        self._sink = sink
        self._budget_id = run_id_for_budget
        self._cap = cap
        self._max_output = max_output_tokens
        self._meter_run_id = meter_run_id

    async def _call_llm(self, litellm_mod, resolved):
        """Admission-check, then issue the call with a hard output cap.

        Mirrors the base implementation's kwargs and adds ``max_tokens`` plus
        metadata. The estimate charged at admission is prompt + worst-case
        output; the meter reconciles it to the truth when the response lands.
        """
        estimate = estimate_messages_tokens(self._messages) + self._max_output
        reservation = self._envelope.admit(self._budget_id, estimate, cap=self._cap)
        if reservation is None:
            self._envelope.set_stop_reason(self._budget_id, "budget_exhausted")
            raise BudgetExhausted(
                f"subagent {self._budget_id} cannot afford another call "
                f"(estimate {estimate}, cap {self._cap}, "
                f"envelope remaining {self._envelope.remaining})"
            )
        self._sink.park(self._budget_id, reservation)

        kwargs: dict[str, Any] = {
            "model": resolved.model,
            "messages": self._messages,
            "temperature": 1.0,
            "max_tokens": self._max_output,
            "metadata": call_metadata(self._meter_run_id, self._budget_id),
            **self._thinking_params,
        }
        if self._tool_schemas:
            kwargs["tools"] = self._tool_schemas
        try:
            return await litellm_mod.acompletion(**kwargs)
        except Exception:
            # The meter never saw this call, so hand the reservation back
            # rather than leaving the envelope permanently committed.
            self._sink.take(self._budget_id)
            self._envelope.release(reservation)
            raise


async def run_budgeted_subagent(
    *,
    task: str,
    model: str,
    tools: list,
    registry: SubagentRegistry,
    run_id: str,
    summary_model: str,
    parent_session_dir: str | Path,
    envelope: TokenEnvelope,
    sink: Any,
    cap: int,
    max_output_tokens: int,
    meter_run_id: str,
    memory_store: Any | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    thinking_params: dict[str, Any] | None = None,
    initial_screenshot_paths: list[str] | None = None,
) -> None:
    """Drive one budgeted sub-agent and report it back to the registry.

    Differs from ``run_general_subagent`` in two ways: it binds the meter
    identity so the session's own helper calls (compaction) are attributed to
    it, and it reports budget exhaustion through ``registry.fail(..., usage)``
    rather than ``kill()``, because ``kill()`` takes no usage and would discard
    the sub-agent's spend from the registry's record.
    """
    set_identity(meter_run_id, run_id)
    session: BudgetedGeneralSubagentSession | None = None
    try:
        registry.mark_running(run_id)
        session = BudgetedGeneralSubagentSession(
            envelope=envelope,
            sink=sink,
            run_id_for_budget=run_id,
            cap=cap,
            max_output_tokens=max_output_tokens,
            meter_run_id=meter_run_id,
            run_id=run_id,
            task=task,
            model=model,
            tools=tools,
            registry=registry,
            summary_model=summary_model,
            parent_session_dir=Path(parent_session_dir),
            memory_store=memory_store,
            max_steps=max_steps,
            thinking_params=thinking_params,
            initial_screenshot_paths=initial_screenshot_paths,
        )
        registry.attach_inbox(run_id, session.inbox)
        result_text = await session.run()
        logger.info(
            "[Subagent] budgeted subagent %s completed (%d+%d tokens)",
            run_id, session.usage.input_tokens, session.usage.output_tokens,
        )
        registry.complete(run_id, result_text, session.usage)
    except BudgetExhausted as e:
        usage = session.usage if session is not None else SubagentUsage()
        logger.info("[Subagent] budgeted subagent %s hit its budget: %s", run_id, e)
        registry.fail(run_id, f"budget_exhausted: {e}", usage)
    except asyncio.CancelledError:
        logger.info("[Subagent] budgeted subagent %s cancelled", run_id)
        registry.kill(run_id)
        raise
    except Exception as e:  # noqa: BLE001 - mirrors run_general_subagent
        usage = session.usage if session is not None else SubagentUsage()
        logger.warning("[Subagent] budgeted subagent %s failed: %s", run_id, e)
        registry.fail(run_id, str(e), usage)
