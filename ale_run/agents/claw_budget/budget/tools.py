"""Budget-aware ``delegate_general`` tool.

Subclasses the stock tool so ``ale_claw`` keeps its exact behaviour; the forked
deployer swaps the instance in the list that ``build_tools`` returns rather than
changing how the list is built.

Two additions over the base tool: a hard ceiling on how many sub-agents a run
may spawn (the ``N`` of the experiment, which the base tool has no notion of -
its registry cap is on *concurrency*, not lifetime count), and a per-sub-agent
share of the envelope computed at spawn time.

Deliberately not re-decorated with ``@register_tool``: the subclass inherits
``name == "delegate_general"`` from the base class, and re-registering could
collide with the entry ``ale_claw`` itself relies on.

Source: ``ale_run/agents/ale_claw/harness/subagent/subagent_tools.py``
(``DelegateGeneralTool.call``) at the commit this fork was taken from.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from ale_run.agents.ale_claw.harness.subagent.subagent_registry import (
    SubagentLimitError,
    SubagentType,
)
from ale_run.agents.ale_claw.harness.subagent.subagent_tools import (
    _ACCEPTED_NOTE,
    DELEGATE_GENERAL_DEFAULT_MAX_STEPS,
    DelegateGeneralTool,
    _sanitize_subagent_model,
)

from .envelope import TokenEnvelope
from .subagent import run_budgeted_subagent

logger = logging.getLogger(__name__)


class BudgetedDelegateGeneralTool(DelegateGeneralTool):
    """``delegate_general`` with a spawn cap and a share of the envelope."""

    def __init__(
        self,
        *,
        envelope: TokenEnvelope,
        sink: Any,
        meter_run_id: str,
        max_subagents: int,
        subagent_max_output_tokens: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._envelope = envelope
        self._sink = sink
        self._meter_run_id = meter_run_id
        self._max_subagents = max_subagents
        self._sub_max_output = subagent_max_output_tokens
        self._spawned = 0

    @property
    def description(self) -> str:
        return (
            f"{super().description} This run may spawn at most "
            f"{self._max_subagents} subagents in total, and each one draws from "
            f"a shared token budget, so delegate deliberately."
        )

    def call(self, params: str | dict, **kwargs) -> dict:
        params_dict = self._verify_json_format_args(params)

        task = params_dict.get("task", "")
        if not isinstance(task, str) or not task.strip():
            return {"status": "error", "reason": "task must be a non-empty string"}

        if self._spawned >= self._max_subagents:
            self._envelope.note(
                "delegation_refused", reason="spawn_cap", cap=self._max_subagents
            )
            return {
                "status": "rejected",
                "reason": (
                    f"spawn cap reached: this run may start at most "
                    f"{self._max_subagents} subagents"
                ),
            }

        cap = self._envelope.sub_cap()
        if cap <= 0 or self._envelope.remaining <= 0:
            self._envelope.note("delegation_refused", reason="budget")
            return {"status": "rejected", "reason": "token budget exhausted"}

        model, model_warning = _sanitize_subagent_model(
            params_dict.get("model"), self._default_model, self._auxiliary_model
        )
        if model_warning:
            logger.warning("delegate_general: %s", model_warning)
        max_steps = int(
            params_dict.get("max_steps", DELEGATE_GENERAL_DEFAULT_MAX_STEPS)
        )
        label = params_dict.get("label", "") or ""
        screenshot_paths_raw = params_dict.get("screenshot_paths") or []
        screenshot_paths: list[str] | None = (
            [p for p in screenshot_paths_raw if isinstance(p, str) and p]
            if isinstance(screenshot_paths_raw, list)
            else None
        )

        try:
            run = self._registry.register(
                type=SubagentType.GENERAL, task=task, label=label, model=model,
            )
        except SubagentLimitError:
            return {"status": "rejected", "reason": "max concurrent subagents reached"}

        self._spawned += 1
        self._envelope.note(
            "delegated", agent_id=run.run_id, cap=cap,
            spawned=self._spawned, max_subagents=self._max_subagents,
        )

        coro = run_budgeted_subagent(
            task=task,
            model=model,
            tools=self._tools,
            registry=self._registry,
            run_id=run.run_id,
            summary_model=self._summary_model,
            parent_session_dir=self._parent_session_dir,
            envelope=self._envelope,
            sink=self._sink,
            cap=cap,
            max_output_tokens=self._sub_max_output,
            meter_run_id=self._meter_run_id,
            memory_store=self._memory_store,
            max_steps=max_steps,
            thinking_params=self._thinking_params,
            initial_screenshot_paths=screenshot_paths,
        )
        task_handle = asyncio.get_running_loop().create_task(coro)
        self._registry.attach_task(run.run_id, task_handle)

        response = {
            "status": "accepted",
            "run_id": run.run_id,
            "token_budget": cap,
            "note": _ACCEPTED_NOTE,
        }
        if model_warning:
            response["model_warning"] = model_warning
        return response
