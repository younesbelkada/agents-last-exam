"""Config for the budget-split harness.

Subclasses :class:`AleClawConfig` rather than copying its ~25 fields: the
experiment only adds knobs, and a copy would silently drift from the harness it
is meant to be comparable with. ``factory.build_config`` filters yaml keys by
``dataclasses.fields``, which walks the MRO, so inherited fields stay settable
from the preset.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ale_run.agents.ale_claw.config import AleClawConfig


@dataclass
class ClawBudgetConfig(AleClawConfig):
    """``AleClawConfig`` plus a token envelope and a delegation ceiling."""

    name: ClassVar[str] = "claw-budget"

    budget_total_tokens: int = 200_000
    """``B``: the run's total ``input + output`` token ceiling, shared by the
    orchestrator, every sub-agent, and every helper call (compaction, memory
    flush, vision). Both arms of the experiment must use the same value."""

    n_subagents: int = 0
    """``N``: how many sub-agents this run may spawn in total. ``0`` is the
    single-agent arm and also disables the delegation tools. Each sub-agent is
    capped at ``min(B/N, remaining)``; the cap is a view on the shared envelope,
    not an allocation, so unused budget returns to the pool."""

    subagent_max_output_tokens: int = 4_096
    """Hard ``max_tokens`` on every sub-agent call. The stock sub-agent session
    sets none, which would leave per-call overshoot unbounded and make the
    envelope unenforceable."""

    min_turn_tokens: int = 2_000
    """Orchestrator turns are refused below this much remaining budget. Stops
    the loop from burning the tail of the envelope on a turn it cannot finish."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.budget_total_tokens <= 0:
            raise ValueError(
                f"ClawBudgetConfig.budget_total_tokens must be positive, "
                f"got {self.budget_total_tokens}"
            )
        if self.n_subagents < 0:
            raise ValueError(
                f"ClawBudgetConfig.n_subagents must be >= 0, got {self.n_subagents}"
            )
        if self.subagent_max_output_tokens <= 0:
            raise ValueError(
                f"ClawBudgetConfig.subagent_max_output_tokens must be positive, "
                f"got {self.subagent_max_output_tokens}"
            )
        if self.n_subagents and self.budget_total_tokens // self.n_subagents < self.subagent_max_output_tokens:
            raise ValueError(
                f"budget_total_tokens // n_subagents "
                f"({self.budget_total_tokens // self.n_subagents}) is below "
                f"subagent_max_output_tokens ({self.subagent_max_output_tokens}): "
                f"no sub-agent could complete a single call"
            )
