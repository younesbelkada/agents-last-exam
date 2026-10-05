"""Call-level token meter: the only thing that charges the envelope.

Why this sits at the litellm layer rather than counting in the agent loops:
``harness/model/helper_runtime.py:18-76`` builds its own request and returns
only ``text`` + ``tool_calls``, discarding ``response.usage``. Compaction,
memory-flush and vision spend is therefore recorded nowhere in the harness, and
that omission scales with the number of agents, which is precisely the variable
under test. A ``litellm`` callback sees every request this process makes,
including those helper calls and litellm's own retries, so it is the only
vantage point from which "B tokens" means anything.

Attribution order is metadata on the call, then a contextvar, then an
``unattributed`` bucket. The contextvar is the dependable one: ``asyncio.Task``
copies the context at creation, so a sub-agent spawned with ``create_task``
inherits its identity automatically, and so do the helper calls it makes. Calls
that hop to a worker thread (the vision tool does) start from an empty context
and land in ``unattributed`` - still charged to the envelope, just not to a
sub-budget.

``MeterCore`` deliberately imports nothing beyond the stdlib so it can be tested
without litellm installed; :func:`install` does the lazy binding.
"""
from __future__ import annotations

import contextvars
import json
import logging
import threading
from pathlib import Path
from typing import Any

from .envelope import UNATTRIBUTED, Reservation, TokenEnvelope

logger = logging.getLogger(__name__)

_RUN_ID: contextvars.ContextVar[str] = contextvars.ContextVar("ale_budget_run_id", default="")
_AGENT_ID: contextvars.ContextVar[str] = contextvars.ContextVar("ale_budget_agent_id", default="")


def set_identity(run_id: str, agent_id: str) -> contextvars.Token:
    """Bind the current task to a run + agent. Returns the agent token."""
    _RUN_ID.set(run_id)
    return _AGENT_ID.set(agent_id)


def current_identity() -> tuple[str, str]:
    return _RUN_ID.get(), _AGENT_ID.get()


def clear_identity() -> None:
    """Drop the binding. Each run unit is its own asyncio Task, so contexts do
    not leak between units in production; this exists for teardown and tests."""
    _RUN_ID.set("")
    _AGENT_ID.set("")


def call_metadata(run_id: str, agent_id: str) -> dict[str, str]:
    """Metadata blob to thread through ``litellm`` kwargs where we control them."""
    return {"ale_run_id": run_id, "ale_agent_id": agent_id}


class _RunSink:
    """One run unit's envelope, ledger file, and outstanding reservations."""

    def __init__(self, envelope: TokenEnvelope, ledger_path: Path) -> None:
        self.envelope = envelope
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        # At most one in-flight call per agent: every agent loop awaits its
        # call before issuing the next, so agent_id is a sufficient key.
        self.pending: dict[str, Reservation] = {}
        self._lock = threading.Lock()

    def park(self, agent_id: str, reservation: Reservation) -> None:
        with self._lock:
            self.pending[agent_id] = reservation

    def take(self, agent_id: str) -> Reservation | None:
        with self._lock:
            return self.pending.pop(agent_id, None)

    def append(self, record: dict[str, Any]) -> None:
        """Append one ledger line immediately.

        Per-call rather than buffered: the episode can be cancelled at the wall
        clock at any moment (``lifecycle.py`` wraps launch in ``wait_for``), and
        a ledger that dies with the process is worthless for the experiment.
        """
        try:
            with self.ledger_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
                fh.flush()
        except OSError as e:
            logger.warning("budget ledger write failed: %s", e)


class MeterCore:
    """Routes observed calls to the right run's envelope and ledger."""

    def __init__(self) -> None:
        self._runs: dict[str, _RunSink] = {}
        self._lock = threading.Lock()

    def register(self, run_id: str, envelope: TokenEnvelope, ledger_path: Path) -> _RunSink:
        sink = _RunSink(envelope, ledger_path)
        with self._lock:
            self._runs[run_id] = sink
        return sink

    def unregister(self, run_id: str) -> None:
        with self._lock:
            self._runs.pop(run_id, None)

    def sink(self, run_id: str) -> _RunSink | None:
        with self._lock:
            return self._runs.get(run_id)

    def resolve(self, kwargs: dict[str, Any] | None) -> tuple[str, str]:
        """Attribution: call metadata, else contextvar, else unattributed."""
        meta = _extract_metadata(kwargs or {})
        run_id = str(meta.get("ale_run_id") or "") or _RUN_ID.get()
        agent_id = str(meta.get("ale_agent_id") or "") or _AGENT_ID.get() or UNATTRIBUTED
        return run_id, agent_id

    def record(
        self,
        *,
        kwargs: dict[str, Any] | None,
        response: Any = None,
        error: str | None = None,
    ) -> dict[str, Any] | None:
        """Charge one observed call. Returns the ledger record, or None if the
        call belongs to no registered run."""
        run_id, agent_id = self.resolve(kwargs)
        sink = self.sink(run_id) if run_id else None
        if sink is None:
            # Not ours: another experiment's run, or a call outside launch().
            return None

        usage = extract_usage(response, kwargs)
        reservation = sink.take(agent_id)
        sink.envelope.settle(
            reservation,
            agent_id=agent_id,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            cache_read_tokens=usage["cache_read_tokens"],
            cache_write_tokens=usage["cache_write_tokens"],
            cost_usd=usage["cost_usd"],
            model=usage["model"] or str((kwargs or {}).get("model") or ""),
        )
        record = {
            "run_id": run_id,
            "agent_id": agent_id,
            "model": usage["model"] or str((kwargs or {}).get("model") or ""),
            **{k: usage[k] for k in (
                "input_tokens", "output_tokens", "cache_read_tokens",
                "cache_write_tokens", "cost_usd",
            )},
            "spent_total": sink.envelope.spent,
            "attributed": agent_id != UNATTRIBUTED,
        }
        if error:
            record["error"] = error
        sink.append(record)
        return record


def _extract_metadata(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pull our metadata out of a litellm callback kwargs blob.

    litellm has moved this between ``kwargs["metadata"]`` and
    ``kwargs["litellm_params"]["metadata"]`` across versions, so check both and
    fall through quietly. The contextvar covers us when neither is present.
    """
    for candidate in (
        kwargs.get("litellm_params", {}).get("metadata") if isinstance(kwargs.get("litellm_params"), dict) else None,
        kwargs.get("metadata"),
        kwargs.get("optional_params", {}).get("metadata") if isinstance(kwargs.get("optional_params"), dict) else None,
    ):
        if isinstance(candidate, dict) and ("ale_run_id" in candidate or "ale_agent_id" in candidate):
            return candidate
    return {}


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def extract_usage(response: Any, kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalise a litellm response into the five numbers the envelope needs.

    Handles both object and dict shapes, and both the chat-completions
    (``prompt_tokens``) and responses-API (``input_tokens``) spellings, because
    the harness uses ``aresponses`` for some helper purposes.
    """
    usage = _get(response, "usage")
    inp = int(_get(usage, "prompt_tokens", None) or _get(usage, "input_tokens", 0) or 0)
    out = int(_get(usage, "completion_tokens", None) or _get(usage, "output_tokens", 0) or 0)

    details = _get(usage, "prompt_tokens_details") or _get(usage, "input_tokens_details") or {}
    cache_read = int(_get(details, "cached_tokens", 0) or 0)
    cache_write = int(
        _get(details, "cache_write_tokens", 0)
        or _get(usage, "cache_creation_input_tokens", 0)
        or 0
    )

    cost = _get(usage, "cost", None)
    if cost is None:
        hidden = _get(response, "_hidden_params") or {}
        cost = _get(hidden, "response_cost", None)
    if cost is None and kwargs:
        cost = kwargs.get("response_cost")
    try:
        cost = float(cost or 0.0)
    except (TypeError, ValueError):
        cost = 0.0

    return {
        "input_tokens": inp,
        "output_tokens": out,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "cost_usd": cost,
        "model": str(_get(response, "model", "") or ""),
    }


# ---------------------------------------------------------------------------
# litellm binding (lazy: this module must import without litellm present)
# ---------------------------------------------------------------------------

METER = MeterCore()
_installed: list[Any] = []
_refcount = 0
_install_lock = threading.Lock()


def install() -> Any:
    """Append our callback to ``litellm.callbacks``.

    Refcounted, because the callback list is process-wide while runs are not:
    several run units share one orchestrator process, and an unbalanced
    uninstall by the first to finish would silently stop metering every other
    unit still in flight.
    """
    global _refcount
    with _install_lock:
        _refcount += 1
        if _installed:
            return _installed[0]
        return _do_install()


def _do_install() -> Any:
    import litellm
    from litellm.integrations.custom_logger import CustomLogger

    class _BudgetLogger(CustomLogger):
        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            METER.record(kwargs=kwargs, response=response_obj)

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            METER.record(kwargs=kwargs, response=response_obj)

        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            METER.record(kwargs=kwargs, response=response_obj, error="failure")

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            METER.record(kwargs=kwargs, response=response_obj, error="failure")

    handler = _BudgetLogger()
    litellm.callbacks.append(handler)
    _installed.append(handler)
    logger.info("claw_budget: litellm meter installed")
    return handler


def uninstall() -> None:
    """Drop one reference; unhook only when the last run is done."""
    global _refcount
    with _install_lock:
        _refcount = max(_refcount - 1, 0)
        if _refcount > 0 or not _installed:
            return
        # Drop our own state first: if detaching from litellm fails we still
        # want a consistent view here, not a handler we think is installed.
        handler = _installed.pop()
        try:
            import litellm

            if handler in litellm.callbacks:
                litellm.callbacks.remove(handler)
        except Exception as e:  # noqa: BLE001 - teardown must never fail a run
            logger.warning("claw_budget: meter uninstall failed: %s", e)
