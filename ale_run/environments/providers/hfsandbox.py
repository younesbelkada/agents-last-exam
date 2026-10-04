"""HfSandboxProvider — one Hugging Face Jobs sandbox per task.

Boots the ALE sandbox image (``agentslastexam/ale-ubuntu22-docker``) on HF
infrastructure, one container per run unit, and reaches its cua-server
(:5000, started by the image entrypoint together with Xvfb + the desktop)
over HTTPS. Two transports, selected by ``transport:``:

* ``job`` (default) — ``HfApi.run_job(image, command=[entrypoint],
  expose=[5000])``. The port is reachable at
  ``https://<job_id>--5000.hf.jobs`` through the HF Jobs proxy, which needs
  ``Authorization: Bearer <HF token>``.
* ``sandbox`` — ``huggingface_hub.Sandbox.create`` (the HF Sandbox API) plus
  ``Sandbox.proxy_url_for(5000)`` / ``proxy_headers``. As of 2026-09 that
  port-proxy returns FastAPI ``404 Not Found`` for roughly every other
  request to cua-server (repro: alternating ``GET /status`` → 200, 404, 200,
  404 while in-sandbox curl is 200 every time), so it is not the default.

Either way the upstream needs auth headers, while everything downstream —
:class:`SandboxHandle`'s wire impl, cua-bench's ``RemoteDesktopSession`` and
the node MCP bridges (``CUA_SERVER_URL``) — talks to a plain unauthenticated
``http://host:port``. So each acquired sandbox gets a tiny local reverse
proxy (aiohttp) on ``127.0.0.1:<random port>`` that injects the headers and
pipes HTTP (incl. SSE streams) and WebSockets through. The handle's
``endpoint`` is that local URL; nothing else in the framework knows about HF.

Requires ``huggingface_hub>=1.x`` and a HF token (``HF_TOKEN`` /
``hf auth login``) with the ``jobs`` scope.
"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import re
import threading
import time
from typing import Any

from ...base_interface import (
    Provider,
    ReleaseMode,
    SandboxHandle,
    SandboxSpec,
)

logger = logging.getLogger(__name__)

_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
})

# The API rejects anything else, dots included: a task id like
# `legal/legal_dr_fees_01` has a slash, a model id like `Qwen/Qwen3.5-9B` both.
_LABEL_UNSAFE = re.compile(r"[^a-zA-Z0-9_-]")


def _job_label(value: str) -> str:
    """HF job label values must match ``^[a-zA-Z0-9_-]*$``."""
    return _LABEL_UNSAFE.sub("_", value)[:60]


# ======================================================================
# Config
# ======================================================================


@dataclasses.dataclass(frozen=True)
class HfSandboxProviderConfig:
    image: str = "ale-ubuntu22-docker"
    """Image NAME in :mod:`ale_run.environments.images` (paths, cua port,
    container ref, entrypoint)."""

    image_ref: str = ""
    """Override the container ref to boot (default: the Image's ``docker_image``)."""

    transport: str = "job"
    """``job`` (HF Job + exposed port, default) or ``sandbox`` (HF Sandbox API
    port-proxy; buggy as of 2026-09, see module docstring)."""

    job_timeout: str = "24h"
    """(transport=job) Hard wall-clock cap on the sandbox job. The framework
    kills the job on ``release`` anyway; this is the safety net."""

    flavor: str = "cpu-upgrade"
    """HF Jobs hardware flavor (``hf jobs hardware``). The ALE image wants
    ~4 vCPU / 16 GB per task; cpu-upgrade is 8 vCPU / 32 GB."""

    idle_timeout: int | float | str | None = 3600
    """(transport=sandbox) Auto-shutdown after this much inactivity (seconds or
    a duration string like ``"2h"``). The sandbox's hard cap is 24h regardless."""

    start_timeout: float = 1800.0
    """Seconds to wait for the sandbox server to come up. Includes the image
    pull — the ALE image is ~40 GB compressed, so a cold node takes 10-15 min."""

    cua_ready_timeout: float = 300.0
    """Seconds to wait for cua-server (:5000) after starting the entrypoint."""

    namespace: str | None = None
    """HF namespace (user/org) to bill the sandbox job to. None → token owner."""

    resolution: tuple[int, int] = (1024, 768)
    """Virtual display size (``ALE_SCREEN_RESOLUTION`` read by the entrypoint)."""

    enable_dind: bool = False
    """Ask the entrypoint to start nested Docker (unsupported on HF Sandboxes)."""

    volumes: tuple[str, ...] = ()
    """Volumes to mount into every sandbox job, ``hf jobs run -v`` syntax:
    ``hf://buckets/<ns>/<name>[/<prefix>]:/<mount>[:ro|:rw]`` or
    ``hf://datasets/<ns>/<name>[/<prefix>]:/<mount>``. Pair a read-only bucket
    holding the extracted ``ale-tasks-data.tar.gz`` with
    ``task_data_source: mounted:/<mount>``."""


def _build_provider_config(raw: dict[str, Any]) -> HfSandboxProviderConfig:
    res = raw.get("resolution")
    if isinstance(res, (list, tuple)) and len(res) == 2:
        res = (int(res[0]), int(res[1]))
    else:
        res = (1024, 768)
    transport = str(raw.get("transport") or "job")
    if transport not in ("job", "sandbox"):
        raise ValueError(f"hfsandbox: transport must be 'job' or 'sandbox', got {transport!r}")
    return HfSandboxProviderConfig(
        image=str(raw.get("image") or "ale-ubuntu22-docker"),
        image_ref=str(raw.get("image_ref") or ""),
        transport=transport,
        job_timeout=str(raw.get("job_timeout") or "24h"),
        flavor=str(raw.get("flavor") or "cpu-upgrade"),
        idle_timeout=raw.get("idle_timeout", 3600),
        start_timeout=float(raw.get("start_timeout") or 1800),
        cua_ready_timeout=float(raw.get("cua_ready_timeout") or 300),
        namespace=raw.get("namespace") or None,
        resolution=res,
        enable_dind=bool(raw.get("enable_dind", False)),
        volumes=tuple(str(v) for v in (raw.get("volumes") or [])),
    )


def parse_volume_spec(spec: str) -> Any:
    """``hf://<type>/<ns>/<name>[/<prefix>]:/<mount>[:ro|:rw]`` → ``Volume``.

    Same grammar as ``hf jobs run -v``. ``<type>`` is ``buckets``, ``datasets``,
    ``models`` or ``spaces`` (repos are always read-only; buckets default to
    read-write unless ``:ro``).
    """
    from huggingface_hub import Volume

    if not spec.startswith("hf://"):
        raise ValueError(f"volume spec must start with hf://, got {spec!r}")
    body = spec[len("hf://"):]
    parts = body.split(":")
    if len(parts) < 2 or not parts[1].startswith("/"):
        raise ValueError(f"volume spec {spec!r}: expected hf://.../<repo>:/<mount>[:ro|:rw]")
    src, mount = parts[0], parts[1]
    mode = parts[2] if len(parts) > 2 else None
    if mode not in (None, "ro", "rw"):
        raise ValueError(f"volume spec {spec!r}: mode must be ro or rw")
    segs = src.split("/")
    kinds = {"buckets": "bucket", "datasets": "dataset", "models": "model", "spaces": "space"}
    if segs[0] in kinds:
        vtype, segs = kinds[segs[0]], segs[1:]
    else:
        vtype = "model"
    if len(segs) < 2:
        raise ValueError(f"volume spec {spec!r}: source must be <namespace>/<name>")
    source, prefix = "/".join(segs[:2]), "/".join(segs[2:]) or None
    read_only: bool | None
    if vtype != "bucket":
        read_only = True
    else:
        read_only = None if mode is None else (mode == "ro")
    return Volume(type=vtype, source=source, mount_path=mount, read_only=read_only, path=prefix)


# ======================================================================
# Local auth-injecting reverse proxy
# ======================================================================


class _AuthProxy:
    """``http://127.0.0.1:<port>`` → ``<upstream>`` with auth headers added.

    Forwards every HTTP request (streaming the body back, so SSE from
    ``/cmd`` works) and every WebSocket upgrade (``/ws``), adding ``headers``
    on the way out. ``upstream`` is an ``https://`` base URL (no trailing
    slash); WebSocket upgrades use the same host with ``wss://``.

    Runs on its **own thread + event loop**, not the orchestrator's. Agent
    harnesses are free to block the main loop while waiting for a tool result
    (ale_claw's MCP tool calls do), and a proxy living on that loop would then
    deadlock every VM-side call until the tool's timeout fired.
    """

    def __init__(self, upstream: str, headers: dict[str, str]):
        self._upstream_base = upstream.rstrip("/")
        self._headers = dict(headers)
        self._runner = None
        self._client = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self.port: int = 0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    # -- lifecycle (called from the orchestrator loop; work happens on our thread)

    async def start(self) -> None:
        loop = asyncio.new_event_loop()
        ready = threading.Event()
        self._loop = loop
        self._thread = threading.Thread(
            target=self._thread_main, args=(loop, ready), name="hfsandbox-proxy",
            daemon=True,
        )
        self._thread.start()
        await asyncio.to_thread(ready.wait)
        fut = asyncio.run_coroutine_threadsafe(self._start_on_loop(), loop)
        await asyncio.wrap_future(fut)

    async def stop(self) -> None:
        loop, thread = self._loop, self._thread
        if loop is None:
            return
        fut = asyncio.run_coroutine_threadsafe(self._stop_on_loop(), loop)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(asyncio.wrap_future(fut), 15)
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            await asyncio.to_thread(thread.join, 15)
        self._loop = self._thread = None

    @staticmethod
    def _thread_main(loop: asyncio.AbstractEventLoop, ready: threading.Event) -> None:
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        try:
            loop.run_forever()
        finally:
            with contextlib.suppress(Exception):
                loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    async def _start_on_loop(self) -> None:
        import aiohttp
        from aiohttp import web

        # auto_decompress=False: relay bodies byte-for-byte together with their
        # original Content-Encoding, instead of inflating them client-side and
        # then forwarding a header that no longer matches the bytes.
        self._client = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=60),
            auto_decompress=False,
        )
        app = web.Application(client_max_size=1024 ** 3)
        app.router.add_route("*", "/{tail:.*}", self._handle)
        # Bounded shutdown: don't wait on lingering client connections at release.
        self._runner = web.AppRunner(app, access_log=None, shutdown_timeout=2.0)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

    async def _stop_on_loop(self) -> None:
        if self._runner is not None:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.wait_for(self._runner.cleanup(), 10)
            self._runner = None
        if self._client is not None:
            await self._client.close()
            self._client = None

    def _upstream(self, path_qs: str, *, scheme: str) -> str:
        host_and_rest = self._upstream_base.split("://", 1)[-1]
        path_qs = path_qs if path_qs.startswith("/") else "/" + path_qs
        return f"{scheme}{host_and_rest}{path_qs}"

    async def _handle(self, request):
        import aiohttp
        from aiohttp import web

        if request.headers.get("Upgrade", "").lower() == "websocket":
            return await self._handle_ws(request)

        headers = {
            k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP
        }
        headers.update(self._headers)
        body = await request.read()
        url = self._upstream(request.rel_url.path_qs, scheme="https://")
        assert self._client is not None
        t0 = time.monotonic()
        try:
            async with self._client.request(
                request.method, url, headers=headers, data=body or None,
                allow_redirects=False,
            ) as up:
                resp = web.StreamResponse(status=up.status)
                for k, v in up.headers.items():
                    if k.lower() not in _HOP_BY_HOP:
                        resp.headers[k] = v
                await resp.prepare(request)
                async for chunk in up.content.iter_any():
                    await resp.write(chunk)
                await resp.write_eof()
                return resp
        except (ConnectionResetError, aiohttp.ClientConnectionResetError):
            # Our client (ALE wire impl / cua SDK / MCP bridge) gave up — usually
            # its own timeout — before the upstream answered. Not a proxy fault.
            logger.warning(
                "hfsandbox proxy: client dropped %s %s after %.1fs",
                request.method, request.rel_url.path_qs, time.monotonic() - t0,
            )
            raise web.HTTPRequestTimeout() from None

    async def _handle_ws(self, request):
        import aiohttp
        from aiohttp import web

        ws_client = web.WebSocketResponse(max_msg_size=0)
        await ws_client.prepare(request)
        url = self._upstream(request.rel_url.path_qs, scheme="wss://")
        assert self._client is not None
        try:
            async with self._client.ws_connect(
                url, headers=self._headers, max_msg_size=0, heartbeat=30,
            ) as ws_up:

                async def pump(src, dst):
                    async for msg in src:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await dst.send_str(msg.data)
                        elif msg.type == aiohttp.WSMsgType.BINARY:
                            await dst.send_bytes(msg.data)
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING,
                            aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR,
                        ):
                            break
                    await dst.close()

                await asyncio.gather(
                    pump(ws_client, ws_up), pump(ws_up, ws_client),
                    return_exceptions=True,
                )
        except Exception as e:  # noqa: BLE001 — surface as a closed socket
            logger.warning("hfsandbox proxy: websocket relay failed: %s", e)
            if not ws_client.closed:
                await ws_client.close()
        return ws_client


# ======================================================================
# Provider
# ======================================================================


@dataclasses.dataclass
class _Live:
    proxy: _AuthProxy
    kill: Any            # zero-arg sync callable that tears the remote down
    sandbox: Any = None  # huggingface_hub.Sandbox (transport=sandbox) or None


class HfSandboxProvider(Provider):
    """One HF Jobs sandbox per ``acquire``; ``release(delete)`` kills it."""

    def __init__(self, config: HfSandboxProviderConfig | dict[str, Any]):
        if isinstance(config, dict):
            config = _build_provider_config(config)
        self._cfg = config
        self._live: dict[str, _Live] = {}

    @property
    def config(self) -> HfSandboxProviderConfig:
        return self._cfg

    # ------------------------------------------------------------------ acquire

    async def acquire(self, spec: SandboxSpec) -> SandboxHandle:
        from ..images import get as get_image

        image = get_image(self._cfg.image)
        container_ref = self._cfg.image_ref or image.docker_image
        if not container_ref:
            raise RuntimeError(
                f"image {self._cfg.image!r} has no docker_image; set `image_ref`"
            )
        res = self._cfg.resolution
        env = {
            "ALE_SCREEN_RESOLUTION": f"{res[0]}x{res[1]}",
            "ALE_ENABLE_DIND": "1" if self._cfg.enable_dind else "0",
        }
        logger.info(
            "hfsandbox[%s]: creating image=%s flavor=%s (task=%s snapshot=%s)",
            self._cfg.transport, container_ref, self._cfg.flavor,
            spec.task_id, spec.snapshot,
        )
        t0 = time.monotonic()
        if self._cfg.transport == "sandbox":
            remote_id, upstream, headers, kill, sbx, extra = await self._create_via_sandbox(
                container_ref, image, env,
            )
        else:
            remote_id, upstream, headers, kill, sbx, extra = await self._create_via_job(
                container_ref, image, env, spec,
            )
        logger.info(
            "hfsandbox: %s up in %.0fs, cua upstream %s", remote_id,
            time.monotonic() - t0, upstream,
        )
        proxy = _AuthProxy(upstream, headers)
        try:
            await proxy.start()
            handle = SandboxHandle(
                id=remote_id,
                endpoint=proxy.url,
                os=image.os,
                **image.sandbox_paths(),
                metadata={
                    "provider": "hfsandbox",
                    "transport": self._cfg.transport,
                    "job_url": f"https://huggingface.co/jobs/{remote_id}",
                    "flavor": self._cfg.flavor,
                    "image": self._cfg.image,
                    "snapshot": spec.snapshot,
                    "proxy_port": proxy.port,
                    **extra,
                },
            )
            await self._wait_cua_ready(handle)
        except Exception:
            await proxy.stop()
            await asyncio.to_thread(kill)
            raise
        self._live[remote_id] = _Live(proxy=proxy, kill=kill, sandbox=sbx)
        logger.info(
            "hfsandbox: %s cua-server reachable via %s (total %.0fs)",
            remote_id, proxy.url, time.monotonic() - t0,
        )
        return handle

    async def _create_via_job(self, container_ref: str, image: Any, env: dict, spec: SandboxSpec):
        """HF Job running the image entrypoint as its command, port exposed."""
        from huggingface_hub import HfApi, get_token

        api = HfApi()
        token = get_token()
        if not token:
            raise RuntimeError("hfsandbox: no HF token (set HF_TOKEN or `hf auth login`)")
        labels = {
            "ale": "sandbox",
            "ale_task": _job_label(spec.task_id or ""),
            "ale_model": _job_label(spec.model_tag or ""),
            "ale_harness": _job_label(spec.harness or ""),
        }
        job = await asyncio.to_thread(
            api.run_job,
            image=container_ref,
            command=[image.docker_entrypoint],
            env=env,
            flavor=self._cfg.flavor,
            timeout=self._cfg.job_timeout,
            expose=[image.cua_server_port],
            labels={k: v for k, v in labels.items() if v},
            volumes=[parse_volume_spec(v) for v in self._cfg.volumes] or None,
            namespace=self._cfg.namespace,
        )
        job_id, owner = job.id, job.owner.name

        def kill() -> None:
            try:
                api.cancel_job(job_id=job_id, namespace=owner)
            except Exception as e:  # noqa: BLE001
                logger.warning("hfsandbox: cancel_job %s failed: %s", job_id, e)

        deadline = time.monotonic() + self._cfg.start_timeout
        url = None
        while time.monotonic() < deadline:
            info = await asyncio.to_thread(api.inspect_job, job_id=job_id, namespace=owner)
            stage = info.status.stage
            if stage == "RUNNING" and info.status.expose_urls:
                url = info.status.expose_urls[0]
                break
            if stage in ("ERROR", "COMPLETED", "CANCELED", "DELETED"):
                raise RuntimeError(
                    f"hfsandbox: job {job_id} ended during startup: {info.status}"
                )
            await asyncio.sleep(3)
        if url is None:
            await asyncio.to_thread(kill)
            raise RuntimeError(
                f"hfsandbox: job {job_id} not RUNNING after {self._cfg.start_timeout:.0f}s "
                f"(image pull of {container_ref} still in progress?)"
            )
        headers = {"Authorization": f"Bearer {token}"}
        return job_id, url, headers, kill, None, {"job_owner": owner}

    async def _create_via_sandbox(self, container_ref: str, image: Any, env: dict):
        """HF Sandbox API: create, start entrypoint detached, use its port-proxy."""
        from huggingface_hub import Sandbox

        sbx = await asyncio.to_thread(
            Sandbox.create,
            image=container_ref,
            flavor=self._cfg.flavor,
            idle_timeout=self._cfg.idle_timeout,
            env=env,
            volumes=[parse_volume_spec(v) for v in self._cfg.volumes] or None,
            namespace=self._cfg.namespace,
            start_timeout=self._cfg.start_timeout,
        )

        def kill() -> None:
            _kill_quiet(sbx)

        try:
            # The sandbox server replaced the container's own CMD, so start the
            # image entrypoint (X display + cua-server) ourselves, detached.
            proc = await asyncio.to_thread(
                sbx.run, [image.docker_entrypoint], shell=False, background=True,
            )
        except Exception:
            await asyncio.to_thread(kill)
            raise
        upstream = sbx.proxy_url_for(image.cua_server_port, "/").rstrip("/")
        extra = {"sandbox_id": sbx.id, "entrypoint_pid": getattr(proc, "pid", None)}
        return sbx.id, upstream, dict(sbx.proxy_headers), kill, sbx, extra

    async def _wait_cua_ready(self, handle: SandboxHandle) -> None:
        deadline = time.monotonic() + self._cfg.cua_ready_timeout
        last: Exception | None = None
        while time.monotonic() < deadline:
            try:
                await handle.check_reachable(label="hfsandbox")
                return
            except Exception as e:  # noqa: BLE001 — any failure means "not yet"
                last = e
                await asyncio.sleep(3)
        raise RuntimeError(
            f"hfsandbox: cua-server in {handle.id} not ready after "
            f"{self._cfg.cua_ready_timeout:.0f}s: {last}"
        )

    # ------------------------------------------------------------------ release

    async def release(
        self, sandbox: SandboxHandle, *, mode: ReleaseMode = "delete",
    ) -> None:
        live = self._live.pop(sandbox.id, None)
        if live is not None:
            await live.proxy.stop()
        if mode == "keep":
            logger.info(
                "hfsandbox: keeping %s running; stop it with `hf jobs cancel %s`",
                sandbox.id, sandbox.id,
            )
            return
        # "stop" has no cheaper equivalent on HF Jobs (a stopped job still
        # bills), so both stop and delete cancel the job.
        if live is not None:
            await asyncio.to_thread(live.kill)
        else:
            from huggingface_hub import HfApi
            owner = (sandbox.metadata or {}).get("job_owner") or self._cfg.namespace
            await asyncio.to_thread(HfApi().cancel_job, job_id=sandbox.id, namespace=owner)
        logger.info("hfsandbox: cancelled %s", sandbox.id)

    def open_session(self, sandbox: SandboxHandle) -> Any:
        from cua_bench.computers.remote import RemoteDesktopSession

        from .gcloud import _init_computer_skip_wait

        session = RemoteDesktopSession(api_url=sandbox.endpoint, os_type=sandbox.os)
        _init_computer_skip_wait(session)
        return session


def _kill_quiet(sbx: Any) -> None:
    try:
        sbx.kill()
    except Exception as e:  # noqa: BLE001
        logger.warning("hfsandbox: kill failed for %s: %s", getattr(sbx, "id", "?"), e)
    with contextlib.suppress(Exception):
        sbx.close()
