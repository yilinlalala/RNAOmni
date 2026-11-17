#!/usr/bin/env python3
"""Unified agent transport for RNAOmni (Cursor SDK + OpenCode).

Used by ``orchestrator.py`` / ``deploy_feature.py`` via ``prompt()``.
Default backend is OpenCode; set ``AGENT_BACKEND=cursor`` to keep Cursor SDK.

Exit-code contract (matches orchestrator.run_once):
  0  — ok
  1  — startup / transport client failure (counts as agent failure)
  2  — run finished with error status (counts as agent failure)
  75 — transport dead; process restart recommended (not agent failure)
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tools.user_env import load_user_env
from rnaomni.paths import REPO

load_user_env(REPO)

BackendName = Literal["cursor", "opencode"]

EXIT_OK = 0
EXIT_STARTUP_FAIL = 1
EXIT_RUN_FAIL = 2
EXIT_TRANSPORT_DEAD = 75

DEFAULT_BACKEND: BackendName = "opencode"
DEFAULT_CURSOR_MODEL = "composer-2.5"
DEFAULT_OPENCODE_MODEL = "opencode/deepseek-v4-flash-free"
DEFAULT_OPENCODE_BIN = "opencode"


@dataclass
class AgentResult:
    """Normalized one-shot agent outcome."""

    status: Literal["ok", "error"]
    result: str | None = None
    message: str | None = None
    transport_dead: bool = False
    backend: BackendName = "cursor"
    model: str = ""
    exit_code: int = EXIT_OK

    def as_exit_code(self) -> int:
        return self.exit_code


def resolve_backend(name: str | None = None) -> BackendName:
    raw = (name if name is not None else os.environ.get("AGENT_BACKEND", DEFAULT_BACKEND))
    raw = (raw or DEFAULT_BACKEND).strip().lower()
    if raw not in ("cursor", "opencode"):
        raise ValueError(f"Unknown AGENT_BACKEND={raw!r}; expected 'cursor' or 'opencode'")
    return raw  # type: ignore[return-value]


def resolve_model(backend: BackendName, model: str | None = None) -> str:
    if model:
        return model
    if backend == "cursor":
        return os.environ.get("CURSOR_MODEL", DEFAULT_CURSOR_MODEL)
    return os.environ.get("OPENCODE_MODEL", DEFAULT_OPENCODE_MODEL)


def resolve_opencode_bin() -> str:
    override = os.environ.get("OPENCODE_BIN", "").strip()
    if override:
        return override
    found = shutil.which(DEFAULT_OPENCODE_BIN)
    if found:
        return found
    home = Path.home() / ".opencode" / "bin" / "opencode"
    if home.is_file():
        return str(home)
    return DEFAULT_OPENCODE_BIN


def is_bridge_dead_error(message: str) -> bool:
    """True when local Cursor SDK bridge is dead (process restart)."""
    msg = (message or "").lower()
    if "connection refused" in msg or "errno 111" in msg:
        return True
    if "connection reset" in msg or "errno 104" in msg:
        return True
    if "bridge" in msg and (
        "timed out" in msg or "timeout" in msg or "readtimeout" in msg
    ):
        return True
    return False


def is_opencode_transport_dead(message: str, returncode: int | None = None) -> bool:
    """True when OpenCode CLI/server is unreachable (process restart)."""
    msg = (message or "").lower()
    if "connection refused" in msg or "errno 111" in msg:
        return True
    if "connection reset" in msg or "errno 104" in msg:
        return True
    if "timed out" in msg or "timeout" in msg or "readtimeout" in msg:
        # attach to dead serve / hung local server
        if "attach" in msg or "server" in msg or "connect" in msg:
            return True
    if "no such file" in msg and "opencode" in msg:
        return True
    if returncode == 127:  # command not found
        return True
    return False


def prompt(
    text: str,
    *,
    cwd: str | Path,
    model: str | None = None,
    backend: str | None = None,
    api_key: str | None = None,
    auto_approve: bool = True,
    attach: str | None = None,
) -> AgentResult:
    """Run one agent turn via the selected backend."""
    be = resolve_backend(backend)
    mdl = resolve_model(be, model)
    cwd_s = str(Path(cwd).resolve())
    if be == "cursor":
        key = api_key if api_key is not None else os.environ.get("CURSOR_API_KEY")
        return run_cursor(text, cwd=cwd_s, model=mdl, api_key=key)
    return run_opencode(
        text,
        cwd=cwd_s,
        model=mdl,
        auto_approve=auto_approve,
        attach=attach or os.environ.get("OPENCODE_BASE_URL") or None,
    )


def complete_isolated(
    text: str,
    *,
    model: str | None = None,
    backend: str | None = None,
    api_key: str | None = None,
    timeout: float | None = 120.0,
) -> AgentResult:
    """JSON/chat completion in an empty temp cwd so the model cannot edit the repo.

    Same OPENCODE_MODEL / CURSOR_MODEL as Iter. Not an overnight tool session.
    """
    be = resolve_backend(backend)
    mdl = resolve_model(be, model)
    tmp = tempfile.mkdtemp(prefix="rnaomni_note_")
    try:
        if be == "cursor":
            key = api_key if api_key is not None else os.environ.get("CURSOR_API_KEY")
            return run_cursor(text, cwd=tmp, model=mdl, api_key=key)
        return run_opencode(
            text,
            cwd=tmp,
            model=mdl,
            auto_approve=True,
            attach=os.environ.get("OPENCODE_BASE_URL") or None,
            timeout=timeout,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_cursor(
    text: str,
    *,
    cwd: str,
    model: str,
    api_key: str | None,
) -> AgentResult:
    """Cursor SDK one-shot (Agent.prompt)."""
    if not api_key:
        return AgentResult(
            status="error",
            message="CURSOR_API_KEY is not set.",
            backend="cursor",
            model=model,
            exit_code=EXIT_STARTUP_FAIL,
        )
    try:
        from cursor_sdk import Agent, AgentOptions, CursorAgentError, LocalAgentOptions
    except ImportError as err:
        return AgentResult(
            status="error",
            message=f"cursor_sdk import failed: {err}",
            backend="cursor",
            model=model,
            exit_code=EXIT_STARTUP_FAIL,
        )

    logging.info("Starting Cursor Agent.prompt (model=%s cwd=%s)", model, cwd)
    try:
        result = Agent.prompt(
            text,
            AgentOptions(
                api_key=api_key,
                model=model,
                local=LocalAgentOptions(cwd=cwd),
            ),
        )
    except CursorAgentError as err:
        dead = is_bridge_dead_error(err.message)
        if dead:
            logging.error(
                "Bridge dead: %s (retryable=%s) — exit %s",
                err.message,
                err.is_retryable,
                EXIT_TRANSPORT_DEAD,
            )
            return AgentResult(
                status="error",
                message=err.message,
                transport_dead=True,
                backend="cursor",
                model=model,
                exit_code=EXIT_TRANSPORT_DEAD,
            )
        logging.error(
            "Cursor startup failed: %s (retryable=%s)",
            err.message,
            err.is_retryable,
        )
        return AgentResult(
            status="error",
            message=err.message,
            backend="cursor",
            model=model,
            exit_code=EXIT_STARTUP_FAIL,
        )

    status = getattr(result, "status", None) or "ok"
    body = getattr(result, "result", None)
    if status == "error":
        logging.error("Cursor run failed after startup.")
        return AgentResult(
            status="error",
            result=body if isinstance(body, str) else None,
            message="Run failed after startup.",
            backend="cursor",
            model=model,
            exit_code=EXIT_RUN_FAIL,
        )
    return AgentResult(
        status="ok",
        result=body if isinstance(body, str) else None,
        backend="cursor",
        model=model,
        exit_code=EXIT_OK,
    )


def build_opencode_cmd(
    text: str,
    *,
    cwd: str,
    model: str,
    bin_path: str | None = None,
    auto_approve: bool = True,
    attach: str | None = None,
) -> list[str]:
    """Build ``opencode run`` argv (no network)."""
    bin_path = bin_path or resolve_opencode_bin()
    cmd: list[str] = [bin_path, "run", "--dir", cwd, "-m", model]
    if auto_approve:
        cmd.append("--auto")
    if attach:
        cmd.extend(["--attach", attach])
        password = os.environ.get("OPENCODE_SERVER_PASSWORD")
        username = os.environ.get("OPENCODE_SERVER_USERNAME")
        if password:
            cmd.extend(["-p", password])
        if username:
            cmd.extend(["-u", username])
    cmd.append(text)
    return cmd


def run_opencode(
    text: str,
    *,
    cwd: str,
    model: str,
    auto_approve: bool = True,
    attach: str | None = None,
    timeout: float | None = None,
) -> AgentResult:
    """OpenCode CLI one-shot (``opencode run``). Path A."""
    bin_path = resolve_opencode_bin()
    if not shutil.which(bin_path) and not Path(bin_path).is_file():
        msg = f"opencode binary not found: {bin_path}"
        logging.error(msg)
        return AgentResult(
            status="error",
            message=msg,
            transport_dead=True,
            backend="opencode",
            model=model,
            exit_code=EXIT_TRANSPORT_DEAD,
        )

    cmd = build_opencode_cmd(
        text,
        cwd=cwd,
        model=model,
        bin_path=bin_path,
        auto_approve=auto_approve,
        attach=attach,
    )
    # Log without dumping full overnight prompt.
    logging.info(
        "Starting opencode run (model=%s cwd=%s bin=%s auto=%s attach=%s prompt_chars=%d)",
        model,
        cwd,
        bin_path,
        auto_approve,
        attach,
        len(text),
    )
    env = os.environ.copy()
    # Prefer system CA when conda curl CA path is broken (common on this host).
    env.setdefault("CURL_CA_BUNDLE", "/etc/pki/tls/certs/ca-bundle.crt")
    env.setdefault("SSL_CERT_FILE", "/etc/pki/tls/certs/ca-bundle.crt")

    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as err:
        msg = f"opencode run timed out: {err}"
        logging.error(msg)
        return AgentResult(
            status="error",
            message=msg,
            transport_dead=True,
            backend="opencode",
            model=model,
            exit_code=EXIT_TRANSPORT_DEAD,
        )
    except OSError as err:
        msg = f"opencode run failed to start: {err}"
        logging.error(msg)
        dead = is_opencode_transport_dead(str(err), returncode=None)
        return AgentResult(
            status="error",
            message=msg,
            transport_dead=dead,
            backend="opencode",
            model=model,
            exit_code=EXIT_TRANSPORT_DEAD if dead else EXIT_STARTUP_FAIL,
        )

    combined = "\n".join(
        part for part in (proc.stdout or "", proc.stderr or "") if part
    ).strip()
    if proc.returncode == 0:
        return AgentResult(
            status="ok",
            result=combined or None,
            backend="opencode",
            model=model,
            exit_code=EXIT_OK,
        )

    dead = is_opencode_transport_dead(combined, returncode=proc.returncode)
    # Auth / model errors are agent/config failures, not transport death.
    lower = combined.lower()
    if "invalid authentication" in lower or "unauthorized" in lower:
        dead = False
    if dead:
        logging.error(
            "OpenCode transport dead (rc=%s): %s — exit %s",
            proc.returncode,
            combined[:500],
            EXIT_TRANSPORT_DEAD,
        )
        return AgentResult(
            status="error",
            result=combined or None,
            message=combined or f"opencode exit {proc.returncode}",
            transport_dead=True,
            backend="opencode",
            model=model,
            exit_code=EXIT_TRANSPORT_DEAD,
        )

    logging.error("OpenCode run failed (rc=%s): %s", proc.returncode, combined[:500])
    return AgentResult(
        status="error",
        result=combined or None,
        message=combined or f"opencode exit {proc.returncode}",
        backend="opencode",
        model=model,
        # Nonzero after process started → treat like run failure (exit 2),
        # unless it looks like immediate startup (rc 1 with empty body → 1).
        exit_code=EXIT_STARTUP_FAIL if not combined else EXIT_RUN_FAIL,
    )


def _self_test() -> int:
    """Import / resolve / argv smoke (no live agent call)."""
    assert resolve_backend("cursor") == "cursor"
    assert resolve_backend("opencode") == "opencode"
    try:
        resolve_backend("nope")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    assert resolve_model("cursor", None)
    assert resolve_model("opencode", None) == os.environ.get(
        "OPENCODE_MODEL", DEFAULT_OPENCODE_MODEL
    )

    cmd = build_opencode_cmd(
        "hello",
        cwd="/tmp",
        model=DEFAULT_OPENCODE_MODEL,
        bin_path="/fake/opencode",
        auto_approve=True,
    )
    assert cmd[0] == "/fake/opencode"
    assert "run" in cmd
    assert "-m" in cmd and DEFAULT_OPENCODE_MODEL in cmd
    assert "--auto" in cmd
    assert cmd[-1] == "hello"

    assert is_bridge_dead_error("ConnectError: [Errno 111] Connection refused")
    assert is_opencode_transport_dead("connection refused", returncode=1)
    assert not is_opencode_transport_dead("Invalid Authentication", returncode=1)

    print("agent_backend self-test OK")
    print(f"  default_backend={DEFAULT_BACKEND}")
    print(f"  opencode_bin={resolve_opencode_bin()}")
    print(f"  default_opencode_model={DEFAULT_OPENCODE_MODEL}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RNAOmni agent backend utilities")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run local resolve/argv checks (no network).",
    )
    parser.add_argument(
        "--print-opencode-cmd",
        action="store_true",
        help="Print opencode run argv for a sample prompt (no execute).",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--cwd", default=str(Path.cwd()))
    parser.add_argument("--prompt", default="Reply with OK only.")
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()
    if args.print_opencode_cmd:
        cmd = build_opencode_cmd(
            args.prompt,
            cwd=str(Path(args.cwd).resolve()),
            model=resolve_model("opencode", args.model),
        )
        # Redact nothing; prompt is caller-provided sample.
        print(subprocess.list2cmdline(cmd))
        return 0
    parser.error("Specify --self-test or --print-opencode-cmd")
    return 2


if __name__ == "__main__":
    sys.exit(main())
