"""Git helpers for the overnight orchestrator."""

from __future__ import annotations

import subprocess

from rnaomni.paths import REPO

BRANCH_PREFIX = "RNAclaw/"


def git_branch() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_head() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_reset_hard(commit: str) -> None:
    subprocess.check_call(
        ["git", "reset", "--hard", commit],
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def head_unchanged(pre_head: str, post_head: str | None = None) -> bool:
    """True when the agent produced no new commit (HEAD still pre_head)."""
    post = git_head() if post_head is None else post_head
    return not post or post == pre_head
