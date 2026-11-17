"""RNAcentral sequence-search online backend (P1).

API: https://rnacentral.org/sequence-search/api
Submit rate limit: 10 jobs / minute / IP (no limit on status/results).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import requests

from tools.msa_search.assets import make_hit_record
from tools.msa_search.contract import normalize_rna

DEFAULT_SERVER = "https://sequence-search.rnacentral.org/"
# Stay under 10 submits/min: default spacing 6.5s between submits.
DEFAULT_SUBMIT_INTERVAL_S = 6.5
DEFAULT_POLL_INTERVAL_S = 10.0
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_WAIT_S = 1800.0  # 30 min per job
DEFAULT_PAGE_SIZE = 100


class RNAcentralOnlineBackend:
    def __init__(
        self,
        *,
        server: str = DEFAULT_SERVER,
        databases: Optional[Sequence[str]] = None,
        submit_interval_s: float = DEFAULT_SUBMIT_INTERVAL_S,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        request_timeout_s: float = DEFAULT_TIMEOUT_S,
        max_wait_s: float = DEFAULT_MAX_WAIT_S,
        page_size: int = DEFAULT_PAGE_SIZE,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.server = server if server.endswith("/") else server + "/"
        self.databases: List[str] = list(databases) if databases else []
        self.submit_interval_s = float(submit_interval_s)
        self.poll_interval_s = float(poll_interval_s)
        self.request_timeout_s = float(request_timeout_s)
        self.max_wait_s = float(max_wait_s)
        self.page_size = int(page_size)
        self.session = session or requests.Session()
        self._last_submit_unix = 0.0

    def _url(self, path: str) -> str:
        return self.server + path.lstrip("/")

    def _throttle_submit(self) -> None:
        gap = self.submit_interval_s
        if gap <= 0:
            return
        now = time.time()
        wait = gap - (now - self._last_submit_unix)
        if wait > 0:
            time.sleep(wait)

    def submit(self, sequence: str) -> str:
        self._throttle_submit()
        seq = normalize_rna(sequence)
        payload = {"databases": self.databases, "sequence": seq}
        resp = self.session.post(
            self._url("api/submit-job"),
            json=payload,
            timeout=self.request_timeout_s,
        )
        self._last_submit_unix = time.time()
        if resp.status_code == 429:
            time.sleep(60.0)
            resp = self.session.post(
                self._url("api/submit-job"),
                json=payload,
                timeout=self.request_timeout_s,
            )
            self._last_submit_unix = time.time()
        resp.raise_for_status()
        data = resp.json()
        job_id = data.get("job_id")
        if not job_id:
            raise RuntimeError(f"submit-job missing job_id: {data!r}")
        return str(job_id)

    def wait_finished(self, job_id: str) -> Dict[str, Any]:
        t0 = time.time()
        while True:
            if time.time() - t0 > self.max_wait_s:
                return {"status": "timeout", "progress": None}
            try:
                resp = self.session.get(
                    self._url(f"api/job-status/{job_id}"),
                    timeout=self.request_timeout_s,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException:
                time.sleep(self.poll_interval_s)
                continue
            status = data.get("status", "unknown")
            if status == "finished":
                return data
            if status in ("error", "not_found"):
                return data
            time.sleep(self.poll_interval_s)

    def fetch_results(self, job_id: str) -> Dict[str, Any]:
        results: List[Any] = []
        page = 1
        hit_count: Optional[int] = None
        while True:
            url = self._url(
                f"api/job-results/{job_id}?page={page}&page_size={self.page_size}"
            )
            resp = self.session.get(url, timeout=self.request_timeout_s)
            resp.raise_for_status()
            data = resp.json()
            if hit_count is None:
                hit_count = int(data.get("hit_count", 0) or 0)
            entries = data.get("entries") or []
            results.extend(entries)
            if not entries or (hit_count is not None and len(results) >= hit_count):
                break
            page += 1
        return {"hit_count": hit_count or 0, "entries": results}

    def search_one(self, sequence: str) -> Dict[str, Any]:
        """Submit + poll + fetch; return hit record dict (contract schema)."""
        t0 = time.time()
        try:
            job_id = self.submit(sequence)
        except Exception as exc:  # noqa: BLE001 — persist failure for resume
            return make_hit_record(
                sequence,
                backend="online_rnacentral",
                status="submit_error",
                elapsed_s=time.time() - t0,
                error=str(exc),
            )

        status_payload = self.wait_finished(job_id)
        status = str(status_payload.get("status", "unknown"))
        if status != "finished":
            return make_hit_record(
                sequence,
                backend="online_rnacentral",
                status=status,
                job_id=job_id,
                elapsed_s=time.time() - t0,
                error=str(status_payload),
            )

        try:
            fetched = self.fetch_results(job_id)
        except Exception as exc:  # noqa: BLE001
            return make_hit_record(
                sequence,
                backend="online_rnacentral",
                status="fetch_error",
                job_id=job_id,
                elapsed_s=time.time() - t0,
                error=str(exc),
            )

        return make_hit_record(
            sequence,
            backend="online_rnacentral",
            status="ok",
            job_id=job_id,
            elapsed_s=time.time() - t0,
            hits=int(fetched["hit_count"]),
            results=fetched["entries"],
            extra={"job_status": status_payload},
        )
