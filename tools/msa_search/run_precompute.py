#!/usr/bin/env python3
"""Online/local MSA homology precompute (shared dataset directory).

Hits and profiles live under one path per CSV::

  artifacts/msa/<dataset_id>/hits|profiles/...

If a seq_hash already has status=ok hits (from online **or** local), search is
skipped unless --force. Profiles similarly skip when .npy exists.

Local: --backend local --db-id rnacentral_active (see MANUAL_SETUP_LOCAL.md).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.msa_search.assets import (
    AssetRoot,
    TimingStats,
    append_hit_index,
    dump_timing_report,
    is_hit_ok,
    is_profile_ok,
    load_hit_json,
    merge_write_meta,
    unique_sequences_from_csv,
    write_hit_payload,
    write_profile,
)
from tools.msa_search.contract import (
    CACHE_KEY,
    dataset_id_from_csv,
    file_sha256,
    resolve_dataset_asset_dir,
    seq_hash,
)
from tools.msa_search.extract_profile import profile_from_hit_record


def _resolve_csv(path: Path) -> Path:
    if path.is_file():
        return path.resolve()
    cand = (ROOT / "dataset" / path.name).resolve()
    if cand.is_file():
        return cand
    cand2 = (ROOT / "dataset" / path).resolve()
    if cand2.is_file():
        return cand2
    raise FileNotFoundError(f"CSV not found: {path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--backend",
        choices=("online", "local"),
        default="online",
        help="online=RNAcentral API; local=PyHMMER (after MANUAL_SETUP_LOCAL.md)",
    )
    p.add_argument(
        "--db-id",
        default="rnacentral_active",
        help="local DB id under artifacts/msa_dbs/<db_id>/ (ignored for online)",
    )
    p.add_argument(
        "--db-root",
        type=Path,
        default=ROOT / "artifacts" / "msa_dbs",
        help="parent dir for local sequence DBs",
    )
    p.add_argument(
        "--stage",
        choices=("hits", "profiles", "all"),
        default="all",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("HEK_te_prediction_task.csv"),
        help="CSV path or filename under dataset/",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="override asset dir (default: artifacts/msa/<dataset_id>)",
    )
    p.add_argument(
        "--asset-root",
        type=Path,
        default=ROOT / "artifacts" / "msa",
        help="parent of per-dataset dirs",
    )
    p.add_argument("--limit", type=int, default=None, help="max unique seqs (default: all)")
    p.add_argument("--seq-col", default="input_seq")
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--force", action="store_true")
    p.add_argument("--submit-interval", type=float, default=6.5)
    p.add_argument("--poll-interval", type=float, default=10.0)
    p.add_argument("--database", action="append", default=[])
    p.add_argument(
        "--cpus",
        type=int,
        default=0,
        help="PyHMMER worker threads (0=auto); online ignores",
    )
    p.add_argument(
        "--no-preload-db",
        action="store_true",
        help="local: stream fasta each query (less RAM, slower)",
    )
    p.add_argument(
        "--evalue",
        type=float,
        default=10.0,
        help="local nhmmer report E-value threshold",
    )
    return p.parse_args()


def _make_backend(args: argparse.Namespace):
    if args.backend == "online":
        from tools.msa_search.backends.online_rnacentral import RNAcentralOnlineBackend

        return RNAcentralOnlineBackend(
            databases=args.database,
            submit_interval_s=args.submit_interval,
            poll_interval_s=args.poll_interval,
        ), "online_rnacentral"

    from tools.msa_search.backends.local_pyhmmer import (
        LocalPyhmmerBackend,
        resolve_local_fasta,
    )

    fasta = resolve_local_fasta(args.db_root, args.db_id)
    backend = LocalPyhmmerBackend(
        fasta_path=fasta,
        db_id=args.db_id,
        cpus=args.cpus,
        top_k=args.top_k,
        E=args.evalue,
        preload=not args.no_preload_db,
    )
    return backend, "local_pyhmmer"


def _run_hits(args: argparse.Namespace, root: AssetRoot, sequences: list[str]) -> int:
    """Search only missing hashes; online and local share the same hits/ dir."""
    try:
        backend, backend_name = _make_backend(args)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: cannot init backend {args.backend}: {exc}", file=sys.stderr)
        return 2

    if args.backend == "online":
        try:
            import requests  # noqa: F401
        except ImportError:
            print("ERROR: needs requests", file=sys.stderr)
            return 2
    else:
        try:
            import pyhmmer  # noqa: F401
        except ImportError:
            print(
                "ERROR: needs pyhmmer (conda install -c conda-forge pyhmmer)",
                file=sys.stderr,
            )
            return 2

    timing = TimingStats()
    ok_n = skip_n = fail_n = 0
    print(
        f"[hits] backend={args.backend} n={len(sequences)} out={root.root} "
        f"force={args.force} (skip if hits already ok from any backend)",
        flush=True,
    )
    if args.backend == "local":
        print(
            f"[hits] db_id={args.db_id} db_root={args.db_root} "
            f"preload={not args.no_preload_db} cpus={args.cpus}",
            flush=True,
        )

    for i, seq in enumerate(sequences, start=1):
        h = seq_hash(seq)
        if not args.force:
            done, via = is_hit_ok(root, h)
            if done:
                skip_n += 1
                print(
                    f"[{i}/{len(sequences)}] skip existing hit {h[:12]}… "
                    f"(via={via})",
                    flush=True,
                )
                continue
        print(f"[{i}/{len(sequences)}] search {h[:12]}… L={len(seq)}", flush=True)
        rec = backend.search_one(seq)
        write_hit_payload(root, h, rec)
        append_hit_index(
            root,
            h=h,
            status=rec["status"],
            n_hits=int(rec.get("hits") or 0),
            elapsed_s=float(rec.get("elapsed_s") or 0.0),
            backend=rec.get("backend", backend_name),
            job_id=str(rec.get("job_id") or ""),
            seq_len=int(rec.get("seq_len") or len(seq)),
            md5=str(rec.get("rnacentral_md5") or ""),
        )
        elapsed = float(rec.get("elapsed_s") or 0.0)
        timing.add(elapsed)
        if rec["status"] == "ok":
            ok_n += 1
            print(f"  ok hits={rec.get('hits')} elapsed_s={elapsed:.1f}", flush=True)
        else:
            fail_n += 1
            print(f"  FAIL status={rec['status']} err={rec.get('error')!r}", flush=True)

    dump_timing_report(
        root.root / "hits" / "timing_report.json",
        timing,
        backend=backend_name,
        ok=ok_n,
        skip=skip_n,
        fail=fail_n,
        stage="hits",
        dataset_id=args.dataset_id,
    )
    print("[hits] timing:", timing.summary())
    print(f"[hits] done ok={ok_n} skip={skip_n} fail={fail_n}")
    return fail_n


def _run_profiles(args: argparse.Namespace, root: AssetRoot, sequences: list[str]) -> int:
    timing = TimingStats()
    ok_n = skip_n = fail_n = missing_hits = 0
    print(
        f"[profiles] dataset_id={args.dataset_id} n={len(sequences)} "
        f"top_k={args.top_k} force={args.force} "
        f"(skip if profile npy exists)",
        flush=True,
    )

    for i, seq in enumerate(sequences, start=1):
        h = seq_hash(seq)
        if not args.force and is_profile_ok(root, h):
            skip_n += 1
            print(f"[{i}/{len(sequences)}] skip existing profile {h[:12]}…")
            continue
        hit_path = root.hits_dir / f"{h}.json"
        if not hit_path.is_file():
            skip_n += 1
            missing_hits += 1
            if missing_hits <= 5:
                print(
                    f"[{i}/{len(sequences)}] skip no hits yet {h[:12]}… "
                    f"(run --stage hits first)",
                    flush=True,
                )
            elif missing_hits == 6:
                print(
                    f"[{i}/{len(sequences)}] … suppressing further 'no hits' lines "
                    f"(will summarize)",
                    flush=True,
                )
            continue
        t0 = time.time()
        try:
            rec = load_hit_json(root, h)
            if rec.get("status") != "ok":
                fail_n += 1
                print(
                    f"[{i}/{len(sequences)}] FAIL hit status={rec.get('status')} "
                    f"{h[:12]}…"
                )
                continue
            profile, meta = profile_from_hit_record(rec, top_k=args.top_k)
            write_profile(
                root,
                h,
                profile,
                status="ok",
                n_hits=int(meta.get("n_hits_used") or 0),
                n_eff=float(meta.get("n_eff") or 0.0),
                backend=str(rec.get("backend") or "unknown"),
            )
            elapsed = time.time() - t0
            timing.add(elapsed)
            ok_n += 1
            print(
                f"[{i}/{len(sequences)}] ok L={profile.shape[0]} "
                f"hits_used={meta.get('n_hits_used')} n_eff={meta.get('n_eff'):.2f} "
                f"elapsed_s={elapsed:.3f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            fail_n += 1
            print(f"[{i}/{len(sequences)}] FAIL {h[:12]}… {exc!r}", flush=True)

    dump_timing_report(
        root.root / "profiles" / "timing_report.json",
        timing,
        stage="profiles",
        ok=ok_n,
        skip=skip_n,
        fail=fail_n,
        top_k=args.top_k,
        dataset_id=args.dataset_id,
    )
    print("[profiles] timing:", timing.summary())
    print(
        f"[profiles] done ok={ok_n} skip={skip_n} fail={fail_n} "
        f"missing_hits={missing_hits}"
    )
    if missing_hits:
        print(
            f"[profiles] NOTE: {missing_hits}/{len(sequences)} sequences have no "
            f"hits JSON yet. Search first, e.g.\n"
            f"  python tools/msa_search/run_precompute.py --backend local "
            f"--db-id {getattr(args, 'db_id', 'rnacentral_active')} "
            f"--csv {getattr(args, 'csv', 'HEK_te_prediction_task.csv')} "
            f"--stage hits",
            flush=True,
        )
    # missing hits are not fatal for profiles stage (partial coverage OK for train zeros)
    return fail_n


def main() -> int:
    args = _parse_args()
    csv_path = _resolve_csv(args.csv)
    dataset_id = dataset_id_from_csv(csv_path)
    args.dataset_id = dataset_id

    if args.out is not None:
        out_dir = args.out.resolve()
    else:
        out_dir = resolve_dataset_asset_dir(
            repo_root=ROOT,
            dataset_id=dataset_id,
            asset_root=args.asset_root,
        )

    sequences = unique_sequences_from_csv(
        csv_path, seq_col=args.seq_col, limit=args.limit
    )
    if not sequences:
        print("ERROR: no sequences", file=sys.stderr)
        return 1

    root = AssetRoot(out_dir)
    root.ensure_layout()
    csv_hash = file_sha256(csv_path)
    backend_label = (
        "online_rnacentral" if args.backend == "online" else "local_pyhmmer"
    )
    merge_write_meta(
        root,
        backend=backend_label,
        dataset_id=dataset_id,
        csv_name=csv_path.name,
        csv_path=str(csv_path),
        csv_sha256=csv_hash,
        top_k=args.top_k,
        db_id=getattr(args, "db_id", "") or "",
        extra={
            "limit": args.limit,
            "stage": args.stage,
            "shared_hits_dir": True,
            "skip_existing_hits": True,
        },
    )
    print(
        f"dataset_id={dataset_id} csv={csv_path.name} out={out_dir} "
        f"backend={args.backend}"
    )

    fails = 0
    if args.stage in ("hits", "all"):
        hits_code = _run_hits(args, root, sequences)
        if hits_code == 2:
            print(
                "ERROR: hits stage could not start (dependency/DB). "
                "Not running profiles. Activate conda myenv3.11 and check "
                "artifacts/msa_dbs/<db_id>/seqs.fasta",
                file=sys.stderr,
            )
            return 2
        fails += hits_code
    if args.stage in ("profiles", "all"):
        fails += _run_profiles(args, root, sequences)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
