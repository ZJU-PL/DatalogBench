"""Close a coding-agent run under measurement protocol v1.

Synthesis stops when it runs out of cases it is willing to retry.  What it
leaves behind is not yet a measurement: some cases hold a program, some hold
only the schema stub written when the agent returned nothing, and the failure
manifest lumps both "the transport died twice" and "our account is disabled"
into one ``failures`` list -- which is why the evaluator refuses the whole run.
This script turns that into a measurement: it decides, per case and by the
frozen rule in ``measurement_v1``, whether the case is scored from its artifact
or scored zero, and rewrites the manifest so the evaluator can tell the two
kinds of failure apart.

It does not call any model and it never scores anything itself.  The eval-pool
numbers come from ``eval_coding_agent.py`` afterwards, as they always have.

Idempotent: running it twice produces byte-identical artifacts and, in
particular, never re-opens the retry budget.  That property is the reason this
is a separate script rather than a flag on synthesis -- a finalizer that can be
re-run without consequence can be run before anyone is sure it is needed.

Typical use:

    python3 synthesis/agent_finalize_measurement.py \
        --agent claude --method signature --dataset all --run 0 \
        --attempt-log benchmark/qa/run_agent_v5_claude_aicode_ai_resume.log \
        --attempt-log benchmark/qa/run_agent_v5_claude_aicode_ai_retry7.log

``--attempt-log`` is given oldest first; the order is the attempt order.  Cases
whose row came from a checkpoint written before these logs count as one
attempt, since their artifact exists and its attempt history is whatever
produced it.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

SYNTHESIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SYNTHESIS_DIR.parent
BENCHMARK_DIR = REPO_ROOT / "benchmark"
sys.path.insert(0, str(SYNTHESIS_DIR))

import measurement_v1 as mv1  # noqa: E402

AGENT_PROTOCOL_VERSION = 5

_TURN_RE = re.compile(
    r"^\[AGENT:(?P<agent>[^\]]+)\]\[run (?P<run>\d+)\] (?P<case>\S+) iter=(?P<iter>\d+)/(?P<max>\d+)"
)
_FAIL_RE = re.compile(
    r"^\[AGENT-FAIL\]\[run (?P<run>\d+)\] (?P<case>\S+): (?P<kind>\S+) after (?P<turns>\d+) attempt"
)
_WARN_RE = re.compile(
    r"^\[WARN\] (?P<case>\S+) session unhealthy \((?P<kind>[^)]+)\)"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_attempt_log(path: Path, run_id: int):
    """Case attempts recorded in one synthesis log, in log order.

    A case that only appears as ``[RESUME] ... protocol-v5-checkpoint`` was not
    attempted in this run; it contributes no attempt here.  Elapsed time is not
    recoverable -- the harness does not timestamp these lines -- so it is left
    null rather than guessed from the log's mtime.
    """
    digest = _sha256(path)
    order = []
    attempts = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        m = _TURN_RE.match(line)
        if m and int(m.group("run")) == run_id:
            case = m.group("case")
            if case not in attempts:
                order.append(case)
                attempts[case] = {
                    "case_id": case,
                    "agent": m.group("agent"),
                    "run": run_id,
                    "turns": 0,
                    "max_iterations": int(m.group("max")),
                    "outcome": "ok",
                    "failure_kind": "",
                    "source_log": path.name,
                    "source_log_sha256": digest,
                    "elapsed_sec": None,
                }
            attempts[case]["turns"] = max(
                attempts[case]["turns"], int(m.group("iter"))
            )
            continue
        m = _WARN_RE.match(line)
        if m and m.group("case") in attempts:
            attempts[m.group("case")]["failure_kind"] = m.group("kind").strip()
            continue
        m = _FAIL_RE.match(line)
        if m and int(m.group("run")) == run_id and m.group("case") in attempts:
            rec = attempts[m.group("case")]
            rec["outcome"] = "failed"
            rec["failure_kind"] = m.group("kind")
    return [attempts[c] for c in order]


def build_attempt_history(rows, logs, run_id):
    """Per-case attempt list, oldest first.

    Cases with no line in any supplied log keep a single synthetic attempt
    describing the checkpoint they were resumed from: the artifact is real, but
    the log that produced it is not one of the frozen ones, so the record says
    so instead of inventing turn counts.
    """
    per_log = [parse_attempt_log(p, run_id) for p in logs]
    history = {}
    for attempts in per_log:
        for rec in attempts:
            history.setdefault(rec["case_id"], []).append(rec)

    out = {}
    for row in rows:
        cid = row["id"]
        recs = [dict(r) for r in history.get(cid, [])]
        if not recs:
            recs = [{
                "case_id": cid,
                "agent": row.get("agent"),
                "run": run_id,
                "turns": row.get("session_turns", 0),
                "max_iterations": row.get("max_iterations"),
                "outcome": "ok" if row.get("checkpoint_valid") else "failed",
                "failure_kind": row.get("failure_kind", ""),
                "source_log": None,
                "source_log_sha256": None,
                "elapsed_sec": None,
                "note": "reconstructed from the checkpoint; no frozen log covers this attempt",
            }]
        for n, rec in enumerate(recs, 1):
            rec["attempt"] = n
        out[cid] = recs
    return out


def finalize(agent, method, model_tag, dataset, run_id, logs, dry_run=False):
    scope = dataset or "all"
    run_dir = (BENCHMARK_DIR / "infer_data" / model_tag /
               f"{agent}_{method}" / scope / f"run_{run_id}")
    if not run_dir.is_dir():
        raise SystemExit(f"[ABORT] no such run directory: {run_dir}")

    progress_path = run_dir / "progress.json"
    aggregate_path = run_dir / f"{scope}.json"

    # progress.json wins when both exist.  The aggregate is written only on a
    # clean finish, so an aggregate sitting next to a progress file is a
    # leftover from an earlier, different run -- exactly the situation that put
    # a Sep-4 aggregate reporting 96 exact matches next to a Sep-9 progress
    # file reporting 123 on the same directory.
    if progress_path.is_file():
        rows = json.loads(progress_path.read_text(encoding="utf-8"))
        source = progress_path
    elif aggregate_path.is_file():
        rows = json.loads(aggregate_path.read_text(encoding="utf-8"))
        source = aggregate_path
    else:
        raise SystemExit(f"[ABORT] neither progress.json nor {scope}.json in {run_dir}")

    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"[ABORT] {source} is not a non-empty list of case rows")

    print(f"[SOURCE] {source} ({len(rows)} case rows)")
    for path in logs:
        print(f"[LOG]    {path.name}  sha256={_sha256(path)}")

    history = build_attempt_history(rows, logs, run_id)

    scored_terminal, invalid = [], []
    finalized = []
    for row in rows:
        cid = row["id"]
        attempts = len(history[cid])
        fields = mv1.classify_case(row, attempts)
        new_row = dict(row)
        new_row.update(fields)
        # The record's tp/fp/fn/f1/perfect_match come from the demo pool: it is
        # the only pool the synthesis loop is allowed to see under protocol v5.
        # The field said scoring_pool=eval and nothing said which pool the
        # numbers were from, so they were read as exact-match rates twice.
        new_row["metrics_pool"] = "demo"
        finalized.append(new_row)

        if fields["artifact_valid"]:
            continue
        entry = {
            "id": cid,
            "kind": row.get("failure_kind") or "invalid_checkpoint",
            "error": row.get("inference_error", ""),
            "case_attempts": attempts,
            "turns_in_final_attempt": row.get("attempts", 0),
        }
        if fields["score_policy"] == mv1.SCORE_POLICY_TERMINAL_ZERO:
            scored_terminal.append(entry)
        else:
            entry["reason"] = (
                f"failure kind {entry['kind']!r} is never scorable"
                if mv1.is_invalidating_kind(entry["kind"]) else
                "retry budget not exhausted"
            )
            invalid.append(entry)

    measurement_complete = all(r["measurement_complete"] for r in finalized)

    manifest_path = run_dir / "agent_failures.json"
    manifest = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) or {}
        except json.JSONDecodeError:
            manifest = {}

    new_manifest = dict(manifest)
    new_manifest.update({
        "agent": agent,
        "method": method,
        "run": run_id,
        "agent_protocol_version": AGENT_PROTOCOL_VERSION,
        "measurement_protocol_version": mv1.MEASUREMENT_PROTOCOL_VERSION,
        "complete_case_set": not manifest.get("not_attempted"),
        "measurement_complete": measurement_complete and not invalid,
        "scored_terminal_failures": scored_terminal,
        "invalid_failures": invalid,
        "attempt_log_sha256": {p.name: _sha256(p) for p in logs},
        "max_case_attempts": mv1.MAX_CASE_ATTEMPTS,
        "retryable_failure_kinds": sorted(mv1.RETRYABLE_FAILURE_KINDS),
    })
    # ``failures``/``complete`` are the pre-v1 keys.  They stay, holding the
    # union, so an older reader still sees every failure -- but they no longer
    # decide anything, because "there were failures" and "the run is
    # unmeasurable" stopped being the same statement.
    new_manifest["failures"] = scored_terminal + invalid
    new_manifest["complete"] = bool(measurement_complete and not invalid)

    print(f"\n[MEASUREMENT-v1] cases={len(finalized)} "
          f"artifact={sum(1 for r in finalized if r['artifact_valid'])} "
          f"terminal_zero={len(scored_terminal)} invalid={len(invalid)}")
    for e in scored_terminal:
        print(f"  [terminal-zero] {e['id']:20s} {e['kind']:16s} "
              f"case_attempts={e['case_attempts']}")
    for e in invalid:
        print(f"  [INVALID]       {e['id']:20s} {e['kind']:16s} {e['reason']}")

    retried = {c: h for c, h in history.items() if len(h) > 1}
    print(f"\n[ATTEMPTS] cases with more than one case attempt: {len(retried)}")
    for cid, recs in sorted(retried.items()):
        trail = " -> ".join(
            f"{r['outcome']}{'/' + r['failure_kind'] if r['failure_kind'] else ''}"
            f"({r['turns']} turns)" for r in recs
        )
        print(f"  {cid:20s} {trail}")

    if dry_run:
        print("\n[DRY-RUN] nothing written")
        return {
            "cases": len(finalized),
            "terminal_zero": len(scored_terminal),
            "invalid": len(invalid),
            "measurement_complete": new_manifest["measurement_complete"],
        }

    for row in finalized:
        path = run_dir / f"{row['id']}.agent.json"
        existing = {}
        if path.is_file():
            try:
                existing = json.loads(path.read_text(encoding="utf-8")) or {}
            except json.JSONDecodeError:
                existing = {}
        merged = dict(existing)
        merged.update({k: row[k] for k in (
            "artifact_valid", "measurement_complete", "score_policy",
            "retry_budget_exhausted", "case_attempts",
            "measurement_protocol_version", "metrics_pool",
        )})
        _atomic_json(path, merged)

    hist_path = run_dir / "attempt_history.jsonl"
    lines = []
    for cid in sorted(history):
        for rec in history[cid]:
            lines.append(json.dumps(rec, ensure_ascii=False, sort_keys=True))
    hist_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _atomic_json(manifest_path, new_manifest)
    _atomic_json(aggregate_path, finalized)

    # progress.json is the resume cursor.  Removing it once the measurement is
    # complete is what makes a second finalize (or a stray ``--resume``) a
    # no-op instead of a fresh attempt on cases whose budget is spent.
    if new_manifest["measurement_complete"]:
        progress_path.unlink(missing_ok=True)
        print(f"\n[DONE] aggregate -> {aggregate_path}")
        print(f"[DONE] attempts  -> {hist_path} ({len(lines)} attempt records)")
        print(f"[DONE] manifest  -> {manifest_path}")
        print("[DONE] progress.json removed; the retry budget is closed")
    else:
        print(f"\n[INCOMPLETE] {len(invalid)} invalid failure(s) remain; "
              "progress.json kept, resume synthesis before evaluating")

    return {
        "cases": len(finalized),
        "terminal_zero": len(scored_terminal),
        "invalid": len(invalid),
        "measurement_complete": new_manifest["measurement_complete"],
    }


def _atomic_json(path: Path, payload) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agent", required=True)
    ap.add_argument("--method", required=True, choices=["signature", "description"])
    ap.add_argument("--model-tag", default=None,
                    help="infer_data directory name; defaults to the agent's default model")
    ap.add_argument("--dataset", default="all")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--attempt-log", action="append", default=[], type=Path,
                    help="synthesis log, oldest first; repeatable")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    model_tag = args.model_tag
    if not model_tag:
        sys.path.insert(0, str(REPO_ROOT / "evaluation"))
        from model_inventory import resolve_default_model_for_agent
        model_tag = re.sub(r"[^A-Za-z0-9._-]+", "_",
                           resolve_default_model_for_agent(args.agent).strip())

    for p in args.attempt_log:
        if not p.is_file():
            raise SystemExit(f"[ABORT] no such attempt log: {p}")

    finalize(
        agent=args.agent, method=args.method, model_tag=model_tag,
        dataset=args.dataset, run_id=args.run, logs=args.attempt_log,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
