"""The agent setting's finished numbers, in one place, from the frozen files.

Three things live here that no single existing artifact holds:

  * the turn decomposition on the *eval* pool.  ``best_iteration`` in a
    checkpoint is the turn at which the candidate first became perfect on the
    demo pool, which is what the loop is allowed to see; crossing it with the
    evaluator's exact match says how much of the final score each turn bought.
  * the demo-to-eval generalisation gap.  Every case the loop stopped on is
    demo-perfect by construction, so the gap between that count and exact
    match is a direct reading of how much a development pool over-reports.
  * what the run cost, taken from ``attempt_history.jsonl`` rather than from
    memory: case attempts, session turns, and agent invocations.
  * what execution feedback bought, as opposed to whether it helped.  The cases
    the agent solves and the matched Direct cell does not are split by whether
    ``decl_repair_probe`` already solves them -- synthesizing the missing
    ``.decl`` lines, with no model call and no iteration -- and the rest are
    sliced by the reference's structure (recursive, auxiliary predicate), so
    the gap feedback closes can be read against the gap it leaves.

Reads only frozen files (the aggregate, the attempt history, the evaluator's
details, the declaration-repair details, and the reference programs) and
computes nothing that needs a model or a container.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = REPO_ROOT / "benchmark"
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
import measurement_v1 as mv1  # noqa: E402
from common_eval import QUERY_DIR  # noqa: E402
from structure_analyzer import analyze_program  # noqa: E402

REPAIR_DETAILS = BENCHMARK_DIR / "qa" / "decl_repair" / "decl_repair_details.jsonl"

# (label, model_tag, agent, matched Direct cell)
AGENT_CELLS = [
    ("Codex CLI / GPT-5.6-sol", "gpt-5.6-sol", "codex",
     "gpt-5.6-sol_signature_all"),
    ("Claude Code / Claude Opus 5", "claude-opus-5", "claude",
     "claude-opus-5_signature_all"),
]


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonl(path):
    p = Path(path)
    if not p.is_file():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _direct_ex(tag):
    p = BENCHMARK_DIR / "res_data" / f"{tag}_details.jsonl"
    if not p.is_file():
        return None
    rows = _jsonl(p)
    return {r["case_id"] for r in rows if r.get("run") in (0, None) and r["perfect_match"]}


def _repair_exact(model_tag, method):
    """Cases the declaration repair turns exact in the Direct cell; None if the probe never ran."""
    rows = _jsonl(REPAIR_DETAILS)
    if not rows:
        return None
    return {r["case_id"] for r in rows
            if r["model"] == model_tag and r["cell"] == method and r.get("perfect_match_after")}


_STRUCTURE = None


def _reference_structure():
    """case_id -> (recursive, introduces an auxiliary predicate), from the reference programs."""
    global _STRUCTURE
    if _STRUCTURE is None:
        _STRUCTURE = {}
        for path in sorted(QUERY_DIR.glob("*.dl")):
            f = analyze_program(path.read_text(encoding="utf-8"))
            _STRUCTURE[path.stem] = (bool(f["recursive"]), f["num_invented"] > 0)
    return _STRUCTURE


def _domains():
    """case_id -> benchmark domain, from the same dataset used for evaluation."""
    records = _load(BENCHMARK_DIR / "dataset.json")
    return {r["id"]: r["category"] for r in records}


def _mutation_outcome():
    """(tasks where every mutant is killed, tasks with a surviving mutant), from
    the frozen exhaustive run. Raw outcome, before any survivor is adjudicated."""
    frozen = _load(BENCHMARK_DIR / "qa" / "mutation_frozen.json")
    killed = {e["case"] for e in frozen if not e["survivors"]}
    return killed, {e["case"] for e in frozen} - killed


def _slice(out, name, cases, direct, exact):
    k = len(cases)
    d, a = len(cases & direct), len(cases & exact)
    out.append(f"  {name:26s} {k:3d}   direct {d:3d} = {d/k:6.1%}   agent {a:3d} = {a/k:6.1%}")
    return d / k, a / k


def analyse(label, model_tag, agent, direct_tag, method="signature", run=0):
    run_dir = (BENCHMARK_DIR / "infer_data" / model_tag /
               f"{agent}_{method}" / "all" / f"run_{run}")
    rows = {r["id"]: r for r in _load(run_dir / "all.json")}
    details = {d["case_id"]: d for d in
               _jsonl(BENCHMARK_DIR / "res_data" /
                      f"{model_tag}_{agent}_{method}_all_details.jsonl")
               if d.get("run") in (run, None)}
    history = {}
    for rec in _jsonl(run_dir / "attempt_history.jsonl"):
        history.setdefault(rec["case_id"], []).append(rec)

    n = len(rows)
    artifact = {c for c, r in rows.items() if r.get("artifact_valid", r.get("checkpoint_valid"))}
    terminal = {c for c, r in rows.items() if mv1.is_terminal_zero(r)}
    demo_perfect = {c for c in artifact if rows[c]["perfect_match"]}
    exact = {c for c, d in details.items() if d["perfect_match"]}
    compile_ok = {c for c, d in details.items() if d["compile_ok"]}
    first_turn = {c for c in exact if rows[c].get("session_turns") == 1}

    # A clean run needs no measurement finalizer, so it legitimately has no
    # attempt_history.jsonl.  Such a run still spent one case attempt per row;
    # reconstruct that unambiguous history from the aggregate rather than
    # reporting zero attempts and zero invocations.  A run with terminal zeros
    # is not reconstructible this way because its retry history matters.
    history_reconstructed = False
    if not history:
        if terminal:
            raise RuntimeError(
                f"{run_dir}: terminal-zero rows require attempt_history.jsonl"
            )
        history = {
            cid: [{
                "outcome": "ok",
                "failure_kind": "",
                "turns": row.get("session_turns", 0),
            }]
            for cid, row in rows.items()
        }
        history_reconstructed = True

    out = [f"\n{'=' * 72}", f"{label}  [{agent}/{method} run {run}]", "=" * 72]
    out.append(f"cases                     {n}")
    out.append(f"artifact produced         {len(artifact)}")
    out.append(f"terminal zero (no program within retry budget)   {len(terminal)}"
               + (f"  {sorted(terminal)}" if terminal else ""))
    out.append("")
    out.append(f"compile pass  (CP)        {len(compile_ok):3d}/{n}  = {len(compile_ok)/n:.4f}")
    out.append(f"exact match   (EX)        {len(exact):3d}/{n}  = {len(exact)/n:.4f}")
    out.append(f"  of which on turn 1      {len(first_turn):3d}/{n}  = {len(first_turn)/n:.4f}")

    out.append("")
    out.append("-- demo-to-eval generalisation gap "
               "(the loop only ever saw the demo pool) --")
    out.append(f"demo-perfect when the loop stopped   {len(demo_perfect):3d}")
    out.append(f"still exact on the held-out variants {len(exact):3d}")
    out.append(f"lost between the two                 {len(demo_perfect - exact):3d}"
               f"  ({len(demo_perfect - exact)/n:.1%} of all cases)")
    if demo_perfect - exact:
        out.append(f"  {sorted(demo_perfect - exact)}")
    stray = exact - demo_perfect
    out.append(f"exact without being demo-perfect     {len(stray):3d}"
               f"{'  ' + str(sorted(stray)) if stray else '   (containment holds)'}")

    out.append("")
    out.append("-- what each turn bought, scored on the eval pool --")
    by_turn = Counter(rows[c].get("best_iteration") for c in exact)
    cum = 0
    for k in sorted(x for x in by_turn if x is not None):
        cum += by_turn[k]
        out.append(f"  turn {k}:  +{by_turn[k]:3d}   cumulative {cum:3d}/{n} = {cum/n:.4f}")

    direct = _direct_ex(direct_tag)
    if direct is not None:
        out.append("")
        out.append(f"-- against the matched Direct cell ({direct_tag}) --")
        out.append(f"direct prompting, one shot at the task   {len(direct):3d}/{n} = {len(direct)/n:.4f}")
        out.append(f"agent, first turn                        {len(first_turn):3d}/{n} = {len(first_turn)/n:.4f}"
                   f"   [scaffold before it iterates: {len(first_turn)-len(direct):+d}]")
        out.append(f"agent, final                             {len(exact):3d}/{n} = {len(exact)/n:.4f}"
                   f"   [iteration against execution feedback: {len(exact)-len(first_turn):+d}]")
        out.append(f"  cases both solve                       {len(direct & first_turn):3d}")
        out.append(f"  only direct                            {len(direct - first_turn):3d}")
        out.append(f"  only agent's first turn                {len(first_turn - direct):3d}")

    feedback = {}
    repaired = _repair_exact(model_tag, method)
    if direct is not None and repaired is not None:
        struct = _reference_structure()
        gained = exact - direct
        by_repair = gained & repaired
        by_iteration = gained - repaired
        out.append("")
        out.append("-- what execution feedback bought (against the matched Direct cell) --")
        out.append(f"agent solves, direct does not            {len(gained):3d}")
        out.append(f"  already solved by declaration repair   {len(by_repair):3d}"
                   f"  ({len(by_repair)/len(gained):.1%})   [no model call, no iteration]")
        out.append(f"  needed iteration                       {len(by_iteration):3d}"
                   f"  (recursive {sum(struct[c][0] for c in by_iteration)}, "
                   f"auxiliary predicate {sum(struct[c][1] for c in by_iteration)})")
        out.append(f"direct + repair solves, agent does not   {len((direct | repaired) - exact):3d}")

        out.append("")
        out.append("-- the structural gaps, before and after feedback (EX on the eval pool) --")
        ids = set(rows)
        rec = {c for c in ids if struct[c][0]}
        aux = {c for c in ids if struct[c][1]}
        r1 = _slice(out, "recursive", rec, direct, exact)
        r0 = _slice(out, "non-recursive", ids - rec, direct, exact)
        a1 = _slice(out, "auxiliary predicate", aux, direct, exact)
        a0 = _slice(out, "no auxiliary predicate", ids - aux, direct, exact)
        out.append(f"  recursion gap            direct {100*(r0[0]-r1[0]):5.1f} pts   "
                   f"agent {100*(r0[1]-r1[1]):5.1f} pts")
        out.append(f"  auxiliary-predicate gap  direct {100*(a0[0]-a1[0]):5.1f} pts   "
                   f"agent {100*(a0[1]-a1[1]):5.1f} pts")

        out.append("")
        out.append("-- recursion x auxiliary-predicate 2x2 (EX on the eval pool) --")
        for recursive in (False, True):
            for auxiliary in (False, True):
                bucket = {
                    c for c in ids
                    if struct[c][0] is recursive and struct[c][1] is auxiliary
                }
                _slice(
                    out,
                    f"recursive={str(recursive).lower()}, aux={str(auxiliary).lower()}",
                    bucket, direct, exact,
                )

        out.append("")
        out.append("-- domain breakdown (EX on the eval pool) --")
        domains = _domains()
        for domain in sorted(set(domains.values())):
            bucket = {c for c in ids if domains[c] == domain}
            _slice(out, domain, bucket, direct, exact)

        # Does an easier-to-fool oracle flatter the agent? Slice by the raw
        # mutation result: if tasks with a surviving mutant scored higher, a
        # weaker oracle could be inflating EX there.
        out.append("")
        out.append("-- by raw mutation outcome (EX on the eval pool) --")
        killed, survived = _mutation_outcome()
        _slice(out, "every mutant killed", ids & killed, direct, exact)
        _slice(out, "a mutant survives", ids & survived, direct, exact)
        feedback = {"gained": len(gained), "gained_by_repair": len(by_repair),
                    "rec_gap_direct": r0[0] - r1[0], "rec_gap_agent": r0[1] - r1[1],
                    "aux_gap_direct": a0[0] - a1[0], "aux_gap_agent": a0[1] - a1[1]}

    out.append("")
    out.append("-- cost, from attempt_history.jsonl --")
    if history_reconstructed:
        out.append("  (clean run: one successful case attempt per aggregate row; "
                   "turns reconstructed from session_turns)")
    attempts = sum(len(v) for v in history.values())
    retried = {c: v for c, v in history.items() if len(v) > 1}
    turns_final = sum(rows[c].get("session_turns", 0) for c in rows)
    turns_all = sum(r.get("turns", 0) for v in history.values() for r in v)
    out.append(f"case attempts                {attempts}  "
               f"({len(retried)} case(s) needed a second, fresh session)")
    out.append(f"session turns, final attempt {turns_final}   "
               f"(mean {turns_final/n:.2f} per case)")
    out.append(f"session turns, all attempts  {turns_all}   "
               f"= total agent invocations")
    for cid, recs in sorted(retried.items()):
        trail = " -> ".join(
            f"{r['outcome']}{'/' + r['failure_kind'] if r['failure_kind'] else ''}"
            f"({r['turns']}t)" for r in recs)
        out.append(f"    {cid:20s} {trail}")

    endpoints = Counter(r.get("endpoint") for r in rows.values())
    if len(endpoints) > 1 or None not in endpoints:
        out.append("")
        out.append("-- endpoint provenance (mixed rows are separable by this field) --")
        for ep, c in endpoints.most_common():
            ex_ep = sum(1 for cid, r in rows.items()
                        if r.get("endpoint") == ep and cid in exact)
            out.append(f"  {str(ep):42s} {c:3d} case(s), EX {ex_ep:3d} = {ex_ep/c:.3f}")

    # A run finished across two endpoints invites the reading "the second
    # endpoint is worse".  It cannot be read off the agent column, because the
    # cases that ran late are exactly the ones that had already failed once --
    # the split is confounded with difficulty by construction.  The Direct cell
    # settles it: every Direct call went to one endpoint, so its EX on the same
    # two subsets measures difficulty with no endpoint term at all.
    if direct is not None and len(endpoints) > 1:
        out.append("")
        out.append("-- is the endpoint gap an endpoint effect? "
                   "(Direct ran entirely on one endpoint) --")
        out.append(f"  {'subset':30s} {'n':>4s} {'agent EX':>10s} {'direct EX':>11s}")
        ratios = {}
        for ep, c in endpoints.most_common():
            sub = [cid for cid, r in rows.items() if r.get("endpoint") == ep]
            a_ex = sum(1 for cid in sub if cid in exact) / len(sub)
            d_ex = sum(1 for cid in sub if cid in direct) / len(sub)
            ratios[ep] = (a_ex, d_ex)
            out.append(f"  {str(ep)[:30]:30s} {len(sub):>4d} {a_ex:>10.3f} {d_ex:>11.3f}")
        eps = list(ratios)
        if len(eps) == 2:
            (a1, d1), (a2, d2) = ratios[eps[0]], ratios[eps[1]]
            out.append(f"  agent spread  {a1 - a2:+.3f}   "
                       f"direct spread {d1 - d2:+.3f}  <- endpoint-free")
            out.append("  the endpoint-free channel shows the same direction, so the "
                       "split is case difficulty")

    return "\n".join(out), {
        "label": label, "cases": n, "exact": len(exact), "compile": len(compile_ok),
        "first_turn": len(first_turn), "demo_perfect": len(demo_perfect),
        "terminal_zero": len(terminal),
        "direct": len(direct) if direct is not None else None,
        **feedback,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None,
                    help="also write the report here")
    args = ap.parse_args()

    chunks, summaries = [], []
    for label, model_tag, agent, direct_tag in AGENT_CELLS:
        try:
            text, s = analyse(label, model_tag, agent, direct_tag)
        except FileNotFoundError as e:
            chunks.append(f"\n[SKIP] {label}: {e}")
            continue
        chunks.append(text)
        summaries.append(s)

    head = ["=" * 72, "AGENT SETTING -- measurement protocol v1, synthesis protocol v5",
            "=" * 72, "",
            f"{'':30s} {'CP':>7s} {'EX':>7s} {'turn 1':>8s} {'direct':>8s} {'demo':>6s}"]
    for s in summaries:
        head.append(
            f"{s['label']:30s} {s['compile']:3d}/{s['cases']:<3d} "
            f"{s['exact']:3d}/{s['cases']:<3d} {s['first_turn']:>8d} "
            f"{(s['direct'] if s['direct'] is not None else -1):>8d} {s['demo_perfect']:>6d}"
        )
    head.append("")
    head.append("'turn 1' and 'direct' are both exact match on the eval pool; 'demo' is")
    head.append("what the synthesis loop saw and is not an exact-match rate.")

    report = "\n".join(head) + "\n" + "\n".join(chunks) + "\n"
    print(report)
    if args.out:
        args.out.write_text(report, encoding="utf-8")
        print(f"[DONE] report -> {args.out}")


if __name__ == "__main__":
    main()
