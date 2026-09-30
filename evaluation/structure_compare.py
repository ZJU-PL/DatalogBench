"""Compare generated programs against the reference program, structurally.

Execution accuracy says whether a program got the right answer; it says nothing
about *how*. This answers the questions R2 asked -- does the model use recursion
when the task needs it, and does it reconstruct the intermediate concepts the
reference program introduces -- by diffing the structure of each generated `.dl`
against its golden.

Two families of metric:

**Recursion / negation / aggregation awareness.** A confusion matrix per
feature: the reference needs recursion and the generated program is recursive,
or it is not, and vice versa. The interesting cell is *needed but absent*: a
non-recursive answer to a fixpoint task is a specific, nameable failure, not an
undifferentiated wrong answer.

**Predicate invention precision / recall.** Matching by *name* would be
meaningless -- a model is free to call the transitive closure `TC`, `Reach` or
`Step` -- so predicates are matched **extensionally**: run both programs on the
same input with their intermediate relations exported, and call two predicates
the same concept when they derive the same tuples. Column permutations are
tried too, so a model that stores the transpose of the reference's relation is
credited rather than penalised for a cosmetic choice.

    python3 evaluation/structure_compare.py --model gpt-5.6-sol --tag signature
    python3 evaluation/structure_compare.py --model gpt-5.6-sol --tag signature_fs1 --run 0
    python3 evaluation/structure_compare.py --selftest       # no generated data needed

The self-test builds generated programs with known relationships to a real
golden (renamed predicate, transposed predicate, no predicate at all) and
checks the metrics come out as they must, so the tooling can be validated
before any rerun produces data.
"""

import argparse
import itertools
import json
import statistics
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "evaluation"))

from mutation_score import run as souffle_run  # noqa: E402
from structure_analyzer import analyze_program  # noqa: E402

QUERY_DIR = _REPO / "benchmark" / "query"
IO_DIR = _REPO / "benchmark" / "io_data"
INFER_DIR = _REPO / "benchmark" / "infer_data"

BOOL_FEATURES = ["recursive", "uses_negation", "uses_aggregation"]
MAX_PERM_ARITY = 5          # 120 permutations; beyond this, identity only


def _with_outputs(dl_text, rels):
    """Add `.output R` for relations that are declared but not exported.

    Intermediate relations are invisible by default; the whole point here is to
    observe them, so they are exported for this analysis run only.
    """
    existing = {line.split()[1].strip(". ")
                for line in dl_text.splitlines()
                if line.strip().startswith(".output") and len(line.split()) > 1}
    extra = [f".output {r}" for r in rels if r not in existing]
    if not extra:
        return dl_text
    return dl_text.rstrip("\n") + "\n" + "\n".join(extra) + "\n"


def extensions(dl_text, facts_dir, rels):
    """{relation: frozenset of tuples} for `rels`, or None if the program fails."""
    out = souffle_run(_with_outputs(dl_text, rels), facts_dir)
    if out is None:
        return None
    got = {}
    for r in rels:
        rows = out.get(f"{r}.csv")
        if rows is None:
            got[r] = frozenset()
        else:
            got[r] = frozenset(tuple(x.split("\t")) for x in rows if x.strip())
    return got


def _perm_equal(a, b):
    """True when some column permutation of `b` equals `a`.

    A model that stores `Reach(y, x)` where the reference stores `Reach(x, y)`
    has invented the same concept; only the argument order differs.
    """
    if a == b:
        return True
    if not a or not b or len(a) != len(b):
        return False
    arity = len(next(iter(a)))
    if arity != len(next(iter(b))) or arity > MAX_PERM_ARITY:
        return False
    for perm in itertools.permutations(range(arity)):
        if perm == tuple(range(arity)):
            continue
        if a == frozenset(tuple(t[i] for i in perm) for t in b):
            return True
    return False


def match_invented(gold_ext, gen_ext):
    """Greedy one-to-one matching of invented predicates by extension.

    Returns (matched_pairs, unmatched_gold, unmatched_gen). Empty extensions are
    never matched to each other: two predicates that both derived nothing on
    this input are not evidence of the same concept.
    """
    pairs, used = [], set()
    for g, gext in sorted(gold_ext.items()):
        if not gext:
            continue
        for c, cext in sorted(gen_ext.items()):
            if c in used or not cext:
                continue
            if _perm_equal(gext, cext):
                pairs.append((g, c))
                used.add(c)
                break
    matched_gold = {g for g, _ in pairs}
    return pairs, sorted(set(gold_ext) - matched_gold), sorted(set(gen_ext) - used)


def _variant_dir(case_id):
    """One scored input to compute predicate extensions on.

    Delegates to common_eval.case_variants_dirs rather than scanning io_data,
    for the same reason the evaluator does: under the eval/ + demo/ layout a
    scan for "a subdirectory holding .facts" finds neither (the facts live one
    level further down), returns None, and every case then reports matched=0 --
    which reads as "the model never reproduces the reference's invented
    concepts" rather than as "the extensions were never computed".
    """
    import sys as _sys
    _eval = str(Path(__file__).resolve().parent)
    if _eval not in _sys.path:
        _sys.path.insert(0, _eval)
    from common_eval import case_variants_dirs
    for v in case_variants_dirs(case_id):
        if any(v.glob("*.facts")):
            return v
    return None


def compare_case(case_id, gen_text, facts_dir=None):
    """Structural diff of one generated program against its reference."""
    gold_path = QUERY_DIR / f"{case_id}.dl"
    if not gold_path.exists():
        return {"case_id": case_id, "status": "no_golden"}
    gold_text = gold_path.read_text(errors="ignore")
    gold = analyze_program(gold_text)
    try:
        gen = analyze_program(gen_text)
    except Exception:
        return {"case_id": case_id, "status": "unparsable_generated"}

    # A program with no rules is not a structural choice. Counting it as
    # "did not use recursion" would inflate the recursion-awareness metric with
    # answers that are not programs at all -- the parser extracts nothing from
    # prose or a truncated response and would report exactly that shape.
    if gen["num_rules"] == 0:
        return {"case_id": case_id, "status": "empty_generated"}

    facts_dir = facts_dir or _variant_dir(case_id)
    if facts_dir is not None:
        # One run doubles as the compile check: structure is only meaningful
        # for a program Soufflé accepts.
        gen_ext = extensions(gen_text, facts_dir, gen["invented"])
        if gen_ext is None:
            return {"case_id": case_id, "status": "generated_failed"}
    else:
        gen_ext = None

    row = {"case_id": case_id, "status": "ok"}
    for f in BOOL_FEATURES:
        row[f"gold_{f}"] = bool(gold[f])
        row[f"gen_{f}"] = bool(gen[f])
    row["gold_max_scc"] = gold["max_scc_size"]
    row["gen_max_scc"] = gen["max_scc_size"]
    row["gold_invented"] = gold["invented"]
    row["gen_invented"] = gen["invented"]

    if gen_ext is None or not (gold["invented"] or gen["invented"]):
        # Nothing to match extensionally: report counts only.
        row["invention_status"] = "no_invented" if not (gold["invented"] or gen["invented"]) else "no_input"
        row["matched"] = 0
        return row

    gold_ext = extensions(gold_text, facts_dir, gold["invented"])
    if gold_ext is None:
        row["invention_status"] = "golden_failed"
        row["matched"] = 0
        return row

    pairs, miss_gold, extra_gen = match_invented(gold_ext, gen_ext)
    row["invention_status"] = "ok"
    row["matched"] = len(pairs)
    row["matches"] = pairs
    row["missed_concepts"] = miss_gold
    row["spurious_concepts"] = extra_gen
    return row


def _rate(num, den):
    return round(num / den, 4) if den else None


def summarize(rows):
    ok = [r for r in rows if r.get("status") == "ok"]
    out = {"cases": len(ok)}
    for f in BOOL_FEATURES:
        need = [r for r in ok if r[f"gold_{f}"]]
        no_need = [r for r in ok if not r[f"gold_{f}"]]
        out[f] = {
            "golden_needs_it": len(need),
            "used_when_needed": sum(1 for r in need if r[f"gen_{f}"]),
            "awareness_rate": _rate(sum(1 for r in need if r[f"gen_{f}"]), len(need)),
            "spurious": sum(1 for r in no_need if r[f"gen_{f}"]),
            "spurious_rate": _rate(sum(1 for r in no_need if r[f"gen_{f}"]), len(no_need)),
        }
    scored = [r for r in ok if r.get("invention_status") == "ok"]
    tm = sum(r["matched"] for r in scored)
    tg = sum(len(r["gold_invented"]) for r in scored)
    tc = sum(len(r["gen_invented"]) for r in scored)
    out["invention"] = {
        "cases_scored": len(scored),
        "golden_concepts": tg,
        "generated_concepts": tc,
        "matched": tm,
        "recall_micro": _rate(tm, tg),
        "precision_micro": _rate(tm, tc),
        "recall_macro": round(statistics.mean(
            [_rate(r["matched"], len(r["gold_invented"])) or 0.0
             for r in scored if r["gold_invented"]]), 4) if any(r["gold_invented"] for r in scored) else None,
    }
    exact = [r for r in ok if r["gold_max_scc"] == r["gen_max_scc"]]
    out["max_scc_exact_match"] = _rate(len(exact), len(ok))
    return out


def report(rows, summary):
    print(f"\n[STRUCTURE] {summary['cases']} case(s) compared")
    print(f"\n  {'feature':16}{'golden needs':>14}{'used':>7}{'aware':>8}{'spurious':>10}")
    for f in BOOL_FEATURES:
        s = summary[f]
        aw = f"{s['awareness_rate']:.2f}" if s["awareness_rate"] is not None else "n/a"
        sp = f"{s['spurious_rate']:.2f}" if s["spurious_rate"] is not None else "n/a"
        print(f"  {f:16}{s['golden_needs_it']:>14}{s['used_when_needed']:>7}{aw:>8}{sp:>10}")

    inv = summary["invention"]
    print(f"\n  predicate invention (matched extensionally, permutation-tolerant):")
    print(f"    cases scored     {inv['cases_scored']}")
    print(f"    golden concepts  {inv['golden_concepts']}   generated {inv['generated_concepts']}   matched {inv['matched']}")
    print(f"    recall  (micro)  {inv['recall_micro']}   precision (micro) {inv['precision_micro']}")
    print(f"    max-SCC exact    {summary['max_scc_exact_match']}")

    worst = [r for r in rows if r.get("status") == "ok"
             and r["gold_recursive"] and not r["gen_recursive"]]
    if worst:
        print(f"\n  needed recursion but did not use it ({len(worst)}):")
        for r in worst[:15]:
            print(f"    {r['case_id']}")
    print("\n  NOTE: extensional matching is evidence on the inputs used, not a proof of "
          "equivalence; a predicate matched on one variant may diverge on another.")


def load_generated(model, tag, scope, run_id):
    d = INFER_DIR / model / tag / scope / f"run_{run_id}"
    if not d.is_dir():
        return None, d
    return {p.stem: p.read_text(errors="ignore") for p in sorted(d.glob("*.dl"))}, d


# --------------------------------------------------------------------------
# Self-test: validates the metrics before any generated data exists.

SELFTEST_CASE = "SCC"

_SELF_VARIANTS = {
    # Same program, invented predicate renamed: must still match.
    "renamed": """.decl Edge(src_node: symbol, dst_node: symbol)
.input Edge
.decl SCC(src_node: symbol, dst_node: symbol)
.output SCC
.decl TC(a: symbol, b: symbol)
TC(x, y) :- Edge(x, y).
TC(x, z) :- TC(x, y), Edge(y, z).
SCC(x, y) :- TC(x, y), TC(y, x).
""",
    # Invented predicate stores the transpose: same concept, permuted columns.
    "transposed": """.decl Edge(src_node: symbol, dst_node: symbol)
.input Edge
.decl SCC(src_node: symbol, dst_node: symbol)
.output SCC
.decl Back(a: symbol, b: symbol)
Back(y, x) :- Edge(x, y).
Back(z, x) :- Back(y, x), Edge(y, z).
SCC(x, y) :- Back(y, x), Back(x, y).
""",
    # No intermediate concept and no recursion: the specific failure the
    # recursion-awareness metric exists to name.
    "flat": """.decl Edge(src_node: symbol, dst_node: symbol)
.input Edge
.decl SCC(src_node: symbol, dst_node: symbol)
.output SCC
SCC(x, y) :- Edge(x, y), Edge(y, x).
""",
}


def selftest():
    expect = {
        "renamed":    {"gen_recursive": True,  "matched": 1},
        "transposed": {"gen_recursive": True,  "matched": 1},
        "flat":       {"gen_recursive": False, "matched": 0},
    }
    print(f"[SELFTEST] golden = {SELFTEST_CASE}")
    ok = True
    rows = []
    for name, text in _SELF_VARIANTS.items():
        r = compare_case(SELFTEST_CASE, text)
        rows.append(r)
        exp = expect[name]
        good = all(r.get(k) == v for k, v in exp.items())
        ok &= good
        print(f"  {name:12} recursive={r.get('gen_recursive')} "
              f"invented={r.get('gen_invented')} matched={r.get('matched')} "
              f"{'OK' if good else 'FAIL (expected ' + str(exp) + ')'}")
    s = summarize(rows)
    print(f"\n  aggregate: recursion awareness {s['recursive']['awareness_rate']} "
          f"(2 of 3 generated programs are recursive, golden needs it)")
    print(f"             invention recall {s['invention']['recall_micro']}")
    print("\n[SELFTEST]", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="Structural comparison of generated vs reference programs.")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--tag", type=str, default="signature", help="Prompt-method tag under infer_data/<model>/")
    ap.add_argument("--dataset", type=str, default="all", help="Scope directory name")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--selftest", action="store_true", help="Validate the metrics on synthetic generated programs")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not args.model:
        ap.error("give --model (or --selftest)")

    gen, d = load_generated(args.model, args.tag, args.dataset, args.run)
    if gen is None:
        ap.error(f"no generated programs at {d}")
    rows = [compare_case(cid, text) for cid, text in sorted(gen.items())]
    summary = summarize(rows)
    report(rows, summary)

    skipped = [r for r in rows if r.get("status") != "ok"]
    if skipped:
        from collections import Counter
        print("  skipped:", dict(Counter(r["status"] for r in skipped)))

    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "cases": rows}, indent=2))
        print(f"[DONE] {args.json}")


if __name__ == "__main__":
    main()
