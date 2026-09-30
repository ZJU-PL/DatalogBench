#!/usr/bin/env python3
"""Measure the discriminative power of DatalogBench's I/O variants.

For each sampled case: build semantic mutants of the golden Datalog program,
run every mutant against all 6 I/O variants, and report

  - mutation score: fraction of mutants that at least one variant kills
  - cumulative kill by variant prefix: how much variants 1..5 add over variant 0

**What counts as observable.** A mutant is killed when its output differs from
the golden's *on the relations the evaluator actually grades* -- dataset.json's
`output_relation`, the same set `common_eval.get_case_output_relations` uses.
This matters: a reference program may `.output` a relation that the record does
not grade, and comparing everything Soufflé emits then counts kills the real
evaluation cannot see, overstating the oracle. Measured on `Traffic`, whose
`Crashable` was exported but ungraded: 7 kills observing all exports, 6 observing
the graded set. Pass --observe all for the looser basis (useful for
quantifying exactly this gap); the summary always states which basis was used.

Usage:
    python3 mutation_score.py                    # 14 random cases, both operators
    python3 mutation_score.py --op swap          # join mis-binding only
    python3 mutation_score.py --cases Path SCC
    python3 mutation_score.py --observe all      # legacy basis: every exported relation
"""

import argparse
import json
import random
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QDIR = ROOT / "benchmark" / "query"
IODIR = ROOT / "benchmark" / "io_data"
NUM_VARIANTS = 6          # legacy flat layout; see _variant_dirs


def _variant_dirs(case_id):
    """The variants to run mutants against: whatever the case currently ships.

    Hardcoding `<case>/0..5` silently stopped working once the frozen set moved
    variants under `eval/` -- every case reported `missing_variants`, which is at
    least loud, but the number of variants is no longer six either. Delegates to
    the same enumerator the evaluator uses so a mutant is judged on exactly the
    inputs that will score real programs.
    """
    from common_eval import case_variants_dirs
    return case_variants_dirs(case_id)


def graded_relations(case_record):
    """The relations the evaluator grades, from dataset.json.

    Mirrors `common_eval.get_case_output_relations` deliberately: this script
    must observe exactly what scoring observes, or its numbers describe an
    oracle that does not exist.
    """
    return sorted({x.split("(")[0] for x in (case_record.get("output_relation") or {})})


def observed_keys(case_id):
    """A case's graded relations as `<Rel>.csv` keys, or None if unknown.

    The single definition of "what is observable", shared by every tool that
    judges a mutant: mutation_score, select_variants and input_generator. They
    must agree, or variant selection would optimise for discrimination that
    scoring never sees.
    """
    records = json.load(open(ROOT / "benchmark" / "dataset.json"))
    rec = next((c for c in records if c["id"] == case_id), {})
    return [f"{r}.csv" for r in graded_relations(rec)] or None


def observed(result, keys):
    """Restrict a run's output to `keys` (None = everything Soufflé emitted).

    Missing relations normalise to the empty tuple list, matching
    `common_eval.load_tuples`, which returns the empty set for an absent file.
    So "produced nothing" and "produced no file" compare equal here, exactly as
    they do during scoring.
    """
    if result is None or keys is None:
        return result
    return {k: result.get(k, []) for k in keys}


def statements(txt):
    """Split a .dl file into (directive lines, rule/fact statements)."""
    dirs, rules, buf = [], [], ""
    for line in txt.splitlines():
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        if not buf and s.startswith("."):
            dirs.append(s)
            continue
        buf += " " + s
        if s.endswith("."):
            rules.append(buf.strip())
            buf = ""
    return dirs, rules


def run(prog_text, facts_dir):
    """Run a program over one variant's facts; return {relation: sorted tuples} or None."""
    with tempfile.TemporaryDirectory() as td:
        prog = Path(td) / "p.dl"
        prog.write_text(prog_text)
        out = Path(td) / "out"
        out.mkdir()
        try:
            r = subprocess.run(
                ["souffle", "-F", str(facts_dir), "-D", str(out), str(prog)],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            return None
        if r.returncode != 0:
            return None
        return {f.name: sorted(f.read_text().splitlines()) for f in out.glob("*.csv")}


def mutants_delete(dirs, rules):
    """Statement deletion and body-atom deletion, which move a program in opposite directions.

    Deleting a whole statement can only lose derivations (under-derivation).
    Deleting one atom from a rule body relaxes that rule's antecedent, so the
    rule fires on more bindings and can derive tuples the reference does not
    (over-derivation). The only guard is that a body keeps at least one atom;
    nothing restricts which variables stay bound. Neither operator is strictly
    weakening.

    Statement deletion covers ground facts as well as rules. A golden that seeds
    its recursion with `Fact(0, 1).` states a base case, and omitting the base
    case is one of the most common ways to get a recursive Datalog program
    wrong -- if no variant can tell that program from the reference, the oracle
    does not check the base case at all.
    """
    out = []
    for i, r in enumerate(rules):
        out.append("\n".join(dirs + [x for j, x in enumerate(rules) if j != i]))
    for i, r in enumerate(rules):
        if ":-" not in r:
            continue
        head, body = r.split(":-", 1)
        atoms = re.findall(r"!?\w+\([^()]*\)", body)
        if len(atoms) < 2:
            continue
        m = body.replace(atoms[-1], "", 1)
        m = re.sub(r",\s*,", ",", m).strip().strip(",")
        m = re.sub(r",\s*\.", ".", m)
        new = head + ":-" + m
        if not new.rstrip().endswith("."):
            new = new.rstrip() + "."
        out.append("\n".join(dirs + [x if j != i else new for j, x in enumerate(rules)]))
    return out


def body_variables(body):
    """Variables occurring in a rule body.

    A variable is an identifier appearing as an atom *argument* -- relation
    names, quoted constants and numbers are excluded. DatalogBench golden
    programs use lowercase variables, so casing cannot be used to identify
    them; position inside the argument list is what matters.
    """
    seen = []
    for args in re.findall(r"!?\w+\(([^()]*)\)", body):
        for tok in args.split(","):
            tok = tok.strip()
            if not re.fullmatch(r"[A-Za-z_]\w*", tok):
                continue  # number, quoted string, expression, wildcard
            if tok == "_":
                continue
            if tok not in seen:
                seen.append(tok)
    return seen


def mutants_swap(dirs, rules):
    """Join-variable mis-binding: the dominant real LLM failure mode.

    Swaps two variables *inside the body only*, so a head-shared variable
    becomes bound to the wrong column. Pure renamings are avoided by
    requiring the two variables to differ in whether they appear in the head.
    """
    out = []
    for i, r in enumerate(rules):
        if ":-" not in r:
            continue
        head, body = r.split(":-", 1)
        vs = body_variables(body)
        head_vars = set(body_variables(head))
        # Prefer swapping a head-shared variable with a body-local one: that
        # always changes the derived relation, never a pure alpha-renaming.
        pairs = [(a, b) for a in vs for b in vs
                 if a < b and ((a in head_vars) != (b in head_vars))]
        if not pairs:
            pairs = [(vs[0], vs[1])] if len(vs) >= 2 else []
        for a, b in pairs[:2]:
            nb = re.sub(rf"\b{a}\b", "@TMP@", body)
            nb = re.sub(rf"\b{b}\b", a, nb).replace("@TMP@", b)
            out.append("\n".join(dirs + [x if j != i else head + ":-" + nb
                                         for j, x in enumerate(rules)]))
    return out


def analyse_case(job):
    """Evaluate every mutant of one case against all variants. Runs in a worker."""
    cid, ops, max_mutants, graded, observe_all = job
    golden_file = QDIR / f"{cid}.dl"
    if not golden_file.exists():
        return {"case": cid, "status": "no_golden"}
    txt = golden_file.read_text(errors="ignore")
    dirs, rules = statements(txt)
    try:
        vdirs = _variant_dirs(cid)
    except FileNotFoundError:
        return {"case": cid, "status": "missing_variants"}
    if not vdirs:
        return {"case": cid, "status": "missing_variants"}
    n_variants = len(vdirs)

    gold = [run(txt, d) for d in vdirs]
    if any(g is None for g in gold):
        return {"case": cid, "status": "golden_failed"}

    keys = None if observe_all else [f"{r}.csv" for r in graded]
    if keys == []:
        # No graded relations at all: nothing to compare, and silently scoring
        # every mutant as "killed" or "survived" would both be wrong.
        return {"case": cid, "status": "no_graded_relations"}
    # Relations the program exports but nobody grades: kills that depend on them
    # would be invisible to the evaluator. qa_check gates this, but report it
    # here too so a stale checkout cannot quote an inflated score unnoticed.
    ungraded_exports = (sorted(set(gold[0]) - set(keys or [])) if keys is not None else [])

    gold_view = [observed(g, keys) for g in gold]

    builders = {"delete": [mutants_delete], "swap": [mutants_swap],
                "both": [mutants_delete, mutants_swap]}[ops]
    ms = [m for b in builders for m in b(dirs, rules)]
    if max_mutants is not None:
        ms = ms[:max_mutants]
    if not ms:
        return {"case": cid, "status": "no_mutants"}

    cum = [0] * n_variants
    killed = 0
    survivors = []
    for mt in ms:
        res = [observed(run(mt, d), keys) for d in vdirs]
        diff = [(res[i] is None) or (res[i] != gold_view[i]) for i in range(n_variants)]
        if any(diff):
            killed += 1
        else:
            # Keep the surviving program text: it must be hand-checked for
            # semantic equivalence before it counts as an oracle weakness.
            # Store every non-directive line, not just the rules: a golden that
            # seeds its computation with ground facts (Again(0). in MinPathSrc,
            # ValidStep(0). in MinSpanTree) is not reconstructible from the rules
            # alone, and a survivor rebuilt without its seeds collapses to the
            # empty result -- which reads as a kill on any input.
            survivors.append([ln for ln in mt.splitlines()
                              if ln.strip() and not ln.strip().startswith(".")])
        for k in range(n_variants):
            if any(diff[: k + 1]):
                cum[k] += 1

    return {"case": cid, "status": "ok", "mutants": len(ms), "killed": killed,
            "cumulative": cum, "survivors": survivors,
            "observed": ("all_exports" if keys is None else graded),
            "ungraded_exports": ungraded_exports}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", choices=["delete", "swap", "both"], default="both")
    ap.add_argument("--cases", nargs="*", default=None)
    # 0 means "no limit" for both, so passing 0 is the way to ask for an
    # exhaustive run. The defaults give a sampled, truncated run.
    ap.add_argument("--sample", type=int, default=14,
                    help="cases to draw; 0 or >= the dataset size means all of them")
    ap.add_argument("--max-mutants", type=int, default=10,
                    help="cap on mutants per case; 0 means no cap")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=None, help="write per-case results as JSON")
    ap.add_argument("--observe", choices=["graded", "all"], default="graded",
                    help="Which relations count as observable: 'graded' = dataset.json's "
                         "output_relation, i.e. what the evaluator scores (default); "
                         "'all' = every relation the program exports (looser, overstates "
                         "the oracle when a case exports something ungraded)")
    args = ap.parse_args()

    random.seed(args.seed)
    records = json.load(open(ROOT / "benchmark" / "dataset.json"))
    by_id = {c["id"]: c for c in records}
    all_ids = [c["id"] for c in records]
    if args.cases:
        cases = args.cases
    elif args.sample <= 0 or args.sample >= len(all_ids):
        cases = all_ids
    else:
        cases = random.sample(all_ids, args.sample)

    observe_all = args.observe == "all"
    max_mutants = None if args.max_mutants <= 0 else args.max_mutants

    # Say which kind of run this is, up front. Two mutation scores are only
    # comparable when both are exhaustive, and a truncated run is not visibly
    # different afterwards -- it is simply a smaller number.
    exhaustive = len(cases) == len(all_ids) and max_mutants is None
    if exhaustive:
        print(f"[MUTATION] exhaustive: all {len(cases)} cases, no per-case cap")
    else:
        print(f"[MUTATION] *** SAMPLED *** {len(cases)}/{len(all_ids)} case(s), "
              f"per-case cap {max_mutants if max_mutants is not None else 'none'}. "
              "Not comparable to the exhaustive figure; do not commit this as the "
              "frozen artifact.")

    jobs = [(cid, args.op, max_mutants,
             graded_relations(by_id.get(cid, {})), observe_all) for cid in cases]
    results = []
    print(f"{'case':<26}{'mutants':>8}{'killed':>8}{'kill%':>7}   cumulative kill by variant 0..5",
          flush=True)

    def report(res):
        if res["status"] != "ok":
            print(f"{res['case']:<26}  ({res['status']})", flush=True)
            return
        print(f"{res['case']:<26}{res['mutants']:>8}{res['killed']:>8}"
              f"{100 * res['killed'] / res['mutants']:>6.0f}%   "
              f"{' '.join(map(str, res['cumulative']))}", flush=True)

    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for res in ex.map(analyse_case, jobs):
                results.append(res)
                report(res)
    else:
        for job in jobs:
            res = analyse_case(job)
            results.append(res)
            report(res)

    ok = [r for r in results if r["status"] == "ok"]
    total = sum(r["mutants"] for r in ok)
    total_killed = sum(r["killed"] for r in ok)
    extra = sum(r["cumulative"][-1] - r["cumulative"][0] for r in ok)
    v0 = sum(r["cumulative"][0] for r in ok)

    basis = ("every exported relation (--observe all; LOOSER than scoring)"
             if observe_all else "the graded relations from dataset.json (what scoring sees)")
    print(f"\nobservation basis: {basis}")
    inflated = [r for r in ok if r.get("ungraded_exports")]
    if inflated:
        print(f"WARNING: {len(inflated)} case(s) export relations nobody grades; "
              "kills depending on them are invisible to the evaluator:")
        for r in inflated[:10]:
            print(f"  {r['case']}: {', '.join(r['ungraded_exports'])}")
    print(f"cases analysed={len(ok)}/{len(results)}  mutants={total}")
    print(f"killed by variant 0 alone : {v0} ({100 * v0 / max(total, 1):.1f}%)")
    print(f"killed by all 6 variants  : {total_killed} ({100 * total_killed / max(total, 1):.1f}%)")
    print(f"variants 1-5 marginal gain: {extra} ({100 * extra / max(total, 1):.1f}% of all mutants)")

    surv = {r["case"]: len(r["survivors"]) for r in ok if r["survivors"]}
    if surv:
        print(f"\ncases with surviving mutants: {len(surv)}")
        for cid, n in sorted(surv.items(), key=lambda kv: -kv[1])[:25]:
            print(f"  {cid:<26} {n}")
    skipped = [r for r in results if r["status"] != "ok"]
    if skipped:
        from collections import Counter
        print("\nskipped:", dict(Counter(r["status"] for r in skipped)))

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))
        print(f"\nwrote {args.out}")
    print("\nNote: survivors include equivalent mutants, which no input can kill. "
          "Hand-check them before quoting a mutation score.")


if __name__ == "__main__":
    main()
