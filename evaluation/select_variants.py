"""Greedy set-cover selection of I/O variants.

Every case currently ships 6 variants that were random siblings of variant 0.
Measured over the library they are largely redundant: for 117 of the 140 cases
the library then held, variants 1-5 killed nothing that variant 0 did not already
kill. (That measurement predates the removals and golden repairs; re-measure
before quoting it.) This picks, per
case, the *smallest* set of inputs that preserves (or improves) discriminative
power, drawing from the existing variants plus the targeted candidates from
`input_generator`.

Selection is greedy over marginal kill, which is the standard approximation for
set cover: repeatedly take the candidate that kills the most mutants nothing
selected so far kills. Existing variants win ties, so a case only churns when a
generated input genuinely adds coverage.

One variant is held out as the few-shot *demonstration* pool and excluded from
the evaluation pool. Demonstrating with an input that is also scored leaks the
oracle, which is a threat independent of oracle strength.

    python3 evaluation/select_variants.py --case SCC
    python3 evaluation/select_variants.py --cases SCC MutualRecursion --out /tmp/sel
    python3 evaluation/select_variants.py --all --jobs 8 --json /tmp/sel.json
"""

import argparse
import json
import multiprocessing
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "evaluation"))

from input_generator import (build_mutants, mine_constants, parse_input_schema,  # noqa: E402
                             random_scenarios, scenarios, write_facts,
                             domain_pools)
from mutation_score import observed, observed_keys, run  # noqa: E402

QUERY_DIR = _REPO / "benchmark" / "query"
IO_DIR = _REPO / "benchmark" / "io_data"
NUM_VARIANTS = 6          # only a fallback for the legacy flat layout


def _kill_vector(dl_text, facts_dir, mutants, keys=None):
    """(golden_output, frozenset of mutant indices this input kills)."""
    golden = observed(run(dl_text, facts_dir), keys)
    if golden is None:
        return None, frozenset()
    killed = set()
    for i, mt in enumerate(mutants):
        out = observed(run(mt, facts_dir), keys)
        if out is None or out != golden:
            killed.add(i)
    return golden, frozenset(killed)


def _blind_relations(golden_out, keys):
    """Graded relations the reference program derives nothing for on this input.

    Computed from the golden's actual output rather than from `.expected` files
    on disk, because generated candidates have no `.expected` yet -- theirs is
    regenerated at materialisation time. Same notion of blindness qa_check
    reports, just evaluated one step earlier.
    """
    if golden_out is None or not keys:
        return []
    return [k[:-len(".csv")] for k in keys if not golden_out.get(k)]


def _candidate_pool(case_id, dl_text, mutants, tmp, extra_random=0):
    """[(name, facts_dir, kills, blind)] over existing variants then generated ones."""
    keys = observed_keys(case_id)
    pool = []
    # Whatever the case ships now, in either layout: re-running selection after a
    # freeze must start from the frozen set, not from a `<case>/0..5` that no
    # longer exists.
    from common_eval import case_variants_dirs
    for d in case_variants_dirs(case_id):
        g, kills = _kill_vector(dl_text, d, mutants, keys)
        pool.append((f"existing/{d.relative_to(IO_DIR / case_id)}", d, kills,
                     _blind_relations(g, keys)))

    # Generating and running candidates is the expensive part, so skip it when
    # the existing variants already kill every mutant: nothing could improve on
    # that, and only the *selection* among them is still open.
    covered = set()
    for _n, _d, kills, _b in pool:
        covered |= kills
    if len(covered) == len(mutants):
        return pool

    schema = parse_input_schema(dl_text)
    mined = mine_constants(dl_text)
    cands = scenarios(schema, mined)
    if extra_random:
        # Draw each join class from one pool: values that cannot meet across
        # relations produce candidates that derive nothing and rank last.
        cands += random_scenarios(schema, mined, count=extra_random,
                                  pools=domain_pools(dl_text, mined))
    for name, facts in cands:
        d = Path(tmp) / name
        write_facts(d, facts)
        golden, kills = _kill_vector(dl_text, d, mutants, keys)
        if golden is None:
            continue                       # candidate breaks the golden program
        pool.append((f"gen/{name}", d, kills, _blind_relations(golden, keys)))
    return pool


def greedy_cover(pool, n_mutants, budget=None):
    """Pick candidates by marginal kill; existing variants break ties."""
    remaining = set(range(n_mutants))
    chosen, order = [], sorted(
        range(len(pool)), key=lambda i: (0 if pool[i][0].startswith("existing/") else 1, pool[i][0]))
    while remaining and (budget is None or len(chosen) < budget):
        best, best_gain = None, 0
        for i in order:
            if i in [c[0] for c in chosen]:
                continue
            gain = len(pool[i][2] & remaining)
            if gain > best_gain:
                best, best_gain = i, gain
        if best is None:
            break
        chosen.append((best, best_gain))
        remaining -= pool[best][2]
    return chosen, remaining


def select_case(case_id, budget=None, extra_random=0, max_mutants=20):
    # Selection is a one-time operation, not idempotent. Run again on a set it
    # already produced and it selects from the eval pool alone -- shrinking it
    # further and holding out yet another variant as a demo, on top of the demo
    # that already exists. The result looks like a normal, smaller selection.
    if (IO_DIR / case_id / "demo").is_dir():
        return {"case": case_id, "status": "already_selected",
                "note": "case already has a demo/ pool; re-selecting would shrink "
                        "the eval set and strand the existing demo. Start from the "
                        "pre-selection variants if you mean to redo it."}

    dl_text = (QUERY_DIR / f"{case_id}.dl").read_text(errors="ignore")
    _d, _r, mutants = build_mutants(dl_text, max_mutants)
    if not mutants:
        return {"case": case_id, "status": "no_mutants"}

    with tempfile.TemporaryDirectory(prefix=f"sel_{case_id}_") as tmp:
        pool = _candidate_pool(case_id, dl_text, mutants, tmp, extra_random)
        if not pool:
            return {"case": case_id, "status": "no_usable_candidates"}

        existing_kill = set()
        for name, _d, kills, _b in pool:
            if name.startswith("existing/"):
                existing_kill |= kills

        chosen, unkilled = greedy_cover(pool, len(mutants), budget)
        eval_pool = [pool[i][0] for i, _g in chosen]

        # Hold out a demonstration input, disjoint from the evaluation pool.
        # Two requirements, in order of importance:
        #   1. no blind slot -- a demo whose golden output is empty for some
        #      graded relation teaches the model that the relation produces
        #      nothing, which is worse than showing no example at all;
        #   2. prefer an existing variant, so few-shot prompts keep real-looking
        #      data rather than synthetic shapes.
        free = [(n, b) for n, _d, _k, b in pool if n not in eval_pool]
        demo = next((n for n, b in free if not b and n.startswith("existing/")), None)
        if demo is None:
            demo = next((n for n, b in free if not b), None)
        demo_blind = []
        if demo is None:
            # Every candidate has a blind slot: take the least-bad one and say
            # so, rather than silently shipping a misleading demonstration.
            fallback = min(free, key=lambda nb: len(nb[1]), default=(None, []))
            demo, demo_blind = fallback

        return {
            "case": case_id,
            "status": "ok",
            "mutants": len(mutants),
            "pool_size": len(pool),
            # How many variants the case shipped before selection -- read from
            # the pool rather than assumed to be six, which stops being true the
            # moment a frozen set is re-selected.
            "existing_count": sum(1 for n, *_ in pool if n.startswith("existing/")),
            "existing_kills": len(existing_kill),
            "selected_kills": len(mutants) - len(unkilled),
            "eval_pool": eval_pool,
            "eval_size": len(eval_pool),
            "marginal": [{"variant": pool[i][0], "gain": g} for i, g in chosen],
            "demo_pool": demo,
            "demo_blind_relations": demo_blind,
            "unkillable": len(unkilled),
        }


def _worker(args):
    cid, budget, extra_random, max_mutants = args
    try:
        return select_case(cid, budget, extra_random, max_mutants)
    except Exception as exc:                       # keep one bad case from killing the sweep
        return {"case": cid, "status": f"error: {exc}"}


def _warn_dropped_artifacts(case_id, kept_variant_names):
    """Shout if collapsing this case's variants would strand mined artifacts.

    `materialize` rebuilds a variant from its facts alone: it copies `*.facts`
    and regenerates `*.expected` by running the reference program. Anything else
    living in a variant directory -- notably the `*.undesired` negative examples
    mined by `negative_examples.py` -- does not come along, and nothing fails
    when it goes missing: `common_eval` reads negatives from the variant being
    evaluated and degrades to "no signal" for a relation that has none. The
    whole negative-exclusion column would quietly read zero-information.

    Negatives are a function of (program, inputs), so the fix is not to copy
    them across but to mine them again once the variant set is final. Selection
    therefore has to run *before* `negative_examples.py --apply`, and this warns
    when that order was not followed.
    """
    stranded = []
    case_dir = IO_DIR / case_id
    if not case_dir.is_dir():
        return
    for vd in sorted(p for p in case_dir.iterdir() if p.is_dir()):
        hits = list(vd.glob("*.undesired"))
        if hits and f"existing/{vd.name}" not in kept_variant_names:
            stranded.append((vd.name, len(hits)))
    if stranded:
        total = sum(n for _v, n in stranded)
        print(f"  WARNING [{case_id}] {total} mined negative-example file(s) in "
              f"{len(stranded)} dropped variant(s) will NOT carry over "
              f"({', '.join(v for v, _n in stranded)}).")
        print("           Negatives depend on the input set, so re-run "
              "negative_examples.py --apply AFTER this selection is final, "
              "rather than before it.")


def materialize(case_id, result, out_root):
    """Write the selected variants (facts + regenerated .expected) under out_root."""
    import subprocess
    kept = set(result.get("eval_pool") or [])
    if result.get("demo_pool"):
        kept.add(result["demo_pool"])
    _warn_dropped_artifacts(case_id, kept)
    dl_text = (QUERY_DIR / f"{case_id}.dl").read_text(errors="ignore")
    schema = parse_input_schema(dl_text)
    mined = mine_constants(dl_text)
    by_name = {f"gen/{n}": f for n, f in scenarios(schema, mined)}

    dest_case = Path(out_root) / case_id
    for kind, names in (("eval", result["eval_pool"]),
                        ("demo", [result["demo_pool"]] if result.get("demo_pool") else [])):
        for idx, name in enumerate(names):
            dest = dest_case / kind / str(idx)
            dest.mkdir(parents=True, exist_ok=True)
            if name.startswith("existing/"):
                src = IO_DIR / case_id / name.split("/", 1)[1]
                for f in src.glob("*.facts"):
                    (dest / f.name).write_text(f.read_text())
            else:
                write_facts(dest, by_name[name])
            with tempfile.TemporaryDirectory() as tmp:
                prog = Path(tmp) / "p.dl"; prog.write_text(dl_text)
                o = Path(tmp) / "o"; o.mkdir()
                subprocess.run(["souffle", "-F", str(dest), "-D", str(o), str(prog)],
                               capture_output=True, text=True, timeout=60)
                for csv in o.glob("*.csv"):
                    (dest / f"{csv.stem}.expected").write_text(csv.read_text())
    return dest_case


def main():
    ap = argparse.ArgumentParser(description="Greedy set-cover selection of I/O variants.")
    ap.add_argument("--case", type=str, default=None)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--budget", type=int, default=None, help="Cap the evaluation pool size")
    ap.add_argument("--random", type=int, default=0, help="Also sample N random candidate inputs")
    ap.add_argument("--max-mutants", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", type=str, default=None, help="Materialize selected variants here")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    if args.all:
        ids = [c["id"] for c in json.load(open(_REPO / "benchmark" / "dataset.json"))]
    else:
        ids = args.cases or ([args.case] if args.case else [])
    if not ids:
        ap.error("give --case / --cases / --all")

    jobs = [(cid, args.budget, args.random, args.max_mutants) for cid in ids]
    if args.jobs > 1:
        with multiprocessing.Pool(args.jobs) as pool:
            results = pool.map(_worker, jobs)
    else:
        results = [_worker(j) for j in jobs]

    tot_old = tot_new = tot_kill_old = tot_kill_new = 0
    for r in results:
        if r.get("status") != "ok":
            print(f"[{r['case']}] {r.get('status')}")
            continue
        print(f"[{r['case']}] {r['mutants']} mutants | pool {r['pool_size']} "
              f"| existing 6 kill {r['existing_kills']}, selected {r['eval_size']} kill {r['selected_kills']}")
        for m in r["marginal"]:
            print(f"    + {m['variant']:32} +{m['gain']}")
        blind = r.get("demo_blind_relations") or []
        warn = (f"   WARNING: demo has no output for {', '.join(blind)} -- every candidate "
                "was blind; the few-shot example will show an empty relation" if blind else "")
        print(f"    demo held out: {r['demo_pool']}   unkillable: {r['unkillable']}{warn}")
        tot_old += r.get("existing_count", NUM_VARIANTS)
        tot_new += r["eval_size"]
        tot_kill_old += r["existing_kills"]
        tot_kill_new += r["selected_kills"]
        if args.out:
            materialize(r["case"], r, args.out)

    ok = [r for r in results if r.get("status") == "ok"]
    if ok:
        print(f"\n[TOTAL] {len(ok)} case(s): variants {tot_old} -> {tot_new} "
              f"({tot_new / tot_old:.0%} of current); kills {tot_kill_old} -> {tot_kill_new}")
    if args.out:
        print(f"[DONE] selected variants -> {args.out}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"[DONE] report -> {args.json}")


if __name__ == "__main__":
    main()
