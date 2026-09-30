"""Generate the negative example set O- from surviving/killed mutants.

O+ is the set of tuples the target program must derive and O- the set it must
not. Exact match scores against O+ (`.expected`); O- (`.undesired`) lets
positive coverage and negative exclusion be reported separately.

Negatives are mined rather than invented: run each semantic mutant of the golden
on a variant and keep the tuples it derives that the golden does not. These are
exactly the wrong answers a *plausibly* wrong program produces, which makes them
far sharper than randomly perturbed tuples -- a candidate that avoids all of
them has demonstrably avoided the common failure modes.

Written as `<Relation>.undesired` next to `<Relation>.expected`, matching the
convention GenSynth already uses (`--use_neg`), so the symbolic baselines can
consume them directly.

    python3 evaluation/negative_examples.py --case SCC            # report only
    python3 evaluation/negative_examples.py --all --jobs 8 --out /tmp/neg
    python3 evaluation/negative_examples.py --all --jobs 8 --apply   # write into io_data
"""

import argparse
import json
import multiprocessing
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "evaluation"))

from input_generator import build_mutants  # noqa: E402
from mutation_score import run  # noqa: E402

QUERY_DIR = _REPO / "benchmark" / "query"
IO_DIR = _REPO / "benchmark" / "io_data"
NUM_VARIANTS = 6          # legacy flat layout only; see _scored_variant_dirs


def _scored_variant_dirs(case_id):
    """Variant dirs to mine, matching what the evaluator scores.

    Delegates to `common_eval.case_variants_dirs` so the two cannot disagree:
    negatives placed anywhere the evaluator does not read are silently inert.
    """
    import sys as _sys
    from pathlib import Path as _P
    _eval = str(_P(__file__).resolve().parent)
    if _eval not in _sys.path:
        _sys.path.insert(0, _eval)
    from common_eval import case_variants_dirs
    return case_variants_dirs(case_id)


def _variant_key(case_id, d):
    """Path of the variant relative to its case, e.g. "3" or "eval/1"."""
    return str(d.relative_to(IO_DIR / case_id))


def _mine_run(prog_text, facts_dir, timeout):
    """Run one program for mining; return (relations, timed_out).

    mutation_score.run collapses "timed out" and "failed to compile" into None,
    which is the right call for scoring -- a program that will not terminate is
    wrong either way. For mining it is not: a mutant that times out contributes
    no negatives and leaves no trace, so the negative set silently depends on
    how loaded the machine was. Array/eval/1 lost two relations' negatives that
    way when the sweep ran eight-wide. Timeouts are therefore counted and
    reported, and the timeout is generous enough that a clean run has none.
    """
    import subprocess as _sp
    with tempfile.TemporaryDirectory() as td:
        prog = Path(td) / "p.dl"
        prog.write_text(prog_text)
        out = Path(td) / "out"
        out.mkdir()
        try:
            r = _sp.run(["souffle", "-F", str(facts_dir), "-D", str(out), str(prog)],
                        capture_output=True, text=True, timeout=timeout)
        except _sp.TimeoutExpired:
            return None, True
        if r.returncode != 0:
            return None, False
        return {f.name: sorted(f.read_text().splitlines()) for f in out.glob("*.csv")}, False


def negatives_for_variant(dl_text, facts_dir, mutants, cap=50, timeout=120):
    """Tuples some mutant derives that the golden does not, capped *per mutant*.

    Each mutant is one plausible way to get the program wrong, so the unit that
    carries information is the failure mode, not the tuple. Capping per relation
    and keeping the lexicographically smallest rows -- what this did before --
    silently dropped whole failure modes when one mutant over-derived far more
    than the others: AccessPolicy's ViewEmployee over-derives 1.6M tuples from
    four mutants, and the first 200 in sort order all came from one of them, so
    three of the four modes vanished. Taking up to `cap` from each contributing
    mutant keeps every mode represented and makes the tuple count proportional
    to how many modes the set covers.

    Returns (negatives, modes, stalled) where `stalled` lists the mutants that
    did not terminate. Those are recorded rather than counted so the set is
    auditable: at a 120s budget the same mutants stall on every run (verified
    against a serial re-run), and any drift in that list is a signal that the
    data changed for a reason other than the benchmark.
    """
    golden, gto = _mine_run(dl_text, facts_dir, timeout)
    if golden is None:
        return None, None, ["golden"] if gto else []
    per_mutant, stalled = {}, []
    for i, mt in enumerate(mutants):
        out, to = _mine_run(mt, facts_dir, timeout)
        if to:
            stalled.append(i)
        if out is None:
            continue                      # does not compile, or timed out (counted above)
        for rel, rows in out.items():
            extra = set(rows) - set(golden.get(rel, []))
            if extra:
                # run() keys relations by output filename ("Rel.csv"); the
                # benchmark addresses them by bare relation name.
                key = rel[:-4] if rel.endswith(".csv") else rel
                per_mutant.setdefault(key, {})[i] = extra
    neg, modes = {}, {}
    for rel, by_mutant in per_mutant.items():
        rows = set()
        for extra in by_mutant.values():
            rows.update(sorted(extra)[:cap])
        neg[rel] = sorted(rows)
        modes[rel] = sorted(by_mutant)          # which mutants, not how many
    return neg, modes, stalled


def process_case(case_id, max_mutants=20, cap=200):
    dl_path = QUERY_DIR / f"{case_id}.dl"
    if not dl_path.exists():
        return {"case": case_id, "status": "no_golden"}
    dl_text = dl_path.read_text(errors="ignore")
    _d, _r, mutants = build_mutants(dl_text, max_mutants)
    if not mutants:
        return {"case": case_id, "status": "no_mutants"}

    # Mine against whatever variants the case actually ships, in either layout.
    # Hardcoding `<case>/0..5` breaks once greedy selection has reduced the set
    # and moved it under `eval/`: nothing would be found and the case would look
    # like it simply has no negatives, rather than like it was never scanned.
    # Negatives are mined per input, so this must run *after* the variant set is
    # final -- see the freeze order.
    per_variant, per_variant_modes, total, stalled = {}, {}, 0, set()
    for d in _scored_variant_dirs(case_id):
        neg, modes, st = negatives_for_variant(dl_text, d, mutants, cap)
        stalled.update(st)
        if neg is None:
            continue
        key = _variant_key(case_id, d)
        per_variant[key] = neg
        per_variant_modes[key] = modes
        total += sum(len(rows) for rows in neg.values())

    # How many positives exist, for context: a case with many negatives per
    # positive gives the negative-exclusion score real resolution.
    pos = 0
    for v in per_variant:
        for f in (IO_DIR / case_id / v).glob("*.expected"):
            pos += len([l for l in f.read_text().splitlines() if l.strip()])

    # How many distinct failure modes the set covers, which is what the count of
    # tuples only proxies: 200 negatives from one mutant say less than 20 from
    # four. Counted as the union of over-deriving mutants across variants -- the
    # same mutant firing on two inputs is one failure mode, not two.
    seen = set()
    for m in per_variant_modes.values():
        for idxs in m.values():
            seen.update(idxs)
    modes = len(seen)
    return {"case": case_id, "status": "ok", "mutants": len(mutants),
            "negatives": total, "positives": pos, "modes": modes,
            "stalled_mutants": sorted(stalled, key=str),
            "per_variant": per_variant, "per_variant_modes": per_variant_modes}


def write_negatives(case_id, result, root):
    """Write `<Rel>.undesired` files under root/<case>/<variant>/.

    Also removes the ones this mining run did *not* produce. A re-mine after a
    golden is corrected can legitimately yield nothing for a variant that
    previously had negatives; writing only the fresh files would leave the old
    ones in place, still excluding tuples the corrected reference now derives.
    That is the same staleness `.expected` suffers, and qa_check's manifest
    drift check is what catches it.
    """
    written = 0
    fresh = set()
    for v, rels in result["per_variant"].items():
        dest = Path(root) / case_id / v
        dest.mkdir(parents=True, exist_ok=True)
        for rel, rows in rels.items():
            # rows are whole output lines (already tab-separated), not tuples
            (dest / f"{rel}.undesired").write_text(
                "".join(r + "\n" for r in rows), encoding="utf-8")
            fresh.add((v, rel))
            written += 1
    case_root = Path(root) / case_id
    if case_root.is_dir():
        for stale in case_root.glob("*/*/*.undesired"):
            key = (f"{stale.parent.parent.name}/{stale.parent.name}", stale.stem)
            if key not in fresh:
                stale.unlink()
    return written


def _worker(args):
    cid, max_mutants, cap = args
    try:
        return process_case(cid, max_mutants, cap)
    except Exception as exc:
        return {"case": cid, "status": f"error: {exc}"}


def main():
    ap = argparse.ArgumentParser(description="Mine the negative example set O- from mutants.")
    ap.add_argument("--case", type=str, default=None)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-mutants", type=int, default=20)
    ap.add_argument("--cap", type=int, default=50,
                    help="Max negatives kept per contributing mutant, per relation per variant")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", type=str, default=None, help="Write .undesired files under this staging dir")
    ap.add_argument("--apply", action="store_true",
                    help="Write .undesired directly into benchmark/io_data (modifies benchmark data)")
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--manifest", action="store_true",
                    help="Also refresh benchmark/negative_manifest.json, the committed record of "
                         "how many negatives and failure modes each case carries")
    args = ap.parse_args()

    if args.all:
        ids = [c["id"] for c in json.load(open(_REPO / "benchmark" / "dataset.json"))]
    else:
        ids = args.cases or ([args.case] if args.case else [])
    if not ids:
        ap.error("give --case / --cases / --all")

    jobs = [(cid, args.max_mutants, args.cap) for cid in ids]
    if args.jobs > 1:
        with multiprocessing.Pool(args.jobs) as pool:
            results = pool.map(_worker, jobs)
    else:
        results = [_worker(j) for j in jobs]

    root = IO_DIR if args.apply else (Path(args.out) if args.out else None)
    ok = [r for r in results if r.get("status") == "ok"]
    empty = [r for r in ok if r["negatives"] == 0]

    for r in results:
        if r.get("status") != "ok":
            print(f"[{r['case']}] {r['status']}")
            continue
        if len(ids) <= 12:
            ratio = (r["negatives"] / r["positives"]) if r["positives"] else 0
            print(f"[{r['case']}] {r['negatives']} negatives vs {r['positives']} positives "
                  f"({ratio:.1f}x) covering {r.get('modes', 0)} failure mode(s) "
                  f"of {r['mutants']} mutants")
        if root:
            write_negatives(r["case"], r, root)

    tot_neg = sum(r["negatives"] for r in ok)
    tot_pos = sum(r["positives"] for r in ok)
    tot_modes = sum(r.get("modes", 0) for r in ok)
    print(f"\n[O-] {len(ok)} case(s): {tot_neg} negatives vs {tot_pos} positives "
          f"({tot_neg / tot_pos:.1f}x)" if tot_pos else f"\n[O-] {tot_neg} negatives")
    print(f"  covering {tot_modes} distinct failure mode(s) -- one per mutant that "
          "over-derives; the tuple count only proxies this")
    stalled = [r for r in ok if r.get("stalled_mutants")]
    if stalled:
        n = sum(len(r["stalled_mutants"]) for r in stalled)
        print(f"  {n} mutant(s) did not terminate, in {len(stalled)} case(s); their negatives "
              f"are missing, so those counts are lower bounds: "
              f"{', '.join(r['case'] for r in stalled[:8])}")
    # "No negatives" only means something when every mutant actually ran. A case
    # that also stalled a mutant has an unproven zero, not a demonstrated one.
    clean = [r for r in empty if not r.get("stalled_mutants")]
    unproven = [r for r in empty if r.get("stalled_mutants")]
    if clean:
        print(f"  {len(clean)} case(s) yielded no negatives (every mutant ran, and they only ever "
              f"derive *fewer* tuples there): {', '.join(r['case'] for r in clean[:8])}")
    if unproven:
        print(f"  {len(unproven)} case(s) yielded no negatives but stalled a mutant, so the zero "
              f"is unproven: {', '.join(r['case'] for r in unproven)}")
    if root:
        print(f"[DONE] .undesired files -> {root}")
    if args.manifest:
        # The tuple counts alone do not say how much the negative set covers, and
        # the .undesired files cannot carry that. Keep a committed manifest so a
        # results table can quote failure modes, and so a case whose zero is
        # unproven (a mutant stalled) is visible without re-mining.
        #
        # Merge, never overwrite. A partial run (--cases with a handful of ids,
        # which is what a golden fix calls for) used to rewrite the file with
        # only those ids and silently drop every other case's record. The
        # manifest is a committed record of the whole library, so entries this
        # run did not recompute must survive it.
        keep = ("case", "status", "negatives", "positives", "modes", "mutants", "stalled_mutants")
        fresh = {r["case"]: {k: v for k, v in r.items() if k in keep} for r in results}
        path = _REPO / "benchmark" / "negative_manifest.json"
        merged = {}
        if path.exists():
            for row in json.loads(path.read_text(encoding="utf-8")):
                merged[row["case"]] = row
        kept = len(set(merged) - set(fresh))
        merged.update(fresh)
        man = [merged[c] for c in sorted(merged)]
        path.write_text(json.dumps(man, indent=1), encoding="utf-8")
        print(f"[DONE] manifest -> {path} ({len(fresh)} refreshed, {kept} carried over)")

    if args.json:
        # keep the report small: drop the tuple payloads
        slim = [{k: v for k, v in r.items() if k != "per_variant"} for r in results]
        Path(args.json).write_text(json.dumps(slim, indent=2))
        print(f"[DONE] report -> {args.json}")


if __name__ == "__main__":
    main()
