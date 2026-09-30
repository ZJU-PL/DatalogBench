"""Difficulty calibration: rank-based tiers over structural and specification load.

Phase 2 asks for evidence that the benchmark spans an easy-to-hard continuum
rather than clustering at one difficulty. This derives that from the structural
metrics `structure_analyzer` already extracts, plus the specification side.

Two axes are scored *separately* on purpose:

  program complexity -- how hard the target program is to write once you know
      what to write: rule count, recursion (max SCC size), stratification depth,
      join width, arity
  specification load -- how much the solver must supply that the prompt does
      not: invented predicates, number of input relations to wire up, and how
      terse the NL question is

Merging them would conflate "this program is intricate" with "this task is
underspecified", and the second must stay separately measurable. Their
correlation is reported so the two axes can be shown to be non-redundant.

Scoring is percentile-rank based: each metric is converted to its rank among the
every case and averaged. That avoids inventing weights for quantities measured in
incomparable units (rules vs. arity vs. words).

    python3 evaluation/difficulty.py                       # tiers + distribution
    python3 evaluation/difficulty.py --tiers 4 --csv out.csv
    python3 evaluation/difficulty.py --validate benchmark/res_data/<...>_details.jsonl
"""

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "evaluation"))

from common_eval import BENCHMARK_DIR, QUERY_DIR  # noqa: E402
from common_eval import save_csv_rows  # noqa: E402
from structure_analyzer import analyze_program  # noqa: E402

# metric -> higher means harder
COMPLEXITY_METRICS = ["num_rules", "max_scc_size", "num_strata", "avg_body_atoms", "max_arity"]
# Named `spec_complexity_tier` downstream, never `spec_tier`. Two different
# things were called the same name: this one bins cases by *how much structure a
# spec has to convey* -- invented concepts, input arity, how terse the sentence
# is -- into equal-count quantiles, so every tier is populated by construction.
# `dataset.json`'s `spec_tier` is the hand-audited L1/L2/L3 *sufficiency* label:
# what a spec is missing, with a distribution that can be arbitrarily lopsided
# (after this round's repairs, ideally almost nothing is L3). Slicing results by
# the wrong one silently answers a different question.
SPEC_METRICS = ["num_invented", "num_input", "nl_terseness"]

TIER_NAMES = {3: ["easy", "medium", "hard"], 4: ["easy", "moderate", "hard", "very_hard"]}


def percentile_ranks(values):
    """Average-rank percentile in [0,1]; ties share their mean rank."""
    n = len(values)
    if n <= 1:
        return [0.5] * n
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg / (n - 1)
        i = j + 1
    return ranks


def collect(dataset_scope="all"):
    """Per-case structural + specification features."""
    from common_eval import dataset_records
    rows = []
    for rec in dataset_records(dataset_scope):
        cid = rec["id"]
        path = QUERY_DIR / f"{cid}.dl"
        if not path.exists():
            continue
        feats = analyze_program(path.read_text(encoding="utf-8"))
        words = len((rec.get("question") or "").split())
        rows.append({
            "case_id": cid,
            "category": rec.get("category", ""),
            **{k: feats[k] for k in COMPLEXITY_METRICS},
            "num_invented": feats["num_invented"],
            "num_input": len(rec.get("input_relation", {})),
            "nl_words": words,
            # terseness: fewer words for the same job means more is left implicit
            "nl_terseness": -words,
            "recursive": feats["recursive"],
            "uses_negation": feats["uses_negation"],
            "uses_aggregation": feats["uses_aggregation"],
        })
    return rows


def score(rows, metrics):
    """Mean percentile rank across the given metrics."""
    ranked = {m: percentile_ranks([r[m] for r in rows]) for m in metrics}
    return [statistics.mean(ranked[m][i] for m in metrics) for i in range(len(rows))]


def assign_tiers(scores, n_tiers):
    """Equal-count tiers (quantile bins), so every tier is populated."""
    names = TIER_NAMES.get(n_tiers) or [f"tier{i + 1}" for i in range(n_tiers)]
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    tiers = [None] * len(scores)
    per = len(scores) / n_tiers
    for pos, idx in enumerate(order):
        tiers[idx] = names[min(int(pos / per), n_tiers - 1)]
    return tiers, names


def spearman(a, b):
    ra, rb = percentile_ranks(a), percentile_ranks(b)
    n = len(a)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else 0.0


def validate_against(rows, details_path):
    """Correlate difficulty with observed pass@1, if a details.jsonl is given."""
    per_case = defaultdict(list)
    with Path(details_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                per_case[d["case_id"]].append(1.0 if d.get("perfect_match") else 0.0)
    shared = [r for r in rows if r["case_id"] in per_case]
    if len(shared) < 5:
        print(f"[VALIDATE] only {len(shared)} shared cases; skipped")
        return
    obs = [statistics.mean(per_case[r["case_id"]]) for r in shared]
    for axis in ("complexity_score", "spec_score"):
        rho = spearman([r[axis] for r in shared], obs)
        print(f"[VALIDATE] {axis} vs observed pass@1: rho = {rho:+.3f} over {len(shared)} cases "
              f"({'harder -> lower pass@1, as expected' if rho < 0 else 'no expected direction'})")


def main():
    ap = argparse.ArgumentParser(description="Difficulty calibration over structural + spec load.")
    ap.add_argument("--dataset", type=str, default="all")
    ap.add_argument("--tiers", type=int, default=3)
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--validate", type=str, default=None, help="A *_details.jsonl to correlate against")
    args = ap.parse_args()

    rows = collect(args.dataset)
    if not rows:
        ap.error("no cases found")

    comp = score(rows, COMPLEXITY_METRICS)
    spec = score(rows, SPEC_METRICS)
    for r, c, s in zip(rows, comp, spec):
        r["complexity_score"] = round(c, 4)
        r["spec_score"] = round(s, 4)
    comp_tiers, names = assign_tiers(comp, args.tiers)
    spec_tiers, _ = assign_tiers(spec, args.tiers)
    for r, ct, st in zip(rows, comp_tiers, spec_tiers):
        r["complexity_tier"] = ct
        r["spec_complexity_tier"] = st

    print(f"[DIFFICULTY] {len(rows)} cases, {args.tiers} tiers\n")
    # Show the continuum: raw metric spread per complexity tier.
    print("complexity tier profile (median of each raw metric):")
    hdr = f"  {'tier':12}{'n':>4}" + "".join(f"{m:>16}" for m in COMPLEXITY_METRICS)
    print(hdr)
    for name in names:
        sub = [r for r in rows if r["complexity_tier"] == name]
        line = f"  {name:12}{len(sub):>4}"
        for m in COMPLEXITY_METRICS:
            line += f"{statistics.median(r[m] for r in sub):>16.2f}"
        print(line)

    print("\nspecification complexity profile (quantile bins, NOT the L1/L2/L3 audit):")
    print(f"  {'tier':12}{'n':>4}{'invented':>12}{'inputs':>10}{'NL words':>10}")
    for name in names:
        sub = [r for r in rows if r["spec_complexity_tier"] == name]
        print(f"  {name:12}{len(sub):>4}"
              f"{statistics.median(r['num_invented'] for r in sub):>12.1f}"
              f"{statistics.median(r['num_input'] for r in sub):>10.1f}"
              f"{statistics.median(r['nl_words'] for r in sub):>10.1f}")

    rho = spearman(comp, spec)
    if abs(rho) < 0.3:
        verdict = "weakly related -> two largely independent axes"
    elif abs(rho) < 0.6:
        verdict = "moderately related -> overlapping but not redundant; keep them separate"
    else:
        verdict = "strongly related -> the two axes mostly measure the same thing"
    print(f"\ncomplexity vs specification load: rho = {rho:+.3f} ({verdict})")

    print("\ntier x domain (complexity):")
    doms = sorted({r["category"] for r in rows})
    print(f"  {'domain':22}" + "".join(f"{n:>12}" for n in names))
    for d in doms:
        c = Counter(r["complexity_tier"] for r in rows if r["category"] == d)
        print(f"  {d:22}" + "".join(f"{c.get(n, 0):>12}" for n in names))

    if args.validate:
        print()
        validate_against(rows, args.validate)

    out = Path(args.csv) if args.csv else BENCHMARK_DIR / "res_data" / f"difficulty_{args.dataset}.csv"
    save_csv_rows(out, list(rows[0].keys()), rows)
    print(f"\n[DONE] per-case difficulty -> {out}")


if __name__ == "__main__":
    main()
