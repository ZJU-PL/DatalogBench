"""Slice evaluation results by the hand-audited specification tier.

`spec_tier` (benchmark/spec_tiers.json) records whether a specification is
self-contained (L1) or determines its target only for a reader who already
knows a domain fact (L2). Slicing by it asks whether domain knowledge is an
independent source of failure, observationally, on a single run.

    python evaluation/tier_report.py --models gpt-5.6-sol claude-opus-5
    python evaluation/tier_report.py --models gpt-5.6-sol --method description

READ THE RESULT WITH CARE. The label is collinear with two dimensions that are
already measured elsewhere:

  * most L2 cases sit in one domain, and most sit in difficulty.py's `hard`
    complexity bin. The exact counts are computed by `collinearity()` and
    printed with every report rather than written here, because a literal in a
    docstring goes stale the moment a case is relabelled and nobody notices.

So an L2-minus-L1 gap is not evidence that domain knowledge costs anything on
its own: most of it may be the domain, or the complexity. Report this
controlled for both, or do not report it. The tool prints the same warning.

Not to be confused with `difficulty.py`'s `spec_complexity_tier`, which bins
cases by how much structure a spec must convey and is equal-count by
construction.
"""

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "evaluation"))

from common_eval import save_csv_rows, spec_tiers  # noqa: E402
from compare_llm import METRIC_FIELDS, details_path, llm_method_tag, load_case_values  # noqa: E402
from metrics import bootstrap_mean_ci  # noqa: E402

DATASET = _REPO / "benchmark" / "dataset.json"
UNLABELLED = "(unlabelled)"


def case_labels():
    """{case_id: tier} for every case, unlabelled ones included."""
    tiers = spec_tiers()
    return {r["id"]: tiers.get(r["id"], UNLABELLED) for r in json.load(open(DATASET))}


def load_cell(model, method, fewshot, scope, field):
    path = details_path(f"{model}_{llm_method_tag(method, fewshot)}_{scope}")
    return {cid: sum(v) / len(v) for cid, v in load_case_values(path, field).items()}


def summarise(values):
    if not values:
        return None
    boot = bootstrap_mean_ci(values)
    return {"n": len(values), "mean": boot["mean"],
            "ci_low": boot["ci_low"], "ci_high": boot["ci_high"]}


def collinearity() -> dict:
    """How far the tier label is confounded with domain and with complexity.

    Computed, not asserted: counts written as literals go stale the moment a
    case is relabelled, and a warning that misstates its own evidence is worse
    than no warning.
    """
    import difficulty
    labels = case_labels()
    l2 = {c for c, b in labels.items() if b == "L2"}
    l1 = {c for c, b in labels.items() if b == "L1"}

    rows = difficulty.collect("all")
    by_id = {r["case_id"]: r for r in rows}
    comp = difficulty.score(rows, difficulty.COMPLEXITY_METRICS)
    tiers, _names = difficulty.assign_tiers(comp, 3)
    hard = {r["case_id"] for r, tier in zip(rows, tiers) if tier == "hard"}

    domains = {}
    for cid in l2:
        d = (by_id.get(cid) or {}).get("category")
        domains[d] = domains.get(d, 0) + 1
    top_domain, top_n = ("unknown", 0)
    if domains:
        top_domain, top_n = max(domains.items(), key=lambda kv: kv[1])
    in_domain = sum(1 for r in rows if r.get("category") == top_domain) or 1

    return {
        "n_l2": len(l2),
        "top_domain": top_domain,
        "top_domain_n": top_n,
        "top_domain_share": top_n / in_domain,
        "hard_n": len(l2 & hard),
        "hard_share": len(l2 & hard) / len(l2) if l2 else 0.0,
        "l1_hard_share": len(l1 & hard) / len(l1) if l1 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description="Slice results by spec_tier or spec_repaired.")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--method", default="signature", choices=["signature", "description"])
    ap.add_argument("--fewshot", type=int, default=0)
    ap.add_argument("--scope", default="all")
    ap.add_argument("--metric", default="pass_at_1", choices=sorted(METRIC_FIELDS))
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    labels = case_labels()
    field = METRIC_FIELDS[args.metric]
    buckets = sorted({v for v in labels.values()})

    rows, missing = [], []
    for model in args.models:
        try:
            cell = load_cell(model, args.method, args.fewshot, args.scope, field)
        except FileNotFoundError as exc:
            missing.append(f"{model}: {exc}")
            continue
        for bucket in buckets:
            vals = [v for cid, v in cell.items() if labels.get(cid) == bucket]
            stat = summarise(vals)
            if stat:
                rows.append({"model": model, "bucket": bucket, "metric": args.metric, **stat})

    if missing:
        for m in missing:
            print(f"[SKIP] {m}")
    if not rows:
        print("[TIER] no results to slice -- run the matching eval_*.py first")
        return

    print(f"[TIER] {args.metric} by spec_tier  "
          f"(method={args.method}, fewshot={args.fewshot}, scope={args.scope})\n")
    width = max(len(r["bucket"]) for r in rows)
    for model in args.models:
        mine = [r for r in rows if r["model"] == model]
        if not mine:
            continue
        print(f"  {model}")
        for r in mine:
            print(f"    {r['bucket']:{width}}  n={r['n']:>4}  {r['mean']:.3f}"
                  f"  [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")

    if len(args.models) > 1:
        print("\n  across models (CI over models, not cases):")
        for bucket in buckets:
            means = [r["mean"] for r in rows if r["bucket"] == bucket]
            if len(means) > 1:
                boot = bootstrap_mean_ci(means)
                print(f"    {bucket:{width}}  {boot['mean']:.3f}"
                      f"  [{boot['ci_low']:.3f}, {boot['ci_high']:.3f}]"
                      f"  over {len(means)} model(s)")

    n_unlabelled = sum(1 for b in labels.values() if b == UNLABELLED)
    if n_unlabelled:
        print(f"\n  NOTE: {n_unlabelled} of {len(labels)} cases carry no tier, so the "
              f"'{UNLABELLED}' row is\n        a gap in the labelling rather than a finding.")
    try:
        co = collinearity()
        print(f"\n  NOTE: the tier is collinear with domain and with complexity -- "
              f"{co['top_domain_n']} of the {co['n_l2']} L2\n"
              f"        cases are {co['top_domain']} ({co['top_domain_share']:.0%} of that domain), "
              f"and {co['hard_n']} of them ({co['hard_share']:.0%})\n"
              f"        are in the hard complexity bin, against {co['l1_hard_share']:.0%} of the L1 cases.\n"
              "        An L2-minus-L1 gap is therefore not on its own evidence about domain\n"
              "        knowledge. Control for both before reporting it.")
    except Exception as exc:                     # noqa: BLE001
        print(f"\n  NOTE: could not compute the collinearity counts ({exc}); the tier is\n"
              "        collinear with domain and complexity -- control for both before\n"
              "        reporting an L2-minus-L1 gap.")

    if args.csv:
        save_csv_rows(Path(args.csv), list(rows[0]), rows)
        print(f"\n[DONE] {args.csv}")


if __name__ == "__main__":
    main()
