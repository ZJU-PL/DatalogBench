"""Batch ablation table for the prompt-configuration axes.

For each model, compare the levels of one axis (fewshot or method) pairwise on
the *same* benchmark cases, and report the effect size with a bootstrap CI and
a paired permutation p-value. Because the pairing is over cases (one pair per
case in the library), this
needs only a single run per cell -- the statistical power comes from the number
of benchmark tasks, not from repeated runs.

With binary outcomes (e.g. pass@1) the paired sign-flip permutation test on the
per-case differences is the standard McNemar-style comparison of two systems on
a shared item set.

An extra ACROSS-MODELS row aggregates the per-model effects: its CI is taken
over models, which is the right uncertainty for "this prompt setting helps in
general" rather than "it helped for this one model".

Examples:
    python evaluation/ablation_report.py --axis fewshot --method signature \\
        --models gpt-5.6-sol claude-opus-5 --metric pass_at_1

    python evaluation/ablation_report.py --axis method --fewshot 0 \\
        --models gpt-5.6-sol claude-opus-5 --metric pass_at_1
"""

import argparse

from common_eval import BENCHMARK_DIR
from common_eval import save_csv_rows
from compare_llm import METRIC_FIELDS
from compare_llm import details_path
from compare_llm import llm_method_tag
from compare_llm import load_case_values
from metrics import bootstrap_mean_ci
from metrics import mean_std_ci95
from metrics import paired_permutation_test


FEWSHOT_LEVELS = [0, 1, 2]
METHOD_LEVELS = ["signature", "description"]


def _cell_stem(model, method, fewshot, scope):
    return f"{model}_{llm_method_tag(method, fewshot)}_{scope}"


def _load_cell(model, method, fewshot, scope, field):
    """{case_id: mean value over runs} for one grid cell."""
    path = details_path(_cell_stem(model, method, fewshot, scope))
    per_case = load_case_values(path, field)
    return {cid: sum(v) / len(v) for cid, v in per_case.items()}


def compare_cells(cell_a, cell_b):
    """Paired effect of A over B across shared cases."""
    shared = sorted(set(cell_a) & set(cell_b))
    if not shared:
        raise ValueError("no shared cases between the two cells")
    diffs = [cell_a[c] - cell_b[c] for c in shared]
    boot = bootstrap_mean_ci(diffs)
    return {
        "n_cases": len(shared),
        "mean_a": sum(cell_a[c] for c in shared) / len(shared),
        "mean_b": sum(cell_b[c] for c in shared) / len(shared),
        "effect": boot["mean"],
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "p_value": paired_permutation_test(diffs),
    }


def _pairs_for_axis(axis, method, fewshot):
    """Yield (label_a, label_b, kwargs_a, kwargs_b) for each pairwise contrast."""
    if axis == "fewshot":
        for i, a in enumerate(FEWSHOT_LEVELS):
            for b in FEWSHOT_LEVELS[i + 1:]:
                yield (
                    f"{a}-shot", f"{b}-shot",
                    {"method": method, "fewshot": a},
                    {"method": method, "fewshot": b},
                )
    else:
        a, b = METHOD_LEVELS
        yield (a, b, {"method": a, "fewshot": fewshot}, {"method": b, "fewshot": fewshot})


def run_ablation(axis, models, method, fewshot, dataset, metric):
    field = METRIC_FIELDS[metric]
    scope = dataset if dataset else "all"
    rows = []

    for label_a, label_b, kw_a, kw_b in _pairs_for_axis(axis, method, fewshot):
        contrast = f"{label_a} - {label_b}"
        per_model_effects = []

        for model in models:
            try:
                cell_a = _load_cell(model, scope=scope, field=field, **kw_a)
                cell_b = _load_cell(model, scope=scope, field=field, **kw_b)
            except FileNotFoundError as exc:
                print(f"[SKIP] {model} [{contrast}]: {exc}")
                continue

            res = compare_cells(cell_a, cell_b)
            per_model_effects.append(res["effect"])
            rows.append({
                "axis": axis,
                "contrast": contrast,
                "model": model,
                "metric": metric,
                "n_cases": res["n_cases"],
                "mean_a": round(res["mean_a"], 4),
                "mean_b": round(res["mean_b"], 4),
                "effect": round(res["effect"], 4),
                "ci95_low": round(res["ci_low"], 4),
                "ci95_high": round(res["ci_high"], 4),
                "p_value": round(res["p_value"], 4),
                "significant": res["p_value"] < 0.05,
            })
            sig = "*" if res["p_value"] < 0.05 else " "
            print(f"  {model:<28} {res['mean_a']:.4f} vs {res['mean_b']:.4f}  "
                  f"effect={res['effect']:+.4f} [{res['ci_low']:+.4f},{res['ci_high']:+.4f}] "
                  f"p={res['p_value']:.4f} {sig}")

        # Across-models aggregate: uncertainty over models, not over cases/runs.
        if len(per_model_effects) >= 2:
            agg = mean_std_ci95(per_model_effects)
            rows.append({
                "axis": axis,
                "contrast": contrast,
                "model": "ACROSS-MODELS",
                "metric": metric,
                "n_cases": len(per_model_effects),  # here: number of models
                "mean_a": "",
                "mean_b": "",
                "effect": round(agg["mean"], 4),
                "ci95_low": round(agg["mean"] - agg["ci95_half"], 4),
                "ci95_high": round(agg["mean"] + agg["ci95_half"], 4),
                "p_value": "",
                "significant": "",
            })
            print(f"  {'ACROSS-MODELS (' + str(len(per_model_effects)) + ' models)':<28} "
                  f"effect={agg['mean']:+.4f} +/- {agg['ci95_half']:.4f} (95% CI over models)")
        print()

    if not rows:
        raise ValueError("No comparable cells found. Run eval_llm.py for the grid first.")

    fixed = f"method-{method}" if axis == "fewshot" else f"fewshot-{fewshot}"
    out_path = BENCHMARK_DIR / "res_data" / f"ablation_{axis}_{fixed}_{scope}_{metric}.csv"
    save_csv_rows(out_path, list(rows[0].keys()), rows)
    print(f"[DONE] ablation table -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Paired ablation table over prompt-configuration axes (single run per cell is sufficient)."
    )
    parser.add_argument("--axis", required=True, choices=["fewshot", "method"],
                        help="Which axis to ablate; the other one is held fixed.")
    parser.add_argument("--models", nargs="+", required=True, help="Models to include")
    parser.add_argument("--method", type=str, default="signature", choices=METHOD_LEVELS,
                        help="Held fixed when --axis is fewshot")
    parser.add_argument("--fewshot", type=int, default=0, choices=FEWSHOT_LEVELS,
                        help="Held fixed when --axis is method")
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--metric", type=str, default="pass_at_1", choices=sorted(METRIC_FIELDS.keys()))
    args = parser.parse_args()

    run_ablation(
        axis=args.axis,
        models=args.models,
        method=args.method,
        fewshot=args.fewshot,
        dataset=args.dataset,
        metric=args.metric,
    )


if __name__ == "__main__":
    main()
