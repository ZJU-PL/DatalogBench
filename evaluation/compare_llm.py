import argparse
import json
import re

from collections import defaultdict
from pathlib import Path

from common_eval import BENCHMARK_DIR
from metrics import bootstrap_ci_mean
from metrics import bootstrap_mean_ci
from metrics import mean_std_ci95
from metrics import paired_permutation_test


METRIC_FIELDS = {
    "pass_at_1": "perfect_match",
    "compile_pass_rate": "compile_ok",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
    # symbolic-only (details rows from eval_symbolic.py carry synth_ok)
    "synth_success": "synth_ok",
}

from model_inventory import AGENT_DEFAULT_MODEL as _CODING_AGENT_DEFAULT_MODEL  # noqa: E402


def llm_method_tag(method: str, fewshot: int) -> str:
    if fewshot == 0:
        return method
    return f"{fewshot}-shot_{method}"


def model_to_path_tag(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model.strip())


def resolve_coding_model(model, agent):
    if model:
        return model
    head = (agent or "").split()[0] if agent else ""
    if head in _CODING_AGENT_DEFAULT_MODEL:
        return _CODING_AGENT_DEFAULT_MODEL[head]
    raise ValueError(f"No default model for coding agent '{agent}', please pass --model")


def side_stem(kind, model, agent, method, fewshot, tool, scope):
    """Filename stem (without the _details.jsonl suffix) for one result set."""
    if kind == "llm":
        return f"{model}_{llm_method_tag(method, fewshot)}_{scope}"
    if kind == "agent":
        return f"{model}_{agent}_{method}_{scope}"
    if kind == "coding":
        return f"{model_to_path_tag(resolve_coding_model(model, agent))}_{agent}_{method}_{scope}"
    if kind == "symbolic":
        return f"{tool}_{scope}"
    raise ValueError(f"Unknown kind: {kind}")


def side_label(kind, model, agent, method, fewshot, tool):
    if kind == "llm":
        return f"{model}/{llm_method_tag(method, fewshot)}"
    if kind == "agent":
        return f"{model}/{agent}_{method}"
    if kind == "coding":
        return f"{model_to_path_tag(resolve_coding_model(model, agent))}/{agent}_{method}"
    if kind == "symbolic":
        return f"{tool}"
    raise ValueError(f"Unknown kind: {kind}")


def details_path(stem: str) -> Path:
    return BENCHMARK_DIR / "res_data" / f"{stem}_details.jsonl"


def load_case_values(path: Path, field: str):
    """Return {case_id: [per-run values]} from a details jsonl file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Details file not found: {path}. Run the matching eval_*.py first."
        )
    per_case = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            value = row[field]
            per_case[row["case_id"]].append(float(bool(value)) if isinstance(value, bool) else float(value))
    return per_case


def per_run_values(path: Path, field: str):
    """Return [per-run aggregate value] (mean over cases within each run)."""
    runs = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            value = row[field]
            runs[row.get("run", 0)].append(float(bool(value)) if isinstance(value, bool) else float(value))
    return [sum(v) / len(v) for _, v in sorted(runs.items())]


def compare(spec_a, spec_b, dataset, metric):
    field = METRIC_FIELDS[metric]
    scope = dataset if dataset else "all"

    stem_a = side_stem(scope=scope, **spec_a)
    stem_b = side_stem(scope=scope, **spec_b)
    path_a = details_path(stem_a)
    path_b = details_path(stem_b)

    cases_a = load_case_values(path_a, field)
    cases_b = load_case_values(path_b, field)

    shared_cases = sorted(set(cases_a) & set(cases_b))
    if not shared_cases:
        raise ValueError("No shared cases between the two result sets.")
    missing = (set(cases_a) | set(cases_b)) - set(shared_cases)
    if missing:
        print(f"[WARN] {len(missing)} case(s) present on only one side are excluded: {sorted(missing)}")

    # Per-case value (mean over repeated runs), restricted to the shared cases so
    # each side's mean is over exactly the item set the paired test uses.
    vals_a = [sum(cases_a[cid]) / len(cases_a[cid]) for cid in shared_cases]
    vals_b = [sum(cases_b[cid]) / len(cases_b[cid]) for cid in shared_cases]
    diffs = [a - b for a, b in zip(vals_a, vals_b)]

    mean_diff = sum(diffs) / len(diffs)
    p_value = paired_permutation_test(diffs)
    ci_low, ci_high = bootstrap_ci_mean(diffs)

    # Primary error bar per side: bootstrap over cases (valid with a single run).
    boot_a = bootstrap_mean_ci(vals_a)
    boot_b = bootstrap_mean_ci(vals_b)
    # Secondary: across-run spread, reported only when repetitions exist.
    runs_a = mean_std_ci95(per_run_values(path_a, field))
    runs_b = mean_std_ci95(per_run_values(path_b, field))

    label_a = side_label(**spec_a)
    label_b = side_label(**spec_b)

    report = {
        "metric": metric,
        "dataset": scope,
        "system_a": label_a,
        "system_b": label_b,
        "kind_a": spec_a["kind"],
        "kind_b": spec_b["kind"],
        "num_shared_cases": len(shared_cases),
        # per-side means with the across-case (bootstrap) CI -- the primary error bar
        "mean_a": boot_a["mean"],
        "ci95_cases_a": boot_a["ci95_half"],
        "cases_ci_a": [boot_a["ci_low"], boot_a["ci_high"]],
        "mean_b": boot_b["mean"],
        "ci95_cases_b": boot_b["ci95_half"],
        "cases_ci_b": [boot_b["ci_low"], boot_b["ci_high"]],
        # across-run reliability; n/a for single-run result sets
        "runs_a": runs_a["n"],
        "runs_b": runs_b["n"],
        "std_runs_a": "n/a" if runs_a["n"] <= 1 else runs_a["std"],
        "ci95_runs_a": "n/a" if runs_a["n"] <= 1 else runs_a["ci95_half"],
        "std_runs_b": "n/a" if runs_b["n"] <= 1 else runs_b["std"],
        "ci95_runs_b": "n/a" if runs_b["n"] <= 1 else runs_b["ci95_half"],
        # paired comparison over cases
        "mean_diff_a_minus_b": mean_diff,
        "diff_bootstrap_ci95": [ci_low, ci_high],
        "p_value_paired_permutation": p_value,
        "significant_at_0.05": p_value < 0.05,
    }

    out_name = f"compare_{stem_a}_vs_{stem_b}_{metric}.json"
    out_path = BENCHMARK_DIR / "res_data" / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    def _runs_note(stats):
        if stats["n"] <= 1:
            return "across-run: n/a (1 run)"
        return f"across-run +/- {stats['ci95_half']:.4f}"

    print(f"[COMPARE] metric={metric} dataset={scope} ({len(shared_cases)} shared cases)")
    print(f"  A [{spec_a['kind']}] {label_a}: {boot_a['mean']:.4f} +/- {boot_a['ci95_half']:.4f} "
          f"(95% CI over cases; {_runs_note(runs_a)})")
    print(f"  B [{spec_b['kind']}] {label_b}: {boot_b['mean']:.4f} +/- {boot_b['ci95_half']:.4f} "
          f"(95% CI over cases; {_runs_note(runs_b)})")
    print(f"  diff (A-B) = {mean_diff:+.4f}, bootstrap 95% CI [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"  paired permutation test p = {p_value:.4f} -> {'significant' if p_value < 0.05 else 'not significant'} at alpha=0.05")
    print(f"[DONE] report -> {out_path}")


def _validate_spec(spec, side):
    if spec["kind"] in {"agent", "coding"} and not spec["agent"]:
        raise ValueError(f"--agent_{side} is required when --kind_{side} is 'agent' or 'coding'")
    if spec["kind"] in {"llm", "agent"} and not spec["model"]:
        raise ValueError(f"--model_{side} is required when --kind_{side} is 'llm' or 'agent'")
    if spec["kind"] == "symbolic" and not spec["tool"]:
        raise ValueError(f"--tool_{side} is required when --kind_{side} is 'symbolic'")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Paired significance test between two evaluated configurations across any of the "
            "three pipelines (llm / agent framework / coding agent)."
        )
    )
    for side in ("a", "b"):
        parser.add_argument(f"--kind_{side}", choices=["llm", "agent", "coding", "symbolic"],
                            default=("llm" if side == "a" else None),
                            help=f"Pipeline of side {side.upper()}. --kind_b defaults to --kind_a.")
        parser.add_argument(f"--model_{side}", type=str, default=None,
                            help=f"Model of side {side.upper()} (coding: optional, resolved from agent).")
        parser.add_argument(f"--agent_{side}", type=str, default=None,
                            help=f"Agent of side {side.upper()} (react/reflection, or codex/claude for coding).")
        parser.add_argument(f"--tool_{side}", type=str, default=None, choices=["gensynth", "egs", "prosynth"],
                            help=f"Symbolic tool of side {side.upper()} (required when --kind_{side} is 'symbolic').")
        parser.add_argument(f"--method_{side}", type=str, default=None, choices=["signature", "description"],
                            help=f"Prompt method of side {side.upper()}. --method_b defaults to --method_a.")
        parser.add_argument(f"--fewshot_{side}", type=int, default=None, choices=[0, 1, 2],
                            help=f"Few-shot of side {side.upper()} (llm only). --fewshot_b defaults to --fewshot_a.")
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--metric", type=str, default="pass_at_1", choices=sorted(METRIC_FIELDS.keys()))
    args = parser.parse_args()

    kind_a = args.kind_a
    kind_b = args.kind_b if args.kind_b is not None else kind_a
    method_a = args.method_a if args.method_a is not None else "signature"
    method_b = args.method_b if args.method_b is not None else method_a
    fewshot_a = args.fewshot_a if args.fewshot_a is not None else 0
    fewshot_b = args.fewshot_b if args.fewshot_b is not None else fewshot_a

    spec_a = {"kind": kind_a, "model": args.model_a, "agent": args.agent_a,
              "tool": args.tool_a, "method": method_a, "fewshot": fewshot_a}
    spec_b = {"kind": kind_b, "model": args.model_b, "agent": args.agent_b,
              "tool": args.tool_b, "method": method_b, "fewshot": fewshot_b}

    _validate_spec(spec_a, "a")
    _validate_spec(spec_b, "b")

    compare(spec_a=spec_a, spec_b=spec_b, dataset=args.dataset, metric=args.metric)


if __name__ == "__main__":
    main()
