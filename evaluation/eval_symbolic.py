import argparse
import json
import time

from common_eval import BENCHMARK_DIR
from common_eval import dataset_index_by_id
from common_eval import dataset_records
from common_eval import evaluate_query_file_for_case
from common_eval import is_exact
from common_eval import grading_provenance
from common_eval import save_csv_rows
from common_eval import save_json
from common_eval import save_jsonl
from common_eval import save_summary_csv
from metrics import calculate_from_confusion
from metrics import case_level_bootstrap
from metrics import mean_std_ci95


DEFAULT_NUM_RUNS = {"gensynth": 5, "egs": 1, "prosynth": 1}


def normalize_dataset_scope(dataset: str) -> str:
    return dataset if dataset else "all"


def query_file_path(tool, scope_dir, case_id, run_id):
    base_dir = BENCHMARK_DIR / "infer_data" / tool / scope_dir
    run_path = base_dir / f"run_{run_id}" / f"{case_id}.dl"
    if run_path.exists():
        return run_path
    legacy_path = base_dir / f"{case_id}.dl"
    if run_id == 0 and legacy_path.exists():
        return legacy_path
    return run_path


def aggregate_json_path(tool, scope_dir, run_id):
    base_dir = BENCHMARK_DIR / "infer_data" / tool / scope_dir
    run_path = base_dir / f"run_{run_id}" / f"{scope_dir}.json"
    if run_path.exists():
        return run_path
    return base_dir / f"{scope_dir}.json"


def load_synth_meta(tool, scope_dir, run_id):
    agg_path = aggregate_json_path(tool, scope_dir, run_id)
    if not agg_path.exists():
        return {}
    with agg_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        return {}
    out = {}
    for row in rows:
        cid = row.get("id")
        if cid:
            out[cid] = {
                "synth_ok": bool(row.get("rules")),
                "synth_sec": row.get("synth_sec"),
                "synth_error": row.get("error", ""),
            }
    return out


def evaluate_one_case(case_record, query_file, meta):
    case_id = case_record["id"]
    start_time = time.time()
    result = evaluate_query_file_for_case(case_record, query_file)
    metric = calculate_from_confusion(result["tp"], result["fp"], result["fn"])
    elapsed_ms = int((time.time() - start_time) * 1000)

    return {
        "case_id": case_id,
        "synth_ok": meta.get("synth_ok", False),
        "compile_ok": result["compile_ok"],
        "perfect_match": is_exact(result),
        "timed_out": bool(result.get("timed_out")),
        "precision": metric["precision"],
        "recall": metric["recall"],
        "f1": metric["f1"],
        "tp": result["tp"],
        "fp": result["fp"],
        "fn": result["fn"],
        "neg_total": result.get("neg_total", 0),
        "neg_hit": result.get("neg_hit", 0),
        "neg_exclusion": (1.0 - result["neg_hit"] / result["neg_total"])
        if result.get("neg_total") else None,
        "variant_count": result["variant_count"],
        "latency_ms": round(elapsed_ms / result["variant_count"], 2) if result["variant_count"] else elapsed_ms,
        "synth_sec": meta.get("synth_sec"),
        "error": " | ".join(result["errors"]) or meta.get("synth_error", ""),
        "per_variant": result["per_variant"],
    }


def summarize_run(run_id, details):
    total = len(details)
    synth_ok = sum(1 for x in details if x["synth_ok"])
    compile_pass = sum(1 for x in details if x["compile_ok"])
    perfect = sum(1 for x in details if x["perfect_match"])
    return {
        "run": run_id,
        "cases": total,
        "synth_ok": synth_ok,
        "compile_pass": compile_pass,
        "perfect_match": perfect,
        "synth_success_rate": synth_ok / total if total else 0.0,
        "compile_pass_rate": compile_pass / total if total else 0.0,
        "pass_at_1": perfect / total if total else 0.0,
        "avg_precision": sum(x["precision"] for x in details) / total if total else 0.0,
        "avg_recall": sum(x["recall"] for x in details) / total if total else 0.0,
        "avg_f1": sum(x["f1"] for x in details) / total if total else 0.0,
        "neg_cases": sum(1 for x in details if x["neg_total"]),
        "avg_neg_exclusion": (
            sum(x["neg_exclusion"] for x in details if x["neg_exclusion"] is not None)
            / sum(1 for x in details if x["neg_exclusion"] is not None)
        ) if any(x["neg_exclusion"] is not None for x in details) else 0.0,
        "avg_latency_ms": round(sum(x["latency_ms"] for x in details) / total, 2) if total else 0.0,
    }


AGGREGATED_METRICS = [
    "synth_success_rate", "compile_pass_rate", "pass_at_1",
    "avg_precision", "avg_recall", "avg_f1", "avg_neg_exclusion",
]

CASE_LEVEL_FIELDS = {
    "compile_pass_rate": "compile_ok",
    "pass_at_1": "perfect_match",
    "avg_precision": "precision",
    "avg_recall": "recall",
    "avg_f1": "f1",
    "avg_neg_exclusion": "neg_exclusion",
    "synth_success_rate": "synth_ok",
}


def add_uncertainty_columns(summary_row, run_summaries, all_details, metrics, case_fields, num_runs):
    """Across-run stats (n/a for a single run) plus across-case bootstrap CIs."""
    single_run = num_runs <= 1
    for metric_name in metrics:
        stats = mean_std_ci95([r[metric_name] for r in run_summaries])
        summary_row[f"{metric_name}_mean"] = stats["mean"]
        summary_row[f"{metric_name}_std_runs"] = "n/a" if single_run else stats["std"]
        summary_row[f"{metric_name}_ci95_runs"] = "n/a" if single_run else stats["ci95_half"]
        field = case_fields.get(metric_name)
        if field:
            boot = case_level_bootstrap(all_details, field)
            summary_row[f"{metric_name}_ci95_cases"] = boot["ci95_half"]
            summary_row[f"{metric_name}_cases_low"] = boot["ci_low"]
            summary_row[f"{metric_name}_cases_high"] = boot["ci_high"]
    return summary_row


def eval_symbolic(tool, dataset="all", case_id=None, num_runs=None):
    if tool not in {"gensynth", "egs", "prosynth"}:
        raise ValueError("--tool must be 'gensynth', 'egs', or 'prosynth'")
    if num_runs is None:
        num_runs = DEFAULT_NUM_RUNS[tool]
    if num_runs <= 0:
        raise ValueError("--num_runs must be > 0")

    dataset_scope = normalize_dataset_scope(dataset)
    scope_dir = case_id if case_id else dataset_scope

    if case_id:
        record = dataset_index_by_id().get(case_id)
        records = [record] if record else []
    else:
        records = dataset_records(dataset_scope)
    if not records:
        raise ValueError("No cases to evaluate. Please check --dataset/--case_id.")

    all_details = []
    run_summaries = []

    for run_id in range(num_runs):
        meta_map = load_synth_meta(tool, scope_dir, run_id)
        run_details = []
        for record in records:
            cid = record["id"]
            qpath = query_file_path(tool, scope_dir, cid, run_id)
            detail = evaluate_one_case(record, qpath, meta_map.get(cid, {}))
            detail["run"] = run_id
            run_details.append(detail)
            print(f"[EVAL:{tool}][run {run_id}] {cid}: synth_ok={detail['synth_ok']} "
                  f"compile_ok={detail['compile_ok']} F1={detail['f1']:.4f}")

        run_details.sort(key=lambda x: x["case_id"])
        all_details.extend(run_details)

        rs = summarize_run(run_id, run_details)
        run_summaries.append(rs)
        print(f"[RUN {run_id}] synth={rs['synth_success_rate']:.4f} pass@1={rs['pass_at_1']:.4f} "
              f"compile={rs['compile_pass_rate']:.4f} avg_f1={rs['avg_f1']:.4f}")

    compact = [{k: v for k, v in d.items() if k != "per_variant"} for d in all_details]

    details_path = BENCHMARK_DIR / "res_data" / f"{tool}_{scope_dir}_details.jsonl"
    runs_path = BENCHMARK_DIR / "res_data" / f"{tool}_{scope_dir}_runs.csv"
    summary_path = BENCHMARK_DIR / "res_data" / f"{tool}_{scope_dir}_summary.csv"

    save_jsonl(details_path, compact)
    save_csv_rows(runs_path, list(run_summaries[0].keys()), run_summaries)

    summary_row = {"tool": tool, "num_runs": num_runs, "cases": run_summaries[0]["cases"]}
    add_uncertainty_columns(
        summary_row, run_summaries, all_details,
        AGGREGATED_METRICS, CASE_LEVEL_FIELDS, num_runs,
    )
    summary_row["avg_latency_ms_mean"] = round(
        sum(r["avg_latency_ms"] for r in run_summaries) / num_runs, 2
    )
    save_summary_csv(summary_path, list(summary_row.keys()), summary_row)

    # The symbolic rows are compared against the LLM and agent rows in the same
    # table, so "which Souffle graded this" has to be answerable for all three.
    # These baselines synthesise with their own toolchains on the host, but the
    # grading step here is the same execution-based oracle the other settings
    # use and can run in the pinned image like they do.
    prov_path = (BENCHMARK_DIR / "res_data" /
                 f"{tool}_{scope_dir}_provenance.json")
    prov = grading_provenance()
    prov.update({"tool": tool, "num_runs": num_runs,
                 "cases": run_summaries[0]["cases"],
                 "note": "synthesis ran on the host with the tool's own "
                         "toolchain; this records the grading step only"})
    save_json(prov_path, prov)

    pass1_boot = case_level_bootstrap(all_details, "perfect_match")
    print(
        f"[SUMMARY] {tool} pass@1 = {pass1_boot['mean']:.4f} "
        f"+/- {pass1_boot['ci95_half']:.4f} (95% CI over {pass1_boot['n']} cases)"
    )
    if num_runs > 1:
        pass1_runs = mean_std_ci95([r["pass_at_1"] for r in run_summaries])
        print(
            f"[SUMMARY] across {num_runs} runs: +/- {pass1_runs['ci95_half']:.4f} "
            f"(95% CI, std={pass1_runs['std']:.4f})"
        )
    else:
        print("[SUMMARY] across-run CI: n/a (single run; case-level CI above is the error bar)")
    print(f"[DONE] details -> {details_path}")
    print(f"[DONE] per-run  -> {runs_path}")
    print(f"[DONE] summary -> {summary_path}")


def report_overlap(tool, dataset="all"):
    """Read-only: which tasks the symbolic tool solves that no direct-prompting cell
    does, and the reverse.  Aggregates cannot show this -- two interfaces with
    similar EX can solve disjoint task sets -- so it is computed on paired tasks."""
    from model_inventory import MODELS
    scope = normalize_dataset_scope(dataset)

    def exact(path):
        if not path.exists():
            return None
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        return {r["case_id"] for r in rows if r.get("run", 0) in (0, None) and r.get("perfect_match")}

    sym = exact(BENCHMARK_DIR / "res_data" / f"{tool}_{scope}_details.jsonl")
    if sym is None:
        raise SystemExit(f"[ERR] no {tool} details for scope {scope}")
    language = set()
    cells = 0
    for model in MODELS:
        for tag in ("signature", "description", "1-shot_signature", "1-shot_description"):
            got = exact(BENCHMARK_DIR / "res_data" / f"{model}_{tag}_{scope}_details.jsonl")
            if got is not None:
                language |= got
                cells += 1
    print(f"[OVERLAP] {tool} vs the union of {cells} direct-prompting cells ({scope})")
    print(f"  {tool} exact                      {len(sym):4d}")
    print(f"  any direct cell exact           {len(language):4d}")
    print(f"  both                            {len(sym & language):4d}")
    print(f"  {tool} only (examples suffice)    {len(sym - language):4d}  {sorted(sym - language)}")
    print(f"  language only                   {len(language - sym):4d}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate symbolic-synthesizer baselines (gensynth/egs).")
    parser.add_argument("--tool", required=True, choices=["gensynth", "egs", "prosynth"])
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--case_id", type=str, default=None)
    parser.add_argument("--num_runs", type=int, default=None, help="Repetitions (default: gensynth=5, egs=1)")
    parser.add_argument("--overlap", action="store_true",
                        help="read-only: task-level overlap with the direct-prompting grid, then exit")
    args = parser.parse_args()

    if args.overlap:
        report_overlap(args.tool, args.dataset)
        return

    eval_symbolic(tool=args.tool, dataset=args.dataset, case_id=args.case_id, num_runs=args.num_runs)


if __name__ == "__main__":
    main()
