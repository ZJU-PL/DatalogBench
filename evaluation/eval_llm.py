import argparse
import hashlib
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
from model_inventory import (
    api_key_env_for,
    check_model_supported,
    endpoint_for,
    max_output_tokens_for,
    requested_model_for,
    reasoning_effort_for,
    stream_response_for,
    thinking_mode_for,
)


def evaluate_one_case(case_record, query_file):
    case_id = case_record["id"]
    start_time = time.time()
    result = evaluate_query_file_for_case(case_record, query_file)
    metric = calculate_from_confusion(result["tp"], result["fp"], result["fn"])
    elapsed_ms = int((time.time() - start_time) * 1000)

    detail = {
        "case_id": case_id,
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
        "latency_ms": elapsed_ms / result["variant_count"],
        "error": " | ".join(result["errors"]),
        "per_variant": result["per_variant"],
    }
    return detail


from common_eval import method_tag  # noqa: E402,F401


def query_file_path(
    model: str,
    method: str,
    fewshot: int,
    dataset: str,
    case_id: str,
    single_case_mode: bool,
    run_id: int,
    self_repair_k: int = 1,
):
    tag = method_tag(method=method, fewshot=fewshot, repair_k=self_repair_k)
    scope = case_id if single_case_mode else dataset
    base_dir = BENCHMARK_DIR / "infer_data" / model / tag / scope
    run_path = base_dir / f"run_{run_id}" / f"{case_id}.dl"
    if run_path.exists():
        return run_path
    # Legacy single-run layout wrote directly under the scope directory.
    legacy_path = base_dir / f"{case_id}.dl"
    if run_id == 0 and legacy_path.exists():
        return legacy_path
    return run_path


def normalize_dataset_scope(dataset: str) -> str:
    return dataset if dataset else "all"


def _load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def validate_inference_artifact(model, method, fewshot, scope, run_id, expected_ids,
                                self_repair_k=1, trust_run_provenance=False):
    """Refuse to score provider failures as model-generated programs."""
    tag = method_tag(method=method, fewshot=fewshot, repair_k=self_repair_k)
    run_dir = BENCHMARK_DIR / "infer_data" / model / tag / scope / f"run_{run_id}"
    aggregate = run_dir / f"{scope}.json"
    failures_path = run_dir / "inference_failures.json"
    if not aggregate.exists():
        raise RuntimeError(
            f"missing completed synthesis aggregate: {aggregate}; "
            "resume synthesis before evaluation"
        )
    try:
        rows = json.loads(aggregate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid synthesis aggregate {aggregate}: {exc}") from exc
    if not isinstance(rows, list):
        raise RuntimeError(f"invalid synthesis aggregate {aggregate}: expected a list")

    by_id = {
        row.get("id"): row for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    missing = sorted(set(expected_ids) - set(by_id))
    extra = sorted(set(by_id) - set(expected_ids))
    duplicates = len(rows) - len(by_id)
    expected_max_tokens = max_output_tokens_for(model)
    expected_reasoning_effort = reasoning_effort_for(model)
    expected_stream = stream_response_for(model)
    expected_requested_model = requested_model_for(model)
    expected_thinking_mode = thinking_mode_for(model)
    expected_base_url = endpoint_for(model)
    expected_api_key_env = api_key_env_for(model)
    provider_contract_required = (
        expected_thinking_mode is not None or expected_requested_model != model
    )
    # Runs recorded before the per-row provider contract existed (and runs whose
    # endpoint differs from what the inventory routes to today) cannot satisfy the
    # per-row check.  Scoring them is still legitimate -- the generations are
    # fixed -- but the exception has to be asked for, and it is only granted when
    # the run's own provenance.json states which model the service served.
    if trust_run_provenance and provider_contract_required:
        prov = _load_json(run_dir / "provenance.json")
        served = (prov or {}).get("served_models") or {}
        if not served:
            raise RuntimeError(
                f"--trust-run-provenance given but {run_dir}/provenance.json records no served model"
            )
        print(f"[CONTRACT] per-row provider contract skipped for {model}: "
              f"run provenance records base_url={prov.get('base_url')} served={sorted(served)}")
        provider_contract_required = False
    direct_seed_hashes = {}
    if self_repair_k > 1:
        direct_run_dir = (
            BENCHMARK_DIR / "infer_data" / model /
            method_tag(method=method, fewshot=fewshot, repair_k=1) /
            scope / f"run_{run_id}"
        )
        for cid in expected_ids:
            seed = _load_json(direct_run_dir / f"{cid}.inference.json")
            if seed is None and scope != "all":
                seed = _load_json(
                    BENCHMARK_DIR / "infer_data" / model /
                    method_tag(method=method, fewshot=fewshot, repair_k=1) /
                    "all" / f"run_{run_id}" / f"{cid}.inference.json"
                )
            if isinstance(seed, dict) and isinstance(seed.get("response"), str):
                direct_seed_hashes[cid] = hashlib.sha256(
                    seed["response"].encode("utf-8")
                ).hexdigest()
    invalid = sorted(
        cid for cid in expected_ids
        if cid in by_id and (
            by_id[cid].get("inference_ok") is not True
            or bool(by_id[cid].get("inference_error"))
            or not (by_id[cid].get("response") or "").strip()
            or not by_id[cid].get("prompt_sha256")
            or by_id[cid].get("max_tokens") != expected_max_tokens
            or by_id[cid].get("reasoning_effort") != expected_reasoning_effort
            or bool(by_id[cid].get("stream")) != expected_stream
            or (provider_contract_required and (
                by_id[cid].get("experiment_model") != model
                or by_id[cid].get("requested_model") != expected_requested_model
                or by_id[cid].get("thinking_mode") != expected_thinking_mode
                or by_id[cid].get("base_url") != expected_base_url
                or by_id[cid].get("api_key_env") != expected_api_key_env
            ))
            or not (run_dir / f"{cid}.dl").is_file()
            or (self_repair_k > 1 and (
                by_id[cid].get("self_repair_protocol_version") != 2
                or by_id[cid].get("self_repair_k") != self_repair_k
                or by_id[cid].get("selection_pool") != "demo"
                or by_id[cid].get("scoring_pool") != "eval"
                or by_id[cid].get("repair_seed") != "direct_checkpoint"
                or by_id[cid].get("direct_seed_sha256") != direct_seed_hashes.get(cid)
                or not isinstance(by_id[cid].get("trajectory"), list)
                or not by_id[cid]["trajectory"]
                or by_id[cid]["trajectory"][0].get("seeded_from_direct") is not True
                or any(
                    step.get("selection_pool") != "demo"
                    for step in by_id[cid]["trajectory"]
                    if step.get("inference_ok") is True
                )
            ))
        )
    )
    if missing or extra or duplicates or invalid:
        raise RuntimeError(
            f"incomplete/legacy inference artifact {aggregate}: missing={len(missing)} "
            f"extra={len(extra)} duplicates={duplicates} invalid={len(invalid)}; "
            f"run synthesis with --resume. "
            f"Sample: {', '.join((missing + extra + invalid)[:8]) or 'n/a'}"
        )

    if not failures_path.exists():
        raise RuntimeError(
            f"missing inference failure manifest {failures_path}; legacy results must be "
            "migrated with synthesis --resume before evaluation"
        )
    failures = json.loads(failures_path.read_text(encoding="utf-8"))
    if failures.get("failures") or failures.get("not_attempted") or not failures.get("complete_case_set"):
        raise RuntimeError(
            f"synthesis failure manifest is not clean: {failures_path}; "
            "resume failed cases before evaluation"
        )


def summarize_run(run_id: int, details):
    total_cases = len(details)
    compile_pass = sum(1 for x in details if x["compile_ok"])
    perfect_match = sum(1 for x in details if x["perfect_match"])
    return {
        "run": run_id,
        "cases": total_cases,
        "compile_pass": compile_pass,
        "perfect_match": perfect_match,
        "compile_pass_rate": compile_pass / total_cases if total_cases else 0.0,
        "pass_at_1": perfect_match / total_cases if total_cases else 0.0,
        "avg_precision": sum(x["precision"] for x in details) / total_cases if total_cases else 0.0,
        "avg_recall": sum(x["recall"] for x in details) / total_cases if total_cases else 0.0,
        "avg_f1": sum(x["f1"] for x in details) / total_cases if total_cases else 0.0,
        "neg_cases": sum(1 for x in details if x["neg_total"]),
        "avg_neg_exclusion": (
            sum(x["neg_exclusion"] for x in details if x["neg_exclusion"] is not None)
            / sum(1 for x in details if x["neg_exclusion"] is not None)
        ) if any(x["neg_exclusion"] is not None for x in details) else 0.0,
        "avg_latency_ms": round(sum(x["latency_ms"] for x in details) / total_cases, 2) if total_cases else 0,
    }


AGGREGATED_METRICS = ["compile_pass_rate", "pass_at_1", "avg_precision", "avg_recall", "avg_f1",
                      "avg_neg_exclusion"]

# Metric -> per-case detail field, for the across-case bootstrap CI. This CI is
# computable from a single run and is the uncertainty a benchmark number should
# carry ("what if we had sampled different tasks"); the across-run CI only
# captures API non-determinism and is n/a when num_runs == 1.
CASE_LEVEL_FIELDS = {
    "compile_pass_rate": "compile_ok",
    "pass_at_1": "perfect_match",
    "avg_precision": "precision",
    "avg_recall": "recall",
    "avg_f1": "f1",
    "avg_neg_exclusion": "neg_exclusion",
}


def add_uncertainty_columns(summary_row, run_summaries, all_details, metrics, case_fields, num_runs):
    """Fill in across-run stats (n/a for a single run) and across-case bootstrap CIs."""
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


def eval_llm(model, method, fewshot=0, dataset="all", case_id=None, num_runs=10,
             self_repair_k=1, trust_run_provenance=False):
    if method not in {"signature", "description"}:
        raise ValueError("--method must be 'signature' or 'description'")
    if fewshot not in {0, 1, 2}:
        raise ValueError("--fewshot must be one of {0, 1, 2}")
    if num_runs <= 0:
        raise ValueError("--num_runs must be > 0")
    if self_repair_k <= 0:
        raise ValueError("--self_repair_k must be > 0")

    dataset_scope = normalize_dataset_scope(dataset)

    if case_id:
        case_map = dataset_index_by_id()
        record = case_map.get(case_id)
        records = [record] if record else []
    else:
        records = dataset_records(dataset_scope)

    if not records:
        raise ValueError("No cases to evaluate. Please check --dataset/--case_id.")

    single_case_mode = case_id is not None

    all_details = []
    run_summaries = []

    for run_id in range(num_runs):
        validate_inference_artifact(
            model=model,
            method=method,
            fewshot=fewshot,
            scope=(case_id if single_case_mode else dataset_scope),
            run_id=run_id,
            expected_ids=[record["id"] for record in records],
            self_repair_k=self_repair_k,
            trust_run_provenance=trust_run_provenance,
        )
        run_details = []
        for record in records:
            cid = record["id"]
            qpath = query_file_path(
                model=model,
                method=method,
                fewshot=fewshot,
                dataset=dataset_scope,
                case_id=cid,
                single_case_mode=single_case_mode,
                run_id=run_id,
                self_repair_k=self_repair_k,
            )
            detail = evaluate_one_case(record, qpath)
            detail["run"] = run_id
            run_details.append(detail)
            print(f"[EVAL][run {run_id}] {cid}: compile_ok={detail['compile_ok']} F1={detail['f1']:.4f}")

        run_details.sort(key=lambda x: x["case_id"])
        all_details.extend(run_details)

        run_summary = summarize_run(run_id, run_details)
        run_summaries.append(run_summary)
        print(
            f"[RUN {run_id}] pass@1={run_summary['pass_at_1']:.4f} "
            f"compile={run_summary['compile_pass_rate']:.4f} avg_f1={run_summary['avg_f1']:.4f}"
        )

    compact_details = [
        {k: v for k, v in d.items() if k != "per_variant"}
        for d in all_details
    ]

    summary_scope = case_id if single_case_mode else dataset_scope
    tag = method_tag(method=method, fewshot=fewshot, repair_k=self_repair_k)
    details_path = BENCHMARK_DIR / "res_data" / f"{model}_{tag}_{summary_scope}_details.jsonl"
    runs_path = BENCHMARK_DIR / "res_data" / f"{model}_{tag}_{summary_scope}_runs.csv"
    summary_path = BENCHMARK_DIR / "res_data" / f"{model}_{tag}_{summary_scope}_summary.csv"

    save_jsonl(details_path, compact_details)
    save_csv_rows(runs_path, list(run_summaries[0].keys()), run_summaries)

    summary_row = {
        "model": model,
        "method": method,
        "fewshot": fewshot,
        "self_repair_k": self_repair_k,
        "num_runs": num_runs,
        "cases": run_summaries[0]["cases"],
    }
    add_uncertainty_columns(
        summary_row, run_summaries, all_details,
        AGGREGATED_METRICS, CASE_LEVEL_FIELDS, num_runs,
    )
    summary_row["avg_latency_ms_mean"] = round(
        sum(r["avg_latency_ms"] for r in run_summaries) / num_runs, 2
    )
    save_summary_csv(summary_path, list(summary_row.keys()), summary_row)

    # Which compiler produced this cell. A grid is graded one cell at a time
    # over many hours, so "the image was pinned" is a claim about every cell,
    # not about the run's first minute, and an image can change under a running
    # grid. The sidecar makes it checkable per cell after the fact.
    prov_path = (BENCHMARK_DIR / "res_data" /
                 f"{model}_{tag}_{summary_scope}_provenance.json")
    prov = grading_provenance()
    prov.update({"model": model, "method": method, "fewshot": fewshot,
                 "num_runs": num_runs, "cases": run_summaries[0]["cases"]})
    save_json(prov_path, prov)

    pass1_boot = case_level_bootstrap(all_details, "perfect_match")
    print(
        f"[SUMMARY] pass@1 = {pass1_boot['mean']:.4f} "
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated Datalog queries from I/O data with metrics.")
    parser.add_argument("--model", type=str, required=True, help="Model name used for generation")
    parser.add_argument("--method", type=str, default="signature", choices=["signature", "description"])
    parser.add_argument("--fewshot", type=int, default=0, choices=[0, 1, 2], help="shot count of the run to score; selects the results directory via method_tag, and must match what synthesis used")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset filter by category, sub category, or all")
    parser.add_argument("--case_id", type=str, default=None, help="Evaluate only one case")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of synthesis repetitions to evaluate (run_0..run_{n-1})")
    parser.add_argument("--trust-run-provenance", dest="trust_run_provenance", action="store_true",
                        help="score a run whose rows predate the per-row provider contract, validating "
                             "against that run's own provenance.json instead (prints what it accepted)")
    parser.add_argument("--self_repair_k", type=int, default=1,
                        help="Repair budget used during synthesis; selects the repair<k> artifact")
    args = parser.parse_args()

    # Before anything reads a results directory. An excluded model may still
    # have artifacts on disk, so scoring one would succeed and silently put a
    # cell back into the grid.
    check_model_supported(args.model)

    eval_llm(
        model=args.model,
        method=args.method,
        fewshot=args.fewshot,
        dataset=args.dataset,
        case_id=args.case_id,
        num_runs=args.num_runs,
        self_repair_k=args.self_repair_k,
        trust_run_provenance=args.trust_run_provenance,
    )
