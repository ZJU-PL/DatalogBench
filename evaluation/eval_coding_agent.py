import argparse
import json
import re
import sys
import time
import shlex
from pathlib import Path

from common_eval import BENCHMARK_DIR
from common_eval import dataset_index_by_id
from common_eval import dataset_records
from common_eval import case_variants_dirs_six
from common_eval import is_exact
from common_eval import evaluate_query_file_for_case
from common_eval import grading_provenance
from common_eval import save_csv_rows
from common_eval import save_jsonl
from common_eval import save_summary_csv
from metrics import calculate_from_confusion
from metrics import case_level_bootstrap
from metrics import mean_std_ci95


from model_inventory import agent_default_help  # noqa: E402
from model_inventory import resolve_default_model_for_agent  # noqa: F401,E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "synthesis"))
import measurement_v1 as mv1  # noqa: E402


AGENT_PROTOCOL_VERSION = 5


def model_to_path_tag(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model.strip())


def method_tag(agent: str, method: str) -> str:
    return f"{agent}_{method}"


def normalize_dataset_scope(dataset: str) -> str:
    return dataset if dataset else "all"


def query_file_path(model_tag, agent, method, dataset, case_id, single_case_mode, run_id):
    tag = method_tag(agent, method)
    scope = case_id if single_case_mode else dataset
    base_dir = BENCHMARK_DIR / "infer_data" / model_tag / tag / scope
    run_path = base_dir / f"run_{run_id}" / f"{case_id}.dl"
    if run_path.exists():
        return run_path
    # Legacy single-run layout wrote directly under the scope directory.
    legacy_path = base_dir / f"{case_id}.dl"
    if run_id == 0 and legacy_path.exists():
        return legacy_path
    return run_path


def aggregate_json_path(model_tag, agent, method, dataset, case_id, run_id):
    tag = method_tag(agent, method)
    scope = case_id if case_id else dataset
    base_dir = BENCHMARK_DIR / "infer_data" / model_tag / tag / scope
    run_path = base_dir / f"run_{run_id}" / f"{scope}.json"
    if run_path.exists():
        return run_path
    return base_dir / f"{scope}.json"


_QUOTA_MARKERS = (
    "balance insufficient", "insufficient balance", "insufficient credit",
    "credit balance", "insufficient_quota", "quota exceeded",
    "quota_exceeded", "余额不足",
)

_ACCOUNT_DISABLED_MARKERS = (
    "organization has been disabled", "organisation has been disabled",
    "organization is disabled", "organisation is disabled",
    "organization has been deactivated", "organisation has been deactivated",
    "account has been disabled", "account is disabled",
    "account has been deactivated", "organization_disabled",
    "organisation_disabled", "account_disabled",
)


def _is_quota_error(message):
    text = str(message or "").lower()
    http_402 = bool(re.search(
        r"(?:api\s*error|http|status|error|code)[^\n]{0,48}\b402\b"
        r"|\b402\b[^\n]{0,48}(?:error|payment|required|balance)",
        text,
    ))
    return http_402 or any(
        marker in text for marker in _QUOTA_MARKERS
    )


def _is_account_disabled_error(message):
    text = str(message or "").lower()
    return any(marker in text for marker in _ACCOUNT_DISABLED_MARKERS)


def _is_provider_fatal_error(message):
    return _is_quota_error(message) or _is_account_disabled_error(message)


def _is_cli_usage_error(message):
    text = str(message or "").lower()
    parser_error = any(marker in text for marker in (
        "unexpected argument", "unexpected option", "unknown argument",
        "unknown option", "unrecognized argument", "unrecognized option",
        "invalid value for",
    ))
    return parser_error and ("usage:" in text or "for more information" in text)


def _load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def validate_agent_artifact(model_tag, agent, method, dataset, case_id, run_id,
                            expected_ids):
    """Refuse to score CLI/provider failures as agent-generated programs."""
    scope = case_id if case_id else dataset
    base_dir = BENCHMARK_DIR / "infer_data" / model_tag / method_tag(agent, method) / scope
    run_dir = base_dir / f"run_{run_id}"
    aggregate = aggregate_json_path(model_tag, agent, method, dataset, case_id, run_id)
    rows = _load_json(aggregate)
    if not isinstance(rows, list):
        raise RuntimeError(
            f"missing/invalid completed agent aggregate: {aggregate}; "
            "resume synthesis before evaluation"
        )

    by_id = {
        row.get("id"): row for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    missing = sorted(set(expected_ids) - set(by_id))
    extra = sorted(set(by_id) - set(expected_ids))
    duplicates = len(rows) - len(by_id)
    invalid = []
    current_contract = False

    for cid in expected_ids:
        row = by_id.get(cid)
        if row is None:
            continue
        checkpoint = _load_json(run_dir / f"{cid}.agent.json")
        trace = _load_json(run_dir / f"{cid}.trace.json")
        dialogue = _load_json(run_dir / f"{cid}.dialogue.json")
        trace = trace if isinstance(trace, list) else []
        dialogue = dialogue if isinstance(dialogue, list) else []
        authoritative = checkpoint if isinstance(checkpoint, dict) else row
        current = (
            authoritative.get("agent_protocol_version") == AGENT_PROTOCOL_VERSION
            or "checkpoint_valid" in authoritative
        )
        current_contract = current_contract or current
        texts = [
            authoritative.get("response"), authoritative.get("inference_error"),
            authoritative.get("failure_kind"),
        ]
        for item in trace + dialogue:
            if isinstance(item, dict):
                texts.extend((item.get("response"), item.get("stderr"),
                              item.get("failure_kind")))
        failed_turn = any(
            isinstance(item, dict) and item.get("inference_ok") is False
            for item in trace
        )
        provider_fatal = any(_is_provider_fatal_error(text) for text in texts)
        cli_usage = any(_is_cli_usage_error(text) for text in texts)
        invalid_current = current and not (
            authoritative.get("checkpoint_valid") is True
            and authoritative.get("inference_ok") is True
            and authoritative.get("terminal_failure") is False
            and not authoritative.get("inference_error")
        )
        protocol_mismatch = (
            agent in {"codex", "claude"}
            and authoritative.get("agent_protocol_version") != AGENT_PROTOCOL_VERSION
        )
        pool_mismatch = (
            authoritative.get("selection_pool") != "demo"
            or authoritative.get("scoring_pool") != "eval"
        )
        # A finalized terminal zero is a *measurement*, not an artifact: the
        # agent produced no program within its retry budget, and measurement-v1
        # counts that as a legitimate benchmark outcome.  It still has to pass
        # everything that says the run itself was sound -- protocol version,
        # pools, and above all provider/CLI failures, which never become a
        # zero -- so only the artifact-shaped requirements are lifted, and only
        # for a row the finalizer has explicitly marked.
        terminal_zero = mv1.is_terminal_zero(authoritative)
        if terminal_zero and not (provider_fatal or cli_usage
                                  or protocol_mismatch or pool_mismatch):
            continue
        if (
            not (authoritative.get("response") or "").strip()
            or not (run_dir / f"{cid}.dl").is_file()
            or not trace
            or not dialogue
            or len(dialogue) != len(trace)
            or failed_turn
            or provider_fatal
            or cli_usage
            or invalid_current
            or protocol_mismatch
            or pool_mismatch
        ):
            invalid.append(cid)

    failures_path = run_dir / "agent_failures.json"
    manifest = _load_json(failures_path)
    # Before measurement-v1 there was one failure list, so "any failure" and
    # "this run cannot be measured" were the same test.  They are not the same
    # statement: a transport failure that exhausted its retry budget is a
    # benchmark outcome, while a disabled account is a fact about our billing.
    # A finalized manifest states which is which, and only the second kind
    # still blocks.  Manifests without the split keep the old, stricter rule --
    # an unfinalized run must not slip through by omitting the new fields.
    if isinstance(manifest, dict) and manifest.get("measurement_protocol_version"):
        dirty_manifest = (
            manifest.get("measurement_protocol_version") != mv1.MEASUREMENT_PROTOCOL_VERSION
            or manifest.get("invalid_failures")
            or manifest.get("not_attempted")
            or manifest.get("complete_case_set") is not True
            or manifest.get("measurement_complete") is not True
            or manifest.get("quota_exhausted") is True
            or bool(manifest.get("fatal_error_kind"))
        )
    else:
        dirty_manifest = isinstance(manifest, dict) and (
            manifest.get("failures")
            or manifest.get("not_attempted")
            or manifest.get("complete") is not True
            or manifest.get("quota_exhausted") is True
            or bool(manifest.get("fatal_error_kind"))
        )
    if current_contract and not isinstance(manifest, dict):
        raise RuntimeError(
            f"missing agent failure manifest {failures_path}; "
            "resume synthesis before evaluation"
        )
    if missing or extra or duplicates or invalid or dirty_manifest:
        raise RuntimeError(
            f"incomplete/invalid agent artifact {aggregate}: missing={len(missing)} "
            f"extra={len(extra)} duplicates={duplicates} invalid={len(invalid)} "
            f"dirty_manifest={bool(dirty_manifest)}; run synthesis with --resume. "
            f"Sample: {', '.join((missing + extra + invalid)[:8]) or 'n/a'}"
        )


def load_iterations_map(model_tag, agent, method, dataset, case_id, run_id):
    agg_path = aggregate_json_path(model_tag, agent, method, dataset, case_id, run_id)
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
                "iterations_used": row.get("iterations_used"),
                "best_iteration": row.get("best_iteration"),
                "session_turns": row.get("session_turns"),
                # These two are the synthesis loop's own numbers and they come
                # from the DEMO pool -- the only pool protocol v5 lets the loop
                # see.  They are carried for the demo-to-eval generalisation
                # gap, never as exact-match rates; the eval-pool figures below
                # are the ones that count.
                "synth_demo_f1": row.get("f1"),
                "synth_demo_perfect_match": row.get("perfect_match"),
                "score_policy": row.get("score_policy"),
                "measurement_protocol_version": row.get("measurement_protocol_version"),
                "measurement_complete": row.get("measurement_complete"),
                "failure_kind": row.get("failure_kind"),
                "case_attempts": row.get("case_attempts"),
            }
    return out


def evaluate_one_case(case_record, query_file, iter_meta):
    case_id = case_record["id"]
    # A terminal zero has no program: what sits at query_file is the schema
    # block the harness writes when the agent returns nothing.  Executing it
    # would score a bare schema, which compiles and emits an empty relation --
    # and an empty relation hits no counterexample, so it collects a perfect
    # negative-exclusion score for producing nothing.
    if mv1.is_terminal_zero(iter_meta):
        return mv1.terminal_zero_detail(
            case_id, iter_meta,
            variant_count=len(case_variants_dirs_six(case_id)),
        )
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
        "variant_count": result["variant_count"],
        "latency_ms": round(elapsed_ms / result["variant_count"], 2) if result["variant_count"] else elapsed_ms,
        "iterations_used": iter_meta.get("iterations_used"),
        "best_iteration": iter_meta.get("best_iteration"),
        "session_turns": iter_meta.get("session_turns"),
        "error": " | ".join(result["errors"]),
        "per_variant": result["per_variant"],
    }
    return detail


def summarize_run(run_id, details):
    total_cases = len(details)
    compile_pass = sum(1 for x in details if x["compile_ok"])
    perfect_match = sum(1 for x in details if x["perfect_match"])
    turn_values = [x["session_turns"] for x in details if isinstance(x.get("session_turns"), (int, float))]
    avg_iterations = sum(turn_values) / len(turn_values) if turn_values else 0.0
    first_pass_success = sum(1 for x in details if x.get("session_turns") == 1 and x["perfect_match"])
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
        "avg_latency_ms": round(sum(x["latency_ms"] for x in details) / total_cases, 2) if total_cases else 0.0,
        "avg_iterations": round(avg_iterations, 4),
        "first_pass_success_rate": first_pass_success / total_cases if total_cases else 0.0,
    }


AGGREGATED_METRICS = [
    "compile_pass_rate", "pass_at_1", "avg_precision", "avg_recall",
    "avg_f1", "avg_neg_exclusion", "avg_iterations", "first_pass_success_rate",
]

CASE_LEVEL_FIELDS = {
    "compile_pass_rate": "compile_ok",
    "pass_at_1": "perfect_match",
    "avg_precision": "precision",
    "avg_recall": "recall",
    "avg_f1": "f1",
    "avg_neg_exclusion": "neg_exclusion",
    "avg_iterations": "session_turns",
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


def eval_coding_agent(agent, method, model=None, dataset="all", case_id=None, num_runs=3):
    if not agent:
        raise ValueError("--agent must be a non-empty coding agent command tag")
    if method not in {"signature", "description"}:
        raise ValueError("--method must be 'signature' or 'description'")
    if num_runs <= 0:
        raise ValueError("--num_runs must be > 0")

    resolved_model = model if model else resolve_default_model_for_agent(agent)
    model_tag = model_to_path_tag(resolved_model)

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
        validate_agent_artifact(
            model_tag=model_tag,
            agent=agent,
            method=method,
            dataset=dataset_scope,
            case_id=case_id,
            run_id=run_id,
            expected_ids=[record["id"] for record in records],
        )
        iter_map = load_iterations_map(model_tag, agent, method, dataset_scope, case_id, run_id)
        run_details = []
        for record in records:
            cid = record["id"]
            qpath = query_file_path(
                model_tag=model_tag,
                agent=agent,
                method=method,
                dataset=dataset_scope,
                case_id=cid,
                single_case_mode=single_case_mode,
                run_id=run_id,
            )
            detail = evaluate_one_case(record, qpath, iter_meta=iter_map.get(cid, {}))
            detail["run"] = run_id
            run_details.append(detail)
            print(
                f"[EVAL][{agent}][run {run_id}] {cid}: compile_ok={detail['compile_ok']} "
                f"F1={detail['f1']:.4f} turns={detail['session_turns']}"
            )

        run_details.sort(key=lambda x: x["case_id"])
        all_details.extend(run_details)

        run_summary = summarize_run(run_id, run_details)
        run_summaries.append(run_summary)
        print(
            f"[RUN {run_id}] pass@1={run_summary['pass_at_1']:.4f} "
            f"compile={run_summary['compile_pass_rate']:.4f} avg_f1={run_summary['avg_f1']:.4f} "
            f"avg_turns={run_summary['avg_iterations']:.2f}"
        )

    compact_details = [{k: v for k, v in d.items() if k != "per_variant"} for d in all_details]

    summary_scope = case_id if single_case_mode else dataset_scope
    details_path = BENCHMARK_DIR / "res_data" / f"{model_tag}_{agent}_{method}_{summary_scope}_details.jsonl"
    runs_path = BENCHMARK_DIR / "res_data" / f"{model_tag}_{agent}_{method}_{summary_scope}_runs.csv"
    summary_path = BENCHMARK_DIR / "res_data" / f"{model_tag}_{agent}_{method}_{summary_scope}_summary.csv"

    save_jsonl(details_path, compact_details)
    save_csv_rows(runs_path, list(run_summaries[0].keys()), run_summaries)

    prov_path = (BENCHMARK_DIR / "res_data" /
                 f"{model_tag}_{agent}_{method}_{summary_scope}_provenance.json")
    terminal_zero_ids = sorted(
        d["case_id"] for d in all_details
        if d.get("score_policy") == mv1.SCORE_POLICY_TERMINAL_ZERO
    )
    prov = grading_provenance()
    prov.update({
        "agent_protocol_version": AGENT_PROTOCOL_VERSION,
        "measurement_protocol_version": mv1.MEASUREMENT_PROTOCOL_VERSION,
        "agent": agent, "model": resolved_model, "method": method,
        "num_runs": num_runs, "cases": run_summaries[0]["cases"],
        "terminal_zero_cases": terminal_zero_ids,
        "terminal_zero_count": len(terminal_zero_ids),
        "note": (
            "terminal zeros are counted, not executed: the .dl on disk is the "
            "schema stub written when the agent returned nothing, and a bare "
            "schema compiles and emits an empty relation, which would earn a "
            "compile pass and a perfect negative-example exclusion"
        ),
    })
    prov_path.write_text(json.dumps(prov, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    summary_row = {
        "agent": agent,
        "model": resolved_model,
        "method": method,
        "num_runs": num_runs,
        "cases": run_summaries[0]["cases"],
        "iteration_metric": "session_turns",
    }
    add_uncertainty_columns(
        summary_row, run_summaries, all_details,
        AGGREGATED_METRICS, CASE_LEVEL_FIELDS, num_runs,
    )
    summary_row["avg_latency_ms_mean"] = round(
        sum(r["avg_latency_ms"] for r in run_summaries) / num_runs, 2
    )
    save_summary_csv(summary_path, list(summary_row.keys()), summary_row)

    pass1_boot = case_level_bootstrap(all_details, "perfect_match")
    print(
        f"[SUMMARY] {agent}/{method} pass@1 = {pass1_boot['mean']:.4f} "
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate synthesized queries generated by coding agents.")
    parser.add_argument("--agent", type=str, required=True, help="Coding agent command tag: codex or claude")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Base model used during synthesis. "
            + agent_default_help()
        ),
    )
    parser.add_argument("--method", type=str, default="signature", choices=["signature", "description"], help="Prompt method used for generation")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset filter by category, sub category, or all")
    parser.add_argument("--case_id", type=str, default=None, help="Evaluate only one case")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of synthesis repetitions to evaluate (run_0..run_{n-1})")
    args = parser.parse_args()

    eval_coding_agent(
        agent=args.agent,
        method=args.method,
        model=args.model,
        dataset=args.dataset,
        case_id=args.case_id,
        num_runs=args.num_runs,
    )


if __name__ == "__main__":
    main()
