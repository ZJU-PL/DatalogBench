import argparse
import datetime
import hashlib
import importlib.util
import json
import re
import sys
import tempfile
from collections import Counter

from pathlib import Path
from typing import Dict
from typing import List

from context_synth import BENCHMARK_DIR
from context_synth import write_tasks_json


IODATA_DIR = BENCHMARK_DIR / "io_data"
DATA_DIR = BENCHMARK_DIR / "data"
QUERY_DIR = BENCHMARK_DIR / "query"
DEFAULT_MAX_PROMPT_CHARS = 500_000
LEGACY_MAX_TOKENS = 4096
SELF_REPAIR_PROTOCOL_VERSION = 2


def extract_query(text: str) -> str:
    pattern = r"```(?:datalog)?\s*([\s\S]+?)(?:```|$)"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _read_all_tuple_lines(file_path: Path) -> List[List[str]]:
    if not file_path.exists():
        return []
    tuples = []
    with file_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("//"):
                continue
            if "\t" in line:
                parts = [x.strip() for x in line.split("\t")]
            elif "," in line:
                parts = [x.strip() for x in line.split(",")]
            else:
                parts = [x.strip() for x in line.split()]
            tuples.append(parts)
    return tuples


def _demo_dirs(case_id: str) -> List[Path]:
    """The held-out demonstration inputs for this case, in a stable order.

    Delegates to the same enumerator the evaluator uses so that a prompt can
    never show an input the same run will score. Demonstrating with a scored
    input leaks the oracle, which is why `select_variants.py` holds one out.
    """
    import sys as _sys
    _eval = str(Path(__file__).resolve().parent.parent / "evaluation")
    if _eval not in _sys.path:
        _sys.path.insert(0, _eval)
    from common_eval import case_demo_dirs
    return case_demo_dirs(case_id)


def _load_case_examples(case_id: str, fewshot: int, schema_def: Dict) -> List[Dict]:
    """Load `fewshot` demonstration examples for a case.

    Raises rather than returning fewer than asked for: skipping a missing
    example would silently turn a few-shot cell into a zero-shot one. A short
    prompt is not a failure any checker downstream can see, so this has to fail
    here.
    """
    if fewshot <= 0:
        return []

    demo_dirs = _demo_dirs(case_id)
    if len(demo_dirs) < fewshot:
        raise FileNotFoundError(
            f"{case_id}: {fewshot}-shot requested but the demonstration pool holds "
            f"{len(demo_dirs)} input(s) ({IODATA_DIR / case_id / 'demo'}). "
            "Add demonstration inputs, or run this cell at a lower shot count -- "
            "do not fall back to evaluation variants, which are scored."
        )

    input_relations = [x["relation"] for x in schema_def["input"]]
    output_relations = [x["relation"] for x in schema_def["output"]]
    examples = []

    for variant_dir in demo_dirs[:fewshot]:
        input_rows = {
            rel: _read_all_tuple_lines(variant_dir / f"{rel}.facts")
            for rel in input_relations
        }
        output_rows = {
            rel: _read_all_tuple_lines(variant_dir / f"{rel}.expected")
            for rel in output_relations
        }
        if not any(input_rows.values()):
            raise ValueError(
                f"{case_id}: demonstration input {variant_dir} carries no facts for any "
                "declared input relation; an empty example teaches nothing and would "
                "make this cell indistinguishable from zero-shot."
            )
        examples.append(
            {
                "variant": f"demo/{variant_dir.name}",
                "input": input_rows,
                "output": output_rows,
            }
        )

    return examples


def format_relation_block(schema_items: List[Dict], method: str) -> str:
    lines = []
    for item in schema_items:
        lines.append(f"- Signature: {item['signature']}")
        if method == "description":
            lines.append(f"  Description: {item['description']}")
    return "\n".join(lines)


def format_example_block(examples: Dict[str, List[List[str]]]) -> str:
    lines = []
    for relation_name, rows in examples.items():
        lines.append(f"- {relation_name}:")
        if not rows:
            lines.append("  (empty)")
            continue
        for row in rows:
            lines.append(f"  - ({', '.join(row)})")
    return "\n".join(lines)


def format_fewshot_block(case_examples: List[Dict]) -> str:
    if not case_examples:
        return "Few-shot IO Examples:\n(None)\n\n"

    lines = ["Few-shot IO Examples:"]
    for example in case_examples:
        lines.append(f"Example Variant {example['variant']}:")
        lines.append("Input Tuples:")
        lines.append(format_example_block(example["input"]))
        lines.append("Output Tuples:")
        lines.append(format_example_block(example["output"]))
        lines.append("")
    return "\n".join(lines)


def format_knowledge_block(task: Dict) -> str:
    """The knowledge field, always shown when the case carries one.

    It is not an on/off factor: most of the fields state things any capable
    model already knows (Maven's three coordinates, the Atwater calorie factors,
    that an abstract method has no body), so withholding them removes no
    information, while supplying one also tells the model *which* facts are
    relevant -- confounding information with salience.

    The field is part of the specification -- a spec that needs domain
    vocabulary to determine its target should carry it -- and `spec_tier` L2
    records which cases those are, so results can be sliced by it instead.
    """
    text = (task.get("knowledge") or "").strip()
    if not text:
        return ""
    return f"Domain Knowledge:\n{text}\n\n"


def build_prompt(task: Dict, method: str, case_examples: List[Dict],
                 ) -> str:
    header = (
        "System Instruction:\n"
        "You are an expert Datalog programmer specializing in the Souffle dialect. "
        "Write strict, safe, and efficient Datalog rules to satisfy the task.\n\n"
    )

    task_block = (
        f"Task Case: {task['case_id']}\n"
        f"Natural Language Query:\n{task['nl_query']}\n\n"
        "Input Relations:\n"
        f"{format_relation_block(task['schema_def']['input'], method)}\n\n"
        "Output Relations:\n"
        f"{format_relation_block(task['schema_def']['output'], method)}\n\n"
    )

    strategy_block = (
        "Synthesis Constraints:\n"
        "1. Ensure all head variables appear in body rules (safety).\n"
        "2. Use recursive rules when required by transitive semantics.\n"
        "3. Output only final Datalog rules in one code block.\n"
        "4. Do not include any comments in the generated Datalog code.\n\n"
    )

    knowledge_block = format_knowledge_block(task)
    fewshot_block = format_fewshot_block(case_examples)

    footer = (
        "Final Output Format:\n"
        "Return only one markdown code block in Souffle Datalog format.\n"
        "```datalog\n"
        "<rules>\n"
        "```\n"
    )

    return header + task_block + knowledge_block + strategy_block + fewshot_block + footer


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def prepare_prompts(tasks, method: str, fewshot: int, max_prompt_chars: int):
    """Build every prompt before the first API call and enforce a size guard.

    Merely checking that a demo directory exists missed LUBM's 15.8 MB example;
    the provider rejected it only after the sweep had already reached that case.
    Character count is tokenizer-independent and deliberately conservative.  The
    exact count and digest are recorded per case for provenance and resume checks.
    """
    prepared = []
    too_large = []
    for task in tasks:
        cid = task["case_id"]
        examples = _load_case_examples(cid, fewshot=fewshot, schema_def=task["schema_def"])
        prompt = build_prompt(task, method=method, case_examples=examples)
        chars = len(prompt)
        if chars > max_prompt_chars:
            too_large.append((cid, chars))
        prepared.append((task, prompt, chars, prompt_sha256(prompt)))
    if too_large:
        detail = ", ".join(f"{cid}={chars:,}" for cid, chars in too_large[:8])
        raise ValueError(
            f"{len(too_large)} prompt(s) exceed --max_prompt_chars={max_prompt_chars:,}: "
            f"{detail}. Compact the held-out demo or raise the limit explicitly; "
            "do not send an over-context request and score the empty reply."
        )
    largest = max(prepared, key=lambda item: item[2]) if prepared else None
    if largest:
        print(f"[PREFLIGHT] {len(prepared)} prompts fit max_prompt_chars={max_prompt_chars:,}; "
              f"largest={largest[0]['case_id']} ({largest[2]:,} chars)")
    return prepared


def _atomic_json(path: Path, payload) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def inference_window(rows, prev_prov):
    """When the artifact's responses were produced, and how we know.

    Separate from the clock of the pass that happens to be writing the file. A
    resume can reuse every checkpoint and make no call, and writing that pass's
    clock into `started/finished_at_utc` relabels months-old inference as
    happening now -- invisibly, because prompts, responses and scores are all
    unchanged across such a pass.

    Module level so the regression test calls this rather than restating it;
    a second copy of a rule is how the proxy allowlist came to name a host the
    CLIs never dialled.
    """
    stamps = sorted(
        row["generated_at_utc"] for row in rows if row.get("generated_at_utc")
    )
    prev_prov = prev_prov or {}
    if stamps:
        start, end = stamps[0], stamps[-1]
        # A resume adds new rows beside reused ones, so the artifact's window
        # spans both passes.
        if prev_prov.get("started_at_utc"):
            start = min(start, prev_prov["started_at_utc"])
        if prev_prov.get("finished_at_utc"):
            end = max(end, prev_prov["finished_at_utc"])
        source = "per-case generated_at_utc"
    elif prev_prov.get("started_at_utc"):
        start = prev_prov["started_at_utc"]
        end = prev_prov.get("finished_at_utc")
        source = "carried forward: this pass generated nothing"
    else:
        # Only reachable for a cell whose rows all predate this field. The
        # honest answer is that we do not know, not this invocation's clock.
        start = end = None
        source = "unknown: no row carries a generation timestamp"
    return {
        "started_at_utc": start,
        "finished_at_utc": end,
        "inference_window_source": source,
        "cases_generated_this_invocation": len(stamps),
    }


def _reusable_row(row, *, prompt_digest: str, max_tokens: int,
                  model: str, fewshot: int, case_id: str,
                  reasoning_effort=None, stream_response: bool = False,
                  requested_model=None, thinking_mode=None, base_url=None,
                  api_key_env=None) -> bool:
    """Whether a checkpoint was produced under the current inference contract."""
    if not isinstance(row, dict) or not (row.get("response") or "").strip():
        return False
    if row.get("inference_ok") is False or row.get("inference_error"):
        return False
    if "prompt_sha256" in row:
        # Existing non-DeepSeek checkpoints predate these fields and remain
        # reusable. DeepSeek rows must carry the full provider contract because
        # endpoint, API model, and thinking switch define the experimental arm.
        provider_contract_required = (
            thinking_mode is not None
            or (requested_model is not None and requested_model != model)
        )
        provider_contract_matches = (
            not provider_contract_required
            or (
                row.get("experiment_model") == model
                and row.get("requested_model") == requested_model
                and row.get("thinking_mode") == thinking_mode
                and row.get("base_url") == base_url
                and row.get("api_key_env") == api_key_env
            )
        )
        return (
            row.get("inference_ok") is True
            and row.get("prompt_sha256") == prompt_digest
            and row.get("max_tokens") == max_tokens
            and row.get("reasoning_effort") == reasoning_effort
            and bool(row.get("stream")) == stream_response
            and provider_contract_matches
        )

    # Compatibility for the pre-checkpoint artifacts.  Their known contract was
    # 4096 tokens and the old LUBM demo.  Reuse successful plain-model cases, but
    # force all reasoning-budget changes and every affected LUBM 1-shot case.
    if reasoning_effort is not None or stream_response:
        return False
    if max_tokens != LEGACY_MAX_TOKENS:
        return False
    if fewshot > 0 and case_id == "LUBM":
        return False
    return True


def _direct_seed_rows(*, model, method, fewshot, scope_dir, run_id, prepared,
                      max_tokens, reasoning_effort, stream_response,
                      requested_model, thinking_mode, base_url, api_key_env):
    """Load a complete, contract-matched Direct run before repair starts."""
    direct_tag = method_tag(method=method, fewshot=fewshot, repair_k=1)
    run_dir = BENCHMARK_DIR / "infer_data" / model / direct_tag / scope_dir / f"run_{run_id}"
    fallback_run_dir = (
        BENCHMARK_DIR / "infer_data" / model / direct_tag / "all" / f"run_{run_id}"
    )
    seeds = {}
    invalid = []
    for task, _, _, prompt_digest in prepared:
        cid = task["case_id"]
        checkpoint = _load_json(run_dir / f"{cid}.inference.json", None)
        if checkpoint is None and fallback_run_dir != run_dir:
            checkpoint = _load_json(fallback_run_dir / f"{cid}.inference.json", None)
        candidate = checkpoint
        if not _reusable_row(
            candidate,
            prompt_digest=prompt_digest,
            max_tokens=max_tokens,
            model=model,
            fewshot=fewshot,
            case_id=cid,
            reasoning_effort=reasoning_effort,
            stream_response=stream_response,
            requested_model=requested_model,
            thinking_mode=thinking_mode,
            base_url=base_url,
            api_key_env=api_key_env,
        ):
            invalid.append(cid)
            continue
        seeds[cid] = candidate
    if invalid:
        raise RuntimeError(
            f"Direct seed run is incomplete or does not match the current contract: {run_dir} "
            f"(fallback {fallback_run_dir}); "
            f"invalid={len(invalid)} sample={', '.join(invalid[:8])}. Complete the Direct "
            "signature/zero-shot cell first; repair must not generate a different k=1 baseline."
        )
    return seeds


def build_decl_and_output_block(schema_def: Dict) -> str:
    """Fallback header built from dataset.json signatures (untyped -> all symbol)."""
    lines = []
    for item in schema_def["input"]:
        args = ", ".join([f"{x}:symbol" for x in item["args"]])
        lines.append(f".decl {item['relation']}({args})")
        lines.append(f".input {item['relation']}")
    for item in schema_def["output"]:
        args = ", ".join([f"{x}:symbol" for x in item["args"]])
        lines.append(f".decl {item['relation']}({args})")
        lines.append(f".output {item['relation']}")
    return "\n".join(lines)


def load_golden_decl_block(case_id: str) -> str:
    """Typed declarations of the *graded interface* only, from benchmark/query/<id>.dl.

    The dataset.json signatures carry argument names but not types, so the
    authoritative typed schema lives in the golden program.  Only the relations
    the task declares as inputs or outputs belong to the graded interface: they
    fix what a candidate is run on and scored against, so the harness supplies
    them rather than trusting a candidate to retype them.

    Auxiliary relations of the golden program are deliberately NOT included.
    They are the reference's own decomposition; injecting their declarations
    both hands a candidate part of the reference solution and masks whether the
    candidate declared the relations it actually invented.
    """
    golden_path = QUERY_DIR / f"{case_id}.dl"
    if not golden_path.exists():
        return ""
    decl_lines = {}
    graded = []
    for raw in golden_path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        m = re.match(r"\.decl\s+([A-Za-z_]\w*)", stripped)
        if m:
            decl_lines[m.group(1)] = stripped
            continue
        m = re.match(r"\.(input|output)\s+([A-Za-z_]\w*)", stripped)
        if m:
            graded.append((m.group(2), stripped))
    # A relation that is both an input and an output appears twice in `graded`,
    # and emitting its .decl once per directive makes Souffle reject the composed
    # program with "Redefinition of relation" -- for every candidate, however
    # well formed. Declare each name once, at its first directive; the .input and
    # .output lines are both still emitted.
    lines = []
    declared = set()
    for name, directive in graded:
        if name in decl_lines and name not in declared:
            lines.append(decl_lines[name])
            declared.add(name)
        lines.append(directive)
    return "\n".join(lines)


def split_candidate_directives(rule_text: str, schema_names):
    """Separate a candidate's own declarations from its rules.

    A candidate's ``.decl`` for a relation it invented is part of its program and
    is kept: deleting it manufactures an "undefined relation" the candidate did
    not commit.  Two kinds of line are still dropped, because they would change
    what is graded rather than what is computed: a redeclaration of a relation in
    the graded interface (the golden typed declaration wins), and any
    ``.input``/``.output`` directive, which would let a candidate choose the
    relations it is scored on.
    """
    kept_decls, rules = [], []
    for line in rule_text.splitlines():
        stripped = line.strip()
        m = re.match(r"\.decl\s+([A-Za-z_]\w*)", stripped)
        if m:
            if m.group(1) not in schema_names:
                kept_decls.append(stripped)
            continue
        if re.match(r"\.(input|output)\b", stripped):
            continue
        rules.append(line)
    return kept_decls, "\n".join(rules).strip()


def sanitize_rules(rule_text: str) -> str:
    """Rules only, with every directive removed (kept for callers that want the body)."""
    return split_candidate_directives(rule_text, schema_names=set())[1]


def compose_full_dl(schema_def: Dict, rule_text: str, case_id: str = None) -> str:
    """The program that is compiled and scored: graded interface + candidate program."""
    declarations = load_golden_decl_block(case_id) if case_id else ""
    if not declarations:
        declarations = build_decl_and_output_block(schema_def)
    schema_names = set(re.findall(r"^\.decl\s+([A-Za-z_]\w*)", declarations, re.M))
    own_decls, rules = split_candidate_directives(rule_text, schema_names)
    parts = [declarations]
    if own_decls:
        parts.append("\n".join(own_decls))
    if rules:
        parts.append(rules)
    return "\n\n".join(parts) + "\n"


def _context_file_path(dataset: str, case_id: str = None) -> Path:
    file_name = f"{case_id}.json" if case_id else f"{dataset}.json"
    return DATA_DIR / file_name


def load_or_build_context(dataset: str, case_id: str = None) -> List[Dict]:
    context_path = _context_file_path(dataset=dataset, case_id=case_id)
    if not context_path.exists():
        context_path = write_tasks_json(dataset=dataset, case_id=case_id)

    with context_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError(f"Invalid context format in {context_path}")
    return payload


def method_tag(method: str, fewshot: int, repair_k: int = 1) -> str:
    """Delegates to the shared definition in evaluation/common_eval.py."""
    import sys as _sys
    from pathlib import Path as _P
    _eval = str(_P(__file__).resolve().parent.parent / "evaluation")
    if _eval not in _sys.path:
        _sys.path.insert(0, _eval)
    from common_eval import method_tag as _shared
    return _shared(method, fewshot, repair_k=repair_k)


# --------------------------------------------------------------------------- #
# Self-repair (controlled k-iteration sweep).
#
# For k > 1, each candidate is executed only on the held-out
# demonstration/development instance. The compile error or bounded semantic
# mismatch is fed back and the model revises up to k attempts. The disjoint
# evaluation variants are never touched during generation, candidate selection,
# or early stopping; the frozen candidate is graded on them afterwards.
# --------------------------------------------------------------------------- #
def _load_eval_tools():
    eval_dir = Path(__file__).resolve().parent.parent / "evaluation"

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(f"{name}_dynamic", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    common_eval = _load("common_eval", eval_dir / "common_eval.py")
    metrics = _load("metrics", eval_dir / "metrics.py")
    return (
        common_eval.evaluate_query_file_for_case,
        common_eval.case_demo_dirs,
        metrics.calculate_from_confusion,
    )


def _feedback_from_result(result: Dict) -> str:
    def render_tuple(relation, row):
        fields = ", ".join(json.dumps(value, ensure_ascii=False) for value in row)
        return f"{relation}({fields})"

    lines = []
    for item in result.get("per_variant", []):
        variant = item.get("variant")
        if not item.get("compile_ok"):
            lines.append(f"variant {variant}: compile failed: {item.get('error', '')}")
        elif item.get("fp", 0) != 0 or item.get("fn", 0) != 0:
            lines.append(f"variant {variant}: mismatch fp={item.get('fp', 0)} fn={item.get('fn', 0)}")
            inputs = [
                render_tuple(facts.get("relation", "?"), row)
                for facts in item.get("input_facts", [])
                for row in facts.get("tuples", [])
            ]
            expected_only = [
                render_tuple(diff.get("relation", "?"), row)
                for diff in item.get("counterexamples", [])
                for row in diff.get("expected_only", [])
            ]
            produced_only = [
                render_tuple(diff.get("relation", "?"), row)
                for diff in item.get("counterexamples", [])
                for row in diff.get("produced_only", [])
            ]
            if inputs:
                lines.append("  input facts (bounded sample): " + "; ".join(inputs))
            if expected_only:
                lines.append("  expected but missing: " + "; ".join(expected_only))
            if produced_only:
                lines.append("  produced but unexpected: " + "; ".join(produced_only))
    if not lines:
        return " | ".join(result.get("errors", [])) or "unknown mismatch"
    return "\n".join(lines)


def _build_repair_prompt(base_prompt: str, previous_query: str, feedback: str) -> str:
    return (
        base_prompt
        + "\nYour previous program was executed on the development I/O instance.\n"
        + "Previous Datalog program:\n```datalog\n"
        + previous_query.strip()
        + "\n```\nFeedback:\n"
        + feedback
        + "\nReturn a corrected Datalog program, only one ```datalog``` code block, no comments.\n"
    )


def _repair_selection_rank(*, perfect, compile_ok, f1, fp, fn):
    """Prefer dev exactness, then F1, executability, and fewer disagreements."""
    return (bool(perfect), float(f1), bool(compile_ok), -(int(fp) + int(fn)))


def _evaluate_full_dl(case_record, full_dl, evaluate_query_file_for_case,
                      case_demo_dirs, calculate_from_confusion):
    with tempfile.TemporaryDirectory(prefix=f"repair_{case_record['id']}_") as tmp:
        query_file = Path(tmp) / f"{case_record['id']}.dl"
        query_file.write_text(full_dl, encoding="utf-8")
        result = evaluate_query_file_for_case(
            case_record,
            query_file,
            include_counterexamples=True,
            variant_dirs=case_demo_dirs(case_record["id"]),
            evaluation_pool="demo",
        )
    metric = calculate_from_confusion(result["tp"], result["fp"], result["fn"])
    # Same rule as common_eval.is_exact, which this module does not import:
    # an unfinished (timed-out) program is never exact.
    perfect = bool(result["compile_ok"] and not result.get("timed_out")
                   and result["fp"] == 0 and result["fn"] == 0)
    return result, metric, perfect


def _run_self_repair(
    llm, task, base_prompt, case_record, k, max_tokens, eval_tools,
    quota_exhausted_type, direct_seed=None,
):
    """Up to k generate->execute->feedback iterations. Returns (best, trajectory,
    first_perfect_iteration, inference_error). best is chosen by F1 across
    iterations. Iteration 1 may be the verified Direct checkpoint, making the
    controlled baseline byte-identical. A provider failure invalidates the case
    rather than becoming an empty candidate program."""
    evaluate_query_file_for_case, case_demo_dirs, calculate_from_confusion = eval_tools
    best = None
    trajectory = []
    first_perfect_iteration = None
    feedback = ""
    inference_error = ""
    quota_stop = False

    previous_query = ""
    for iteration in range(1, k + 1):
        seeded = iteration == 1 and direct_seed is not None
        if seeded:
            response = direct_seed["response"]
            served_model = direct_seed.get("served_model")
            attempts = 0
            attempt_elapsed_seconds = []
            finish_reason = direct_seed.get("finish_reason")
            reasoning_chars = direct_seed.get("reasoning_chars", 0)
            completion_tokens = direct_seed.get("completion_tokens")
            reasoning_tokens = direct_seed.get("reasoning_tokens")
        else:
            prompt = (base_prompt if iteration == 1 else
                      _build_repair_prompt(base_prompt, previous_query, feedback))
            try:
                response = llm.infer(prompt, max_tokens=max_tokens)
            except quota_exhausted_type as exc:
                response = ""
                llm.last_error = str(exc)
                llm.last_failure_kind = "quota"
                quota_stop = True
            served_model = llm.last_served_model
            attempts = llm.last_attempts
            attempt_elapsed_seconds = list(llm.last_attempt_elapsed_seconds)
            finish_reason = llm.last_finish_reason
            reasoning_chars = llm.last_reasoning_chars
            completion_tokens = llm.last_completion_tokens
            reasoning_tokens = llm.last_reasoning_tokens
        if not response and llm.last_error:
            inference_error = llm.last_error
            trajectory.append({
                "iteration": iteration,
                "inference_ok": False,
                "inference_error": inference_error,
                "failure_kind": llm.last_failure_kind,
                "attempts": attempts,
                "attempt_elapsed_seconds": attempt_elapsed_seconds,
                "finish_reason": finish_reason,
                "reasoning_chars": reasoning_chars,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": reasoning_tokens,
                "served_model": served_model,
                "seeded_from_direct": seeded,
            })
            break
        query_rules = extract_query(response)
        previous_query = query_rules
        full_dl = compose_full_dl(task["schema_def"], query_rules, case_id=task["case_id"])

        result, metric, perfect = _evaluate_full_dl(
            case_record, full_dl, evaluate_query_file_for_case, case_demo_dirs,
            calculate_from_confusion,
        )
        round_feedback = "" if perfect else _feedback_from_result(result)
        trajectory.append({
            "iteration": iteration,
            "selection_pool": "demo",
            "dev_compile_ok": result["compile_ok"],
            "dev_perfect_match": perfect,
            "dev_precision": metric["precision"],
            "dev_recall": metric["recall"],
            "dev_f1": metric["f1"],
            "dev_tp": result["tp"],
            "dev_fp": result["fp"],
            "dev_fn": result["fn"],
            "query": query_rules,
            "full_dl": full_dl,
            "feedback": round_feedback,
            "served_model": served_model,
            "inference_ok": True,
            "inference_error": "",
            "failure_kind": "",
            "attempts": attempts,
            "attempt_elapsed_seconds": attempt_elapsed_seconds,
            "finish_reason": finish_reason,
            "reasoning_chars": reasoning_chars,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "seeded_from_direct": seeded,
        })
        rank = _repair_selection_rank(
            perfect=perfect,
            compile_ok=result["compile_ok"],
            f1=metric["f1"],
            fp=result["fp"],
            fn=result["fn"],
        )
        if best is None or rank > best["selection_rank"]:
            best = {"query": query_rules, "full_dl": full_dl, "response": response,
                    "f1": metric["f1"], "iteration": iteration,
                    "served_model": served_model, "selection_rank": rank,
                    "finish_reason": finish_reason,
                    "reasoning_chars": reasoning_chars,
                    "completion_tokens": completion_tokens,
                    "reasoning_tokens": reasoning_tokens}
        if perfect:
            first_perfect_iteration = iteration
            break
        feedback = round_feedback

    return best, trajectory, first_perfect_iteration, inference_error, quota_stop


def synthesize_llm(
    dataset: str,
    model: str,
    api_key: str,
    method: str = "signature",
    fewshot: int = 0,
    case_id: str = None,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_tokens: int = None,
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
    num_runs: int = 1,
    self_repair_k: int = 1,
    repair_seed_from_direct: bool = False,
    resume: bool = False,
):
    try:
        from online_llm import OnlineLLM, QuotaExhausted
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency for online model calls. Please install requirements.txt (openai package required)."
        ) from exc

    if method not in {"signature", "description"}:
        raise ValueError("--method must be 'signature' or 'description'")
    if fewshot not in {0, 1, 2}:
        raise ValueError("--fewshot must be one of {0, 1, 2}")
    if num_runs <= 0:
        raise ValueError("--num_runs must be > 0")
    if self_repair_k <= 0:
        raise ValueError("--self_repair_k must be > 0")
    if repair_seed_from_direct and self_repair_k == 1:
        raise ValueError("--repair_seed_from_direct requires --self_repair_k > 1")
    if self_repair_k > 1 and not repair_seed_from_direct:
        raise ValueError(
            "controlled self-repair requires --repair_seed_from_direct so k=1 is "
            "identical to the matching Direct checkpoint"
        )
    if max_prompt_chars <= 0:
        raise ValueError("--max_prompt_chars must be > 0")

    eval_dir = str(Path(__file__).resolve().parent.parent / "evaluation")
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    from model_inventory import max_output_tokens_for
    if max_tokens is None:
        max_tokens = max_output_tokens_for(model)
    if max_tokens <= 0:
        raise ValueError("--max_tokens must be > 0")

    llm = OnlineLLM(model=model, temperature=temperature, top_p=top_p, api_key=api_key)
    tasks = load_or_build_context(dataset=dataset, case_id=case_id)
    prepared = prepare_prompts(tasks, method, fewshot, max_prompt_chars)
    print(f"[CONFIG] experiment_model={model} requested_model={llm.requested_model} "
          f"base_url={llm.base_url} api_key_env={llm.api_key_env} "
          f"thinking_mode={llm.thinking_mode or 'n/a'} max_tokens={max_tokens} "
          f"request_timeout={llm.request_timeout}s max_attempts={llm.max_attempts} "
          f"reasoning_effort={llm.reasoning_effort or 'provider-default'} "
          f"stream={llm.stream_response} "
          f"resume={resume}")

    dataset_scope = dataset if dataset else "all"
    # One tag per configuration, from the shared definition. The self-repair
    # modifier must appear here: two configurations that share a directory would
    # silently overwrite each other's programs.
    tag = method_tag(method=method, fewshot=fewshot, repair_k=self_repair_k)

    eval_tools = _load_eval_tools() if self_repair_k > 1 else None

    scope_dir = case_id if case_id else dataset_scope
    output_base_dir = BENCHMARK_DIR / "infer_data" / model / tag / scope_dir

    direct_seeds_by_run = {}
    if repair_seed_from_direct:
        # Resolve every seed before the first repair API call. A missing Direct
        # case must abort the cell, not silently turn into an unmatched first
        # generation halfway through the run.
        for run_id in range(num_runs):
            direct_seeds_by_run[run_id] = _direct_seed_rows(
                model=model,
                method=method,
                fewshot=fewshot,
                scope_dir=scope_dir,
                run_id=run_id,
                prepared=prepared,
                max_tokens=max_tokens,
                reasoning_effort=llm.reasoning_effort,
                stream_response=llm.stream_response,
                requested_model=llm.requested_model,
                thinking_mode=llm.thinking_mode,
                base_url=llm.base_url,
                api_key_env=llm.api_key_env,
            )
        print(f"[PREFLIGHT] self-repair k=1 seeds verified from Direct: "
              f"{num_runs} run(s) x {len(prepared)} case(s)")

    incomplete_runs = 0
    for run_id in range(num_runs):
        run_dir = output_base_dir / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        aggregate_path = run_dir / f"{scope_dir}.json"
        legacy_rows = _load_json(aggregate_path, []) if resume else []
        legacy_by_id = {
            row.get("id"): row for row in legacy_rows
            if isinstance(row, dict) and row.get("id")
        }

        # One provenance window per run, so run_i/provenance.json describes
        # exactly the calls that produced run_i.
        llm.reset_provenance()
        started_at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        # A resume pass can reuse every row and make no call at all. Overwriting
        # the window with this pass's clock would then relabel months-old
        # inference as having happened now, and nothing downstream would notice:
        # prompts, responses and scores are all unchanged. Keep what the earlier
        # pass recorded so a later audit still has something to audit.
        prev_prov = _load_json(run_dir / "provenance.json", None) or {}

        rows = []
        failures = []
        quota_stop = False
        for task, base_prompt, prompt_chars, prompt_digest in prepared:
            cid = task["case_id"]
            direct_seed = direct_seeds_by_run.get(run_id, {}).get(cid)
            direct_seed_sha256 = (
                hashlib.sha256(direct_seed["response"].encode("utf-8")).hexdigest()
                if direct_seed is not None else None
            )
            checkpoint_path = run_dir / f"{cid}.inference.json"
            checkpoint = _load_json(checkpoint_path, None) if resume else None
            candidate = checkpoint if checkpoint is not None else legacy_by_id.get(cid)

            reusable = resume and _reusable_row(
                candidate,
                prompt_digest=prompt_digest,
                max_tokens=max_tokens,
                model=model,
                fewshot=fewshot,
                case_id=cid,
                reasoning_effort=llm.reasoning_effort,
                stream_response=llm.stream_response,
                requested_model=llm.requested_model,
                thinking_mode=llm.thinking_mode,
                base_url=llm.base_url,
                api_key_env=llm.api_key_env,
            )
            if reusable and self_repair_k > 1:
                reusable = (
                    candidate.get("self_repair_protocol_version") == SELF_REPAIR_PROTOCOL_VERSION
                    and candidate.get("selection_pool") == "demo"
                    and candidate.get("scoring_pool") == "eval"
                    and candidate.get("repair_seed") == "direct_checkpoint"
                    and candidate.get("direct_seed_sha256") == direct_seed_sha256
                )
            if reusable:
                row = dict(candidate)
                row.update({
                    "prompt_sha256": prompt_digest,
                    "prompt_chars": prompt_chars,
                    "max_tokens": max_tokens,
                    "inference_ok": True,
                    "inference_error": "",
                    "failure_kind": "",
                    "resumed": True,
                })
                # Successful checkpoints created before max-attempt provenance
                # was added were all produced by the historical fixed-5 loop.
                # Preserve their actual timeout instead of relabelling them as
                # the current configuration during a mixed resume.
                row.setdefault("max_attempts", 5)
                row.setdefault("attempt_elapsed_seconds", [])
                # Present but null on checkpoints written before this field
                # existed: "we do not know when" is a different statement from
                # "now", and only one of them is true.
                row.setdefault("generated_at_utc", None)
                row.setdefault("reasoning_effort", None)
                row.setdefault("stream", False)
                query_rules = row.get("query") or extract_query(row.get("response") or "")
                row["query"] = query_rules
                (run_dir / f"{cid}.dl").write_text(
                    compose_full_dl(task["schema_def"], query_rules, case_id=cid),
                    encoding="utf-8",
                )
                _atomic_json(checkpoint_path, row)
                rows.append(row)
                print(f"[RESUME][run {run_id}] {cid}: verified checkpoint")
                continue

            if self_repair_k == 1:
                try:
                    response = llm.infer(base_prompt, max_tokens=max_tokens)
                except QuotaExhausted as exc:
                    response = ""
                    llm.last_error = str(exc)
                    llm.last_failure_kind = "quota"
                    quota_stop = True
                query_rules = extract_query(response)
                full_dl = compose_full_dl(task["schema_def"], query_rules, case_id=cid)
                (run_dir / f"{cid}.dl").write_text(full_dl, encoding="utf-8")
                inference_ok = not bool(llm.last_error)
                row = {
                    "id": cid,
                    "run": run_id,
                    "response": response,
                    "query": query_rules,
                    "served_model": llm.last_served_model,
                    "inference_ok": inference_ok,
                    "inference_error": llm.last_error,
                    "failure_kind": llm.last_failure_kind,
                    "attempts": llm.last_attempts,
                    "attempt_elapsed_seconds": list(llm.last_attempt_elapsed_seconds),
                    # When this response was actually produced. Without it the
                    # run's inference window lives only in provenance.json, and
                    # a resume rewrites that file wholesale.
                    "generated_at_utc": datetime.datetime.now(
                        datetime.timezone.utc).isoformat(timespec="seconds"),
                    "prompt_sha256": prompt_digest,
                    "prompt_chars": prompt_chars,
                    "max_tokens": max_tokens,
                    "experiment_model": model,
                    "requested_model": llm.requested_model,
                    "thinking_mode": llm.thinking_mode,
                    "base_url": llm.base_url,
                    "api_key_env": llm.api_key_env,
                    "request_timeout_seconds": llm.request_timeout,
                    "max_attempts": llm.max_attempts,
                    "reasoning_effort": llm.reasoning_effort,
                    "stream": llm.stream_response,
                    "finish_reason": llm.last_finish_reason,
                    "reasoning_chars": llm.last_reasoning_chars,
                    "completion_tokens": llm.last_completion_tokens,
                    "reasoning_tokens": llm.last_reasoning_tokens,
                }
            else:
                case_record = {
                    "id": cid,
                    "output_relation": {
                        x["signature"]: x.get("description", "") for x in task["schema_def"]["output"]
                    },
                }
                best, trajectory, first_perfect, inference_error, quota_stop = _run_self_repair(
                    llm, task, base_prompt, case_record, self_repair_k, max_tokens,
                    eval_tools, QuotaExhausted, direct_seed=direct_seed,
                )
                if best is None:
                    best = {
                        "full_dl": compose_full_dl(task["schema_def"], "", case_id=cid),
                        "response": "",
                        "query": "",
                        "iteration": 0,
                        "served_model": llm.last_served_model,
                        "finish_reason": llm.last_finish_reason,
                        "reasoning_chars": llm.last_reasoning_chars,
                        "completion_tokens": llm.last_completion_tokens,
                        "reasoning_tokens": llm.last_reasoning_tokens,
                    }
                (run_dir / f"{cid}.dl").write_text(best["full_dl"], encoding="utf-8")
                row = {
                    "id": cid,
                    "run": run_id,
                    "response": best["response"],
                    "query": best["query"],
                    "self_repair_k": self_repair_k,
                    "self_repair_protocol_version": SELF_REPAIR_PROTOCOL_VERSION,
                    "selection_pool": "demo",
                    "scoring_pool": "eval",
                    "repair_seed": "direct_checkpoint",
                    "direct_seed_sha256": direct_seed_sha256,
                    "best_iteration": best["iteration"],
                    "first_perfect_iteration": first_perfect,
                    "served_model": best["served_model"],
                    "trajectory": trajectory,
                    "inference_ok": not bool(inference_error),
                    "inference_error": inference_error,
                    "failure_kind": llm.last_failure_kind if inference_error else "",
                    "attempts": sum(x.get("attempts", 0) for x in trajectory),
                    "model_calls": len(trajectory),
                    "model_calls_this_run": len([
                        x for x in trajectory if not x.get("seeded_from_direct")
                    ]),
                    "attempt_elapsed_seconds": [
                        elapsed
                        for step in trajectory
                        for elapsed in step.get("attempt_elapsed_seconds", [])
                    ],
                    "generated_at_utc": datetime.datetime.now(
                        datetime.timezone.utc).isoformat(timespec="seconds"),
                    "prompt_sha256": prompt_digest,
                    "prompt_chars": prompt_chars,
                    "max_tokens": max_tokens,
                    "experiment_model": model,
                    "requested_model": llm.requested_model,
                    "thinking_mode": llm.thinking_mode,
                    "base_url": llm.base_url,
                    "api_key_env": llm.api_key_env,
                    "request_timeout_seconds": llm.request_timeout,
                    "max_attempts": llm.max_attempts,
                    "reasoning_effort": llm.reasoning_effort,
                    "stream": llm.stream_response,
                    "finish_reason": best.get("finish_reason"),
                    "reasoning_chars": best.get("reasoning_chars", 0),
                    "completion_tokens": best.get("completion_tokens"),
                    "reasoning_tokens": best.get("reasoning_tokens"),
                }

            _atomic_json(checkpoint_path, row)
            rows.append(row)
            if row["inference_ok"]:
                if self_repair_k == 1:
                    print(f"[SYNTH][run {run_id}] {cid}: rule_len={len(row['query'])}")
                else:
                    print(f"[SYNTH][run {run_id}][repair] {cid}: "
                          f"first_perfect={first_perfect} best_iter={best['iteration']} "
                          f"best_f1={best.get('f1', 0.0):.4f}")
            else:
                failure = {
                    "id": cid,
                    "kind": row["failure_kind"],
                    "error": row["inference_error"],
                    "attempts": row["attempts"],
                    "attempt_elapsed_seconds": row.get("attempt_elapsed_seconds", []),
                }
                failures.append(failure)
                print(f"[INFERENCE-FAIL][run {run_id}] {cid}: "
                      f"{failure['kind']} after {failure['attempts']} attempt(s)")
            if quota_stop:
                break

        complete_case_set = len(rows) == len(prepared)
        if complete_case_set:
            _atomic_json(aggregate_path, rows)
            (run_dir / "progress.json").unlink(missing_ok=True)
        else:
            _atomic_json(run_dir / "progress.json", rows)

        missing = [task["case_id"] for task, *_ in prepared[len(rows):]]
        failure_payload = {
            "model": model,
            "method": method,
            "fewshot": fewshot,
            "run": run_id,
            "max_tokens": max_tokens,
            "experiment_model": model,
            "requested_model": llm.requested_model,
            "thinking_mode": llm.thinking_mode,
            "base_url": llm.base_url,
            "api_key_env": llm.api_key_env,
            "request_timeout_seconds": llm.request_timeout,
            "max_attempts": llm.max_attempts,
            "reasoning_effort": llm.reasoning_effort,
            "stream": llm.stream_response,
            "complete_case_set": complete_case_set,
            "failures": failures,
            "not_attempted": missing,
        }
        _atomic_json(run_dir / "inference_failures.json", failure_payload)

        prov = llm.provenance()
        served_this_invocation = dict(prov.get("served_models") or {})
        served_from_rows = Counter(
            row.get("served_model") for row in rows if row.get("served_model")
        )
        # A resumed cell can make only one new request while reusing 135 valid
        # rows.  Provenance must describe the final artifact, not just calls in
        # this invocation, or it appears that almost no response identified its
        # served model.
        prov["served_models_this_invocation"] = served_this_invocation
        prov["served_models"] = dict(served_from_rows)
        prov.update({
            "run": run_id,
            "num_runs": num_runs,
            "method": method,
            "fewshot": fewshot,
            "self_repair_k": self_repair_k,
            "self_repair_protocol_version": (
                SELF_REPAIR_PROTOCOL_VERSION if self_repair_k > 1 else None
            ),
            "selection_pool": "demo" if self_repair_k > 1 else None,
            "scoring_pool": "eval",
            "repair_seed": "direct_checkpoint" if self_repair_k > 1 else None,
            "dataset_scope": scope_dir,
            "cases": len(rows),
            "expected_cases": len(prepared),
            "successful_inferences": sum(bool(row.get("inference_ok")) for row in rows),
            "failed_inferences": len(failures),
            "max_tokens": max_tokens,
            # `*_configured` names this invocation. The per-case lists expose
            # mixed resume provenance (old successful 300s/5-attempt rows plus
            # newly recovered 600s/2-attempt rows) instead of implying that all
            # responses were generated under the newest operational ceiling.
            "request_timeout_seconds_configured": llm.request_timeout,
            "max_attempts_configured": llm.max_attempts,
            "case_request_timeout_seconds": sorted({
                row.get("request_timeout_seconds") for row in rows
                if isinstance(row.get("request_timeout_seconds"), (int, float))
            }),
            "case_max_attempts": sorted({
                row.get("max_attempts") for row in rows
                if isinstance(row.get("max_attempts"), int)
            }),
            "max_prompt_chars": max_prompt_chars,
            "complete": complete_case_set and not failures,
            "calls_with_served_model": sum(served_from_rows.values()),
        })

        # Two different windows, named for which is which -- the same split this
        # file already makes for served_models. `started/finished_at_utc`
        # describe when the artifact's responses were produced; the
        # `_this_invocation` pair describes the pass that just ran, which for a
        # pure resume can be seconds long and contain no model call at all.
        # Reporting only the second under the first's name is what made a
        # resumed cell look freshly generated.
        pass_finished = datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds")
        prov.update(inference_window(rows, prev_prov))
        prov.update({
            "started_at_utc_this_invocation": started_at,
            "finished_at_utc_this_invocation": pass_finished,
            "cases_reused_this_invocation": sum(
                1 for row in rows if row.get("resumed")
            ),
        })
        _atomic_json(run_dir / "provenance.json", prov)

        served = prov["served_models"]
        if not served:
            note = "none reported by provider"
        elif len(served) == 1 and llm.requested_model in served:
            note = f"{llm.requested_model} (matches request)"
        else:
            note = f"{served} <- DIFFERS from requested {llm.requested_model!r}"
        if complete_case_set:
            print(f"[DONE][run {run_id}] aggregate -> {aggregate_path}")
        else:
            print(f"[INCOMPLETE][run {run_id}] progress -> {run_dir / 'progress.json'}")
        print(f"[PROVENANCE][run {run_id}] served model(s): {note}")
        if failures or not complete_case_set:
            incomplete_runs += 1
            print(f"[INCOMPLETE][run {run_id}] failures={len(failures)} "
                  f"not_attempted={len(missing)}; rerun with --resume")

    print(f"[DONE] {num_runs} run(s) -> {output_base_dir}")
    return 1 if incomplete_runs else 0


def main():
    parser = argparse.ArgumentParser(description="Synthesize Datalog queries with normalized task context.")
    parser.add_argument("--dataset", type=str, default="all", help="Case filter value from category, sub category, or all")
    parser.add_argument("--model", type=str, required=True, help="Online model name")
    parser.add_argument("--api_key", type=str, default=None,
                        help="Provider key, overriding the environment. Prefer EVAL_API_KEY: "
                             "a key on the command line is visible to every process via `ps`")
    parser.add_argument("--method", type=str, default="signature", choices=["signature", "description"])
    parser.add_argument("--fewshot", type=int, default=0, choices=[0, 1, 2], help="number of held-out demonstration examples to show (drawn from <case>/demo, never from the scored eval pool)")
    parser.add_argument("--case_id", type=str, default=None, help="Optional single case id")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (pass@1 protocol uses 0)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling (ignored when temperature is 0)")
    parser.add_argument(
        "--max_tokens", type=int, default=None,
        help="Max output tokens (default comes from evaluation/model_inventory.py; "
             "reasoning models receive a larger budget)",
    )
    parser.add_argument(
        "--max_prompt_chars", type=int, default=DEFAULT_MAX_PROMPT_CHARS,
        help="Refuse a prompt larger than this before making any API call",
    )
    parser.add_argument("--num_runs", type=int, default=1, help="Independent repetitions; each run writes to run_<i>/")
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse successful prompt-hash-matched checkpoints and rerun only invalid/missing cases",
    )
    parser.add_argument("--self_repair_k", type=int, default=1,
                        help="Max generate->execute->feedback iterations (k=1 disables self-repair). "
                             "k>1 selects candidates only on the demo/development pool and writes "
                             "to a repair<k> tag")
    parser.add_argument(
        "--repair_seed_from_direct", action="store_true",
        help=("For k>1, reuse the matching completed Direct checkpoint as iteration 1. "
              "This is required by the formal controlled self-repair matrix."),
    )
    args = parser.parse_args()

    # The key follows the model's provider, so a probe-side model run through
    # this entry point picks up its own key rather than silently using the
    # evaluation one against an endpoint that does not accept it.
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
    from model_inventory import api_key_env_for, api_key_for, check_model_supported
    # Before the key is resolved, because routing is by membership in MODELS: an
    # excluded model is indistinguishable from a probe-side one to
    # `api_key_env_for`, so without this it would not fail -- it would pick up
    # the probe key, call the probe endpoint, and write results into the
    # evaluation tree.
    check_model_supported(args.model)
    api_key = args.api_key or api_key_for(args.model)
    if not api_key:
        parser.error(f"no API key: export {api_key_env_for(args.model)} (preferred) "
                     "or pass --api_key")
    args.api_key = api_key

    return synthesize_llm(
        dataset=args.dataset,
        model=args.model,
        api_key=args.api_key,
        method=args.method,
        fewshot=args.fewshot,
        case_id=args.case_id,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        max_prompt_chars=args.max_prompt_chars,
        num_runs=args.num_runs,
        self_repair_k=args.self_repair_k,
        repair_seed_from_direct=args.repair_seed_from_direct,
        resume=args.resume,
    )


if __name__ == "__main__":
    raise SystemExit(main())
