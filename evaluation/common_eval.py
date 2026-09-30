import csv
import json
import os
import re
import shutil
import subprocess
import tempfile

from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = ROOT / "benchmark"
IO_DATA_DIR = BENCHMARK_DIR / "io_data"
QUERY_DIR = BENCHMARK_DIR / "query"

# Wall-clock limit for a single Souffle execution (one program on one variant).
# A synthesized program can be non-terminating or blow up combinatorially, so
# every invocation is capped; a timeout counts as a compile/run failure for that
# variant. Overridable via the SOUFFLE_TIMEOUT_SEC environment variable.
SOUFFLE_TIMEOUT_SEC = int(os.environ.get("SOUFFLE_TIMEOUT_SEC", "30"))


# Compile/run failure taxonomy, refined to Datalog-semantic categories rather
# than coarse syntax/undefined buckets. Ordered by priority: the first pattern
# that matches the captured Soufflé stderr wins (a parse error aborts before
# semantic analysis, so it is checked before stratification/type/safety; the
# Datalog-specific categories -- cannot_stratify, type_error,
# ungrounded_variable -- rank above the generic ones). Patterns calibrated
# against Soufflé 2.5 wording.
ERROR_CATEGORIES = [
    "none", "timeout", "souffle_not_found", "missing_output", "syntax_error",
    "cannot_stratify", "type_error", "ungrounded_variable", "undefined_relation",
    "redefinition", "arity_mismatch", "io_error", "other",
]

_ERROR_PATTERNS = [
    ("timeout",             r"timed out after"),
    ("souffle_not_found",   r"souffle command not found"),
    ("missing_output",      r"query file not found|missing io_data"),
    ("syntax_error",        r"syntax error|unexpected (token|end of file|IDENT)"),
    ("cannot_stratify",     r"[Uu]nable to stratify|cannot be stratified|not stratified"),
    ("type_error",          r"[Uu]nable to deduce type|constraints are incompatible|"
                            r"type\s+mismatch|type\s+clash|[Cc]annot find type|"
                            r"[Uu]ndefined type|no type could|does not have a type"),
    ("ungrounded_variable", r"[Uu]ngrounded variable|[Uu]nsafe"),
    ("undefined_relation",  r"[Uu]ndefined relation"),
    ("redefinition",        r"[Rr]edefinition of"),
    ("arity_mismatch",      r"[Mm]ismatch(ing)? arit|wrong number of arguments"),
    ("io_error",            r"[Cc]annot open|[Ff]ailed to (open|read)|fact file"),
]


def classify_souffle_error(text: str) -> str:
    """Map captured Soufflé stderr to one ERROR_CATEGORIES label."""
    if not text or not text.strip():
        return "none"
    for category, pattern in _ERROR_PATTERNS:
        if re.search(pattern, text):
            return category
    return "other"


def normalize_line(line: str):
    """Parse one tuple line into a field tuple.

    Only the line terminator is stripped, never the payload. A leading or
    trailing field can legitimately be an empty cell -- Soufflé writes such a
    symbol as a space -- and stripping the whole line deletes those fields
    outright, shortening the tuple. That silently equated different rows: a
    tic-tac-toe board with four leading blanks and the same marks shifted to
    the other end both collapsed to the same 7-field tuple, so position, which
    is the entire meaning of the relation, stopped being compared.

    Individual fields are still stripped, which is what reconciles the two
    spellings of an empty cell (a space vs nothing between two tabs).
    """
    line = line.rstrip("\r\n")
    if not line.strip() or line.lstrip().startswith("//"):
        return ()
    if "\t" in line:
        parts = line.split("\t")
    elif "," in line:
        parts = line.split(",")
    else:
        parts = line.split()
    return tuple(part.strip() for part in parts)


def load_tuples(file_path: Path):
    tuples_set = set()
    if not file_path.exists():
        return tuples_set
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            norm = normalize_line(line)
            if norm:
                tuples_set.add(norm)
    return tuples_set


SPEC_TIERS = ("L1", "L2", "L3")


def method_tag(method: str, fewshot: int, repair_k: int = 1) -> str:
    """The run directory / result filename tag for one prompt configuration.

    The single definition, imported by both the synthesis and the evaluation
    side. A copy in each would be a silent-failure shape: a modifier added to
    one side only makes synthesis write to one directory and evaluation read
    from another, and the run looks empty rather than wrong.

    Suffix order is fixed and additive so existing tags keep their names:
    `signature`, `2-shot_signature`, `signature_repair3`.
    """
    tag = method if fewshot == 0 else f"{fewshot}-shot_{method}"
    if repair_k > 1:
        tag = f"{tag}_repair{repair_k}"
    return tag


def spec_tiers() -> Dict[str, str]:
    """{case_id: tier} from benchmark/spec_tiers.json, or {} when absent.

    The labels live outside dataset.json because they are not part of a task --
    nothing in the prompt shows them, and a solver is never told which tier a
    case is in. Keeping them here leaves dataset.json holding only what a solver
    is given.
    """
    path = Path(__file__).resolve().parent.parent / "benchmark" / "spec_tiers.json"
    if not path.exists():
        return {}
    tiers = json.loads(path.read_text(encoding="utf-8")).get("tiers", {})
    bad = {c: t for c, t in tiers.items() if t not in SPEC_TIERS}
    if bad:
        raise ValueError(f"spec_tiers.json: not one of {SPEC_TIERS}: {bad}")
    return tiers


def dataset_index_by_id() -> Dict[str, Dict]:
    dataset_file = BENCHMARK_DIR / "dataset.json"
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    with dataset_file.open("r", encoding="utf-8") as f:
        return {x["id"]: x for x in json.load(f)}


def dataset_records(dataset: str = "all") -> List[Dict]:
    dataset_file = BENCHMARK_DIR / "dataset.json"
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    with dataset_file.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not dataset or dataset.lower() == "all":
        return records

    return [
        x for x in records
        if x.get("category") == dataset or x.get("sub_category") == dataset
    ]


def get_case_output_relations(case_record: Dict) -> List[str]:
    return [x.split("(")[0] for x in case_record.get("output_relation", {}).keys()]


def io_case_root(case_id: str) -> Path:
    primary = IO_DATA_DIR / case_id
    if primary.exists():
        return primary

    raise FileNotFoundError(f"Case directory not found: {primary}")


def case_variants_dirs(case_id: str) -> List[Path]:
    """The variant directories this case is *scored* on.

    Two layouts are supported. The original one numbers variants directly under
    the case (`<case>/0` ... `<case>/5`). The one `select_variants.py --out`
    produces splits them into `<case>/eval/<i>` and `<case>/demo/0`, and only
    the eval pool is scored -- the demo variant exists to be shown in few-shot
    prompts, so scoring it would leak the oracle it was held out to protect.

    Getting this wrong is silent rather than loud: with the split layout and a
    numeric-only scan, no directory matches, the case root is returned, it holds
    no `.facts`, and every program scores tp=fp=fn=0 -- which reads as a perfect
    match for everything.
    """
    case_root = io_case_root(case_id)
    if not case_root.exists():
        raise FileNotFoundError(f"Case directory not found: {case_root}")

    eval_root = case_root / "eval"
    if eval_root.is_dir():
        numbered = sorted((x for x in eval_root.iterdir() if x.is_dir() and x.name.isdigit()),
                          key=lambda p: int(p.name))
        if numbered:
            return numbered

    children = [x for x in case_root.iterdir() if x.is_dir()]
    if not children:
        return [case_root]

    numeric_children = [x for x in children if x.name.isdigit()]
    if numeric_children:
        return sorted(numeric_children, key=lambda p: int(p.name))
    return [case_root]


def case_demo_dirs(case_id: str) -> List[Path]:
    """The held-out demonstration variants, never scored."""
    demo_root = io_case_root(case_id) / "demo"
    if not demo_root.is_dir():
        return []
    return sorted((x for x in demo_root.iterdir() if x.is_dir()),
                  key=lambda p: p.name)


def case_variants_dirs_six(case_id: str) -> List[Path]:
    """Retained name, now layout-aware: the count is whatever the case ships.

    It used to return `<case>/0..5` unconditionally, which hardcodes both the
    old layout and the assumption of exactly six variants -- neither survives
    the greedy selection that reduces most cases to one or two.
    """
    return case_variants_dirs(case_id)


TIMEOUT_PREFIX = "souffle timed out after"


def is_exact(result: Dict) -> bool:
    """Exact match: the program compiled, finished on every scored variant, and
    reproduced every expected tuple with nothing extra.

    `timed_out` has to be checked separately: a variant that runs out of time
    produces no output, and when its expected set is empty (a blind slot) that
    leaves fp and fn both zero -- an unfinished program would otherwise count as
    exact.
    """
    return bool(result["compile_ok"] and not result.get("timed_out")
                and result["fp"] == 0 and result["fn"] == 0)


def run_souffle(program_file: Path, facts_dir: Path, output_dir: Path) -> Tuple[bool, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Souffle's preprocessor rejects non-ASCII characters in the program path
    # (e.g. localized OneDrive folders), so run from an ASCII-safe temp copy.
    temp_dir = None
    try:
        str(program_file).encode("ascii")
        safe_program = program_file
    except UnicodeEncodeError:
        temp_dir = tempfile.mkdtemp(prefix="souffle_prog_")
        safe_program = Path(temp_dir) / program_file.name
        shutil.copyfile(program_file, safe_program)

    cmd = ["souffle", "-F", str(facts_dir), "-D", str(output_dir), str(safe_program)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=SOUFFLE_TIMEOUT_SEC)
    except FileNotFoundError:
        return False, "souffle command not found in PATH"
    except subprocess.TimeoutExpired:
        # subprocess.run kills the child on timeout; treat as a failed run.
        return False, f"{TIMEOUT_PREFIX} {SOUFFLE_TIMEOUT_SEC}s"
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)

    if proc.returncode != 0:
        return False, (proc.stderr.strip() or proc.stdout.strip() or "unknown souffle error")
    return True, ""


def evaluate_query_file_for_case(case_record: Dict, query_file: Path,
                                 include_counterexamples: bool = False,
                                 variant_dirs: List[Path] = None,
                                 evaluation_pool: str = "eval") -> Dict:
    """Execute one candidate on an explicitly named I/O pool.

    Normal scoring uses the disjoint ``eval`` variants.  Iterative systems may
    pass ``case_demo_dirs(case_id)`` and ``evaluation_pool="demo"`` to obtain
    development feedback without adapting to the final test oracle.  Keeping
    the default on ``eval`` preserves every non-interactive evaluator, while
    recording the pool in the result makes a feedback trace auditable.
    """
    case_id = case_record["id"]
    output_relations = get_case_output_relations(case_record)
    if variant_dirs is None:
        variant_dirs = case_variants_dirs_six(case_id)
    else:
        variant_dirs = list(variant_dirs)
    if not variant_dirs:
        raise ValueError(f"{case_id}: evaluation pool {evaluation_pool!r} is empty")

    aggregate_tp = 0
    aggregate_fp = 0
    aggregate_fn = 0
    aggregate_neg_total = 0
    aggregate_neg_hit = 0
    compile_ok = True
    timed_out = False
    errors: List[str] = []
    per_variant = []

    if not query_file.exists():
        return {
            "case_id": case_id,
            "evaluation_pool": evaluation_pool,
            "compile_ok": False,
            "errors": [f"query file not found: {query_file}"],
            "neg_total": 0,
            "neg_hit": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "variant_count": len(variant_dirs),
            # The loop variable has to be the one the row names. This branch
            # runs only when a submission is missing, so a NameError here would
            # surface on the first synthesis failure of a run rather than in any
            # smoke test that feeds it real programs.
            "per_variant": [
                {
                    "variant": d.name,
                    "compile_ok": False,
                    "error": f"query file not found: {query_file}",
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                }
                for d in variant_dirs
            ],
        }

    with tempfile.TemporaryDirectory(prefix=f"eval_{case_id}_") as temp_root:
        temp_root_path = Path(temp_root)

        for idx, variant_dir in enumerate(variant_dirs):
            variant_name = str(idx)
            if not variant_dir.exists() or not variant_dir.is_dir():
                compile_ok = False
                err = f"missing io_data variant dir: {variant_dir}"
                errors.append(f"{variant_name}: {err}")
                per_variant.append(
                    {
                        "variant": variant_name,
                        "compile_ok": False,
                        "error": err,
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                    }
                )
                continue

            variant_out = temp_root_path / "out" / variant_name
            ok, err = run_souffle(query_file, variant_dir, variant_out)
            if not ok and err.startswith(TIMEOUT_PREFIX):
                # The program compiled -- Souffle checks and stratifies before it
                # evaluates, and every timeout in the reported grid also runs on
                # empty input -- but evaluation did not finish, so it produced
                # nothing. Every expected tuple is missing; the variant says
                # nothing about negative exclusion, so its negatives are not
                # counted (a program that outputs nothing would exclude them all).
                timed_out = True
                errors.append(f"{variant_name}: {err}")
                missing = sum(len(load_tuples(variant_dir / f"{rel}.expected"))
                              for rel in output_relations)
                aggregate_fn += missing
                per_variant.append(
                    {
                        "variant": variant_name,
                        "compile_ok": True,
                        "timed_out": True,
                        "error": err,
                        "tp": 0,
                        "fp": 0,
                        "fn": missing,
                        "neg_total": 0,
                        "neg_hit": 0,
                    }
                )
                continue
            if not ok:
                compile_ok = False
                errors.append(f"{variant_name}: {err}")
                per_variant.append(
                    {
                        "variant": variant_name,
                        "compile_ok": False,
                        "error": err,
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                    }
                )
                continue

            variant_tp = 0
            variant_fp = 0
            variant_fn = 0
            variant_neg_total = 0
            variant_neg_hit = 0
            counterexamples = []

            for rel in output_relations:
                expected_file = variant_dir / f"{rel}.expected"
                predicted_file = variant_out / f"{rel}.csv"
                exp_set = load_tuples(expected_file)
                pred_set = load_tuples(predicted_file)
                tp = len(exp_set & pred_set)
                fp = len(pred_set - exp_set)
                fn = len(exp_set - pred_set)
                variant_tp += tp
                variant_fp += fp
                variant_fn += fn

                # SyGuS-style negative exclusion: O- holds tuples that plausible
                # wrong programs derive but the target must not. Deriving one is
                # a sharper signal than a generic false positive.
                neg_set = load_tuples(variant_dir / f"{rel}.undesired")
                variant_neg_total += len(neg_set)
                variant_neg_hit += len(pred_set & neg_set)

                if include_counterexamples:
                    # Stable bounded witnesses make repair feedback actionable
                    # without copying an unbounded held-out dataset into the
                    # model context.
                    counterexamples.append({
                        "relation": rel,
                        "expected_only": [list(row) for row in sorted(exp_set - pred_set)[:8]],
                        "produced_only": [list(row) for row in sorted(pred_set - exp_set)[:8]],
                    })

            aggregate_tp += variant_tp
            aggregate_fp += variant_fp
            aggregate_fn += variant_fn
            aggregate_neg_total += variant_neg_total
            aggregate_neg_hit += variant_neg_hit
            variant_result = {
                    "variant": variant_name,
                    "compile_ok": True,
                    "error": "",
                    "tp": variant_tp,
                    "fp": variant_fp,
                    "fn": variant_fn,
                    "neg_total": variant_neg_total,
                    "neg_hit": variant_neg_hit,
                }
            if include_counterexamples:
                input_facts = []
                remaining = 20
                for facts_file in sorted(variant_dir.glob("*.facts")):
                    rows = sorted(load_tuples(facts_file))
                    shown = rows[:remaining]
                    input_facts.append({
                        "relation": facts_file.stem,
                        "tuples": [list(row) for row in shown],
                        "total": len(rows),
                    })
                    remaining -= len(shown)
                    if remaining <= 0:
                        break
                variant_result["input_facts"] = input_facts
                variant_result["counterexamples"] = counterexamples
            per_variant.append(variant_result)

    return {
        "case_id": case_id,
        "evaluation_pool": evaluation_pool,
        "compile_ok": compile_ok,
        "timed_out": timed_out,
        "errors": errors,
        "neg_total": aggregate_neg_total,
        "neg_hit": aggregate_neg_hit,
        "tp": aggregate_tp,
        "fp": aggregate_fp,
        "fn": aggregate_fn,
        "variant_count": len(variant_dirs),
        "per_variant": per_variant,
    }


def grading_provenance() -> Dict:
    """Which compiler produced these numbers, recorded next to them.

    Every score is decided by Souffle, so the output has to say which Souffle.
    Two ways that goes wrong: grading on the host because Docker was
    unavailable, and a grid whose image changed between cells because the tag
    was re-read from the working tree each time. Neither is visible in a CSV of
    scores. This makes it visible.

    ``harness_image_id`` is the pin that matters. The tag is a mutable pointer,
    so two runs can name the same image and mean different bits;
    ``docker/freeze-image.sh`` resolves it once per grid and exports both.

    Lives here rather than in each evaluator so the agent and Direct rows cannot
    end up describing their provenance differently -- one fact, one definition.
    """
    prov = {
        "grading_host": "container" if Path("/opt/dlb").is_dir() else "host",
        "harness_image": os.environ.get("HARNESS_IMAGE") or None,
        "harness_image_id": os.environ.get("HARNESS_IMAGE_ID") or None,
    }
    try:
        out = subprocess.run(["souffle", "--version"], capture_output=True,
                             text=True, timeout=30).stdout
        for key, prefix in (("souffle_version", "Version:"),
                            ("souffle_word_size", "Word size:")):
            prov[key] = next((ln.strip() for ln in out.splitlines()
                              if ln.strip().startswith(prefix)), "unknown")
    except (OSError, subprocess.SubprocessError):
        prov["souffle_version"] = "unavailable"
        prov["souffle_word_size"] = "unavailable"
    return prov


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_summary_csv(path: Path, headers: List[str], row: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerow(row)


def save_csv_rows(path: Path, headers: List[str], rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
