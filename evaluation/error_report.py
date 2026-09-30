"""Compile/run failure breakdown by Datalog-semantic category.

Post-hoc: reads a `*_details.jsonl` produced by any eval_*.py (the captured
Soufflé stderr is already stored in each row's `error` field) and reports how the
failures distribute over the taxonomy in common_eval.ERROR_CATEGORIES -- e.g.
`cannot_stratify`, `type_error`, `ungrounded_variable` -- instead of a single
compile-pass rate. No re-evaluation is needed.

    # by result-set (llm)
    python evaluation/error_report.py --model gpt-5.6-sol --method signature --fewshot 0 --dataset all
    # or point straight at a details file (any pipeline)
    python evaluation/error_report.py --details benchmark/res_data/<...>_details.jsonl
    # the fine taxonomy and the structure cross-tab over every reported cell
    python evaluation/error_report.py --fine --out <path>.csv
"""

import argparse
import csv
import json
import re

from collections import Counter, defaultdict
from pathlib import Path

from common_eval import BENCHMARK_DIR
from common_eval import dataset_index_by_id
from common_eval import TIMEOUT_PREFIX
from common_eval import ERROR_CATEGORIES
from common_eval import classify_souffle_error
from common_eval import save_csv_rows


def _timed_out(row):
    """Souffle ran out of time evaluating the program. Newer details carry the
    flag; older ones only the evaluator's message, and recorded the program as not
    compiling. Match that message exactly: symbolic rows also say "timed out
    after", but about synthesis (`egs timed out after 300s`), not evaluation."""
    return bool(row.get("timed_out")) or TIMEOUT_PREFIX in str(row.get("error") or "")


def llm_details_path(model, method, fewshot, dataset):
    tag = method if fewshot == 0 else f"{fewshot}-shot_{method}"
    scope = dataset if dataset else "all"
    return BENCHMARK_DIR / "res_data" / f"{model}_{tag}_{scope}_details.jsonl"


def load_rows(path):
    if not path.exists():
        raise FileNotFoundError(f"Details file not found: {path}. Run the matching eval_*.py first.")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def error_report(details_path, label):
    rows = load_rows(details_path)

    # per (case,run) row category, and a per-case category (a case counts as
    # failed if it failed in any run; "none" only if it compiled in every run).
    row_cat = Counter()
    per_case = defaultdict(list)
    for r in rows:
        # a row that compiled cleanly has category "none" regardless of stray text
        if _timed_out(r):
            cat = "timeout"
        else:
            cat = "none" if r.get("compile_ok") else classify_souffle_error(r.get("error", ""))
        row_cat[cat] += 1
        per_case[r["case_id"]].append(cat)

    case_cat = Counter()
    for cid, cats in per_case.items():
        failed = [c for c in cats if c != "none"]
        case_cat[failed[0] if failed else "none"] += 1

    n_rows = len(rows)
    n_cases = len(per_case)
    n_failed_cases = sum(v for k, v in case_cat.items() if k != "none")

    print(f"[ERROR-BREAKDOWN] {label}")
    print(f"  {n_cases} cases, {n_rows} rows; {n_failed_cases} cases fail to compile in >=1 run")
    out_rows = []
    for cat in ERROR_CATEGORIES:
        rc, cc = row_cat.get(cat, 0), case_cat.get(cat, 0)
        if rc == 0 and cc == 0:
            continue
        share = cc / n_failed_cases if (cat != "none" and n_failed_cases) else ""
        out_rows.append({
            "category": cat,
            "cases": cc,
            "rows": rc,
            "share_of_failures": round(share, 4) if share != "" else "",
        })
        share_str = f"{share:.1%}" if share != "" else "-"
        print(f"  {cat:20} cases={cc:4d} rows={rc:4d} share_of_failures={share_str}")

    out_path = details_path.parent / (details_path.name.replace("_details.jsonl", "_errorcats.csv"))
    save_csv_rows(out_path, ["category", "cases", "rows", "share_of_failures"], out_rows)
    print(f"[DONE] error taxonomy -> {out_path}")


# --------------------------------------------------------------------------- #
# Fine taxonomy: what the program got wrong, read off the program as well as the
# message.
#
# The coarse labels above come from the error message alone, which is enough to
# name the compiler's complaint but not its cause. Two causes need the program:
# an aggregate written in a rule head is reported as a generic syntax error
# ("expecting :" in about two thirds of cases, something else in the rest), and a
# type failure means something different when the rule it sits in defines a
# relation the model invented. So each failure is traced to the statement
# Souffle points at, in the composed program that was actually compiled.
# --------------------------------------------------------------------------- #

FINE_CATEGORIES = [
    "undeclared_invented",      # uses a relation it invented, never declared it
    "type_inference",           # Souffle cannot type a variable
    "aggregate_in_head",        # an aggregate computed inside a rule head, SQL-style
    "reserved_word_as_name",    # count/sum/min/max/mean used as a variable or attribute
    "untyped_declaration",      # a .decl attribute with no :type
    "negation_keyword",         # `not p(...)` / `NOT p(...)`; Souffle negates with `!`
    "malformed_aggregate",      # an aggregate in a body, but not in Souffle's `op x : {...}` form
    "foreign_operator",         # C/SQL syntax: >>, <<, &, |, a..b, x.field, Rel(...) = v
    "truncated_response",       # the program stops mid-statement: the response was cut off
    "constant_type_mix",        # a number constant where a symbol is due, or back
    "other_dialect",            # any other syntax Souffle does not accept
    "ungrounded_variable",
    "unstratifiable_negation",
    "non_code_in_answer",       # prose or markup left in the extracted program
    "undefined_schema_relation",
    "other",
]

# Also not a compile failure: the program compiled, and Souffle ran out of the
# 30-second budget while evaluating it, so it produced no output. It is an
# evaluation-stage failure: counted as compiling in Compile Pass, never exact, and
# kept out of the compile-failure denominator here.
TIMEOUT = "evaluation_timeout"

# The fine categories grouped by what went wrong. A type failure counts toward
# predicate invention only when the rule it sits in defines an invented relation.
FAMILIES = [
    ("predicate invention left incomplete", ["undeclared_invented", "type_inference:invented_head"]),
    ("Souffle dialect", ["aggregate_in_head", "reserved_word_as_name", "untyped_declaration",
                         "negation_keyword", "malformed_aggregate", "foreign_operator",
                         "other_dialect"]),
    ("typing of constants and schema relations", ["constant_type_mix", "type_inference:schema_head"]),
    ("safety and stratification", ["ungrounded_variable", "unstratifiable_negation"]),
    ("response not a clean program", ["truncated_response", "non_code_in_answer"]),
    ("other", ["undefined_schema_relation", "other"]),
]

# Not a compile failure: the agent never produced a program, and the protocol
# counts the task as zero without executing anything. Kept apart so it does not
# inflate a compile-failure rate.
NO_PROGRAM = "no_program"

_LINE_RE = re.compile(r"at line (\d+)")
_UNDEF_RE = re.compile(r"[Uu]ndefined relation (\w+)")
_AGG = r"(count|sum|min|max|mean)"
# An aggregate *computed* in a head: `min(x)`, `min x`, or `count : {...}` misplaced.
_AGG_HEAD_RE = re.compile(rf"\b{_AGG}\b\s*(\(|<|:\s*\{{|[A-Za-z_])")
_STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"')
_NEGATION_RE = re.compile(r"(?<![\w!])(not|NOT)\s+[A-Za-z_]\w*\s*\(")
_BODY_AGG_RE = re.compile(rf"\b{_AGG}\b|\bGROUP\s+BY\b", re.I)
_FOREIGN_OP_RE = re.compile(
    r">>|<<|(?<![&])&(?!&)|\|\||(?<!\|)\|(?!\|)|==|\s(or|and|OR|AND)\s|\.\.|\b[A-Za-z_]\w*\.[A-Za-z_]\w*"   # bit ops, range, x.field
    r"|\b(?:[A-Z]\w*|contains|match)\s*\([^()]*\)\s*!?=(?!=)")                   # relation used as a value
# The same keyword used as a plain name: an argument (`Rel(n, count)`), or a
# declared attribute (`sum: number`). Souffle reserves these words.
_RESERVED_NAME_RE = re.compile(rf"\b{_AGG}\b\s*([,)]|:\s*(number|symbol|unsigned|float)\b)")
_UNTYPED_DECL_RE = re.compile(r"^\s*\.decl\s+\w+\s*\(([^)]*)\)")
_PROSE_RE = re.compile(r"^\s*([#*>`-]|\d+[.)]\s|[A-Za-z][a-z]+( [A-Za-z][a-z]+){3,})")
# Text that cannot be Datalog wherever it appears: backticks, a question mark, a
# JSON or list opener, a shell command.
_NON_CODE_RE = re.compile(r"`|\?|^\s*[{\[]|^\s*(bash|ls|cd|cat|python3?|souffle)\b")
# A weaker sign, four bare words in a row, is checked only after the aggregate
# forms: SQL's `sum val group by` is four bare words too.
_SENTENCE_RE = re.compile(r"\b[A-Za-z]+ [A-Za-z]+ [A-Za-z]+ [A-Za-z]+\b")
_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)


def schema_names(case_record):
    """Relation names the task schema gives the model; anything else is invented."""
    names = set()
    for side in ("input_relation", "output_relation"):
        for sig in case_record.get(side, {}):
            names.add(sig.split("(", 1)[0].strip())
    return names


def statement_at(program, line_no):
    """The whole statement containing 1-based `line_no`.

    Rules may span lines and end in '.'; directives (.decl, .input, ...) are one
    line each and do not. Walk back to the end of the previous statement and
    forward to the end of this one.
    """
    lines = program.splitlines()
    if not 1 <= line_no <= len(lines):
        return ""

    def ends(i):
        text = lines[i].strip()
        return text.startswith(".") or text.endswith(".")

    lo = line_no - 1
    while lo > 0 and not ends(lo - 1):
        lo -= 1
    hi = line_no - 1
    while hi < len(lines) - 1 and not ends(hi):
        hi += 1
    return " ".join(x.strip() for x in lines[lo:hi + 1])


def rule_head(statement):
    head = statement.split(":-", 1)[0].strip()
    m = re.match(r"(\w+)\s*\(", head)
    return head, (m.group(1) if m else None)


def fine_category(error, program, schema):
    """(category, detail) for one failed program; detail is a short evidence string."""
    error = error or ""
    if not error.strip():
        return "other", ""
    if re.search(r"timed out after", error):
        return "timeout", ""

    m = _LINE_RE.search(error)
    # Comments are blanked, line breaks kept, so line numbers still match Souffle's.
    program = _COMMENT_RE.sub(lambda c: "\n" * c.group(0).count("\n"), program or "")
    stmt = statement_at(program, int(m.group(1))) if (m and program) else ""
    head, head_rel = rule_head(stmt) if stmt else ("", None)

    if re.search(r"syntax error|unexpected", error):
        bare = _STRING_RE.sub('""', stmt)
        if stmt and ((_PROSE_RE.match(stmt) and ":-" not in stmt) or _NON_CODE_RE.search(bare)):
            return "non_code_in_answer", stmt[:60]
        # The program ends inside a statement. Every such response in the
        # reported grid also ends mid-token with its code fence unclosed, so the
        # answer was cut off rather than written wrong.
        if "unexpected end of file" in error:
            return "truncated_response", stmt[-60:]
        if head and not head.startswith(".") and _AGG_HEAD_RE.search(head):
            return "aggregate_in_head", head[:60]
        if stmt and _RESERVED_NAME_RE.search(stmt):
            return "reserved_word_as_name", _RESERVED_NAME_RE.search(stmt).group(1)
        d = _UNTYPED_DECL_RE.match(stmt) if stmt else None
        if d and any(":" not in a for a in d.group(1).split(",") if a.strip()):
            return "untyped_declaration", stmt[:60]
        code = _STRING_RE.sub('""', stmt)          # no operator inside a literal counts
        body = code.split(":-", 1)[1] if ":-" in code else ""
        if _NEGATION_RE.search(code):
            return "negation_keyword", _NEGATION_RE.search(code).group(0)
        if body and _BODY_AGG_RE.search(body):
            return "malformed_aggregate", body.strip()[:60]
        if _SENTENCE_RE.search(bare):
            return "non_code_in_answer", stmt[:60]
        if _FOREIGN_OP_RE.search(code):
            return "foreign_operator", _FOREIGN_OP_RE.search(code).group(0)
        return "other_dialect", stmt[:60]

    m = _UNDEF_RE.search(error)
    if m:
        rel = m.group(1)
        return ("undefined_schema_relation" if rel in schema else "undeclared_invented"), rel

    if re.search(r"constant \(unable to deduce type\)", error):
        return "constant_type_mix", stmt[:60]
    if re.search(r"[Uu]nable to deduce type|constraints are incompatible|type\s+mismatch|"
                 r"[Cc]annot find type|[Uu]ndefined type|no type could|does not have a type", error):
        at_invented = bool(head_rel) and head_rel not in schema
        return "type_inference", "invented_head" if at_invented else "schema_head"
    if re.search(r"[Uu]ngrounded variable|[Uu]nsafe", error):
        return "ungrounded_variable", ""
    if re.search(r"[Uu]nable to stratify|cannot be stratified|not stratified", error):
        return "unstratifiable_negation", ""
    return "other", error.strip().splitlines()[0][:60]


def load_structure():
    """case_id -> (recursive, has_auxiliary) from the reference-program analysis."""
    path = BENCHMARK_DIR / "res_data" / "structure_golden_all.csv"
    out = {}
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["case_id"]] = (r["recursive"] in ("True", "1"), int(r["num_invented"]) > 0)
    return out


def program_path(details_path, case_id, run):
    """The composed program a details row graded, beside its generations."""
    stem = details_path.name[:-len("_all_details.jsonl")]
    for model_len in range(len(stem.split("_")), 0, -1):
        model = "_".join(stem.split("_")[:model_len])
        tag = "_".join(stem.split("_")[model_len:])
        cand = BENCHMARK_DIR / "infer_data" / model / tag / "all" / f"run_{run}" / f"{case_id}.dl"
        if cand.exists():
            return cand
    return None


STRUCTURE_CELLS = [
    ("non-recursive, no auxiliary", False, False),
    ("non-recursive, auxiliary", False, True),
    ("recursive, no auxiliary", True, False),
    ("recursive, auxiliary", True, True),
]


def fine_report(groups, out_path=None):
    """Fine failure taxonomy and the structure cross-tab, per group of details files.

    `groups` maps a label (e.g. "Direct (24 cells)") to a list of details paths;
    rows within a group are pooled.
    """
    index = dataset_index_by_id()
    structure = load_structure()
    out_rows = []
    for label, paths in groups.items():
        cats, type_split = Counter(), Counter()
        by_cell = {name: Counter() for name, _, _ in STRUCTURE_CELLS}
        outcome = Counter()
        for path in paths:
            for r in load_rows(path):
                cid = r["case_id"]
                rec, aux = structure[cid]
                cell = next(name for name, a, b in STRUCTURE_CELLS if (a, b) == (rec, aux))
                error = str(r.get("error") or "")
                if _timed_out(r):
                    kind = TIMEOUT
                elif r.get("compile_ok"):
                    kind = "exact" if r.get("perfect_match") else "compiles_wrong"
                elif error.startswith("terminal inference failure"):
                    kind = NO_PROGRAM
                else:
                    kind = "compile_fail"
                outcome[kind] += 1
                by_cell[cell]["n"] += 1
                by_cell[cell][kind] += 1
                if kind != "compile_fail":
                    continue
                prog = program_path(path, cid, r.get("run", 0))
                text = prog.read_text(encoding="utf-8") if prog else ""
                cat, detail = fine_category(error, text, schema_names(index[cid]))
                cats[cat] += 1
                if cat == "type_inference":
                    type_split[detail] += 1

        n, n_fail = sum(outcome.values()), outcome["compile_fail"]
        print(f"\n[FINE] {label}: {n} tasks over {len(paths)} cell(s)")
        for kind, what in (("exact", "exact"),
                           ("compiles_wrong", "compile but are wrong"),
                           ("compile_fail", "fail to compile"),
                           (TIMEOUT, "compile, then exceed the evaluation timeout"),
                           (NO_PROGRAM, "produced no program (counted as zero, never compiled)")):
            if outcome[kind]:
                print(f"  {outcome[kind]:5d}  {outcome[kind] / n:6.1%}  {what}")
            out_rows.append({"group": label, "table": "outcome", "key": kind,
                             "count": outcome[kind], "share": round(outcome[kind] / n, 4)})
        if not n_fail:
            print("  (no compile failures)")
        else:
            print(f"  compile failures by cause (share of the {n_fail}):")
        for cat in FINE_CATEGORIES:
            if cats[cat]:
                print(f"    {cat:28s} {cats[cat]:5d}  {cats[cat] / n_fail:6.1%}")
                out_rows.append({"group": label, "table": "category", "key": cat,
                                 "count": cats[cat], "share": round(cats[cat] / n_fail, 4)})
        if type_split:
            print(f"    type_inference, rule defines an invented relation: "
                  f"{type_split['invented_head']} of {sum(type_split.values())}")
        if n_fail:
            print(f"  grouped (share of the {n_fail}):")
            for family, members in FAMILIES:
                count = sum(type_split[m.split(":")[1]] if ":" in m else cats[m] for m in members)
                if count:
                    print(f"    {family:42s} {count:5d}  {count / n_fail:6.1%}")
                out_rows.append({"group": label, "table": "family", "key": family,
                                 "count": count, "share": round(count / n_fail, 4)})
        print(f"  {'structure':28s} {'n':>5s} {'compile fail':>13s} {'timeout':>8s} {'compiles, wrong':>16s}")
        for name, _, _ in STRUCTURE_CELLS:
            c = by_cell[name]
            if not c["n"]:
                continue
            fail, tout, wrong = (c["compile_fail"] / c["n"], c[TIMEOUT] / c["n"],
                                 c["compiles_wrong"] / c["n"])
            print(f"  {name:28s} {c['n']:5d} {fail:13.1%} {tout:8.1%} {wrong:16.1%}")
            out_rows.append({"group": label, "table": "structure", "key": name,
                             "count": c["n"], "compile_fail": round(fail, 4),
                             "timeout": round(tout, 4), "compiles_wrong": round(wrong, 4)})
    if out_path:
        save_csv_rows(Path(out_path), ["group", "table", "key", "count", "share",
                                       "compile_fail", "timeout", "compiles_wrong"], out_rows)
        print(f"\n[DONE] fine taxonomy -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Datalog compile-error taxonomy from a details.jsonl.")
    parser.add_argument("--details", type=str, default=None, help="Path to a *_details.jsonl (any pipeline)")
    parser.add_argument("--model", type=str, default=None, help="LLM result-set: model")
    parser.add_argument("--method", type=str, default="signature", choices=["signature", "description"])
    parser.add_argument("--fewshot", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--fine", action="store_true",
                        help="Fine taxonomy and structure cross-tab over the reported cells: "
                             "the 24 Direct cells pooled, then each coding agent")
    parser.add_argument("--out", type=str, default=None, help="With --fine: write the tables to this CSV")
    args = parser.parse_args()

    if args.fine:
        import model_inventory
        rd = BENCHMARK_DIR / "res_data"
        direct = [rd / f"{m}_{t}_all_details.jsonl" for m in model_inventory.MODELS
                  for t in ("signature", "description", "1-shot_signature", "1-shot_description")]
        missing = [p.name for p in direct if not p.exists()]
        if missing:
            parser.error(f"missing Direct details: {missing[:3]}; run eval_llm.py first")
        fine_report({
            "Direct (24 cells)": direct,
            "Codex": [rd / "gpt-5.6-sol_codex_signature_all_details.jsonl"],
            "CC": [rd / "claude-opus-5_claude_signature_all_details.jsonl"],
        }, out_path=args.out)
        return

    if args.details:
        from pathlib import Path
        path = Path(args.details)
        label = path.name
    elif args.model:
        path = llm_details_path(args.model, args.method, args.fewshot, args.dataset)
        label = f"{args.model}/{args.method}/fewshot{args.fewshot}/{args.dataset}"
    else:
        parser.error("provide --details PATH or --model (+ --method/--fewshot/--dataset)")

    error_report(path, label)


if __name__ == "__main__":
    main()
