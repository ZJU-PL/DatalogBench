"""Declaration repair: how many compile failures are only missing declarations?

Soufflé rejects a program that uses a relation it never declares with
``Error: Undefined relation``. Often the program does define that relation with
a rule head, and only the type declaration is missing. For every generated
program whose compile errors are *only* undefined relations, this tool
synthesizes the missing ``.decl`` lines from how each relation is used,
recompiles, and rescores on the same evaluation variants. Each repaired program
ends in one of three outcomes:

  * it compiles and becomes exact -- the missing declaration was the only fault;
  * it compiles but stays wrong -- the rules themselves derive the wrong relation;
  * it still does not compile -- the undefined relation was a symptom of another
    error.

Repairs are written to a separate tree and never touch ``infer_data``, so the
graded generations stay exactly what the models produced.

    python3 evaluation/decl_repair_probe.py --limit 20      # smoke
    python3 evaluation/decl_repair_probe.py --share         # read-only: share of compile rejections this covers
    bash docker/run-harness.sh python3 evaluation/decl_repair_probe.py
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common_eval import (  # noqa: E402
    BENCHMARK_DIR,
    dataset_index_by_id,
    evaluate_query_file_for_case,
    is_exact,
    save_json,
    save_jsonl,
)
from model_inventory import MODELS  # noqa: E402

CELLS = ["signature", "description", "1-shot_signature", "1-shot_description"]

RE_UNDEF = re.compile(r"Error: Undefined relation\s+(\w+)")
RE_ERROR = re.compile(r"Error: ([A-Za-z][^\n]{0,60})")
RE_DECL = re.compile(r"^\s*\.decl\s+([A-Za-z_]\w*)\s*\(([^)]*)\)", re.M)
RE_NUMBER = re.compile(r"^-?\d+$")
RE_STRING = re.compile(r'^".*"$')
# Souffle functors and aggregates are not relations; an atom-shaped match on one
# would otherwise be declared as a relation and change the program's meaning.
NOT_RELATIONS = {
    "match", "contains", "substr", "strlen", "ord", "to_number", "to_string",
    "cat", "min", "max", "count", "sum", "mean", "autoinc", "range", "cross",
    "as", "nil", "number", "symbol", "unsigned", "float",
}


def iter_atoms(text):
    """Yield relation-shaped calls, including arguments with nested parens.

    The earlier ``[^()]*`` regex silently skipped a complete atom whenever an
    argument contained arithmetic parentheses. That made otherwise visible
    undefined relations look as if they had no inferable argument list.
    """
    head = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\s*\(")
    for match in head.finditer(text):
        depth = 1
        quote = None
        escaped = False
        pos = match.end()
        while pos < len(text) and depth:
            ch = text[pos]
            if quote:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == quote:
                    quote = None
            elif ch in {'"', "'"}:
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            pos += 1
        if depth == 0:
            yield match.group(1), text[match.end():pos - 1]


def split_statements(program):
    """Split rules on terminating dots without splitting quoted regexes.

    A plain ``program.split('.')`` cut ``matches(x, "a.*")`` in half, so its
    undefined relation disappeared from type inference. A terminator is a dot
    outside strings/brackets followed by whitespace or end-of-file; directive
    prefixes such as ``.decl`` therefore remain intact.
    """
    out, start, depths = [], 0, {"(": 0, "[": 0, "{": 0}
    closing = {")": "(", "]": "[", "}": "{"}
    quote = None
    escaped = False
    line_comment = False
    for pos, ch in enumerate(program):
        if line_comment:
            if ch == "\n":
                line_comment = False
            continue
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in {'"', "'"}:
            quote = ch
            continue
        if ch == "/" and pos + 1 < len(program) and program[pos + 1] == "/":
            line_comment = True
            continue
        if ch in depths:
            depths[ch] += 1
        elif ch in closing:
            depths[closing[ch]] = max(0, depths[closing[ch]] - 1)
        elif (ch == "." and not any(depths.values())
              and (pos + 1 == len(program) or program[pos + 1].isspace())):
            out.append(program[start:pos])
            start = pos + 1
    if program[start:].strip():
        out.append(program[start:])
    return out


def split_args(text):
    """Split an atom's argument list on top-level commas."""
    args, depth, cur = [], 0, ""
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        args.append(cur.strip())
    return args


def declared_types(program):
    """Map relation -> list of argument types taken from existing .decl lines."""
    out = {}
    for name, arglist in RE_DECL.findall(program):
        types = []
        for a in split_args(arglist):
            types.append(a.split(":")[-1].strip() if ":" in a else "symbol")
        out[name] = types
    return out


def numeric_variables(rule):
    """Variables forced to numeric type by literals, arithmetic, or counts."""
    clean = re.sub(r'"(?:\\.|[^"\\])*"', '""', rule)
    out = set()
    # count/sum/mean always produce numbers. min/max inherit their input type,
    # so leave those to relation-position propagation rather than guessing.
    for match in re.finditer(
        r"\b([A-Za-z_]\w*)\s*=\s*(?:count|sum|mean)\s*:", clean
    ):
        out.add(match.group(1))
    # Numeric literals on either side of assignment/comparison constrain the
    # variable even when it never appears in an already-declared relation.
    for match in re.finditer(
        r"\b([A-Za-z_]\w*)\s*(?:=|<=|>=|<|>)\s*-?\d+\b|"
        r"-?\d+\s*(?:=|<=|>=|<|>)\s*\b([A-Za-z_]\w*)\b",
        clean,
    ):
        out.add(match.group(1) or match.group(2))
    # Every variable participating in arithmetic must be numeric. This covers
    # both assignments (`next = pos + 1`) and arithmetic relation arguments.
    for match in re.finditer(
        r"(?:\b([A-Za-z_]\w*)\b|-?\d+)\s*[+\-*/%]\s*"
        r"(?:\b([A-Za-z_]\w*)\b|-?\d+)",
        clean,
    ):
        out.update(x for x in match.groups() if x)
    return out


def infer_types(program, relation, known):
    """Infer one relation's argument types from every atom that mentions it.

    A variable's type is taken from a declared relation that binds it in the
    same rule; a literal supplies its own.  Where nothing constrains an
    argument we fall back to ``symbol``, which is Souffle's widest sane
    default and cannot silently coerce a number.
    """
    var_type, arity, seen = {}, None, False
    for rule in split_statements(program):
        atoms = list(iter_atoms(rule))
        if not any(n == relation for n, _ in atoms):
            continue
        for name, arglist in atoms:
            if name not in known:
                continue
            for pos, arg in enumerate(split_args(arglist)):
                if (pos < len(known[name]) and known[name][pos]
                        and re.fullmatch(r"[A-Za-z_]\w*", arg)):
                    var_type.setdefault(arg, known[name][pos])
        for variable in numeric_variables(rule):
            var_type.setdefault(variable, "number")
        for name, arglist in atoms:
            if name != relation:
                continue
            seen = True
            args = split_args(arglist)
            arity = len(args) if arity is None else max(arity, len(args))
            for pos, arg in enumerate(args):
                key = ("pos", pos)
                if RE_NUMBER.match(arg):
                    var_type.setdefault(key, "number")
                elif RE_STRING.match(arg):
                    var_type.setdefault(key, "symbol")
                elif arg in var_type:
                    var_type.setdefault(key, var_type[arg])
                elif re.search(r"[+\-*/%]", arg):
                    var_type.setdefault(key, "number")
    if not seen or arity is None:
        return None
    return [var_type.get(("pos", i)) for i in range(arity)]


def repair(program, missing):
    """Prepend synthesized declarations for the relations Souffle rejected."""
    known = declared_types(program)
    inferred, unresolved = {}, []
    for rel in sorted(missing):
        arities = [len(split_args(args)) for name, args in iter_atoms(program)
                   if name == rel]
        if not arities:
            unresolved.append(rel)
        else:
            inferred[rel] = [None] * max(arities)

    # Missing relations often constrain one another: e.g. Composite gets its
    # numeric type from NumberUpToLimit, which is itself undeclared. Propagate
    # evidence to a fixed point before falling back to symbol.
    known.update(inferred)
    changed = True
    while changed:
        changed = False
        for rel, slots in inferred.items():
            evidence = infer_types(program, rel, known)
            if evidence is None:
                continue
            for pos, type_name in enumerate(evidence):
                if type_name and slots[pos] is None:
                    slots[pos] = type_name
                    changed = True

    lines = []
    for rel, slots in inferred.items():
        types = [type_name or "symbol" for type_name in slots]
        params = ", ".join(f"a{i}:{t}" for i, t in enumerate(types))
        lines.append(f".decl {rel}({params})")
    if not lines:
        return None, unresolved
    banner = "// declarations synthesized by decl_repair_probe.py\n"
    return banner + "\n".join(lines) + "\n\n" + program, unresolved


def candidates(models, cells):
    """Yield generated programs whose only compile errors are undefined relations."""
    for model in models:
        for cell in cells:
            details = BENCHMARK_DIR / "res_data" / f"{model}_{cell}_all_details.jsonl"
            if not details.exists():
                continue
            for line in details.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("compile_ok"):
                    continue
                err = str(row.get("error") or "")
                undef = set(RE_UNDEF.findall(err))
                if not undef:
                    continue
                other = [e for e in set(RE_ERROR.findall(err))
                         if not e.startswith("Undefined relation")]
                if other:
                    continue
                dl = (BENCHMARK_DIR / "infer_data" / model / cell / "all" / "run_0"
                      / f"{row['case_id']}.dl")
                if dl.exists():
                    yield model, cell, row, dl, undef


def compile_failures(models, cells):
    """model -> number of generated programs that fail to compile, over all causes."""
    counts = Counter()
    for model in models:
        for cell in cells:
            details = BENCHMARK_DIR / "res_data" / f"{model}_{cell}_all_details.jsonl"
            if not details.exists():
                continue
            for line in details.open(encoding="utf-8"):
                if line.strip() and not json.loads(line).get("compile_ok"):
                    counts[model] += 1
    return counts


def report_share(models, cells):
    """How much of what CP rejects is undeclared relations alone -- no compiler run needed."""
    failures = compile_failures(models, cells)
    only_undef = Counter(model for model, *_ in candidates(models, cells))
    print("[SHARE] compile failures whose only errors are undefined relations")
    for model in models:
        if failures[model]:
            print(f"  {model:32s} {only_undef[model]:4d}/{failures[model]:<4d} "
                  f"= {only_undef[model] / failures[model]:6.1%}")
    total, part = sum(failures.values()), sum(only_undef.values())
    if total:
        print(f"  {'all':32s} {part:4d}/{total:<4d} = {part / total:6.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--cells", nargs="*", default=CELLS)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N programs (smoke runs)")
    ap.add_argument("--out-dir", default=str(BENCHMARK_DIR / "qa" / "decl_repair"))
    ap.add_argument("--share", action="store_true",
                    help="only report the share of compile failures the probe targets, then exit")
    args = ap.parse_args()

    if args.share:
        report_share(args.models, args.cells)
        return

    out_dir = Path(args.out_dir)
    (out_dir / "programs").mkdir(parents=True, exist_ok=True)
    index = dataset_index_by_id()

    rows, tally = [], Counter()
    per_model = defaultdict(Counter)

    for i, (model, cell, row, dl, undef) in enumerate(candidates(args.models, args.cells)):
        if args.limit and i >= args.limit:
            break
        case_id = row["case_id"]
        program = dl.read_text(encoding="utf-8", errors="replace")
        patched, unresolved = repair(program, undef)
        tally["candidates"] += 1
        if patched is None:
            tally["no_decl_synthesized"] += 1
            per_model[model]["unrepairable"] += 1
            rows.append({"model": model, "cell": cell, "case_id": case_id,
                         "missing": sorted(undef), "repaired": False,
                         "reason": "no argument list could be inferred"})
            continue

        target = out_dir / "programs" / f"{model}__{cell}__{case_id}.dl"
        target.write_text(patched, encoding="utf-8")

        record = index.get(case_id)
        if record is None:
            tally["case_missing_from_dataset"] += 1
            continue
        result = evaluate_query_file_for_case(record, target)
        exact = is_exact(result)
        tally["repaired"] += 1
        tally["compiles_after_repair"] += int(result["compile_ok"])
        tally["exact_after_repair"] += int(exact)
        per_model[model]["repaired"] += 1
        per_model[model]["compiles"] += int(result["compile_ok"])
        per_model[model]["exact"] += int(exact)
        rows.append({
            "model": model, "cell": cell, "case_id": case_id,
            "missing": sorted(undef), "unresolved": unresolved, "repaired": True,
            "compile_ok_after": result["compile_ok"],
            "perfect_match_after": exact,
            "tp": result["tp"], "fp": result["fp"], "fn": result["fn"],
            # Keep the complete compiler diagnostics. Souffle emits warnings
            # before errors, so truncating this string hid the actual cause of
            # every failed repair during the smoke test.
            "error_after": " | ".join(result.get("errors") or []),
        })

    save_jsonl(out_dir / "decl_repair_details.jsonl", rows)
    summary = {
        "counts": dict(tally),
        "per_model": {m: dict(c) for m, c in per_model.items()},
        "note": ("Repairs live under this directory only; benchmark/infer_data is "
                 "never modified, so the graded generations remain what the models "
                 "produced."),
    }
    save_json(out_dir / "decl_repair_summary.json", summary)

    n = tally["repaired"] or 1
    print(f"[CANDIDATES] {tally['candidates']} programs whose only compile errors "
          f"are undefined relations")
    print(f"[REPAIRED]   {tally['repaired']}  "
          f"({tally['no_decl_synthesized']} had no inferable argument list)")
    print(f"[COMPILES]   {tally['compiles_after_repair']}/{n} = "
          f"{tally['compiles_after_repair'] / n * 100:.1f}%")
    print(f"[EXACT]      {tally['exact_after_repair']}/{n} = "
          f"{tally['exact_after_repair'] / n * 100:.1f}%")
    print()
    print("Read the two rates together. A high compile rate with a low exact rate "
          "means the rules were still wrong once declared; a high exact rate means "
          "the missing declaration was the whole failure.")
    print(f"\n[OUT] {out_dir}")


if __name__ == "__main__":
    main()
