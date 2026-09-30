#!/usr/bin/env python3
"""Scan generated programs for predicate names recalled from the upstream artifacts.

Every DatalogBench task derives from a public artifact, so asking whether a model
"has seen" a task is not answerable. This asks a narrower question that has a
sharp answer: did a model emit a predicate name that appears in the upstream
sources but was never shown to it?

Normalization renamed relations when the benchmark was built, so a memorized
program cannot be reproduced verbatim -- the model is given the new schema and
must use it. That makes the reverse observation informative: a name that occurs
upstream, does not occur in the prompt, and is not a Souffle builtin cannot have
been inferred from what the model was given. Each occurrence is direct evidence
of recall rather than an inference from similarity.

The catch is that many relation names are generic. `edge`, `path` and `node` occur
in nearly every graph task upstream, and a model inventing `path` for an auxiliary
relation has demonstrated nothing. We therefore report each hit with the number of
distinct upstream tasks its name occurs in: a name confined to one upstream task
is evidence, a name spread across forty is vocabulary. Nothing is auto-classified
as contamination; the script ranks candidates for a human to judge.

    python3 evaluation/leaked_predicates.py --originals /path/to/Originals
    python3 evaluation/leaked_predicates.py --originals ... --max-ubiquity 3 --json out.json
    python3 evaluation/leaked_predicates.py --originals ... --reported-only   # the run-matrix cells only
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCHMARK = REPO / "benchmark"
INFER = BENCHMARK / "infer_data"

# Souffle builtins, aggregates and type names: a generated program may legitimately
# mention these, and they carry no provenance signal.
BUILTIN = {
    "min", "max", "count", "sum", "mean", "cat", "ord", "strlen", "substr",
    "to_number", "to_string", "match", "contains", "autoinc", "range",
    "number", "symbol", "unsigned", "float", "true", "false", "nil", "not",
}

DECL_RE = re.compile(r"^\s*\.decl\s+(\w+)", re.M)
INPUT_RE = re.compile(r"^\s*\.(?:input|output)\s+(\w+)", re.M)
ATOM_RE = re.compile(r"!?\b([A-Za-z_]\w*)\s*\(")
GDL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
COMMENT_RE = re.compile(r"^\s*(//|#|/\*|\*)")

# A name shorter than this carries no provenance signal: a model inventing an
# auxiliary called `tc` or `sp` has not demonstrated recall of anything.
MIN_NAME_LEN = 3


def upstream_vocabulary(originals: Path):
    """name -> set of upstream task directories that use it.

    Reads every artifact form the sources actually shipped: Datalog rules and
    solutions (.dl), rule templates (.t), Godel programs (.gdl), and the relation
    names implied by fact filenames.
    """
    vocab = defaultdict(set)
    if not originals.exists():
        sys.exit(f"[ERR] originals directory not found: {originals}")

    for path in originals.rglob("*"):
        if not path.is_file():
            continue
        task = path.parent  # one directory per upstream task
        suffix = path.suffix.lower()
        names = set()
        if suffix in (".dl", ".t"):
            # Drop comment lines first: license headers parse as atoms otherwise,
            # which is how "Copyright" ended up looking like a relation.
            text = "\n".join(l for l in path.read_text(errors="ignore").splitlines()
                             if not COMMENT_RE.match(l))
            names |= set(DECL_RE.findall(text))
            names |= set(INPUT_RE.findall(text))
            names |= set(ATOM_RE.findall(text))
            names |= {m.lstrip("*") for m in re.findall(r"^\*?(\w+)\(", text, re.M)}
        elif suffix == ".gdl":
            gtext = "\n".join(l for l in path.read_text(errors="ignore").splitlines()
                               if not COMMENT_RE.match(l))
            names |= set(GDL_RE.findall(gtext))
        elif suffix in (".facts", ".expected", ".csv"):
            names.add(path.stem)
        for n in names:
            if n and len(n) >= MIN_NAME_LEN and n.lower() not in BUILTIN:
                vocab[n].add(str(task))
    return vocab


def schema_names(case):
    """Relation names the model was actually shown for this task."""
    out = set()
    for block in (case.get("input_relation", {}), case.get("output_relation", {})):
        for sig in block:
            m = re.match(r"\s*(\w+)", sig)
            if m:
                out.add(m.group(1))
    return out


def generated_names(rules_text):
    """Predicate names a generated program declares or references."""
    names = set(DECL_RE.findall(rules_text)) | set(INPUT_RE.findall(rules_text))
    names |= set(ATOM_RE.findall(rules_text))
    return {n for n in names
            if len(n) >= MIN_NAME_LEN and n.lower() not in BUILTIN}


# The run-matrix configurations: the direct-prompting grid of every evaluated
# model, plus each agent's signature run.  Smoke scopes, repeated runs and other
# agent conditions may also be on disk and must not enter the count.
REPORTED_MODELS = {"gpt-5.6-sol", "claude-opus-5", "gemini-3.7-flash", "deepseek-v4-pro",
                   "deepseek-v4-flash", "deepseek-v4-flash-non-thinking"}
REPORTED_DIRECT = {"signature", "description", "1-shot_signature", "1-shot_description"}
REPORTED_AGENTS = {("gpt-5.6-sol", "codex_signature"), ("claude-opus-5", "claude_signature")}


def iter_reported(root: Path):
    """Like iter_generations, restricted to scope `all`, run 0, reported configurations,
    and one program per (model, configuration, case)."""
    seen = set()
    for agg in sorted(root.rglob("run_0/*.json")):
        parts = agg.relative_to(root).parts  # model/tag/scope/run_0/file.json
        if len(parts) != 5 or parts[2] != "all":
            continue
        model, tag = parts[0], parts[1]
        if model not in REPORTED_MODELS or not (
                tag in REPORTED_DIRECT or (model, tag) in REPORTED_AGENTS):
            continue
        try:
            rows = json.loads(agg.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            cid, rules = row.get("id"), row.get("query")
            if cid and isinstance(rules, str) and rules.strip() and (model, tag, cid) not in seen:
                seen.add((model, tag, cid))
                yield model, tag, "run_0", cid, rules


def reference_inventions():
    """case_id -> auxiliary predicate names the reference program introduces."""
    sys.path.insert(0, str(REPO / "evaluation"))
    from structure_analyzer import analyze_program
    return {p.stem: set(analyze_program(p.read_text(encoding="utf-8"))["invented"])
            for p in sorted((BENCHMARK / "query").glob("*.dl"))}


def codefuse_split(originals: Path):
    """Direct-prompting EX on tasks drawn from CodeFuse-Query against the rest.

    CodeFuse-Query published Goedel, not Datalog, so for these tasks no Datalog
    text existed upstream; recall of Datalog text should help them less.  Task
    membership is by exact directory name."""
    upstream = originals / "CodeFuseQuery"
    if not upstream.is_dir():
        return
    cases = {c["id"] for c in json.loads((BENCHMARK / "dataset.json").read_text())}
    cf = cases & {p.name for p in upstream.iterdir()}
    rest = cases - cf
    cf_rates, rest_rates = [], []
    for model in sorted(REPORTED_MODELS):
        for tag in sorted(REPORTED_DIRECT):
            path = BENCHMARK / "res_data" / f"{model}_{tag}_all_details.jsonl"
            if not path.exists():
                continue
            exact = {r["case_id"] for r in map(json.loads, path.read_text().splitlines())
                     if r.get("perfect_match")}
            cf_rates.append(len(exact & cf) / len(cf))
            rest_rates.append(len(exact & rest) / len(rest))
    if not cf_rates:
        return
    n = len(cf_rates)
    ahead = sum(a > b for a, b in zip(cf_rates, rest_rates))
    print(f"\n[CODEFUSE] {len(cf)} CodeFuse-Query tasks vs {len(rest)} others, over {n} direct cells")
    print(f"  mean EX {sum(cf_rates)/n:.1%} vs {sum(rest_rates)/n:.1%}; "
          f"CodeFuse-Query ahead in {ahead}/{n} cells")


def iter_generations(root: Path):
    """Yield (model, tag, scope, run, case_id, rules) over infer_data."""
    if not root.exists():
        return
    for agg in sorted(root.rglob("run_*/*.json")):
        if agg.name.startswith("provenance"):
            continue
        try:
            rows = json.loads(agg.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(rows, list):
            continue
        parts = agg.relative_to(root).parts  # model/tag/scope/run_N/file.json
        model = parts[0] if len(parts) > 0 else "?"
        tag = parts[1] if len(parts) > 1 else "?"
        run = agg.parent.name
        for row in rows:
            cid, rules = row.get("id"), row.get("query")
            if cid and isinstance(rules, str) and rules.strip():
                yield model, tag, run, cid, rules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--originals", required=True,
                    help="directory holding the upstream sources (EGS, GenSynth, Souffle, CodeFuseQuery*)")
    ap.add_argument("--infer-dir", default=str(INFER),
                    help="root of generated programs (benchmark/infer_data)")
    ap.add_argument("--max-ubiquity", type=int, default=3,
                    help="only flag names confined to at most this many upstream tasks (0 = no limit)")
    ap.add_argument("--json", default=None, help="write full findings here")
    ap.add_argument("--reported-only", action="store_true",
                    help="scan only the run-matrix configurations (see REPORTED_*)")
    args = ap.parse_args()

    vocab = upstream_vocabulary(Path(args.originals))
    print(f"[UPSTREAM] {len(vocab)} distinct relation names across "
          f"{len({t for ts in vocab.values() for t in ts})} task directories")

    cases = {c["id"]: c for c in json.loads((BENCHMARK / "dataset.json").read_text())}
    schemas = {cid: schema_names(c) for cid, c in cases.items()}

    findings, n_progs = [], 0
    source = iter_reported if args.reported_only else iter_generations
    for model, tag, run, cid, rules in source(Path(args.infer_dir)):
        n_progs += 1
        given = schemas.get(cid, set())
        for name in generated_names(rules) - given:
            tasks = vocab.get(name)
            if not tasks:
                continue                      # invented, not upstream: nothing to say
            if args.max_ubiquity and len(tasks) > args.max_ubiquity:
                continue                      # generic vocabulary, not recall
            findings.append({
                "case_id": cid, "model": model, "config": tag, "run": run,
                "name": name, "upstream_tasks": sorted(tasks)[:5],
                "ubiquity": len(tasks),
            })

    print(f"[SCAN] {n_progs} generated program(s)")
    if not n_progs:
        print("  no generations found -- run the synthesis pipeline first")
        return

    if not findings:
        print("  no upstream-only predicate names appear in any generation.")
        print("  This is the negative result the check is designed to produce; report it as such.")
    else:
        print(f"  {len(findings)} occurrence(s) of an upstream name never shown to the model\n")
        by_name = Counter(f["name"] for f in findings)
        print(f"  {'name':<26}{'hits':>6}{'upstream tasks':>16}   example case")
        for name, hits in by_name.most_common(30):
            ex = next(f for f in findings if f["name"] == name)
            print(f"  {name:<26}{hits:>6}{ex['ubiquity']:>16}   {ex['case_id']}")
        print("\n  Each row is a candidate, not a verdict: confirm by checking whether the name "
              "is plausible as an independent invention before calling it recall.")
        # Two checks on whether a flagged name is recall or the obvious word:
        # did our own annotators choose it for the same task, or anywhere?
        inv = reference_inventions()
        anywhere = set().union(*inv.values())
        same_task = sum(f["name"] in inv.get(f["case_id"], set()) for f in findings)
        in_refs = sum(f["name"] in anywhere for f in findings)
        print(f"\n  also invented by our reference for the same task   {same_task}/{len(findings)}")
        print(f"  invented by some reference program in the benchmark {in_refs}/{len(findings)}")

    if args.reported_only:
        codefuse_split(Path(args.originals))

    if args.json:
        Path(args.json).write_text(json.dumps(findings, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
