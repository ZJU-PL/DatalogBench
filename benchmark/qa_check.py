"""Benchmark QA checks for the golden programs and their I/O variants (CI-friendly).

Seven checks:

1. **Dead-rule liveness analysis.** A rule is dead when its head relation cannot
   reach any `.output` relation in the dependency graph -- the rule can never
   influence an exported result, so mutants of it are unkillable and complexity
   statistics that count it are inflated.

2. **Blind oracle slots.** A `(variant, output relation)` slot whose `.expected`
   is empty or absent scores vacuously: with an empty golden set `tp` and `fn`
   are always 0, so the slot can only ever catch over-approximation. A program
   that omits the rules for that relation entirely and emits nothing scores
   *perfectly* there. An absent file is the same blindness and harder to spot,
   because `common_eval.load_tuples` returns an empty set for a missing path.

   Emptiness is not automatically a defect -- a satisfiable 2-SAT instance
   genuinely has no contradiction variables -- so this check reports rather
   than fails, with one exception: a blind slot in a *demo* variant is always
   wrong, because the few-shot example then demonstrates that the relation
   produces nothing. Use --strict-empty to fail on any blind slot.

3. **Duplicate variants.** Variants are distinguished by their input facts, so
   two variants whose `.facts` hold the same tuples are the same test run
   twice: they cost evaluation time and can never kill a mutant the other
   misses. Equivalence is judged on normalised tuple sets, not bytes, so
   reordered lines and trailing newlines are correctly ignored. Reported rather
   than failed (--strict-duplicates to enforce), with two exceptions that are
   always errors:

   - a *demo* variant equivalent to an *eval* variant, which makes the demo /
     eval split nominal and leaks the scored input into the few-shot prompt;
   - equivalent inputs whose `.expected` differ, which means one of the two was
     not regenerated from the current reference program.

4. **Knowledge-field discipline.** The optional `knowledge` field states what a
   task's terms and values mean; it must not state program structure. A field
   that names the reference program's invented predicates has stopped being
   vocabulary and become a solution outline, which would inflate the
   with-knowledge column of the main table. Naming an *input* or *output*
   relation is fine -- the prompt already shows those. `spec_tier` is validated
   against L1/L2/L3 at the same time. Both are errors, because unlike an empty
   oracle slot there is no benign reading.

5. **Baseline survivor fidelity.** A mutation baseline stores each surviving
   mutant as text so it can be rebuilt and hand-checked. Four goldens seed their
   recursion with ground facts (`Indices(0).`, `Conf(0).`, `Again(0).`,
   `ValidStep(0).`); a baseline that kept only the lines containing `:-` drops
   those seeds, and a survivor rebuilt without them evaluates to the empty
   result -- which reads as "killed" on *every* input. That silently invalidates
   manual survivor review, so it is an error rather than a report. The same
   applies to a bodyless aggregate assignment such as MinSpanTree's
   `MinimumSpanningTree(sum w : { ... }).`. Skipped when no baseline is present
   (the qa/ directory is not committed).

7. **Schema agreement.** `dataset.json`'s `input_relation` / `output_relation`
   are what the prompt shows the solver; the golden's `.input` / `.output`
   directives are what the task actually is. When they disagree the task is
   mis-stated at best and unanswerable at worst -- CommentCodePairsPython
   described a merged `ElementComment` that does not exist while omitting the
   `PyDocstring` / `PyComment` split its question turns on, and FibSymbolic
   never mentioned the `PlusMod` table its reference program reads. Neither is
   reachable by the readback probe, which asks a model to restate the spec and
   therefore measures agreement about whatever schema it was handed. An error,
   because a solver cannot be expected to guess a relation it was not shown.

Reuses the structure_analyzer parser, which already handles two parsing
pitfalls: variables are identified positionally (never by
case), and statements are split on top-level '.' (head-aggregate rules such as
`X(sum w : { R(...) }).` have no ':-' yet still depend on the relations inside
the aggregate body).

Usage:
    python3 benchmark/qa_check.py                     # whole library
    python3 benchmark/qa_check.py --case Escape       # one case, with detail
    python3 benchmark/qa_check.py --json out.json     # machine-readable report
    python3 benchmark/qa_check.py --strict-empty      # blind slots also fail
    python3 benchmark/qa_check.py --strict-duplicates # duplicate variants also fail

Exit code 0 = clean, 1 = a hard failure (dead rules, a graded relation the
program does not export, a blind demo slot, a demo/eval duplicate, or
duplicate inputs with diverging expected output), so it can gate CI.
"""

import argparse
import json
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

# structure_analyzer lives in evaluation/; qa_check is a CI entry point under
# benchmark/, so put the sibling dir on the path before importing.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "evaluation"))

from common_eval import load_tuples  # noqa: E402
from structure_analyzer import analyze_program  # noqa: E402
from structure_analyzer import (  # noqa: E402
    _REF_RE,
    _aggregate_spans,
    _refs_with_context,
    _split_directives_and_body,
    _split_statements,
    _strip_comments,
    _INPUT_RE,
    _OUTPUT_RE,
    _DECL_RE,
)

QUERY_DIR = _REPO / "benchmark" / "query"
IODATA_DIR = _REPO / "benchmark" / "io_data"
DATASET = _REPO / "benchmark" / "dataset.json"


def parse_program(dl_text):
    """Return (inputs, outputs, declared, rules) where rules is a list of
    (head, deps, statement_text). deps are the *known* relations the rule body
    references (including inside aggregate bodies)."""
    text = _strip_comments(dl_text)
    directives, body = _split_directives_and_body(text)

    declared, inputs, outputs = set(), set(), set()
    for d in directives:
        m = _DECL_RE.match(d)
        if m:
            declared.add(m.group(1))
            continue
        m = _INPUT_RE.match(d)
        if m:
            inputs.add(m.group(1))
            continue
        m = _OUTPUT_RE.match(d)
        if m:
            outputs.add(m.group(1))

    statements = _split_statements(body)
    heads = []
    for stmt in statements:
        m = _REF_RE.match(stmt.lstrip("!").lstrip())
        heads.append(m.group(2) if m else None)
    known = declared | {h for h in heads if h}

    rules = []
    for stmt, head in zip(statements, heads):
        if head is None:
            continue
        agg_spans = _aggregate_spans(stmt)
        deps = set()
        for i, (name, _neg, _agg, _idx) in enumerate(_refs_with_context(stmt, agg_spans)):
            if i == 0:
                continue  # the head atom itself
            if name in known:
                deps.add(name)
        rules.append((head, deps, stmt))
    return inputs, outputs, known, rules


def live_relations(outputs, rules):
    """Relations that can reach an output: reverse-reachability from outputs
    over head <- deps edges."""
    dependants = {}  # relation -> set of heads whose rules use it
    for head, deps, _ in rules:
        for d in deps:
            dependants.setdefault(d, set()).add(head)

    live = set(outputs)
    frontier = list(outputs)
    heads_by_rel = {}
    for head, deps, _ in rules:
        heads_by_rel.setdefault(head, []).append(deps)

    # a relation R is live if R is an output, or some live relation's rule
    # depends on R. Walk backwards: for each live relation, its rules' deps
    # become live.
    while frontier:
        rel = frontier.pop()
        for deps in heads_by_rel.get(rel, []):
            for d in deps:
                if d not in live:
                    live.add(d)
                    frontier.append(d)
    return live


def dead_rules_for(dl_text):
    inputs, outputs, known, rules = parse_program(dl_text)
    live = live_relations(outputs, rules)
    dead = [(head, stmt) for head, deps, stmt in rules if head not in live]
    return dead, len(rules), outputs, live


def _norm(stmt):
    """Whitespace/terminator-insensitive form, for matching parsed statements
    back to the raw source lines."""
    return " ".join(stmt.replace("\n", " ").split()).rstrip(".").strip()


def fix_case(case_id, dry_run=True):
    """Delete a case's dead rules (and any .decl left with no purpose).

    Rewrites `benchmark/query/<case>.dl` in place. Dead rules cannot affect any
    exported relation, so the cleaned program must stay behaviourally identical
    -- always re-run the evaluation afterwards to confirm that empirically.
    """
    path = QUERY_DIR / f"{case_id}.dl"
    text = path.read_text(encoding="utf-8")
    dead, total, _outputs, _live = dead_rules_for(text)
    if not dead:
        return None
    dead_norm = {_norm(stmt) for _, stmt in dead}

    kept_lines, dropped, buf, buf_lines = [], [], "", []
    for line in text.splitlines():
        s = line.strip()
        if not buf and (not s or s.startswith("//") or s.startswith(".")):
            kept_lines.append(line)
            continue
        buf = (buf + " " + s).strip()
        buf_lines.append(line)
        if s.endswith("."):
            if _norm(buf) in dead_norm:
                dropped.append(_norm(buf))
            else:
                kept_lines.extend(buf_lines)
            buf, buf_lines = "", []
    if buf_lines:                      # unterminated tail: keep it untouched
        kept_lines.extend(buf_lines)

    if len(dropped) != len(dead_norm):
        raise RuntimeError(
            f"{case_id}: matched {len(dropped)} of {len(dead_norm)} dead rules in the "
            "source; refusing to rewrite (statement layout not line-aligned)"
        )

    # Drop declarations that now serve no purpose: no rules, not .input/.output.
    cleaned = "\n".join(kept_lines)
    inputs, outputs, _known, parsed = parse_program(cleaned)
    still_defined = {h for h, _, _ in parsed}
    orphan_decls = []
    final_lines = []
    for line in kept_lines:
        m = _DECL_RE.match(line.strip())
        if m:
            rel = m.group(1)
            if rel not in inputs and rel not in outputs and rel not in still_defined:
                orphan_decls.append(rel)
                continue
        final_lines.append(line)

    out_text = "\n".join(final_lines).rstrip("\n") + "\n"
    print(f"-- {case_id}: removing {len(dropped)}/{total} dead rule(s)"
          + (f", {len(orphan_decls)} orphan .decl ({', '.join(orphan_decls)})" if orphan_decls else ""))
    if dry_run:
        print("   (dry run; pass --fix to write)")
    else:
        path.write_text(out_text, encoding="utf-8")
        print(f"   written -> {path}")
    return {"case_id": case_id, "removed_rules": len(dropped),
            "orphan_decls": orphan_decls, "total_rules_before": total}


def variant_dirs(case_id):
    """(label, path, role) per variant directory.

    Supports both layouts: the flat `io_data/<case>/<i>/` in use today, and the
    `eval/<i>` + `demo/<i>` split that `select_variants.py --out` materialises.
    Role matters because a blind slot in a demo variant is a different, and
    always-wrong, failure: it feeds the model a few-shot example.
    """
    case_dir = IODATA_DIR / case_id
    if not case_dir.is_dir():
        return []
    out = []
    for role in ("eval", "demo"):
        role_dir = case_dir / role
        if role_dir.is_dir():
            for vd in sorted(role_dir.iterdir(), key=lambda p: p.name):
                if vd.is_dir():
                    out.append((f"{role}/{vd.name}", vd, role))
    if out:
        return out
    for vd in sorted(case_dir.iterdir(), key=lambda p: p.name):
        if vd.is_dir():
            out.append((vd.name, vd, "variant"))
    return out


def scored_relations(case_record):
    """The relations the evaluator actually grades.

    This is dataset.json's `output_relation`, NOT the golden's `.output`
    directives -- `common_eval.get_case_output_relations` reads the record, so
    a relation the program exports but the record omits is never scored at all.
    Basing the slot check on the .dl instead would mis-file that case as a
    blind slot when it is really an orphan output (see orphan_outputs_for).
    """
    return {x.split("(")[0] for x in (case_record.get("output_relation") or {})}


def orphan_outputs_for(declared_outputs, scored):
    """Relations exported by the golden but absent from the graded set, and the
    reverse. The first is dead weight the oracle ignores; the second means the
    record grades something the program never exports, so nothing can pass."""
    return {"unscored_exports": sorted(declared_outputs - scored),
            "ungraded_missing_export": sorted(scored - declared_outputs)}


def blind_slots_for(case_id, scored):
    """Slots that cannot penalise a program for producing nothing.

    `empty` = the .expected exists but holds no tuples; `missing` = there is no
    .expected at all, which the evaluator silently reads as the empty set.
    """
    slots = []
    for label, vdir, role in variant_dirs(case_id):
        for rel in sorted(scored):
            path = vdir / f"{rel}.expected"
            if not path.exists():
                kind = "missing"
            elif not path.read_text(encoding="utf-8").strip():
                kind = "empty"
            else:
                continue
            slots.append({"variant": label, "relation": rel,
                          "role": role, "kind": kind})
    return slots


def _tuple_signature(paths):
    """Content signature as the *evaluator* sees it: {file: set of tuples}.

    Deliberately not a byte hash. Tuple order and trailing newlines carry no
    meaning to `common_eval.load_tuples`, so hashing bytes would report churn
    the scoring pipeline cannot see -- and a CI check that cries wolf on
    reordered lines is worse than no check. Comparing normalised tuple sets
    keeps this aligned with what actually gets graded.
    """
    return frozenset(
        (p.name, frozenset(load_tuples(p))) for p in sorted(paths, key=lambda x: x.name)
    )


def duplicate_variants_for(case_id):
    """Groups of variants whose input facts are equivalent.

    Inputs are what makes one variant differ from another, so equivalent inputs
    mean the same test executed twice -- pure evaluation cost with no possible
    gain in discriminative power.
    """
    groups = defaultdict(list)
    for label, vdir, role in variant_dirs(case_id):
        groups[_tuple_signature(vdir.glob("*.facts"))].append((label, role, vdir))

    out = []
    for members in groups.values():
        if len(members) < 2:
            continue
        roles = {r for _l, r, _d in members}
        # Equivalent inputs must imply equivalent expected output; if not, one
        # of them was not regenerated after the reference program last changed.
        expected = {_tuple_signature(d.glob("*.expected")) for _l, _r, d in members}
        out.append({
            "variants": [l for l, _r, _d in members],
            "demo_eval_leak": {"demo", "eval"} <= roles,
            "expected_diverges": len(expected) > 1,
        })
    return out


def knowledge_violations(case_record, dl_text):
    """Ways a case's knowledge / spec_tier fields break the construction rules.

    The structural half of the rule is mechanically checkable: the reference
    program's *invented* predicates are exactly the intermediate concepts a
    solver is supposed to discover, so naming one in the knowledge field hands
    over the decomposition. Input and output relations are excluded because the
    prompt shows them already.
    """
    problems = []
    # The sufficiency tier used to live here. It moved to
    # benchmark/spec_tiers.json: nothing in the prompt shows it, so keeping it
    # in dataset.json put a field a solver never sees beside the fields a solver
    # is given. `spec_repaired` went with it and was dropped outright -- it
    # recorded what this round changed, which belongs to the history rather than
    # to the artifact.
    text = ((case_record or {}).get("knowledge") or "").strip()
    if not text:
        return problems
    feats = analyze_program(dl_text)
    leaked = [r for r in feats["invented"]
              if re.search(rf"\b{re.escape(r)}\b", text, re.IGNORECASE)]
    if leaked:
        problems.append(
            "knowledge names the reference program's invented predicate(s): "
            + ", ".join(leaked))
    return problems


def check_case(case_id, case_record=None, verbose=False):
    path = QUERY_DIR / f"{case_id}.dl"
    gold_text = path.read_text(encoding="utf-8")
    dead, total, outputs, _ = dead_rules_for(gold_text)
    if verbose and dead:
        print(f"-- {case_id}: {len(dead)}/{total} dead rules (outputs: {sorted(outputs)})")
        for head, stmt in dead:
            preview = stmt if len(stmt) <= 100 else stmt[:97] + "..."
            print(f"   [{head}] {preview}")

    scored = scored_relations(case_record) if case_record is not None else set(outputs)
    orphans = orphan_outputs_for(outputs, scored)
    if verbose and any(orphans.values()):
        for rel in orphans["unscored_exports"]:
            print(f"-- {case_id}: [orphan ] .output {rel} is never graded "
                  "(absent from dataset.json output_relation)")
        for rel in orphans["ungraded_missing_export"]:
            print(f"-- {case_id}: [BROKEN ] graded relation {rel} is not .output "
                  "in the golden program")

    slots = blind_slots_for(case_id, scored)
    n_variants = len(variant_dirs(case_id))
    if verbose and slots:
        print(f"-- {case_id}: {len(slots)}/{n_variants * len(scored)} blind oracle slot(s)")
        for s in slots:
            flag = "  <- DEMO: teaches an empty answer" if s["role"] == "demo" else ""
            print(f"   [{s['kind']:7}] {s['variant']}/{s['relation']}{flag}")

    schema_problems = schema_agreement(case_id, case_record, gold_text)
    unread = unread_inputs(gold_text)
    if verbose and unread:
        print(f"-- {case_id}: [SCHEMA] .input never read by any rule: {unread}")
    if verbose and schema_problems:
        for msg in schema_problems:
            print(f"-- {case_id}: [SCHEMA] {msg}")

    kn_problems = knowledge_violations(case_record, gold_text)
    if verbose and kn_problems:
        for msg in kn_problems:
            print(f"-- {case_id}: [KNOWLEDGE] {msg}")

    dups = duplicate_variants_for(case_id)
    if verbose and dups:
        redundant = sum(len(g["variants"]) - 1 for g in dups)
        print(f"-- {case_id}: {redundant}/{n_variants} redundant variant(s)")
        for g in dups:
            flags = ""
            if g["demo_eval_leak"]:
                flags += "  <- DEMO == EVAL: few-shot leaks a scored input"
            if g["expected_diverges"]:
                flags += "  <- same inputs, DIFFERENT .expected"
            print(f"   [{'dup':7}] {' == '.join(g['variants'])}{flags}")

    return {"case_id": case_id, "total_rules": total, "dead_rules": len(dead),
            "dead_heads": sorted({h for h, _ in dead}),
            "declared_outputs": sorted(outputs), "scored_outputs": sorted(scored),
            "orphan_outputs": orphans["unscored_exports"],
            "ungraded_missing_export": orphans["ungraded_missing_export"],
            "variants": n_variants,
            "total_slots": n_variants * len(scored),
            "blind_slots": slots,
            "blind_demo_slots": sum(1 for s in slots if s["role"] == "demo"),
            "knowledge_problems": kn_problems,
            "schema_problems": schema_problems,
            "unread_inputs": unread,
            "duplicate_groups": dups,
            "redundant_variants": sum(len(g["variants"]) - 1 for g in dups)}


BASELINE = _REPO / "benchmark" / "qa" / "mutation_frozen.json"


def bodyless_statements(dl_text):
    """Statements a golden asserts with no body: ground facts that seed a
    recursion (`Again(0).`) and bodyless aggregate assignments
    (`MinimumSpanningTree(sum w : { ... }).`). Both are part of the program and
    both are lost by any record that keeps only the lines containing `:-`."""
    return [ln.strip() for ln in dl_text.splitlines()
            if ln.strip() and not ln.strip().startswith(".")
            and not ln.strip().startswith("//") and ":-" not in ln]


def baseline_fidelity(baseline_path=BASELINE):
    """Stored survivors must carry the golden's ground facts, or they rebuild wrong.

    Returns (problems, checked, skipped_reason).
    """
    if not Path(baseline_path).exists():
        return [], 0, f"no baseline at {baseline_path}"
    try:
        records = json.load(open(baseline_path))
    except (ValueError, OSError) as exc:
        return [f"baseline unreadable: {exc}"], 0, None

    problems, checked = [], 0
    for rec in records:
        cid = rec.get("case")
        survivors = rec.get("survivors") or []
        if not cid or not survivors:
            continue
        path = QUERY_DIR / f"{cid}.dl"
        if not path.exists():
            continue
        facts = bodyless_statements(path.read_text(encoding="utf-8"))
        if not facts:
            continue
        checked += 1
        # A mutation removes at most one statement, so a survivor legitimately
        # missing a bodyless statement is the "deleted the base case" mutant.
        # A lossy record is different in kind: it drops *every* fact from
        # *every* survivor. Flag that shape, not the single legitimate one.
        gaps = []
        for k, stored in enumerate(survivors):
            kept = {ln.strip() for ln in stored}
            gaps.append([f for f in facts if f not in kept])
        for k, missing in enumerate(gaps):
            if len(missing) > 1:
                problems.append(
                    f"{cid}#{k}: stored survivor is missing {len(missing)} bodyless "
                    f"statements {missing}, but a mutation removes at most one -- "
                    "the record was written by a baseline that kept only the rules")
        if len(gaps) > 1 and all(g for g in gaps):
            problems.append(
                f"{cid}: every one of the {len(gaps)} stored survivors is missing a "
                "bodyless statement; a mutation removes at most one statement, so "
                "this is a baseline that dropped the golden's facts wholesale")
    return problems, checked, None


MANIFEST = _REPO / "benchmark" / "negative_manifest.json"


def negative_manifest_drift():
    """The committed negative manifest must match the .undesired files on disk.

    The manifest records what each case's negative set covers -- tuple counts,
    distinct failure modes, and which mutants stalled. Nothing else records the
    last two, so a manifest that has drifted from the files silently misstates
    the benchmark in any table that quotes it.
    """
    if not MANIFEST.exists():
        return [], 0, f"no manifest at {MANIFEST}"
    import sys as _s
    _s.path.insert(0, str(_REPO / "evaluation"))
    from common_eval import case_variants_dirs
    try:
        entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        return [f"manifest unreadable: {exc}"], 0, None
    problems, checked = [], 0
    for e in entries:
        cid = e.get("case")
        if not cid or e.get("status") != "ok":
            continue
        on_disk = 0
        for vd in case_variants_dirs(cid):
            for f in vd.glob("*.undesired"):
                on_disk += len([l for l in f.read_text(encoding="utf-8").splitlines() if l.strip()])
        checked += 1
        if on_disk != e.get("negatives"):
            problems.append(f"{cid}: manifest says {e.get('negatives')} negative(s), "
                            f"{on_disk} on disk -- re-run negative_examples.py --manifest")
    return problems, checked, None


def schema_agreement(case_id, case_record, dl_text):
    """Relations the prompt advertises vs relations the golden declares."""
    if case_record is None:
        return []
    declared_in = {m.group(1) for line in dl_text.splitlines()
                   if (m := re.match(r"\s*\.input\s+(\w+)", line))}
    declared_out = {m.group(1) for line in dl_text.splitlines()
                    if (m := re.match(r"\s*\.output\s+(\w+)", line))}
    spec_in = {k.split("(")[0].strip() for k in (case_record.get("input_relation") or {})}
    spec_out = {k.split("(")[0].strip() for k in (case_record.get("output_relation") or {})}

    out = []
    for label, spec, declared in (("input", spec_in, declared_in),
                                  ("output", spec_out, declared_out)):
        phantom = sorted(spec - declared)
        hidden = sorted(declared - spec)
        if phantom:
            out.append(f"{label}_relation advertises {phantom}, which the golden does not declare")
        if hidden:
            out.append(f"golden declares .{label} {hidden}, which {label}_relation never mentions")

    # Arity and parameter names, against the golden's own .decl. A signature the
    # solver cannot read positionally is as unusable as a missing one: a case
    # once advertised `Load(var, var)` for a `Load(dst_var, addr_var)`, leaving
    # nothing to say which position was the destination. Parameter names are the
    # only thing the signature arm shows, so a wrong one misleads rather than
    # merely omits -- see Speed's `distance_to_sink`, which held a difference.
    decls = {}
    for line in dl_text.splitlines():
        m = re.match(r"\s*\.decl\s+(\w+)\s*\((.*)\)\s*$", line.strip())
        if m:
            args = [a.split(":")[0].strip() for a in m.group(2).split(",") if a.strip()]
            decls[m.group(1)] = args
    for field in ("input_relation", "output_relation"):
        for key in (case_record.get(field) or {}):
            m = re.match(r"\s*(\w+)\s*\((.*)\)\s*$", key)
            if not m:
                continue
            rel, args = m.group(1), [a.strip() for a in m.group(2).split(",") if a.strip()]
            if len(args) != len(set(args)):
                out.append(f"{field} signature {key!r} repeats a parameter name, so the "
                           "positions cannot be told apart")
            gold_args = decls.get(rel)
            if gold_args is not None and len(gold_args) != len(args):
                out.append(f"{field} signature {key!r} has {len(args)} argument(s); the golden "
                           f"declares {rel} with {len(gold_args)}")
    return out


def unread_inputs(dl_text):
    """.input relations no rule ever reads.

    Not a defect in itself -- a realistic schema carries tables a query does not
    touch, and AccessPolicy's bank tables are plausibly deliberate. It is a
    defect when it is *accidental*, and nothing else in the pipeline would ever
    say so, so it is reported for a human to classify rather than failed on.
    """
    declared = {m.group(1) for line in dl_text.splitlines()
                if (m := re.match(r"\s*\.input\s+(\w+)", line))}
    _dirs, rules = parse_program(dl_text)[3], None
    body = " ".join(line for line in dl_text.splitlines()
                    if ":-" in line or (line.strip() and not line.strip().startswith(".")))
    return sorted(r for r in declared if not re.search(rf"\b{r}\s*\(", body))


def singleton_violations():
    """Declared parameter relations that carry more than one tuple.

    `benchmark/singleton_inputs.json` lists the input relations that hold an
    instance *parameter* -- a limit, a default, a dimension, a designated start
    node. Scaling one does not make a bigger instance, it changes the task, and
    it does so silently: Array's neighbourhood padding was multiplied by each of
    1600 Default rows, Factorial's 1600 Lim rows collapsed to min()=0 so the
    scaled variant derived a single tuple, and MinPathSrc's eval/2 had *no*
    source at all, which made the reference's 'infinity' sentinel the graded
    ground truth. None of that fails loudly, so it is checked here.
    """
    path = _REPO / "benchmark" / "singleton_inputs.json"
    if not path.exists():
        return [], 0, "no benchmark/singleton_inputs.json"
    decl = json.loads(path.read_text(encoding="utf-8")).get("singletons", {})
    bad, checked = [], 0
    for case, rels in sorted(decl.items()):
        for rel in rels:
            for vdir in sorted((d for d in (IODATA_DIR / case).glob("*/*") if d.is_dir()),
                               key=lambda p: (p.parent.name, p.name)):
                f = vdir / f"{rel}.facts"
                if not f.exists():
                    continue
                checked += 1
                n = len([ln for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()])
                if n != 1:
                    bad.append(f"{case}/{vdir.parent.name}/{vdir.name}/{rel}: "
                               f"{n} tuple(s), must be exactly 1")
    return bad, checked, None


def composition_invariants():
    """Grade a synthetic candidate and check the composer did not rewrite it.

    `compose_full_dl` builds the graded program from the graded interface plus
    the candidate's program. Two failures would be invisible in any score: a
    candidate's own `.decl` lines being dropped, so a candidate that declared the
    relations it invented is graded as though it had not; and the reference's
    auxiliary declarations leaking into the composed program, handing them to
    every candidate for free.

    The probe is synthetic on purpose. The real generations are not in the
    repository, so a check that reads them would pass vacuously on a fresh
    clone. Composing a candidate written here tests the composer itself, on every
    case, with nothing on disk.

    Returns (problems, checked, skipped_reason).
    """
    try:
        sys.path.insert(0, str(_REPO / "synthesis"))
        sys.path.insert(0, str(_REPO / "evaluation"))
        from synth_llm import compose_full_dl, load_golden_decl_block
    except Exception as exc:                      # noqa: BLE001
        return [], 0, f"cannot import the composer ({exc})"

    decl_re = re.compile(r"^\s*\.decl\s+([A-Za-z_]\w*)", re.M)
    io_re = re.compile(r"^\s*\.(input|output)\s+([A-Za-z_]\w*)", re.M)
    probe_aux = "QaProbeAuxRelation"
    problems, checked = [], 0

    cases = [r["id"] for r in json.loads(DATASET.read_text(encoding="utf-8"))]
    for case_id in sorted(cases):
        golden = (QUERY_DIR / f"{case_id}.dl")
        if not golden.is_file():
            continue
        gtext = golden.read_text(encoding="utf-8")
        try:
            interface = set(decl_re.findall(load_golden_decl_block(case_id)))
        except Exception:                          # noqa: BLE001
            continue
        aux = (set(decl_re.findall(gtext)) - {n for _, n in io_re.findall(gtext)}) - interface

        # A candidate that declares, and uses, a relation of its own.
        rules = (f".decl {probe_aux}(x:symbol)\n"
                 f"{probe_aux}(\"x\") :- true.\n")
        composed = compose_full_dl(gtext.split("\n\n")[0], rules, case_id=case_id)
        names = set(decl_re.findall(composed))
        checked += 1

        if probe_aux not in names:
            problems.append(f"{case_id}: the candidate's own declaration was dropped")
        leaked = aux & names
        if leaked:
            problems.append(f"{case_id}: reference auxiliary declarations leaked "
                            f"to the candidate: {sorted(leaked)[:4]}")
        stray = {n for _, n in io_re.findall(composed)} - interface
        if stray:
            problems.append(f"{case_id}: the candidate changed the graded interface: "
                            f"{sorted(stray)[:4]}")
        # A relation that is both an input and an output must still be declared
        # once; a second .decl makes Souffle reject every candidate on that case.
        # Cheap to check and it does not need the compiler.
        seen, twice = set(), set()
        for name in decl_re.findall(composed):
            (twice if name in seen else seen).add(name)
        if twice:
            problems.append(f"{case_id}: the composed program declares a relation "
                            f"more than once: {sorted(twice)[:4]}")
    return problems, checked, ""


def check_reference_survives_composition(limit_cases=None):
    """Send each reference program's own rules through the candidate path.

    The composition probe above is synthetic and compiler-free, so it catches
    structural damage but not a program that is merely rejected. This asks the
    end-to-end question instead: if the reference program itself were submitted
    as a candidate, would it still be graded exact?

    That has to hold by construction -- the reference defines the expected
    output -- so any failure is the harness damaging a correct program rather
    than a fact about the case. `regen_expected` executes the reference
    *directly*, never through the path a candidate takes, so it cannot catch
    this.

    It does NOT replace `composition_invariants` above, and cannot. A composer
    that strips a candidate's own declarations but adds the reference's would
    leave the reference untouched, since the reference uses exactly those names;
    only a candidate with *different* auxiliary names exposes it, which is what
    the synthetic probe is for. The two checks guard opposite failures: this
    one, a harness that breaks a correct program; that one, a harness that leaks
    or strips declarations.

    Returns (problems, checked, skipped_reason).
    """
    try:
        sys.path.insert(0, str(_REPO / "synthesis"))
        sys.path.insert(0, str(_REPO / "evaluation"))
        from synth_llm import compose_full_dl
        from common_eval import (evaluate_query_file_for_case, dataset_index_by_id,
                                 case_variants_dirs)
    except Exception as exc:                      # noqa: BLE001
        return [], 0, f"cannot import the composer or evaluator ({exc})"
    if shutil.which("souffle") is None:
        return [], 0, "souffle not on PATH"

    index = dataset_index_by_id()
    problems, checked = [], 0
    cases = sorted(index)
    if limit_cases:
        cases = [c for c in cases if c in set(limit_cases)]

    with tempfile.TemporaryDirectory() as tmp:
        for case_id in cases:
            golden = QUERY_DIR / f"{case_id}.dl"
            if not golden.is_file() or not case_variants_dirs(case_id):
                continue
            gtext = golden.read_text(encoding="utf-8")
            # Submit the reference verbatim, as a candidate that reproduced it
            # exactly would. The composer does the rest on its own: it drops the
            # interface declarations and every .input/.output, keeps auxiliary
            # declarations and rules, and re-adds the interface block -- the same
            # path a real answer takes, which is the point.
            composed = compose_full_dl(gtext.split("\n\n")[0], gtext, case_id=case_id)
            path = Path(tmp) / f"{case_id}.dl"
            path.write_text(composed, encoding="utf-8")
            checked += 1
            try:
                res = evaluate_query_file_for_case(index[case_id], path)
            except Exception as exc:               # noqa: BLE001
                problems.append(f"{case_id}: evaluation raised {exc}")
                continue
            # Exact match exactly as eval_llm.py scores it: compiles, and no
            # spurious or missing tuple on any scored variant.
            if not res["compile_ok"]:
                first = next((e for e in res.get("errors") or [] if e), "")
                problems.append(f"{case_id}: the reference does not compile through "
                                f"the candidate path: {str(first).strip().splitlines()[:1]}")
            elif res.get("timed_out"):
                problems.append(f"{case_id}: the reference exceeds the evaluation timeout "
                                f"through the candidate path")
            elif res["fp"] or res["fn"]:
                problems.append(f"{case_id}: the reference is not exact through the "
                                f"candidate path (fp={res['fp']}, fn={res['fn']})")
    return problems, checked, ""


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark QA: dead rules in the golden programs, and blind oracle slots in their variants.")
    ap.add_argument("--case", type=str, default=None, help="Check a single case and print its dead rules")
    ap.add_argument("--json", type=str, default=None, help="Write the per-case report to this JSON file")
    ap.add_argument("--fix", nargs="*", metavar="CASE",
                    help="Delete dead rules in the named cases (no names = all offenders) and rewrite the .dl files")
    ap.add_argument("--dry-run", action="store_true", help="With --fix: show what would be removed, write nothing")
    ap.add_argument("--strict-empty", action="store_true",
                    help="Exit 1 on any blind oracle slot, not just those in demo variants")
    ap.add_argument("--strict-duplicates", action="store_true",
                    help="Exit 1 on any equivalent variant pair, not just demo/eval leaks")
    args = ap.parse_args()

    if args.fix is not None:
        targets = args.fix
        if not targets:
            ids = [c["id"] for c in json.load(open(DATASET))]
            targets = [cid for cid in sorted(ids) if check_case(cid)["dead_rules"] > 0]
        changed = [fix_case(cid, dry_run=args.dry_run) for cid in targets]
        changed = [c for c in changed if c]
        print(f"\n[FIX] {len(changed)} case(s) "
              f"{'would be' if args.dry_run else ''} cleaned; "
              f"{sum(c['removed_rules'] for c in changed)} rule(s) removed")
        print("[NEXT] re-run the evaluation on these cases to confirm they still match, "
              "then structure_analyzer / mutation_score for clean statistics")
        return

    records = {c["id"]: c for c in json.load(open(DATASET))}
    if args.case:
        report = [check_case(args.case, records.get(args.case), verbose=True)]
    else:
        report = [check_case(cid, records[cid], verbose=True) for cid in sorted(records)]

    offenders = [r for r in report if r["dead_rules"] > 0]
    total_dead = sum(r["dead_rules"] for r in offenders)
    total_rules = sum(r["total_rules"] for r in report)

    print(f"\n[QA] {len(report)} case(s), {total_rules} rules; "
          f"dead: {total_dead} rule(s) in {len(offenders)} case(s)")
    for r in offenders:
        print(f"  {r['case_id']}: {r['dead_rules']}/{r['total_rules']} dead "
              f"(heads: {', '.join(r['dead_heads'])})")

    blind = [r for r in report if r["blind_slots"]]
    total_slots = sum(r["total_slots"] for r in report)
    total_blind = sum(len(r["blind_slots"]) for r in report)
    demo_blind = sum(r["blind_demo_slots"] for r in report)
    print(f"[QA] {total_slots} oracle slot(s) (variant x output relation); "
          f"blind: {total_blind} in {len(blind)} case(s)")
    for r in blind:
        kinds = ", ".join(f"{s['variant']}/{s['relation']}" for s in r["blind_slots"])
        print(f"  {r['case_id']}: {len(r['blind_slots'])}/{r['total_slots']} blind ({kinds})")
    if total_blind:
        print("  NOTE: an empty golden set is not automatically wrong (the true answer "
              "may be empty); it is *half-blind* -- such a slot can never penalise a "
              "program that produces nothing for that relation.")
    if demo_blind:
        print(f"  ERROR: {demo_blind} blind slot(s) in DEMO variants -- the few-shot "
              "example would demonstrate an empty answer.")

    orphaned = [r for r in report if r["orphan_outputs"]]
    broken = [r for r in report if r["ungraded_missing_export"]]
    print(f"[QA] orphan outputs (exported by the golden, never graded): "
          f"{sum(len(r['orphan_outputs']) for r in orphaned)} in {len(orphaned)} case(s)")
    for r in orphaned:
        print(f"  {r['case_id']}: {', '.join(r['orphan_outputs'])}")
    for r in broken:
        print(f"  ERROR {r['case_id']}: graded but not exported: "
              f"{', '.join(r['ungraded_missing_export'])}")

    tier_path = _REPO / "benchmark" / "spec_tiers.json"
    if tier_path.exists():
        tiers = json.loads(tier_path.read_text(encoding="utf-8")).get("tiers", {})
        bad = sorted(c for c, t in tiers.items() if t not in ("L1", "L2", "L3"))
        stray = sorted(set(tiers) - set(records))
        missing = sorted(set(records) - set(tiers))
        print(f"[QA] spec tiers (side file): {len(tiers)}/{len(report)} labelled"
              + (f"; {len(bad)} not L1/L2/L3" if bad else "")
              + (f"; {len(stray)} name a case that is gone" if stray else "")
              + (f"; {len(missing)} unlabelled" if missing else ""))
        for c in bad + stray:
            print(f"  ERROR spec_tiers.json: {c}")
        tier_bad = bad + stray
    else:
        print("[QA] spec tiers: no benchmark/spec_tiers.json")
        tier_bad = []

    kn_bad = [r for r in report if r.get("knowledge_problems")]
    print(f"[QA] knowledge fields: "
          f"{sum(1 for r in report if (records.get(r['case_id'], {}) or {}).get('knowledge'))} case(s) carry knowledge; "
          f"{len(kn_bad)} with problems")
    for r in kn_bad:
        for msg in r["knowledge_problems"]:
            print(f"  ERROR {r['case_id']}: {msg}")

    schema_bad = [r for r in report if r.get("schema_problems")]
    print(f"[QA] schema agreement (prompt vs golden declarations): "
          f"{len(schema_bad)} case(s) disagree")
    for r in schema_bad:
        for msg in r["schema_problems"]:
            print(f"  ERROR {r['case_id']}: {msg}")

    unread_cases = [r for r in report if r.get("unread_inputs")]
    n_unread = sum(len(r["unread_inputs"]) for r in unread_cases)
    print(f"[QA] inputs advertised but never read by the golden: {n_unread} in "
          f"{len(unread_cases)} case(s)")
    for r in unread_cases:
        print(f"  {r['case_id']}: {', '.join(r['unread_inputs'])}")
    if unread_cases:
        print("  NOTE: reported, not failed -- an unused table can be a deliberate "
              "distractor. Confirm each is intended rather than left over.")

    duped = [r for r in report if r["duplicate_groups"]]
    total_redundant = sum(r["redundant_variants"] for r in report)
    total_variants = sum(r["variants"] for r in report)
    leaks = [r for r in duped if any(g["demo_eval_leak"] for g in r["duplicate_groups"])]
    diverging = [r for r in duped if any(g["expected_diverges"] for g in r["duplicate_groups"])]
    print(f"[QA] {total_variants} variant(s); equivalent inputs: "
          f"{total_redundant} redundant in {len(duped)} case(s)")
    for r in duped:
        detail = "; ".join(" == ".join(g["variants"]) for g in r["duplicate_groups"])
        print(f"  {r['case_id']}: {r['redundant_variants']} redundant ({detail})")
    if total_redundant:
        print("  NOTE: duplicates only cost evaluation time -- they cannot kill a mutant "
              "the other misses. Variant selection collapses them; this is the inventory.")
    for r in leaks:
        print(f"  ERROR {r['case_id']}: a demo variant is equivalent to an eval "
              "variant -- the few-shot prompt would show a scored input")
    for r in diverging:
        print(f"  ERROR {r['case_id']}: equivalent inputs with differing .expected -- "
              "one was not regenerated from the current reference program")

    drift, man_checked, man_skipped = negative_manifest_drift()
    if man_skipped:
        print(f"[QA] negative manifest: skipped ({man_skipped})")
    else:
        print(f"[QA] negative manifest: {man_checked} case(s) checked; {len(drift)} out of sync")
        for msg in drift:
            print(f"  ERROR {msg}")

    sing, sing_checked, sing_skipped = singleton_violations()
    if sing_skipped:
        print(f"[QA] singleton inputs: skipped ({sing_skipped})")
    else:
        print(f"[QA] singleton inputs: {sing_checked} declared parameter file(s) checked; "
              f"{len(sing)} violation(s)")
        for msg in sing:
            print(f"  ERROR {msg}")

    fidelity, fid_checked, fid_skipped = baseline_fidelity()
    if fid_skipped:
        print(f"[QA] baseline survivor fidelity: skipped ({fid_skipped})")
    else:
        print(f"[QA] baseline survivor fidelity: {fid_checked} case(s) with bodyless "
              f"statements in the golden; {len(fidelity)} stored survivor(s) incomplete")
        for msg in fidelity:
            print(f"  ERROR {msg}")

    comp_bad, comp_checked, comp_skipped = composition_invariants()
    if comp_skipped:
        print(f"[QA] composition invariants: skipped ({comp_skipped})")
    else:
        print(f"[QA] composition invariants: {comp_checked} case(s) composed; "
              f"{len(comp_bad)} violation(s)")
        for msg in comp_bad:
            print(f"  ERROR {msg}")

    rt_bad, rt_checked, rt_skipped = check_reference_survives_composition(
        [args.case] if args.case else None)
    if rt_skipped:
        print(f"[QA] reference through the candidate path: skipped ({rt_skipped})")
    else:
        print(f"[QA] reference through the candidate path: {rt_checked} case(s); "
              f"{len(rt_bad)} not graded exact")
        for msg in rt_bad:
            print(f"  ERROR {msg}")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[DONE] report -> {args.json}")

    failed = (bool(offenders) or bool(broken) or bool(kn_bad) or bool(schema_bad) or demo_blind > 0
              or bool(comp_bad) or bool(rt_bad)
              or bool(leaks) or bool(diverging) or bool(fidelity) or bool(drift) or bool(sing)
              or bool(tier_bad)
              or (args.strict_empty and total_blind > 0)
              or (args.strict_duplicates and total_redundant > 0))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
