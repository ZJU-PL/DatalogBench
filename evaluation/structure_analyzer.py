"""Datalog structural analyzer (Soufflé dialect).

Extracts, per program, the structural dimensions results are sliced by:
predicate dependency SCCs (recursion), stratification depth,
negation / aggregation usage, predicate invention, arity distribution, and body
size. Runs on the golden programs (benchmark/query/*.dl) and, later, on any
generated .dl.

Two parsing pitfalls are handled deliberately:

1. Variables vs relations are NOT distinguished by case. This benchmark uses
   lowercase variables and Capitalized relations, but nothing here relies on
   that: a *relation reference* is an identifier immediately followed by "(",
   and everything else (bare identifiers = variables/consts) is ignored. Built-in
   functors that also look like `name(` are dropped by intersecting against the
   set of relations that are actually declared or appear as a rule head.

2. Rules are NOT split on ":-". A statement is a top-level "."-terminated unit
   (tracking (), {}, and strings), so head-aggregate rules with no ":-" such as
   `X(sum w : { R(...) }).` are parsed correctly, and the relations referenced
   inside a head aggregate are counted as dependencies.
"""

import argparse
import re
import statistics
from pathlib import Path

from common_eval import BENCHMARK_DIR
from common_eval import QUERY_DIR
from common_eval import save_csv_rows


AGG_KEYWORDS = {"count", "sum", "min", "max", "mean"}
DIRECTIVE_PREFIXES = (".decl", ".input", ".output", ".type", ".pragma",
                      ".number_type", ".symbol_type", ".functor", ".comp",
                      ".init", ".override", ".plan")

_DECL_RE = re.compile(r"^\.decl\s+([A-Za-z_]\w*)\s*\((.*)\)\s*$")
_INPUT_RE = re.compile(r"^\.input\s+([A-Za-z_]\w*)")
_OUTPUT_RE = re.compile(r"^\.output\s+([A-Za-z_]\w*)")
# a relation reference: optional leading '!', a name, then '('
_REF_RE = re.compile(r"(!?)\s*([A-Za-z_]\w*)\s*\(")


def _strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    out = []
    for line in text.splitlines():
        i = line.find("//")
        out.append(line if i < 0 else line[:i])
    return "\n".join(out)


def _split_directives_and_body(text):
    directives, body_lines = [], []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(DIRECTIVE_PREFIXES):
            directives.append(s)
        else:
            body_lines.append(s)
    return directives, " ".join(body_lines)


def _split_statements(body):
    """Split the joined rule text into top-level '.'-terminated statements."""
    stmts, buf = [], []
    depth_paren = depth_brace = 0
    in_str = False
    for ch in body:
        if in_str:
            buf.append(ch)
            if ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace -= 1
        if ch == "." and depth_paren <= 0 and depth_brace <= 0:
            stmt = "".join(buf).strip()
            if stmt:
                stmts.append(stmt)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        stmts.append(tail)
    return stmts


def _arity_of(arg_str):
    arg_str = arg_str.strip()
    if not arg_str:
        return 0
    depth = 0
    n = 1
    for ch in arg_str:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            n += 1
    return n


def _aggregate_spans(stmt):
    """Char spans [start, end] of each aggregate body (`AGG [expr] : body`).

    body is either a `{ ... }` group or a single `Rel(...)` atom. Relations
    inside such a span are aggregate-scoped, which stratifies like negation.
    """
    spans = []
    for m in re.finditer(r"\b(count|sum|min|max|mean)\b", stmt):
        # locate the aggregate colon (depth 0, not part of ':-'), bail at a
        # top-level ',' ';' or ')' -- then this keyword was not an aggregate.
        k, depth, colon = m.end(), 0, -1
        while k < len(stmt):
            ch = stmt[k]
            if ch == "(":
                depth += 1
            elif ch == ")":
                if depth == 0:
                    break
                depth -= 1
            elif depth == 0 and ch in ",;":
                break
            elif depth == 0 and ch == ":" and stmt[k + 1:k + 2] != "-":
                colon = k
                break
            k += 1
        if colon < 0:
            continue
        b = colon + 1
        while b < len(stmt) and stmt[b] == " ":
            b += 1
        if b < len(stmt) and stmt[b] == "{":
            d, e = 0, b
            while e < len(stmt):
                if stmt[e] == "{":
                    d += 1
                elif stmt[e] == "}":
                    d -= 1
                    if d == 0:
                        break
                e += 1
            spans.append((b, e))
        else:
            am = re.compile(r"[A-Za-z_]\w*\s*\(").search(stmt, b)
            if am:
                d, e = 0, am.end() - 1
                while e < len(stmt):
                    if stmt[e] == "(":
                        d += 1
                    elif stmt[e] == ")":
                        d -= 1
                        if d == 0:
                            break
                    e += 1
                spans.append((am.start(), e))
    return spans


def _refs_with_context(stmt, agg_spans):
    """Yield (name, negated, aggregate_scoped, start_index) for each relation ref."""
    for m in _REF_RE.finditer(stmt):
        neg = m.group(1) == "!"
        name = m.group(2)
        idx = m.start(2)
        agg_scoped = any(s <= idx <= e for (s, e) in agg_spans)
        yield name, neg, agg_scoped, idx


def analyze_program(dl_text):
    text = _strip_comments(dl_text)
    directives, body = _split_directives_and_body(text)

    declared, arities, inputs, outputs = {}, {}, set(), set()
    for d in directives:
        m = _DECL_RE.match(d)
        if m:
            name = m.group(1)
            declared[name] = True
            arities[name] = _arity_of(m.group(2))
            continue
        m = _INPUT_RE.match(d)
        if m:
            inputs.add(m.group(1)); continue
        m = _OUTPUT_RE.match(d)
        if m:
            outputs.add(m.group(1))

    statements = _split_statements(body)

    # First pass: head relation of each statement (leading `Ident(`).
    heads = []
    for stmt in statements:
        m = _REF_RE.match(stmt.lstrip("!").lstrip())
        heads.append(m.group(2) if m else None)
    known = set(declared) | {h for h in heads if h}

    # edges: dependency (src -> dst) means dst depends on src; strict = neg/agg.
    pos_edges, strict_edges = set(), set()
    num_rules = 0
    body_atom_counts = []
    num_neg_literals = 0
    uses_negation = uses_aggregation = False

    for stmt, head in zip(statements, heads):
        if head is None:
            continue
        num_rules += 1
        agg_spans = _aggregate_spans(stmt)
        uses_aggregation = uses_aggregation or bool(agg_spans)

        n_body_atoms = 0
        for i, (name, neg, agg_scoped, _idx) in enumerate(_refs_with_context(stmt, agg_spans)):
            if i == 0:
                continue  # the head itself
            if name not in known:
                continue  # functor / builtin
            n_body_atoms += 1
            if neg:
                num_neg_literals += 1
                uses_negation = True
            (strict_edges if (neg or agg_scoped) else pos_edges).add((name, head))
        body_atom_counts.append(n_body_atoms)

    invented = sorted((set(h for h in heads if h) - outputs) - inputs)

    scc_info = _scc_and_strata(known, pos_edges, strict_edges)

    declared_arities = list(arities.values())
    return {
        "num_rules": num_rules,
        "num_relations": len(known),
        "num_input": len(inputs),
        "num_output": len(outputs),
        "num_invented": len(invented),
        # The names themselves, not just the count: comparing a generated
        # program against the golden needs to know *which* intermediate
        # concepts each one introduced (see structure_compare.py).
        "invented": invented,
        "max_arity": max(declared_arities) if declared_arities else 0,
        "mean_arity": round(statistics.mean(declared_arities), 3) if declared_arities else 0.0,
        "avg_body_atoms": round(statistics.mean(body_atom_counts), 3) if body_atom_counts else 0.0,
        "max_body_atoms": max(body_atom_counts) if body_atom_counts else 0,
        "uses_negation": uses_negation,
        "num_neg_literals": num_neg_literals,
        "uses_aggregation": uses_aggregation,
        "recursive": scc_info["recursive"],
        "num_recursive_scc": scc_info["num_recursive_scc"],
        "max_scc_size": scc_info["max_scc_size"],
        "num_strata": scc_info["num_strata"],
        "stratified": scc_info["stratified"],
    }


def _tarjan_scc(nodes, edges):
    """Iterative Tarjan; edges are (src, dst). Returns list of SCC node-sets."""
    adj = {n: [] for n in nodes}
    for u, v in edges:
        if u in adj:
            adj[u].append(v)
    index = {}
    low = {}
    on_stack = set()
    stack = []
    sccs = []
    counter = [0]

    for root in nodes:
        if root in index:
            continue
        work = [(root, 0)]
        while work:
            node, pi = work[-1]
            if pi == 0:
                index[node] = low[node] = counter[0]
                counter[0] += 1
                stack.append(node)
                on_stack.add(node)
            recurse = False
            neighbours = adj[node]
            while pi < len(neighbours):
                w = neighbours[pi]
                if w not in index:
                    work[-1] = (node, pi + 1)
                    work.append((w, 0))
                    recurse = True
                    break
                elif w in on_stack:
                    low[node] = min(low[node], index[w])
                pi += 1
            if recurse:
                continue
            if low[node] == index[node]:
                comp = set()
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    comp.add(w)
                    if w == node:
                        break
                sccs.append(comp)
            work.pop()
            if work:
                parent, _ = work[-1]
                low[parent] = min(low[parent], low[node])
    return sccs


def _scc_and_strata(nodes, pos_edges, strict_edges):
    all_edges = pos_edges | strict_edges
    self_loops = {u for (u, v) in all_edges if u == v}
    sccs = _tarjan_scc(nodes, all_edges)

    comp_of = {}
    for i, comp in enumerate(sccs):
        for n in comp:
            comp_of[n] = i

    recursive_sccs = [c for c in sccs if len(c) > 1 or (next(iter(c)) in self_loops)]
    max_scc_size = max((len(c) for c in sccs), default=0)

    # Stratifiability: a strict (neg/agg) edge inside an SCC breaks stratification.
    stratified = True
    for (u, v) in strict_edges:
        if u in comp_of and v in comp_of and comp_of[u] == comp_of[v]:
            stratified = False

    # Strata: longest path over the SCC-condensed DAG, strict edges weigh +1.
    num_comp = len(sccs)
    dag = {i: [] for i in range(num_comp)}
    indeg = {i: 0 for i in range(num_comp)}
    seen = set()
    for edges, w in ((pos_edges, 0), (strict_edges, 1)):
        for (u, v) in edges:
            if u not in comp_of or v not in comp_of:
                continue
            cu, cv = comp_of[u], comp_of[v]
            if cu == cv:
                continue
            key = (cu, cv, w)
            if key in seen:
                continue
            seen.add(key)
            dag[cu].append((cv, w))
            indeg[cv] += 1

    stratum = {i: 0 for i in range(num_comp)}
    queue = [i for i in range(num_comp) if indeg[i] == 0]
    processed = 0
    while queue:
        u = queue.pop()
        processed += 1
        for (v, w) in dag[u]:
            if stratum[u] + w > stratum[v]:
                stratum[v] = stratum[u] + w
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    num_strata = (max(stratum.values()) + 1) if stratum else 1

    return {
        "recursive": len(recursive_sccs) > 0,
        "num_recursive_scc": len(recursive_sccs),
        "max_scc_size": max_scc_size,
        "num_strata": num_strata if stratified else -1,  # -1 = cannot stratify
        "stratified": stratified,
    }


_SUMMARY_KEYS = ["num_rules", "num_invented", "max_arity", "avg_body_atoms",
                 "num_strata", "max_scc_size"]


def analyze_golden(dataset_scope="all"):
    from common_eval import dataset_records
    records = dataset_records(dataset_scope)
    rows = []
    for rec in records:
        cid = rec["id"]
        path = QUERY_DIR / f"{cid}.dl"
        if not path.exists():
            print(f"[SKIP] missing golden: {cid}")
            continue
        feats = analyze_program(path.read_text(encoding="utf-8"))
        rows.append({"case_id": cid, "category": rec.get("category", ""), **feats})

    rows.sort(key=lambda r: r["case_id"])
    scope = dataset_scope if dataset_scope else "all"
    out_path = BENCHMARK_DIR / "res_data" / f"structure_golden_{scope}.csv"
    save_csv_rows(out_path, list(rows[0].keys()), rows)

    # console summary
    n = len(rows)
    print(f"[STRUCTURE] {n} golden programs")
    print(f"  recursive: {sum(r['recursive'] for r in rows)}  "
          f"uses_negation: {sum(r['uses_negation'] for r in rows)}  "
          f"uses_aggregation: {sum(r['uses_aggregation'] for r in rows)}  "
          f"non-stratifiable: {sum(not r['stratified'] for r in rows)}")
    print(f"  invented>0: {sum(r['num_invented'] > 0 for r in rows)}  "
          f"(>=3: {sum(r['num_invented'] >= 3 for r in rows)}, "
          f">=8: {sum(r['num_invented'] >= 8 for r in rows)})")
    for k in _SUMMARY_KEYS:
        vals = [r[k] for r in rows]
        print(f"  {k:16} mean={statistics.mean(vals):.2f} median={statistics.median(vals):.2f} max={max(vals)}")
    print(f"[DONE] per-case structure -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Datalog structural analyzer (golden or a single .dl).")
    parser.add_argument("--dl", type=str, default=None, help="Analyze a single .dl file and print its features")
    parser.add_argument("--dataset", type=str, default="all", help="Golden scope (category / sub_category / all)")
    args = parser.parse_args()

    if args.dl:
        feats = analyze_program(Path(args.dl).read_text(encoding="utf-8"))
        for k, v in feats.items():
            print(f"{k:20} = {v}")
    else:
        analyze_golden(args.dataset)


if __name__ == "__main__":
    main()
