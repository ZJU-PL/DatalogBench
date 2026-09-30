"""Structure-aware candidate input generator for I/O variant redesign.

The current variants are random siblings of variant 0 and, measured over the
whole library, variants 1-5 add only ~3% marginal mutation kill -- for 117 of the
140 cases measured at the time, nothing at all. (Re-measure: the library has since
lost cases and had goldens corrected.) This generator produces *targeted* candidate
inputs instead: shapes chosen because they expose a specific class of bug
(recursion depth, cycles, self-loops, empty negated relations, numeric
boundaries, duplicate tuples...).

It emits candidates only; the expected output of a candidate is obtained by
running the golden program on it, and its value is measured by how many
currently-surviving mutants it kills (`--score`). A later greedy set-cover step
picks the smallest candidate set that maximises marginal kill.

    # what would be generated for one case
    python3 evaluation/input_generator.py --case SCC --list
    # write candidate fact dirs
    python3 evaluation/input_generator.py --case SCC --out /tmp/cand
    # rank candidates by how many variant-0-surviving mutants they kill
    python3 evaluation/input_generator.py --case SCC --score
"""

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "evaluation"))

from mutation_score import (mutants_delete, mutants_swap, observed,  # noqa: E402
                            observed_keys, run, statements)

QUERY_DIR = _REPO / "benchmark" / "query"
IO_DIR = _REPO / "benchmark" / "io_data"

_DECL_RE = re.compile(r"^\.decl\s+([A-Za-z_]\w*)\s*\((.*)\)\s*$")
_INPUT_RE = re.compile(r"^\.input\s+([A-Za-z_]\w*)")

SYMS = ["a", "b", "c", "d", "e", "f", "g", "h"]
NUMS = [0, 1, 2, 3, -1, 7]


def parse_input_schema(dl_text):
    """[(relation, [arg types])] for every .input relation, declaration order."""
    decls, inputs = {}, []
    for line in dl_text.splitlines():
        s = line.strip()
        m = _DECL_RE.match(s)
        if m:
            args = [a.strip() for a in m.group(2).split(",") if a.strip()]
            decls[m.group(1)] = [(a.split(":")[-1].strip() if ":" in a else "symbol") for a in args]
            continue
        m = _INPUT_RE.match(s)
        if m:
            inputs.append(m.group(1))
    return [(r, decls.get(r, [])) for r in inputs]


_QUOTED_RE = re.compile(r'"([^"]*)"')


def mine_constants(dl_text):
    """Literal constants appearing in the golden's rules.

    Many programs only derive anything for specific domain values -- TwoSAT
    needs "Pos"/"Neg", AccessPolicy needs "Alice"/"Bob". Generic symbols can
    never satisfy those joins, so candidates built from them derive nothing and
    discriminate nothing. Mining the literals out of the rules fixes that.
    """
    _dirs, rules = statements(dl_text)
    body = " ".join(rules)
    syms = []
    for s in _QUOTED_RE.findall(body):
        if s and s not in syms:
            syms.append(s)
    return syms[:6]



# ---------------------------------------------------------------- join domains
#
# Generating each input relation independently is the reason scale variants
# degenerate: a program whose state alphabet lives in Transition and whose tape
# alphabet lives in Tape derives nothing once the two are filled from unrelated
# value pools, and the variant silently tests an empty computation. Worse, the
# resulting facts are semantically absurd -- a river-crossing puzzle whose banks
# are "s_v2_0", a Turing machine in state "v0_1".
#
# The fix is to generate per *join class* rather than per relation. Two argument
# positions that ever share a variable in a rule must range over one domain, so
# union-find over (relation, position) pairs recovers the domains the program
# actually joins on, and every position in a class then draws from one pool.
# The literal values matter far less than the agreement: n/s versus a/b makes no
# difference to what the rules derive, but Opp(a, b) alongside Safe(s, ...) does.

_ATOM_RE = re.compile(r"\b([A-Z]\w*)\s*\(")
_VAR_RE = re.compile(r"[a-z_]\w*\Z")


def _split_args(text):
    """Split an argument list on top-level commas (aggregates nest brackets)."""
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        cur += ch
    if cur.strip():
        out.append(cur)
    return [a.strip() for a in out]


def _atoms(rule, known):
    """(relation, [arg text]) for every atom of a declared relation in `rule`."""
    for m in _ATOM_RE.finditer(rule):
        if m.group(1) not in known:
            continue
        depth, end = 1, None
        rest = rule[m.end():]
        for i, ch in enumerate(rest):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is not None:
            yield m.group(1), _split_args(rest[:end])


def join_classes(dl_text, inputs_only=True):
    """Group input (relation, position) pairs that must range over one domain.

    Classes are the connected components of "shares a variable in some rule",
    computed over *all* relations so that a domain shared only through a derived
    relation is still recovered (TuringMachine's state alphabet reaches
    Configuration only via Transition).

    Returns {(relation, position): class_key}. With inputs_only the map is
    restricted to input relations; the full map is what literal mining needs,
    since a constant may be written on a *derived* relation's position that
    shares the class (AccessPolicy pins "Manager" on ViewEmployee, never on the
    Employee column the generator has to fill).
    """
    arities = {}
    for line in dl_text.splitlines():
        m = _DECL_RE.match(line.strip())
        if m:
            arities[m.group(1)] = len([a for a in m.group(2).split(",") if a.strip()])
    _dirs, rules = statements(dl_text)

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for rule in rules:
        by_var = {}
        for rel, args in _atoms(rule, arities):
            for k, a in enumerate(args):
                if a != "_" and _VAR_RE.match(a):
                    by_var.setdefault(a, []).append((rel, k))
        for slots in by_var.values():
            for other in slots[1:]:
                parent[find(other)] = find(slots[0])

    if not inputs_only:
        return {(rel, k): find((rel, k))
                for rel, n in arities.items() for k in range(n)}
    return {(rel, k): find((rel, k))
            for rel, types in parse_input_schema(dl_text)
            for k in range(len(types))}


def domain_pools(dl_text, mined, size=3):
    """{(relation, position): [values]} -- one shared pool per join class.

    Symbolic pools prefer the literals the golden mentions (a program that only
    fires on "Pos"/"Neg" cannot be exercised by a/b/c), then fall back to the
    generic symbol alphabet. Numeric positions keep numeric pools.
    """
    classes = join_classes(dl_text)
    types = {(rel, k): t
             for rel, ts in parse_input_schema(dl_text) for k, t in enumerate(ts)}
    members = {}
    for slot, key in classes.items():
        members.setdefault(key, []).append(slot)

    literals = class_literals(dl_text, join_classes(dl_text, inputs_only=False))

    pools, fill = {}, 0
    for i, (key, slots) in enumerate(sorted(members.items(), key=lambda kv: str(kv[0]))):
        kinds = {types.get(s, "symbol") for s in slots}
        numeric = kinds <= {"number", "unsigned", "float"}
        if numeric:
            kind = "number" if "number" in kinds else sorted(kinds)[0]
            pool = list(dict.fromkeys(literals.get(key, [])))[:size]
            j = 0
            while len(pool) < size and j < 4 * size:
                cand = _const(kind, i * 2 + j)
                if cand not in pool:
                    pool.append(cand)
                j += 1
        else:
            # Literals seen at *this* class's positions first: they are the only
            # values its joins can match. Generic fill is taken disjointly per
            # class, so two unrelated domains never collapse onto one alphabet.
            pool = list(dict.fromkeys(literals.get(key, [])))[:size]
            guard = 0
            while len(pool) < size and guard < 4 * len(SYMS):
                cand = SYMS[fill % len(SYMS)]
                fill += 1
                guard += 1
                if cand not in pool:
                    pool.append(cand)
        for slot in slots:
            pools[slot] = pool
    return pools


_CMP_RE = re.compile(r"([A-Za-z_]\w*)\s*(<=|>=|!=|<|>|=)\s*(-?\d+)")
_CMP_REV_RE = re.compile(r"(-?\d+)\s*(<=|>=|!=|<|>|=)\s*([A-Za-z_]\w*)")


def class_literals(dl_text, classes):
    """{class_key: [constants]} -- values the golden's rules single out for a class.

    Two sources, both of which random values essentially never hit:

    - constants written as an atom argument (AccessPolicy's "Manager", its 92311
      day, the lowercase "alice" that the self-view rule needs);
    - numeric thresholds a rule compares against. `cnt > 4` makes 4 and 5 the
      only interesting line counts for ASTPrintPython's long-expression branch,
      so the boundary and its two neighbours go into the pool.
    """
    arities = {}
    for line in dl_text.splitlines():
        m = _DECL_RE.match(line.strip())
        if m:
            arities[m.group(1)] = len([a for a in m.group(2).split(",") if a.strip()])
    _dirs, rules = statements(dl_text)
    out = {}
    for rule in rules:
        for rel, args in _atoms(rule, arities):
            for k, a in enumerate(args):
                key = classes.get((rel, k))
                if key is None:
                    continue
                if len(a) >= 2 and a[0] == '"' and a[-1] == '"':
                    out.setdefault(key, []).append(a[1:-1])
                elif re.fullmatch(r"-?\d+", a):
                    # A bare integer written as an argument is a magic value the
                    # rules test for (AccessPolicy's 92311 day); random numbers
                    # never hit it, so the branch stays unexercised.
                    out.setdefault(key, []).append(a)

    # Comparison thresholds, attributed through the variable being compared.
    for rule in rules:
        by_var = {}
        for rel, args in _atoms(rule, arities):
            for k, a in enumerate(args):
                if a != "_" and _VAR_RE.match(a):
                    by_var.setdefault(a, []).append((rel, k))
        pairs = [(m.group(1), int(m.group(3))) for m in _CMP_RE.finditer(rule)]
        pairs += [(m.group(3), int(m.group(1))) for m in _CMP_REV_RE.finditer(rule)]
        for var, bound in pairs:
            for slot in by_var.get(var, []):
                key = classes.get(slot)
                if key is None:
                    continue
                for v in (bound - 1, bound, bound + 1):
                    out.setdefault(key, []).append(str(v))
    return out

def _is_graph_like(types):
    """Binary relation over one homogeneous domain -> treat as an edge set."""
    return len(types) == 2 and types[0] == types[1]


def _const(kind, i):
    """i-th constant of a type; numbers stay numeric, everything else symbolic."""
    if kind == "number":
        return str(NUMS[i % len(NUMS)])
    if kind == "unsigned":
        return str(abs(NUMS[i % len(NUMS)]))
    if kind == "float":
        return f"{NUMS[i % len(NUMS)]}.0"
    return SYMS[i % len(SYMS)]


def _pad(tup, types):
    """Widen an edge pair to the relation's real arity with filler constants."""
    out = list(tup)
    for k in range(len(tup), len(types)):
        out.append(_const(types[k], k))
    return tuple(out[:len(types)])


# Edge shapes, as index pairs over a small vertex domain. Each targets a
# distinct failure mode of the rules that consume the relation.
EDGE_SHAPES = {
    "self_loop":      [(0, 0)],                                    # reflexive derivations
    "two_cycle":      [(0, 1), (1, 0)],                            # symmetric / mutual recursion
    "three_cycle":    [(0, 1), (1, 2), (2, 0)],                    # SCC larger than 2
    "long_chain":     [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],    # recursion depth
    "disconnected":   [(0, 1), (2, 3)],                            # component separation
    "diamond":        [(0, 1), (0, 2), (1, 3), (2, 3)],            # multiple derivation paths
    "duplicate_edge": [(0, 1), (0, 1)],                            # set semantics / dedup
    "asymmetric":     [(0, 1)],                                    # direction sensitivity
    "chain_with_loop": [(0, 1), (1, 2), (2, 2)],                   # loop at the end of a chain
}


def _domain_scenarios(schema, mined):
    """Candidates built from the golden's own literal constants.

    For each mined constant C every relation gets a tuple alternating C with
    generic fillers (the shape that makes rules like `Impl("Pos",u,"Neg",u)`
    fire), and binary relations additionally get all ordered pairs of mined
    constants (for lookup tables such as `Not("Pos","Neg")`).
    """
    out = []
    for c in mined:
        facts = {}
        for rel, ts in schema:
            if not ts:
                facts[rel] = []
                continue
            tup = [c if (t not in ("number", "unsigned", "float") and k % 2 == 0)
                   else _const(t, k) for k, t in enumerate(ts)]
            tuples = [tuple(tup)]
            if len(ts) == 2 and ts[0] == ts[1] and ts[0] not in ("number", "unsigned", "float"):
                tuples += [(a, b) for a in mined for b in mined if a != b]
            facts[rel] = tuples
        out.append((f"domain_{re.sub(r'[^A-Za-z0-9]+', '_', c) or 'const'}", facts))

    if mined:
        # Every relation filled purely with mined constants, cycling.
        facts = {}
        for rel, ts in schema:
            facts[rel] = [tuple(mined[k % len(mined)]
                               if t not in ("number", "unsigned", "float") else _const(t, k)
                               for k, t in enumerate(ts))] if ts else []
        out.append(("domain_all", facts))

        # One row per mined constant, with a *single repeated* filler in the
        # remaining columns. This is what makes self-referential patterns fire:
        # TwoSAT's Incon needs Clause("Pos",x,"Pos",x) and Clause("Neg",x,"Neg",x)
        # present together, which alternating distinct fillers never produces.
        for filler in ("x", "y"):
            facts = {}
            for rel, ts in schema:
                if not ts:
                    facts[rel] = []
                    continue
                rows = []
                for c in mined:
                    rows.append(tuple(
                        _const(t, 0) if t in ("number", "unsigned", "float")
                        else (c if k % 2 == 0 else filler)
                        for k, t in enumerate(ts)))
                if len(ts) == 2 and ts[0] == ts[1] and ts[0] not in ("number", "unsigned", "float"):
                    rows += [(a, b) for a in mined for b in mined if a != b]
                facts[rel] = rows
            out.append((f"domain_repeat_{filler}", facts))
    return out


def scenarios(schema, mined=()):
    """Yield (name, {relation: [tuples]}) candidates for one case's schema."""
    graph_rels = [(r, t) for r, t in schema if _is_graph_like(t)]
    out = []

    # Baselines that apply to every case.
    out.append(("empty_all", {r: [] for r, _ in schema}))
    out.append(("singleton", {r: [tuple(_const(t, i) for i, t in enumerate(ts))]
                              for r, ts in schema}))
    # All columns equal: catches rules that silently assume distinct arguments.
    out.append(("all_equal", {r: [tuple(_const(ts[0], 0) for _ in ts)] if ts else []
                              for r, ts in schema}))

    # Graph shapes: applied to one relation at a time, the others kept minimal
    # so the kill can be attributed to the shape.
    base = {r: [tuple(_const(t, i) for i, t in enumerate(ts))] for r, ts in schema}
    for rel, types in graph_rels:
        for shape, pairs in EDGE_SHAPES.items():
            facts = dict(base)
            facts[rel] = [_pad((_const(types[0], i), _const(types[1], j)), types)
                          for i, j in pairs]
            out.append((f"{rel}__{shape}", facts))

    # Numeric boundaries for any relation carrying a numeric column.
    for rel, types in schema:
        idx = [k for k, t in enumerate(types) if t in ("number", "unsigned", "float")]
        if not idx:
            continue
        for label, val in (("zero", "0"), ("negative", "-1"), ("large", "999")):
            if label == "negative" and any(types[k] == "unsigned" for k in idx):
                continue
            facts = dict(base)
            tup = list(base[rel][0])
            for k in idx:
                tup[k] = val
            facts[rel] = [tuple(tup)]
            out.append((f"{rel}__num_{label}", facts))

    # An empty relation one at a time: the decisive case for negated bodies
    # (`!R(...)` succeeds exactly when R is empty).
    for rel, _ in schema:
        facts = dict(base)
        facts[rel] = []
        out.append((f"{rel}__empty", facts))

    out.extend(_domain_scenarios(schema, list(mined)))

    # Deduplicate by content, keeping the first (most descriptive) name.
    seen, uniq = set(), []
    for name, facts in out:
        key = tuple(sorted((r, tuple(sorted(map(tuple, ts)))) for r, ts in facts.items()))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((name, facts))
    return uniq


def random_scenarios(schema, mined, count=40, seed=0, max_rows=4, pools=None):
    """Random small inputs over the mined + generic constant pools.

    Shape enumeration finds inputs that *exercise* a rule, but some cases only
    discriminate on a narrow semantic boundary -- TwoSAT exports a single
    `Satisfiable("Yes"/"No")` bit, so a mutant is only visible on an input where
    the golden finds exactly the inconsistency the mutant misses. Sampling many
    small inputs is the practical way to hit such boundaries.
    """
    import random as _random
    rng = _random.Random(seed)
    out = []
    for n in range(count):
        facts = {}
        for rel, ts in schema:
            if not ts:
                facts[rel] = []
                continue
            rows = []
            for _ in range(rng.randint(1, max_rows)):
                row = []
                for k, t in enumerate(ts):
                    pool = (pools or {}).get((rel, k))
                    if pool:
                        row.append(rng.choice(pool))
                    elif t in ("number", "unsigned", "float"):
                        row.append(_const(t, rng.randrange(len(NUMS))))
                    elif mined and rng.random() < 0.6:
                        row.append(rng.choice(mined))
                    else:
                        row.append(rng.choice(SYMS[:3]))
                rows.append(tuple(row))
            facts[rel] = sorted(set(rows))
        out.append((f"random_{n:03d}", facts))
    return out


SCALE_LADDER = [10, 25, 50, 100, 200, 400, 800, 1600]


def scale_scenarios(schema, mined, n, pools=None, classes=None):
    """Size-parameterised inputs, for the failure mode small inputs cannot reach.

    Every existing variant is tiny (the library median is 12 input tuples), so a
    program that is wrong only at scale -- a spurious rule that needs a longer
    chain to fire, a join that degenerates into a product as the domain grows,
    a recursion that does not terminate once the graph is big enough to cycle
    through arithmetic -- passes every one of them.

    IMPORTANT: these are *not* an efficiency benchmark. A timeout counts as a
    failure in scoring, so sizing these so that a merely slower-but-correct
    program times out would silently score efficiency as correctness. Sizes are
    therefore calibrated (see calibrate_scale) to leave the reference program a
    large margin under the timeout; a timeout then means genuine pathology, not
    a constant factor.
    """
    graph_rels = [(r, t) for r, t in schema if _is_graph_like(t)]
    pools = pools or {}
    base = {r: [tuple((pools.get((r, i)) or [_const(t, i)])[0]
                      for i, t in enumerate(ts))] for r, ts in schema}
    out = []

    def node(types, i):
        return f"n{i}" if types[0] not in ("number", "unsigned", "float") else str(i)

    for rel, types in graph_rels:
        # A path: recursion depth n, transitive closure O(n^2).
        facts = dict(base)
        facts[rel] = [_pad((node(types, i), node(types, i + 1)), types) for i in range(n)]
        out.append((f"scale_chain_{n}__{rel}", facts))

        # A cycle: every node reaches every node. Catches rules that only look
        # correct while the graph stays acyclic.
        facts = dict(base)
        facts[rel] = [_pad((node(types, i), node(types, (i + 1) % n)), types) for i in range(n)]
        out.append((f"scale_cycle_{n}__{rel}", facts))

        # Denser: two out-edges per node, so joins have real fan-out.
        facts = dict(base)
        edges = [(i, (i + 1) % n) for i in range(n)] + [(i, (i * 2 + 1) % n) for i in range(n)]
        facts[rel] = sorted({_pad((node(types, a), node(types, b)), types) for a, b in edges})
        out.append((f"scale_dense_{n}__{rel}", facts))

    # Applies to every case, graph-like or not: n distinct tuples per relation,
    # which grows the domain each rule joins over.
    # Widening must grow each *join class* as a unit. Filling relations
    # independently was how scale variants ended up asserting a Turing machine
    # in state "v0_1" while its transition table spoke of "a": the join never
    # matched, and the variant tested an empty computation. Values a class's
    # rules single out come first, so magic constants survive the widening.
    classes = classes if classes is not None else {}

    def class_values(widen_closed):
        """n values per class. `widen_closed` decides what happens to a class
        whose values the golden spells out: widening it is right for an open
        domain (more person names) and fatally wrong for a closed one (a board
        has exactly X, O and blank), and nothing in the program distinguishes
        the two. Both flavours are emitted and calibrate_scale keeps whichever
        still derives something."""
        per_class, tag = {}, 0
        for slot, key in sorted(classes.items(), key=lambda kv: (str(kv[1]), kv[0])):
            if key in per_class:
                continue
            seeds = list(dict.fromkeys(pools.get(slot) or []))
            numeric = bool(seeds) and all(c.lstrip("-").isdigit() for c in seeds if c)
            if seeds and not widen_closed:
                per_class[key] = seeds
                tag += 1
                continue
            vals, i = list(seeds), 0
            while len(vals) < n:
                cand = str(i) if numeric else f"w{tag}_{i}"
                if cand not in vals:
                    vals.append(cand)
                i += 1
            per_class[key] = vals[:n]
            tag += 1
        return per_class

    for suffix, widen in (("", True), ("_closed", False)):
        per_class = class_values(widen)
        facts = {}
        for rel, ts in schema:
            if not ts:
                facts[rel] = []
                continue
            rows = []
            for i in range(n):
                row = []
                for k, t in enumerate(ts):
                    vals = per_class.get(classes.get((rel, k)))
                    if vals:
                        # Offset by the column index: two positions of one class
                        # share a *pool*, not a value. Indexing both with i
                        # alone makes every Intersect(a, b) a self-loop and
                        # every Safe(x, x, x, x) a fixed point.
                        row.append(vals[(i + k) % len(vals)])
                    elif t in ("number", "unsigned", "float"):
                        row.append(str(i))
                    else:
                        row.append(f"v{i}_{k}")
                rows.append(tuple(row))
            facts[rel] = sorted(set(rows))
        out.append((f"scale_wide{suffix}_{n}", facts))
    return out


class _ScaleJob:
    """Picklable worker for the parallel library-wide calibration."""

    def __init__(self, budget, max_out, out_root):
        self.budget, self.max_out, self.out_root = budget, max_out, out_root

    def __call__(self, case_id):
        try:
            best = calibrate_scale(case_id, self.budget, self.max_out)
        except Exception:
            best = None
        if not best:
            return (case_id, None, 0.0, 0, 0)
        name, facts, secs, out_t, in_t = best
        if self.out_root:
            write_facts(Path(self.out_root) / case_id / name, facts)
        return (case_id, name, secs, out_t, in_t)


def _timed_run(dl_text, facts_dir, timeout):
    """(seconds, {relation: rows}) or (None, None) on failure/timeout."""
    import subprocess, tempfile, time
    with tempfile.TemporaryDirectory() as td:
        prog = Path(td) / "p.dl"
        prog.write_text(dl_text)
        o = Path(td) / "out"
        o.mkdir()
        t0 = time.time()
        try:
            r = subprocess.run(["souffle", "-F", str(facts_dir), "-D", str(o), str(prog)],
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, None
        if r.returncode != 0:
            return None, None
        return time.time() - t0, {f.name: f.read_text().count("\n") for f in o.glob("*.csv")}


def calibrate_scale(case_id, budget_sec=2.0, max_out=20000, timeout=30):
    """Largest ladder size whose reference run stays well inside the budget.

    Returns (name, facts, seconds, out_tuples) for the biggest candidate that
    keeps the golden under `budget_sec` and its output under `max_out` tuples.
    The budget is deliberately far below the evaluation timeout: the resulting
    variant must be comfortable for any correct program, so that a timeout
    during evaluation is evidence of pathology rather than of slowness. The
    output cap keeps `.expected` files a sane size in the repository.
    """
    dl_text = (QUERY_DIR / f"{case_id}.dl").read_text(errors="ignore")
    schema = parse_input_schema(dl_text)
    mined = mine_constants(dl_text)
    if not schema:
        return None

    # Every graded relation must come out non-empty, or the variant would add a
    # blind slot: an input that derives nothing exercises nothing and cannot
    # penalise a program that produces nothing.
    graded = [k[:-len(".csv")] for k in (observed_keys(case_id) or [])]
    pools = domain_pools(dl_text, mined)
    classes = join_classes(dl_text)

    best = None
    import tempfile
    for n in SCALE_LADDER:
        improved = False
        for name, facts in scale_scenarios(schema, mined, n, pools=pools, classes=classes):
            # Scaling a parameter relation changes the task rather than the size.
            facts = clamp_singletons(facts, case_id)
            with tempfile.TemporaryDirectory(prefix=f"scale_{case_id}_") as tmp:
                d = Path(tmp) / name
                write_facts(d, facts)
                secs, out = _timed_run(dl_text, d, timeout)
            if secs is None or secs > budget_sec:
                continue
            total = sum(out.values())
            if total > max_out or total == 0:
                continue
            if graded and any(out.get(f"{r}.csv", 0) == 0 for r in graded):
                continue                       # would introduce a blind slot
            # Rank by derived tuples, not by input size: the point is how much
            # work the reference program is made to do. A huge input that
            # derives nothing (a long chain fed to a clique finder) is not a
            # stress test.
            cand = (name, facts, secs, total, sum(len(v) for v in facts.values()))
            if best is None or cand[3] > best[3]:
                best, improved = cand, True
        if not improved and best is not None:
            break            # ladder has stopped paying; stay at the last size
    return best


def singleton_inputs(case_id):
    """Input relations of `case_id` that must stay at exactly one tuple.

    A limit, a default, a dimension, a designated start node: scaling these
    does not make a bigger instance, it changes the task. Array's Default
    multiplied the neighbourhood padding by 1600; Factorial's Lim collapsed to
    min()=0 so the scaled variant derived a single tuple; MinPathSrc became a
    1600-source shortest-path problem. Declared by hand in
    benchmark/singleton_inputs.json -- which relations are parameters is a
    semantic judgement no scan decides reliably.
    """
    path = Path(__file__).resolve().parent.parent / "benchmark" / "singleton_inputs.json"
    if not path.exists():
        return set()
    import json as _json
    return set(_json.loads(path.read_text(encoding="utf-8"))
               .get("singletons", {}).get(case_id, []))


def clamp_singletons(facts, case_id):
    """Trim declared singleton relations back to their first tuple."""
    keep = singleton_inputs(case_id)
    if not keep:
        return facts
    return {r: (t[:1] if r in keep and len(t) > 1 else t) for r, t in facts.items()}


def write_facts(target_dir, facts):
    target_dir.mkdir(parents=True, exist_ok=True)
    for rel, tuples in facts.items():
        lines = ["\t".join(str(x) for x in t) for t in tuples]
        (target_dir / f"{rel}.facts").write_text(
            ("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def build_mutants(dl_text, max_mutants=20):
    dirs, rules = statements(dl_text)
    muts = mutants_delete(dirs, rules) + mutants_swap(dirs, rules)
    return dirs, rules, muts[:max_mutants]


def score_case(case_id, max_mutants=20, top=None, random_count=0):
    """Rank candidates by how many mutants surviving the current variants they kill."""
    dl_text = (QUERY_DIR / f"{case_id}.dl").read_text(errors="ignore")
    schema = parse_input_schema(dl_text)
    mined = mine_constants(dl_text)
    _dirs, _rules, muts = build_mutants(dl_text, max_mutants)
    if not muts:
        print(f"[{case_id}] no mutants to target")
        return

    # Which mutants survive the existing variants?
    from common_eval import case_variants_dirs
    existing = case_variants_dirs(case_id)
    # Judge discrimination on the relations the evaluator grades, not on
    # everything the program exports -- an "extra kill" on an ungraded relation
    # would send us hunting for inputs that cannot change any score.
    keys = observed_keys(case_id)
    survivors = []
    for mt in muts:
        killed = False
        for d in existing:
            g, m = observed(run(dl_text, d), keys), observed(run(mt, d), keys)
            if g is None:
                continue
            if m is None or m != g:
                killed = True
                break
        if not killed:
            survivors.append(mt)
    print(f"[{case_id}] {len(muts)} mutants, {len(survivors)} survive the current {len(existing)} variants")
    if not survivors:
        print("  nothing left to target -- current variants already discriminate every mutant")
        return

    cands = scenarios(schema, mined)
    if random_count:
        cands += random_scenarios(schema, mined, count=random_count,
                                  pools=domain_pools(dl_text, mined))
    cands = [(n, clamp_singletons(f, case_id)) for n, f in cands]
    rows = []
    with tempfile.TemporaryDirectory(prefix=f"cand_{case_id}_") as tmp:
        for name, facts in cands:
            d = Path(tmp) / name
            write_facts(d, facts)
            golden_out = observed(run(dl_text, d), keys)
            if golden_out is None:
                continue                      # candidate breaks the golden: unusable
            kills = sum(1 for mt in survivors
                        if (lambda m: m is None or m != golden_out)(observed(run(mt, d), keys)))
            derived = sum(len(v) for v in golden_out.values())
            rows.append((name, kills, derived))

    rows.sort(key=lambda r: (-r[1], r[2]))
    shown = rows if top is None else rows[:top]
    print(f"  {'candidate':34}{'kills':>7}{'derived':>9}")
    for name, kills, derived in shown:
        if kills == 0 and top is not None:
            break
        print(f"  {name:34}{kills:>7}{derived:>9}")
    best = rows[0][1] if rows else 0
    print(f"  best single candidate kills {best}/{len(survivors)} survivor(s)")


def main():
    ap = argparse.ArgumentParser(description="Structure-aware candidate input generator.")
    ap.add_argument("--case", default=None)
    ap.add_argument("--all", action="store_true", help="With --scale: calibrate every case")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", type=str, default=None, help="Write candidate fact dirs here")
    ap.add_argument("--list", action="store_true", help="List candidates without writing")
    ap.add_argument("--score", action="store_true", help="Rank candidates by mutants killed")
    ap.add_argument("--max-mutants", type=int, default=20)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--random", type=int, default=0,
                    help="Also sample N random small inputs (for cases whose output is a coarse aggregate)")
    ap.add_argument("--scale", action="store_true",
                    help="Calibrate the largest scale-stress input the reference program handles comfortably")
    ap.add_argument("--budget", type=float, default=2.0,
                    help="Seconds the reference run may take (kept far below the 30s evaluation timeout)")
    ap.add_argument("--max-out", type=int, default=20000, help="Cap on generated .expected tuples")
    args = ap.parse_args()

    if args.scale and args.all:
        import json as _json
        from concurrent.futures import ProcessPoolExecutor
        ids = [c["id"] for c in _json.load(open(_REPO / "benchmark" / "dataset.json"))]
        fn = _ScaleJob(args.budget, args.max_out, args.out)
        rows = []
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for row in ex.map(fn, ids):
                rows.append(row)
                cid, name, secs, out_t, in_t = row
                print(f"{cid:28}" + (f"{name:30}{in_t:>7} in{out_t:>9} out{secs:>8.2f}s"
                                     if name else "  (no scalable candidate)"), flush=True)
        got = [r for r in rows if r[1]]
        print(f"\n[SCALE] {len(got)}/{len(rows)} case(s) scalable; "
              f"median derived tuples {sorted(r[3] for r in got)[len(got)//2] if got else 0}")
        slow = sorted(got, key=lambda r: -r[2])[:5]
        print("slowest reference runs (margin under the 30s timeout):")
        for cid, name, secs, _o, _i in slow:
            print(f"  {cid:28}{secs:>6.2f}s  {30/max(secs,1e-6):>6.0f}x")
        return

    if not args.case:
        ap.error("give --case (or --scale --all)")

    if args.scale:
        best = calibrate_scale(args.case, args.budget, args.max_out)
        if not best:
            print(f"[{args.case}] no scale candidate fits the budget")
            return
        name, facts, secs, out_tuples, in_tuples = best
        print(f"[{args.case}] {name}: {in_tuples} input tuples -> {out_tuples} output tuples, "
              f"golden {secs:.2f}s ({30 / max(secs, 1e-6):.0f}x margin under the 30s timeout)")
        if args.out:
            d = Path(args.out) / args.case / name
            write_facts(d, facts)
            print(f"[DONE] {d}")
        return

    dl_text = (QUERY_DIR / f"{args.case}.dl").read_text(errors="ignore")
    schema = parse_input_schema(dl_text)

    if args.score:
        score_case(args.case, args.max_mutants, args.top, args.random)
        return

    mined = mine_constants(dl_text)
    cands = [(n, clamp_singletons(f, args.case)) for n, f in scenarios(schema, mined)]
    if mined:
        print(f"[{args.case}] mined constants: {mined}")
    print(f"[{args.case}] inputs: " + ", ".join(f"{r}({','.join(t)})" for r, t in schema))
    print(f"[{args.case}] {len(cands)} candidate input set(s)")
    for name, facts in cands:
        size = sum(len(v) for v in facts.values())
        print(f"  {name:34} {size} tuple(s)")
        if args.out:
            write_facts(Path(args.out) / args.case / name, facts)
    if args.out:
        print(f"[DONE] candidates -> {Path(args.out) / args.case}")


if __name__ == "__main__":
    main()
