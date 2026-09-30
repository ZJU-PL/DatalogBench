"""Pre-screen surviving mutants into equivalent / dead_rule / real_gap.

A mutation score is only meaningful once *equivalent mutants* -- mutants that no
input can ever kill -- are removed: leaving them in understates oracle strength
and wastes effort chasing unreachable targets. This script does the mechanical
part of that triage so only genuinely ambiguous survivors need a human.

Categories
    dead_rule   the mutated rule's head cannot reach any .output relation, so
                the mutation is unobservable by construction (reuses the
                liveness analysis in benchmark/qa_check.py)
    equivalent  the mutant program is provably the same as the golden one:
                  alpha_identical  -- same rule set up to variable renaming
                                      (covers pure renamings and symmetric
                                      rules such as H(a,b) :- B(a,b), B(b,a))
                  subsumed_rule    -- the deleted rule was theta-subsumed by a
                                      rule that remains, so it derived nothing new
    real_gap    none of the above -- *not provably equivalent by these checks*

The screen is **sound but incomplete**: an `equivalent` verdict is justified by
an explicit witness (a renaming, or a subsuming rule), but `real_gap` is only an
*upper bound* on genuine oracle gaps. In particular a rule that is redundant
because it is derivable in several steps (e.g. a one-step reachability rule
implied by a base rule plus transitivity) is not caught by single-rule
theta-subsumption and lands in `real_gap`. Treat `real_gap` as the manual-review
queue, not as a count of confirmed gaps.

Usage:
    python3 evaluation/triage_survivors.py                       # whole baseline
    python3 evaluation/triage_survivors.py --case Escape -v      # one case, verbose
    python3 evaluation/triage_survivors.py --out benchmark/qa/survivor_triage_frozen.json
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "evaluation"))
sys.path.insert(0, str(_REPO / "benchmark"))

# Parse the golden exactly the way the mutants were produced, otherwise the
# rule-by-rule diff will not line up.
from mutation_score import statements  # noqa: E402
# Liveness comes from the structure_analyzer-based parser (handles head
# aggregates and identifies variables positionally, not by case).
from qa_check import live_relations, parse_program  # noqa: E402

QUERY_DIR = _REPO / "benchmark" / "query"
# Default to the current mutation run, not the historical pre-cleanup baseline:
# triaging survivors against a stale run silently reports gaps for cases that
# have since been repaired (Escape showed 19 phantom survivors this way).
DEFAULT_BASELINE = _REPO / "benchmark" / "qa" / "mutation_frozen.json"
DEFAULT_OUT = _REPO / "benchmark" / "qa" / "survivor_triage_frozen.json"

_ATOM_RE = re.compile(r"(!?)(\w+)\(([^()]*)\)")
_VAR_RE = re.compile(r"[A-Za-z_]\w*")


def _balanced_args(text, open_at):
    """Argument text between the parenthesis at `open_at` and its partner, or None."""
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_at + 1:i]
    return None


def _split_top_level(text):
    """Split on commas that are not inside parentheses."""
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        cur += ch
    if cur.strip():
        out.append(cur)
    return tuple(a.strip() for a in out)


def _head_atom(head_txt):
    """(neg, relation, args) of a rule head, tolerating functors in the arguments.

    `_ATOM_RE` matches `Name(...)` with a bracket-free argument list, so a head
    like `Permutation(cat(x, cat(",", y)))` does not match at `Permutation` and
    `search` settles on the inner `cat(...)` instead. The relation then reads as
    `cat`, which reaches no output, and the rule is classified `dead_rule` --
    for the two goldens that build a head with a functor (Permutations uses
    `cat`, MinSpanTree uses `autoinc`) that verdict was pure parser artifact.
    """
    m = re.search(r"(!?)(\w+)\s*\(", head_txt)
    if not m:
        return None
    args = _balanced_args(head_txt, head_txt.index("(", m.start(2)))
    if args is None:
        return None
    return (m.group(1), m.group(2), _split_top_level(args) if args.strip() else ())


def parse_rule(text):
    """(head, body_atoms, constraints) with atoms as (neg, rel, args tuple).

    Anything in the body that is not a relational atom (comparisons, arithmetic)
    is kept as an opaque constraint string and compared after renaming.
    """
    text = text.strip().rstrip(".")
    if ":-" not in text:
        return None
    head_txt, body_txt = text.split(":-", 1)
    head = _head_atom(head_txt)
    if head is None:
        return None

    body = []
    leftover = body_txt
    for m in _ATOM_RE.finditer(body_txt):
        args = tuple(a.strip() for a in m.group(3).split(",")) if m.group(3).strip() else ()
        body.append((m.group(1), m.group(2), args))
        leftover = leftover.replace(m.group(0), " ", 1)
    constraints = [c.strip() for c in re.split(r"[,;]", leftover) if c.strip()]
    return head, body, constraints


def _is_var(tok):
    return bool(_VAR_RE.fullmatch(tok)) and tok != "_"


def _extend(mapping, a_args, b_args):
    """Extend a variable mapping so a_args -> b_args; None if inconsistent."""
    if len(a_args) != len(b_args):
        return None
    out = dict(mapping)
    for x, y in zip(a_args, b_args):
        if x == "_" or y == "_":
            if x != y:
                return None
            continue
        if _is_var(x):
            if not _is_var(y):
                return None
            if out.get(x, y) != y:
                return None
            out[x] = y
        else:                      # constant must match literally
            if x != y:
                return None
    return out


def _match_atoms(a_atoms, b_atoms, mapping, used, injective=True):
    """Yield every multiset match of a_atoms onto b_atoms under `mapping`.

    All matches are enumerated, not just the first: a mapping that satisfies the
    atoms may still fail on the constraints, in which case another one may work.
    """
    if not a_atoms:
        yield mapping
        return
    (neg, rel, args), rest = a_atoms[0], a_atoms[1:]
    for i, (bneg, brel, bargs) in enumerate(b_atoms):
        if i in used or bneg != neg or brel != rel:
            continue
        ext = _extend(mapping, args, bargs)
        if ext is None:
            continue
        if injective and len(set(ext.values())) != len(ext):
            continue
        yield from _match_atoms(rest, b_atoms, ext, used | {i}, injective)


_SYMMETRIC_OPS = {"=", "!=", "<>"}
_FLIP_OPS = {">": "<", ">=": "<=", "<": ">", "<=": ">="}
_CMP_RE = re.compile(r"^(.*?)\s*(!=|<>|<=|>=|=|<|>)\s*(.*)$")


def normalize_constraint(text):
    """Canonicalize a comparison so that equivalent orderings compare equal.

    `a != b` and `b != a` are the same constraint; `a > b` is `b < a`. Without
    this, a body-variable swap on a symmetric comparison looks like a change.
    """
    text = " ".join(text.split())
    m = _CMP_RE.match(text)
    if not m:
        return text
    lhs, op, rhs = m.group(1).strip(), m.group(2), m.group(3).strip()
    if op in _SYMMETRIC_OPS:
        lhs, rhs = sorted([lhs, rhs])
        op = "!=" if op in {"!=", "<>"} else "="
    elif op in (">", ">="):
        lhs, rhs, op = rhs, lhs, _FLIP_OPS[op]
    return f"{lhs} {op} {rhs}"


def alpha_equivalent(r1, r2):
    """True if the two rules are identical up to a bijective variable renaming."""
    p1, p2 = parse_rule(r1), parse_rule(r2)
    if p1 is None or p2 is None:
        return False
    (h1, b1, c1), (h2, b2, c2) = p1, p2
    if h1[1] != h2[1] or len(b1) != len(b2) or len(c1) != len(c2):
        return False
    base = _extend({}, h1[2], h2[2])
    if base is None:
        return False
    target = sorted(normalize_constraint(c) for c in c2)
    for mapping in _match_atoms(b1, b2, base, frozenset(), injective=True):
        renamed = sorted(
            normalize_constraint(_VAR_RE.sub(lambda m: mapping.get(m.group(0), m.group(0)), c))
            for c in c1
        )
        if renamed == target:
            return True
    return False


def subsumes(general, specific):
    """theta-subsumption: `general` derives everything `specific` does.

    Same head (under a mapping) and body atoms a sub-multiset of the other's,
    so deleting `specific` cannot change the fixpoint.
    """
    pg, ps = parse_rule(general), parse_rule(specific)
    if pg is None or ps is None:
        return False
    (hg, bg, cg), (hs, bs, cs) = pg, ps
    if hg[1] != hs[1] or len(bg) > len(bs) or cg:
        return False  # constraints on the general rule could restrict it
    base = _extend({}, hg[2], hs[2])
    if base is None:
        return False
    return any(True for _ in _match_atoms(bg, bs, base, frozenset(), injective=False))


def diff_rules(golden_rules, mutant_rules):
    """Greedily pair rules by alpha-equivalence; return the unmatched leftovers."""
    remaining = list(range(len(golden_rules)))
    unmatched_mut = []
    for mr in mutant_rules:
        hit = None
        for idx in remaining:
            if alpha_equivalent(golden_rules[idx], mr):
                hit = idx
                break
        if hit is None:
            unmatched_mut.append(mr)
        else:
            remaining.remove(hit)
    return [golden_rules[i] for i in remaining], unmatched_mut


# Deliberately outside benchmark/qa/, which is ignored: everything else in that
# directory is a regenerable analysis artifact, whereas these are human
# judgements that cannot be recomputed and must travel with the repository.
REVIEWED = Path(__file__).resolve().parent.parent / "benchmark" / "reviewed_survivors.json"


def _reviewed():
    """Human verdicts recorded from earlier review passes.

    The automatic checks are sound but incomplete -- in particular they test
    single-step subsumption, so a rule implied only through a chain of others
    lands in `real_gap`. Once a person has settled such a case the answer has to
    survive the next run, or every re-triage puts the same entries back on the
    worklist and the review never converges.

    Keyed "<case>#<mutant index>". Indices are stable for a fixed golden program
    and mutant budget; a change to either invalidates them, which is why each
    entry records the rule text it was judged on.
    """
    if not REVIEWED.exists():
        return {}
    import json as _json
    return _json.loads(REVIEWED.read_text(encoding="utf-8")).get("verdicts", {})


def symmetric_positions(rules):
    """{relation: (i, j)} for relations the program explicitly symmetrises.

    A rule whose single body atom is its own head with exactly two argument
    positions exchanged closes the relation under that transposition -- Datalog
    spelling of "R is symmetric". Every other argument must ride through
    unchanged, which is why MinPathGen's Edge(a, b, w) :- Edge(b, a, w) counts
    while a rule that also permuted the weight would not.
    """
    out = {}
    for r in rules:
        parsed = parse_rule(r)
        if not parsed:
            continue
        (_hneg, hrel, hargs), body, cons = parsed
        if cons or len(body) != 1:
            continue
        bneg, brel, bargs = body[0]
        if bneg or brel != hrel or len(bargs) != len(hargs):
            continue
        if not all(_is_var(a) for a in hargs) or len(set(hargs)) != len(hargs):
            continue
        diff = [k for k in range(len(hargs)) if hargs[k] != bargs[k]]
        if len(diff) == 2:
            i, j = diff
            if hargs[i] == bargs[j] and hargs[j] == bargs[i]:
                out[hrel] = (i, j)
    return out


def _swapped(args, pos):
    i, j = pos
    out = list(args)
    out[i], out[j] = out[j], out[i]
    return tuple(out)


def _render_rule(head, body, cons):
    def atom(neg, rel, args):
        return f"{neg}{rel}({', '.join(args)})"
    parts = [atom(*a) for a in body] + list(cons)
    return f"{atom(*head)} :- {', '.join(parts)}."


def symmetry_variants(rule, symrels, cap=6):
    """Rules that denote the same thing as `rule`, given which relations are symmetric.

    Two independent reasons a swap can be harmless:

    - an atom of a symmetric relation reads the same either way, so its two
      arguments can be exchanged in place;
    - if the *head* is symmetric, exchanging its two arguments makes the rule
      derive the transpose of what it derived before, and the transpose closes
      back to the same relation.
    """
    parsed = parse_rule(rule)
    if not parsed:
        return []
    head, body, cons = parsed
    slots = ([("head", None)] if head[1] in symrels else [])
    slots += [("body", k) for k, (_n, rel, _a) in enumerate(body) if rel in symrels]
    if not slots or len(slots) > cap:
        return []
    out = []
    for mask in range(1, 1 << len(slots)):
        h, b = head, list(body)
        for bit, (kind, k) in enumerate(slots):
            if not (mask >> bit) & 1:
                continue
            if kind == "head":
                h = (head[0], head[1], _swapped(head[2], symrels[head[1]]))
            else:
                neg, rel, args = b[k]
                b[k] = (neg, rel, _swapped(args, symrels[rel]))
        out.append(_render_rule(h, b, cons))
    return out


_EQ_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*$")


def equality_normalized(rule):
    """Rewrite variables joined by an `X = Y` body constraint to one representative.

    Such variables denote the same value in every derivation, so which of them
    the head carries is a matter of spelling. GetMybatisDOClass writes
    `DoClass(class_name) :- JavaClass(_, class_name), DbElementType(_, type_name),
    class_name = type_name` -- reading the head off the other side of the
    equality is the same program, and only looks like a join mis-binding.
    """
    parsed = parse_rule(rule)
    if not parsed:
        return rule
    head, body, cons = parsed
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    linked = False
    for c in cons:
        m = _EQ_RE.match(c)
        if m and _is_var(m.group(1)) and _is_var(m.group(2)):
            a, b = find(m.group(1)), find(m.group(2))
            if a != b:
                lo, hi = sorted((a, b))
                parent[hi] = lo
                linked = True
    if not linked:
        return rule

    def rw(args):
        return tuple(find(a) if _is_var(a) else a for a in args)

    head = (head[0], head[1], rw(head[2]))
    body = [(n, r, rw(a)) for n, r, a in body]
    kept = []
    for c in cons:
        m = _EQ_RE.match(c)
        if m and _is_var(m.group(1)) and _is_var(m.group(2)) and find(m.group(1)) == find(m.group(2)):
            continue                      # the equality is now X = X
        kept.append(_VAR_RE.sub(lambda mm: find(mm.group(0)), c))
    return _render_rule(head, body, kept)


def _sym_swap(removed, added, golden_rules, mutant_rules):
    """True if `added` is `removed` with arguments swapped on symmetric relations.

    The symmetry must hold in the mutant as well: a mutation that damaged the
    transposition rule itself leaves the relation no longer symmetric, and the
    swap would then be observable.
    """
    sym_g = symmetric_positions(golden_rules)
    if not sym_g:
        return False
    sym_m = symmetric_positions(mutant_rules)
    symrels = {r: p for r, p in sym_g.items() if sym_m.get(r) == p}
    if not symrels:
        return False
    return any(alpha_equivalent(v, added) for v in symmetry_variants(removed, symrels))


def triage_case(case_id, survivors, verbose=False):
    golden_text = (QUERY_DIR / f"{case_id}.dl").read_text(errors="ignore")
    _dirs, all_rules = statements(golden_text)
    golden_rules = [r for r in all_rules if ":-" in r]
    golden_facts = [r for r in all_rules if ":-" not in r]

    _inp, outputs, _known, parsed = parse_program(golden_text)
    live = live_relations(outputs, parsed)

    results = []
    for k, stored in enumerate(survivors):
        # Rules and ground facts are diffed separately: alpha-equivalence is
        # defined on rules, and a fact carries no body to rename. Diffing only
        # the rules would make a deleted base case look like an untouched
        # program -- "alpha_identical", i.e. equivalent -- and silently drop a
        # genuine gap from the queue.
        mutant_rules = [r for r in stored if ":-" in r]
        removed, added = diff_rules(golden_rules, mutant_rules)

        kept_facts = [r.strip() for r in stored if ":-" not in r]
        removed_facts = [f for f in golden_facts if f.strip() not in kept_facts]

        # A single mutation removes one statement, or replaces one with its
        # weakened form. A diff wider than that means the stored survivor was
        # built from a different golden -- classifying it would describe a
        # program that no longer exists.
        if len(removed) + len(removed_facts) > 1 or len(added) > 1:
            results.append({
                "index": k, "verdict": "stale",
                "reason": (f"STALE: diff is {len(removed) + len(removed_facts)} removed / "
                           f"{len(added)} added, but one mutation changes at most one "
                           "statement; re-run the baseline for this case"),
                "removed": removed + removed_facts, "added": added, "source": "auto",
            })
            if verbose:
                print(f"  [{k}] stale      (baseline predates the current golden)")
            continue

        if removed_facts:
            # Deleting a ground fact strictly weakens the program. It is an
            # oracle gap unless the fact cannot reach an output at all.
            heads = {parse_rule(f + " :- true.")[0][1] for f in removed_facts
                     if parse_rule(f + " :- true.")}
            if heads and all(h not in live for h in heads):
                verdict, reason = "dead_rule", "fact_cannot_reach_output"
            else:
                verdict, reason = ("real_gap",
                                   "base case unchecked: no variant distinguishes the "
                                   "program that omits this fact")
            removed = removed + removed_facts
        elif not removed and not added:
            verdict, reason = "equivalent", "alpha_identical"
        else:
            heads = {parse_rule(r)[0][1] for r in removed + added if parse_rule(r)}
            if heads and all(h not in live for h in heads):
                verdict, reason = "dead_rule", "head_cannot_reach_output"
            elif not added and removed and all(
                any(subsumes(other, r) for other in mutant_rules) for r in removed
            ):
                verdict, reason = "equivalent", "subsumed_rule"
            elif len(removed) == 1 and len(added) == 1 and _sym_swap(
                removed[0], added[0], golden_rules, mutant_rules
            ):
                verdict, reason = "equivalent", "symmetric_relation_swap"
            elif len(removed) == 1 and len(added) == 1 and alpha_equivalent(
                equality_normalized(removed[0]), equality_normalized(added[0])
            ):
                verdict, reason = "equivalent", "equality_constrained_swap"
            else:
                verdict, reason = "real_gap", "semantic_difference_unobserved"

        # A recorded human verdict overrides the automatic one, and must be
        # applied before the row is built -- the checks are incomplete, so the
        # whole point is that a settled case stays settled across re-runs.
        # Match on the recorded rule text, not on the index: adding an input
        # kills some survivors and renumbers the rest, so an index-keyed lookup
        # would silently drop a settled verdict. The "#k" in the key is only for
        # readability. An entry for this case whose rule matches nothing in the
        # current survivor set is reported as STALE by check_reviewed_entries().
        human = None
        if removed:
            target = removed[0].strip()
            for key, entry in _reviewed().items():
                if key.split("#", 1)[0] == case_id and (entry.get("rule") or "").strip() == target:
                    human = entry
                    break
        if human:
            verdict, reason = human["verdict"], f"reviewed: {human['reason']}"

        results.append({
            "index": k, "verdict": verdict, "reason": reason,
            "removed": removed, "added": added,
            "source": "human" if human else "auto",
        })
        if verbose:
            print(f"  [{k}] {verdict:10} ({reason})")
            for r in removed:
                print(f"       - {r}")
            for r in added:
                print(f"       + {r}")
    return results


def check_reviewed_entries():
    """Report recorded verdicts whose rule no longer exists in the golden program.

    Verdicts are matched by rule text rather than by survivor index, so an index
    shift is harmless -- but an edit to a golden program can invalidate a human
    judgement silently. This sweep is the guard: every recorded rule must still
    be present (up to variable renaming) in the case it was judged against.
    """
    stale = []
    for key, entry in _reviewed().items():
        case_id = key.split("#", 1)[0]
        recorded = (entry.get("rule") or "").strip()
        if not recorded:
            continue
        path = QUERY_DIR / f"{case_id}.dl"
        if not path.exists():
            stale.append((key, "case file missing"))
            continue
        _dirs, all_rules = statements(path.read_text(encoding="utf-8"))
        golden_rules = [r for r in all_rules if ":-" in r]
        if not any(alpha_equivalent(g, recorded) for g in golden_rules):
            stale.append((key, "rule no longer in the golden program"))
    return stale


def main():
    ap = argparse.ArgumentParser(description="Pre-screen surviving mutants (equivalent / dead_rule / real_gap).")
    ap.add_argument("--baseline", type=str, default=str(DEFAULT_BASELINE))
    ap.add_argument("--case", type=str, default=None, help="Triage a single case")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    baseline = json.load(open(args.baseline))
    cases = [c for c in baseline if c.get("survivors")]
    if args.case:
        cases = [c for c in cases if c["case"] == args.case]
        if not cases:
            print(f"[INFO] {args.case} has no surviving mutants.")
            return

    report, tally = {}, Counter()
    for c in cases:
        if args.verbose:
            print(f"-- {c['case']} ({len(c['survivors'])} survivors)")
        res = triage_case(c["case"], c["survivors"], verbose=args.verbose)
        report[c["case"]] = res
        tally.update(r["verdict"] for r in res)

    total = sum(tally.values())
    print(f"\n[TRIAGE] {total} survivors across {len(report)} case(s)")
    for verdict in ("equivalent", "dead_rule", "real_gap"):
        n = tally.get(verdict, 0)
        print(f"  {verdict:12} {n:4d}  ({n / total:.1%})" if total else f"  {verdict}: 0")
    remaining = tally.get("real_gap", 0)
    print(f"  -> {remaining} survivor(s) queued for manual review (upper bound on real gaps:")
    print("     multi-step-derivable redundant rules are not detected by these checks)")

    by_case = sorted(((cid, sum(1 for r in rs if r["verdict"] == "real_gap"))
                      for cid, rs in report.items()), key=lambda x: -x[1])
    top = [f"{cid}({n})" for cid, n in by_case if n][:8]
    if top:
        print(f"  top real_gap cases: {', '.join(top)}")

    stale = check_reviewed_entries()
    if stale:
        print(f"\n[STALE] {len(stale)} recorded verdict(s) no longer match the golden program:")
        for key, why in stale:
            print(f"  {key}: {why}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[DONE] triage -> {args.out}")


if __name__ == "__main__":
    main()
