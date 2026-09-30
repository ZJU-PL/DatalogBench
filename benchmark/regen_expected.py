"""Re-derive every `.expected` file from the reference program.

A `.expected` file is not independent evidence -- it is whatever the golden
produced when the instance was built. So when a golden is corrected, its
`.expected` files are stale by construction and silently keep grading against
the old, wrong answer. This regenerates them.

    python benchmark/regen_expected.py --check                 # verify every case
    python benchmark/regen_expected.py --check --cases Grid
    python benchmark/regen_expected.py --write --cases Grid UnionFind

`--check` never writes: it reports where the committed `.expected` disagrees
with what the golden produces now, which is exactly the set a golden fix
invalidates. Run it before and after a fix -- before, it should be silent.

Graded relations are the golden's own `.output` directives. A relation that is
both `.input` and `.output` (TuringMachine's tape, Josephus's ring) is still
graded, so it is regenerated like any other.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
QUERY = _REPO / "benchmark" / "query"
IO = _REPO / "benchmark" / "io_data"


def graded_relations(dl_text):
    out = []
    for line in dl_text.splitlines():
        s = line.strip()
        if s.startswith(".output"):
            name = s[len(".output"):].strip().split("(")[0].strip()
            if name and name not in out:
                out.append(name)
    return out


def variant_dirs(case):
    root = IO / case
    if not root.is_dir():
        return []
    return sorted((d for d in root.glob("*/*") if d.is_dir()),
                  key=lambda p: (p.parent.name, p.name))


def run_golden(case, facts_dir):
    """{relation: file text} produced by the reference on these facts."""
    prog = QUERY / f"{case}.dl"
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        r = subprocess.run(["souffle", "-F", str(facts_dir), "-D", str(out), str(prog)],
                           capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"{case}: souffle failed on {facts_dir}\n{r.stderr.strip()[:400]}")
        return {p.stem: p.read_text() for p in out.glob("*.csv")}


def main():
    ap = argparse.ArgumentParser(description="Re-derive .expected from the reference programs.")
    ap.add_argument("--cases", nargs="+", default=None, help="default: every case with a golden")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="report disagreements, write nothing")
    g.add_argument("--write", action="store_true", help="overwrite .expected in place")
    args = ap.parse_args()

    cases = args.cases or sorted(p.stem for p in QUERY.glob("*.dl"))
    stale, wrote, errors = [], 0, []

    for case in cases:
        prog = QUERY / f"{case}.dl"
        if not prog.exists():
            errors.append(f"{case}: no golden at {prog}")
            continue
        rels = graded_relations(prog.read_text())
        for vdir in variant_dirs(case):
            try:
                produced = run_golden(case, vdir)
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                errors.append(str(exc).splitlines()[0])
                continue
            for rel in rels:
                target = vdir / f"{rel}.expected"
                fresh = produced.get(rel, "")
                current = target.read_text() if target.exists() else None
                if current == fresh:
                    continue
                where = f"{case}/{vdir.parent.name}/{vdir.name}/{rel}"
                n_old = len(current.splitlines()) if current else 0
                n_new = len(fresh.splitlines())
                stale.append(f"{where}  {n_old} -> {n_new} tuples"
                             + ("  (was missing)" if current is None else ""))
                if args.write:
                    target.write_text(fresh)
                    wrote += 1

    for e in errors:
        print(f"[ERROR] {e}")
    if stale:
        print(f"[STALE] {len(stale)} .expected file(s) disagree with the reference:")
        for s in stale[:60]:
            print(f"  {s}")
        if len(stale) > 60:
            print(f"  ... and {len(stale) - 60} more")
    else:
        print(f"[OK] every .expected across {len(cases)} case(s) matches its reference")
    if args.write:
        print(f"[DONE] rewrote {wrote} file(s)")
    return 1 if (errors or (stale and args.check)) else 0


if __name__ == "__main__":
    sys.exit(main())
