"""Synthesize Datalog programs with symbolic (non-LLM) synthesizers as baselines.

Two backends, both introduced as git submodules under baselines/:
- gensynth (https://github.com/jonomendelson/gensynth): genetic search, Python +
  Souffle, runs locally. Stochastic -> repeated runs give real variance.
- egs (https://github.com/aalok-thakkar/egs-artifact): example-guided synthesis.
  The EGS core is a standalone Scala/sbt project -- build the fat jar locally
  with `cd baselines/egs/egs && sbt assembly` (no Docker needed; the upstream
  artifact's Docker image only bundles other tools) and run it with
  `java -jar`. Deterministic -> a single run suffices.

Protocol (matches the other pipelines for comparability): for each case we
synthesize from I/O variant 0 only, then the composed .dl is evaluated over all
of the case's evaluation variants by eval_symbolic.py. Each run writes to
run_<i>/<case>.dl.
"""

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import time

from pathlib import Path

from context_synth import BENCHMARK_DIR
from synth_llm import load_or_build_context


ROOT = BENCHMARK_DIR.parent
BASELINES_DIR = ROOT / "baselines"
IODATA_DIR = BENCHMARK_DIR / "io_data"

DEFAULT_NUM_RUNS = {"gensynth": 5, "egs": 1, "prosynth": 1}

# ProSynth (rule-selection synthesizer) is its own submodule; it needs its
# patched Souffle built (see baselines/setup.sh) plus the z3 python module.
# The scripts must run with cwd = the submodule/repo root (prepare resolves
# souffle as $PWD/prosynth/souffle/...); the tool itself lives one level in.
PROSYNTH_REPO = BASELINES_DIR / "prosynth"
PROSYNTH_ROOT = PROSYNTH_REPO / "prosynth"
PROSYNTH_SOUFFLE = PROSYNTH_ROOT / "souffle" / "src" / "souffle"

# GenSynth writes its final program here (VERSION_NUMBER=102, trial number 0).
GENSYNTH_FINAL_LOG = "{name}-parallel_log_final102.txt"

# Local EGS fat jar produced by `sbt assembly` (the default, no Docker).
DEFAULT_EGS_JAR = BASELINES_DIR / "egs" / "egs" / "target" / "scala-2.13" / "egs-assembly-0.1.0-SNAPSHOT.jar"

# Optional command-template override (e.g. Docker); placeholders: {problem}, {jar}.
# Example Docker form: "docker run --rm -v {problem}:/problem egs scala /path/to/jar /problem"

_LITERAL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(([^()]*)\)")
_RULE_LINE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\([^\n]*\)\s*(:-[^\n]*)?\.\s*$")


def _terminate_process_group(proc, grace_sec=3):
    """Stop a timed-out synthesizer and every compiler/worker it spawned.

    ``subprocess.run(timeout=...)`` kills only its direct child.  GenSynth
    starts a multiprocessing pool, while ProSynth's ``prepare`` shell starts
    Souffle, ``souffle-compile``, g++, and cc1plus.  Killing only the wrapper
    leaves those descendants reparented to PID 1, consuming cores long after
    the case has already been recorded as a timeout.
    """
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_sec
    while time.monotonic() < deadline:
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.wait()
    except ChildProcessError:
        pass


def _run_process(cmd, *, timeout, **kwargs):
    """Run a local tool in its own session and clean its whole group on exit."""
    if kwargs.pop("capture_output", False):
        if kwargs.get("stdout") is not None or kwargs.get("stderr") is not None:
            raise ValueError("stdout and stderr may not be used with capture_output")
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    proc = subprocess.Popen(cmd, start_new_session=True, **kwargs)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_process_group(proc)
        stdout, stderr = proc.communicate()
        raise subprocess.TimeoutExpired(cmd, timeout, output=stdout, stderr=stderr)
    except BaseException:
        _terminate_process_group(proc)
        raise
    # A tool can return while a pool worker is still alive.  It has no useful
    # work after the parent result is collected, so do not let it become a PID
    # 1 orphan even on the nominal-success path.
    try:
        os.killpg(proc.pid, 0)
    except ProcessLookupError:
        pass
    else:
        _terminate_process_group(proc)
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def _arity(item) -> int:
    return len(item["args"])


def _copy_variant0_examples(case_id, schema_def, dest_dir: Path):
    """Copy variant-0 facts (inputs) and expected (outputs) into dest_dir."""
    # Use the evaluator's layout-aware enumerator.  The benchmark now stores
    # scored inputs under <case>/eval/<n>, while older checkouts used
    # <case>/<n>; reading <case>/0 directly silently supplied empty examples
    # after the split into eval/ and demo/ pools.
    import sys
    evaluation_dir = str(ROOT / "evaluation")
    if evaluation_dir not in sys.path:
        sys.path.insert(0, evaluation_dir)
    from common_eval import case_variants_dirs

    variants = case_variants_dirs(case_id)
    if not variants:
        raise FileNotFoundError(f"{case_id}: no scored I/O variants")
    src = variants[0]
    for item in schema_def["input"]:
        rel = item["relation"]
        s = src / f"{rel}.facts"
        d = dest_dir / f"{rel}.facts"
        if s.exists():
            shutil.copyfile(s, d)
        else:
            d.write_text("", encoding="utf-8")
    for item in schema_def["output"]:
        rel = item["relation"]
        s = src / f"{rel}.expected"
        d = dest_dir / f"{rel}.expected"
        if s.exists():
            shutil.copyfile(s, d)
        else:
            d.write_text("", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Shared: parse synthesized rules and compose a souffle-runnable program.
# --------------------------------------------------------------------------- #
def _extract_rule_lines(text: str) -> str:
    rules = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("DNA") or line.startswith("PROGRAM"):
            continue
        if _RULE_LINE_RE.match(line):
            rules.append(line)
    return "\n".join(rules)


def parse_gensynth_program(text: str) -> str:
    """Take the last 'PROGRAM ...' block (the final/reduced solution) and keep
    only its Datalog rule lines. GenSynth's saved __str__ has no Rule() literal.
    """
    parts = text.split("PROGRAM ")
    block = parts[-1] if len(parts) > 1 else text
    return _extract_rule_lines(block)


def parse_egs_program(text: str) -> str:
    """EGS prints the target program's rules (lines containing ':-') to stdout."""
    return _extract_rule_lines(text)


_RULE_META_RE = re.compile(r"\bRule\(\s*\d+\s*\)")


def parse_prosynth_program(text: str) -> str:
    """ProSynth prints selected rules to stdout, each carrying a `Rule(n)`
    selection tag in the body (e.g. `H(..) :- Rule(27), B1(..), B2(..).`).
    Strip the tag and normalize the leftover punctuation.
    """
    rules = []
    for raw in text.splitlines():
        line = raw.strip()
        if ":-" not in line:
            continue
        cleaned = _RULE_META_RE.sub("", line)
        # remove the comma/space the tag left behind, in any body position
        cleaned = re.sub(r",\s*,", ",", cleaned)
        cleaned = re.sub(r":-\s*,", ":-", cleaned)
        cleaned = re.sub(r",\s*\.", ".", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if _RULE_LINE_RE.match(cleaned):
            rules.append(cleaned)
    return "\n".join(rules)


def compose_symbolic_dl(schema_def, rules_text: str) -> str:
    """Wrap synthesized rules with our standard decl block (all symbol types),
    declaring any invented predicates the tool introduced so souffle can run it.
    """
    input_rels = {it["relation"]: _arity(it) for it in schema_def["input"]}
    output_rels = {it["relation"]: _arity(it) for it in schema_def["output"]}
    known = set(input_rels) | set(output_rels)

    seen = {}
    for m in _LITERAL_RE.finditer(rules_text):
        name = m.group(1)
        args = m.group(2).strip()
        arity = len([a for a in args.split(",") if a.strip()]) if args else 0
        seen[name] = max(seen.get(name, 0), arity)

    def decl(name, arity):
        cols = ", ".join(f"x{i}:symbol" for i in range(max(arity, 1)))
        return f".decl {name}({cols})"

    # Declare each relation once: one that is both an input and an output would
    # otherwise be declared twice and Souffle rejects the program outright.
    lines = []
    declared = set()
    for rels, directive in ((input_rels, "input"), (output_rels, "output")):
        for name, ar in rels.items():
            if name not in declared:
                lines.append(decl(name, ar))
                declared.add(name)
            lines.append(f".{directive} {name}")
    for name, ar in sorted(seen.items()):
        if name in known:
            continue
        lines.append(decl(name, ar))

    body = rules_text.strip()
    header = "\n".join(lines)
    return f"{header}\n\n{body}\n" if body else f"{header}\n"


# --------------------------------------------------------------------------- #
# GenSynth backend (local).
# --------------------------------------------------------------------------- #
def build_gensynth_rules_t(schema_def, type_name: str = "V") -> str:
    lines = []
    for it in schema_def["input"]:
        sig = ", ".join([type_name] * _arity(it))
        lines.append(f"*{it['relation']}({sig})")
    for it in schema_def["output"]:
        sig = ", ".join([type_name] * _arity(it))
        lines.append(f"{it['relation']}({sig})")
    return "\n".join(lines) + "\n"


def run_gensynth(schema_def, case_id, threads, timeout_sec, souffle_path, target_score):
    gensynth_bin = BASELINES_DIR / "gensynth" / "gensynth"
    if not gensynth_bin.exists():
        return "", "gensynth submodule not found (git submodule update --init)"

    # GenSynth terminates its multiprocessing pool from a success callback.
    # On Linux a worker can still finish a log write while the parent process
    # is exiting, racing TemporaryDirectory's recursive cleanup and raising
    # ENOTEMPTY after an otherwise successful synthesis.  Cleanup failure must
    # not abort the whole benchmark grid; the OS temp area can safely retain a
    # rare raced directory for later cleanup.
    with tempfile.TemporaryDirectory(
        prefix=f"gensynth_{case_id}_", ignore_cleanup_errors=True
    ) as tmp:
        root = Path(tmp)
        prob = root / "0" / case_id
        prob.mkdir(parents=True)
        (prob / "rules.t").write_text(build_gensynth_rules_t(schema_def), encoding="utf-8")
        _copy_variant0_examples(case_id, schema_def, prob)

        cmd = [
            "python3", str(gensynth_bin), str(root), case_id,
            "-n", str(threads), "-l", "0", "-t", str(target_score),
            "--souffle_path", souffle_path,
        ]
        try:
            proc = _run_process(
                cmd, capture_output=True, text=True, timeout=timeout_sec,
                cwd=str(gensynth_bin.parent),
            )
        except subprocess.TimeoutExpired:
            return "", f"gensynth timed out after {timeout_sec}s"
        except FileNotFoundError as exc:
            return "", f"gensynth launch failed: {exc}"

        log_file = root / "logs" / "trial0" / GENSYNTH_FINAL_LOG.format(name=case_id)
        if log_file.exists():
            rules = parse_gensynth_program(log_file.read_text(encoding="utf-8"))
        else:
            rules = parse_gensynth_program(proc.stdout or "")

        if rules:
            return rules, ""
        err = (proc.stderr or proc.stdout or "no program synthesized").strip()
        return "", err[:300]


# --------------------------------------------------------------------------- #
# EGS backend (Docker; not built in this environment yet).
# --------------------------------------------------------------------------- #
def build_egs_rules_small(schema_def, type_name: str = "V") -> str:
    lines = [f".type {type_name}", ""]
    for it in schema_def["input"]:
        cols = ", ".join(f"v{i}: {type_name}" for i in range(_arity(it)))
        lines.append(f".decl {it['relation']}({cols})")
        lines.append(f".input {it['relation']}")
    for it in schema_def["output"]:
        cols = ", ".join(f"v{i}: {type_name}" for i in range(_arity(it)))
        lines.append(f".decl {it['relation']}({cols})")
        lines.append(f".output {it['relation']}")
    return "\n".join(lines) + "\n"


def run_egs(schema_def, case_id, timeout_sec, egs_jar=None, egs_cmd_template=None, java_bin="java"):
    """Run EGS locally via its assembly jar (default) or a custom command template.

    The jar is compiled against Java 11 APIs (String.strip), so java_bin must be a
    Java 11+ runtime even though the rest of the benchmark runs fine on Java 8.
    """
    with tempfile.TemporaryDirectory(prefix=f"egs_{case_id}_") as tmp:
        prob = Path(tmp) / case_id
        prob.mkdir(parents=True)
        (prob / "rules.small.dl").write_text(build_egs_rules_small(schema_def), encoding="utf-8")
        _copy_variant0_examples(case_id, schema_def, prob)

        jar = Path(egs_jar) if egs_jar else DEFAULT_EGS_JAR
        if egs_cmd_template:
            cmd = shlex.split(egs_cmd_template.format(problem=str(prob), jar=str(jar), java=java_bin))
        else:
            if not jar.exists():
                return "", (
                    f"EGS jar not found at {jar}. Build it locally with "
                    "`cd baselines/egs/egs && sbt assembly` (JDK 11+, no Docker), or pass --egs_jar / --egs_cmd."
                )
            cmd = [java_bin, "-jar", str(jar), str(prob)]

        try:
            proc = _run_process(cmd, capture_output=True, text=True, timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            return "", f"egs timed out after {timeout_sec}s"
        except FileNotFoundError as exc:
            return "", f"egs launch failed: {exc}"

        rules = parse_egs_program(proc.stdout or "")
        if rules:
            return rules, ""
        err = (proc.stderr or proc.stdout or "no program synthesized").strip()
        return "", err[:300]


# --------------------------------------------------------------------------- #
# ProSynth backend (local; rule-selection, needs patched Souffle + z3).
# --------------------------------------------------------------------------- #
def _z3_importable(python_bin: str) -> bool:
    try:
        return subprocess.run([python_bin, "-c", "import z3"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def run_prosynth(schema_def, case_id, timeout_sec, rule_width, python_bin="python3"):
    """Three-step ProSynth pipeline: rule-gen (enumerate candidates) -> prepare
    (compile candidates via patched Souffle) -> prosynth (z3 rule selection).
    The candidate space blows up with rule_width; the `prepare` compile is the
    usual bottleneck (their own driver caps each step at ~5 min).
    """
    if not PROSYNTH_SOUFFLE.exists():
        return "", (
            f"patched Souffle not built at {PROSYNTH_SOUFFLE}. Build it under "
            "baselines/egs/prosynth/souffle (see README)."
        )
    if not _z3_importable(python_bin):
        return "", "z3 python module not available (pip install z3-solver)"

    gen = PROSYNTH_ROOT / "scripts" / "rule-gen" / "generate"
    prepare = PROSYNTH_ROOT / "scripts" / "prepare"
    prosynth = PROSYNTH_ROOT / "scripts" / "prosynth"

    with tempfile.TemporaryDirectory(prefix=f"prosynth_{case_id}_") as tmp:
        pd = Path(tmp) / case_id
        pd.mkdir(parents=True)
        # rules.t uses the same GenSynth signature format the generator expects.
        (pd / "rules.t").write_text(build_gensynth_rules_t(schema_def), encoding="utf-8")
        _copy_variant0_examples(case_id, schema_def, pd)

        # 1. Enumerate candidate rules (generate writes rules.dl/ruleNames.txt to cwd).
        try:
            _run_process(
                [python_bin, str(gen), str(pd), str(rule_width)],
                cwd=str(pd), capture_output=True, text=True, timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return "", f"rule-gen timed out after {timeout_sec}s (rule_width={rule_width})"
        if not (pd / "rules.dl").exists():
            return "", "rule-gen produced no rules.dl"
        # prepare/prosynth read the `.small` variant names.
        shutil.copyfile(pd / "rules.dl", pd / "rules.small.dl")
        shutil.copyfile(pd / "ruleNames.txt", pd / "ruleNames.small.txt")

        # 2. Compile the candidate set into souffle.small.out (patched Souffle -t explain).
        #    prepare resolves souffle via $PWD/prosynth/souffle, so cwd = repo root.
        try:
            p = _run_process(
                ["bash", str(prepare), str(pd)],
                cwd=str(PROSYNTH_REPO), capture_output=True, text=True, timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return "", f"prepare (Souffle compile of width-{rule_width} candidates) timed out after {timeout_sec}s"
        if not (pd / "souffle.small.out").exists():
            return "", "prepare failed: " + (p.stderr or p.stdout or "no souffle.small.out")[:200]

        # 3. Select rules via provenance-guided synthesis (coprov=0, delta=1).
        #    Run in the problem dir so stray artifacts (data.log) stay in tmp.
        try:
            r = _run_process(
                [python_bin, str(prosynth), str(pd), "0", "1", "1", str(pd / "pslog.txt")],
                cwd=str(pd), capture_output=True, text=True, timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return "", f"prosynth (z3 selection) timed out after {timeout_sec}s"

        rules = parse_prosynth_program(r.stdout or "")
        if rules:
            return rules, ""
        return "", (r.stderr or r.stdout or "no program synthesized").strip()[:300]


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #
def synthesize_symbolic(
    tool,
    dataset,
    case_id=None,
    num_runs=None,
    threads=8,
    timeout_sec=300,
    souffle_path="souffle",
    target_score=1.0,
    egs_jar=None,
    egs_cmd=None,
    java_bin="java",
    rule_width=2,
    python_bin="python3",
    resume=False,
):
    if tool not in {"gensynth", "egs", "prosynth"}:
        raise ValueError("--tool must be 'gensynth', 'egs', or 'prosynth'")
    if num_runs is None:
        num_runs = DEFAULT_NUM_RUNS[tool]
    if num_runs <= 0:
        raise ValueError("--num_runs must be > 0")

    tasks = load_or_build_context(dataset=dataset, case_id=case_id)
    dataset_scope = dataset if dataset else "all"
    scope_dir = case_id if case_id else dataset_scope
    output_base_dir = BENCHMARK_DIR / "infer_data" / tool / scope_dir

    for run_id in range(num_runs):
        run_dir = output_base_dir / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for task in tasks:
            cid = task["case_id"]
            schema = task["schema_def"]
            case_path = run_dir / f"{cid}.dl"
            if resume and case_path.exists():
                full_dl = case_path.read_text(encoding="utf-8")
                # Symbolic outputs contain only directives followed by rules;
                # recover the latter for the aggregate checkpoint.  A blank
                # body is an already-recorded failed synthesis and must also be
                # skipped, otherwise a resumed long grid silently reruns every
                # expensive timeout.
                rules = "\n".join(
                    line for line in full_dl.splitlines()
                    if line.strip() and not line.lstrip().startswith(".")
                ).strip()
                rows.append({
                    "id": cid,
                    "run": run_id,
                    "tool": tool,
                    "rules": rules,
                    "error": "" if rules else "resumed existing empty synthesis",
                    "synth_sec": None,
                    "resumed": True,
                })
                print(f"[RESUME:{tool}][run {run_id}] {cid}: existing output")
                continue
            t0 = time.time()
            if tool == "gensynth":
                rules, err = run_gensynth(schema, cid, threads, timeout_sec, souffle_path, target_score)
            elif tool == "egs":
                rules, err = run_egs(schema, cid, timeout_sec, egs_jar=egs_jar, egs_cmd_template=egs_cmd, java_bin=java_bin)
            else:
                rules, err = run_prosynth(schema, cid, timeout_sec, rule_width, python_bin=python_bin)
            full_dl = compose_symbolic_dl(schema, rules)
            case_path.write_text(full_dl, encoding="utf-8")

            rows.append({
                "id": cid,
                "run": run_id,
                "tool": tool,
                "rules": rules,
                "error": err,
                "synth_sec": round(time.time() - t0, 2),
            })
            status = "ok" if rules else f"FAIL({err[:60]})"
            print(f"[SYNTH:{tool}][run {run_id}] {cid}: {status}")

        agg = run_dir / f"{scope_dir}.json"
        agg.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[DONE][run {run_id}] aggregate -> {agg}")

    print(f"[DONE] {num_runs} run(s) -> {output_base_dir}")


def main():
    parser = argparse.ArgumentParser(description="Synthesize Datalog with symbolic baselines (gensynth/egs).")
    parser.add_argument("--tool", required=True, choices=["gensynth", "egs", "prosynth"])
    parser.add_argument("--dataset", type=str, default="all", help="Case filter: category, sub category, or all")
    parser.add_argument("--case_id", type=str, default=None, help="Optional single case id")
    parser.add_argument("--num_runs", type=int, default=None, help="Repetitions (default: gensynth=5, egs=1, prosynth=1)")
    parser.add_argument("--rule_width", type=int, default=2, help="prosynth: max body literals per candidate rule (bigger = broader hypothesis space but much slower prepare; 3+ can exceed the timeout even on simple cases)")
    parser.add_argument("--python_bin", type=str, default="python3", help="prosynth: python interpreter with z3 installed")
    parser.add_argument("--threads", type=int, default=8, help="gensynth: number of populations/threads (-n)")
    parser.add_argument("--timeout_sec", type=int, default=300, help="Per-case wall-clock timeout")
    parser.add_argument("--souffle_path", type=str, default="souffle", help="gensynth: path to souffle")
    parser.add_argument("--target_score", type=float, default=1.0, help="gensynth: minimum F1 to accept (-t)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip case outputs already present in each run directory")
    parser.add_argument("--egs_jar", type=str, default=None,
                        help="EGS: path to the assembly jar (default: baselines/egs/egs/target/... or env EGS_JAR)")
    parser.add_argument("--egs_cmd", type=str, default=None,
                        help="EGS: optional run-command template ({problem}, {jar}, {java}); overrides local jar (e.g. Docker). Env EGS_RUN_CMD")
    parser.add_argument("--java_bin", type=str, default=None,
                        help="EGS: Java 11+ runtime for the jar (default: env EGS_JAVA or 'java')")
    args = parser.parse_args()

    import os
    egs_jar = args.egs_jar or os.environ.get("EGS_JAR")
    egs_cmd = args.egs_cmd or os.environ.get("EGS_RUN_CMD")
    java_bin = args.java_bin or os.environ.get("EGS_JAVA") or "java"

    synthesize_symbolic(
        tool=args.tool,
        dataset=args.dataset,
        case_id=args.case_id,
        num_runs=args.num_runs,
        threads=args.threads,
        timeout_sec=args.timeout_sec,
        souffle_path=args.souffle_path,
        target_score=args.target_score,
        egs_jar=egs_jar,
        egs_cmd=egs_cmd,
        java_bin=java_bin,
        rule_width=args.rule_width,
        python_bin=args.python_bin,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
