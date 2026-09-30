# DatalogBench

**DatalogBench** is a benchmark for text-to-Datalog synthesis: **136 tasks**
across four domains, each with a natural-language question, a relation schema, a
reference program, and held-out input/output variants. A synthesized program is
graded by executing it under Soufflé, against an oracle validated by mutation
analysis.

> **Paper:** [*DatalogBench: Evaluating Large Language Models on Text-to-Datalog Synthesis*](https://arxiv.org/abs/2609.37233)

---

## Overview

A system receives what a Datalog programmer would receive -- a question and the
input/output relation schema -- and never sees the reference rules. For example
(`EvenPath`):

```
Question  Find all node pairs connected by a path of even length in a directed graph.
Input     Edge(src_node, dst_node)       A directed edge from src_node to dst_node.
Output    EvenPath(src_node, dst_node)   A path of even length from src_node to dst_node.
```

The reference program needs an auxiliary predicate the schema does not mention,
and mutual recursion:

```prolog
OddPath(x, z)  :- Edge(x, z).
OddPath(x, z)  :- Edge(x, y), EvenPath(y, z).
EvenPath(x, z) :- Edge(x, y), OddPath(y, z).
```

Grading proceeds as follows:

```
question + schema (+ relation descriptions, + one demonstration instance)
        │
        ▼  synthesis: Direct prompting | coding agent | symbolic baseline
   candidate rules
        │
        ▼  composition with the graded interface (.decl / .input / .output)
   <Case>.dl
        │
        ▼  Soufflé 2.5 on every evaluation variant (30 s each)
   CP · EX · tuple-level P/R/F1 · negative exclusion
```

A task counts as solved (**exact match, EX**) only if its program compiles and
reproduces the expected output on **every** evaluation variant. **Compile pass
(CP)** is the fraction of programs Soufflé accepts.

---

## Repository Structure

```
DatalogBench/
├── benchmark/
│   ├── dataset.json              # 136 tasks: question, input/output relations and descriptions
│   ├── query/<Case>.dl           # reference programs
│   ├── io_data/<Case>/
│   │   ├── demo/0/               #   demonstration instance (one-shot example, agents' development instance), never graded
│   │   └── eval/<i>/             #   evaluation variants: *.facts, *.expected, *.undesired
│   ├── spec_tiers.json           # specification-sufficiency labels (L1/L2), never shown to a solver
│   ├── negative_manifest.json    # provenance of the mined negative tuples
│   ├── singleton_inputs.json     # single-instance parameters that must not be scaled like data
│   ├── reviewed_survivors.json   # hand verdicts on surviving mutants
│   ├── qa/                       # the two frozen oracle-validation files
│   ├── qa_check.py               # integrity checks over the released files (run in CI)
│   └── regen_expected.py         # re-derives every .expected file from its reference program
│
├── synthesis/
│   ├── synth_llm.py              # Direct prompting
│   ├── synth_coding_agent.py     # coding agents (Codex CLI, Claude Code)
│   ├── agent_container.py        #   runs an agent inside the sandbox container; --verify checks it
│   ├── agent_finalize_measurement.py  # closes an agent run before it is graded
│   ├── measurement_v1.py         #   how a task that never produced a program is counted
│   ├── synth_symbolic.py         # symbolic baselines (EGS, GenSynth, ProSynth)
│   ├── context_synth.py          # prompt context built from dataset.json
│   └── online_llm.py             # API client
│
├── evaluation/
│   ├── eval_llm.py, eval_coding_agent.py, eval_symbolic.py   # grading, one per setting
│   ├── common_eval.py, metrics.py                            # Soufflé execution, metrics, CIs
│   ├── model_inventory.py        # the evaluated models, endpoints, budgets and timeouts
│   ├── compare_llm.py            # paired permutation test and bootstrap CI between two cells
│   ├── structure_analyzer.py, structure_compare.py           # structural statistics of programs
│   ├── mutation_score.py, triage_survivors.py                # oracle validation by mutation
│   ├── input_generator.py, select_variants.py, negative_examples.py  # variant and negative-set construction
│   └── ablation_report.py, tier_report.py, difficulty.py, error_report.py,
│       decl_repair_probe.py, agent_measurement_report.py, leaked_predicates.py  # analyses in the paper
│
├── docker/                       # pinned grading image and agent sandbox (Dockerfiles, egress proxy)
├── baselines/                    # symbolic baselines: pinned submodules, build patches, setup.sh
├── run_llm.sh, run_agent.sh, run_symbolic.sh   # full grids, one per setting
└── reproduce.sh                  # integrity checks, then grades whatever generations are present
```

The pipelines write `benchmark/infer_data/` (what each system generated) and
`benchmark/res_data/` (how it scored); neither is part of the repository.

---

## Quick Start

Python 3.10+ and Soufflé 2.5. Soufflé is the grader, so its build decides every
number:

```bash
pip3 install -r requirements.txt

curl -fsSL -o /tmp/souffle.deb \
  https://github.com/souffle-lang/souffle/releases/download/2.5/x86_64-ubuntu-2204-souffle-2.5-Linux.deb
sudo apt-get install -y /tmp/souffle.deb
souffle --version    # expect: Version 2.5, Word size: 64 bits
```

The word size matters: the official `.deb` is 64-bit, while a source build and
Homebrew default to 32-bit. Integer arithmetic also differs across architectures
-- division by zero traps on x86 and returns 0 on arm64 -- which is why the check
below re-derives the expected outputs rather than trusting the files on disk.

```bash
bash reproduce.sh              # a few minutes; no API key, network or Docker
```

`qa_check.py` verifies the schema, oracle and negative-set invariants, and sends
every reference program through the path a candidate takes, requiring it to be
graded exact. `regen_expected.py --check` re-derives every `.expected` file by
executing the reference program. Once the pipelines have written
`benchmark/infer_data/`, `reproduce.sh` also grades each cell present and prints
it next to the value reported in the paper; the values match exactly only on the
generations the reported numbers came from.

Add `HARNESS="bash docker/run-harness.sh"` to grade inside the pinned image
rather than with the host's Soufflé; each cell records which was used in its
`res_data/*_provenance.json`.

---

## The Benchmark

| | |
|---|---|
| Tasks | 136 |
| Domains | program analysis 47, graph analytics 33, formal reasoning 29, knowledge discovery 27 |
| Evaluation variants | 247 (1--7 per task, median 2) |
| Demonstration instances | 136, one per task, never graded |
| Recursive reference programs | 78 |
| Reference programs with auxiliary predicates | 59 |
| Negative tuples mined for the variants | 4,867 across 236 files |
| Specification tiers | L1 111, L2 25; 15 tasks carry a knowledge note |

Tasks derive from four public sources -- Soufflé's test suite, EGS, GenSynth and
CodeFuse-Query. Relations were renamed and retyped and the facts regenerated, so
solving a task requires reading the schema in the prompt. The structural
statistics are regenerated from `dataset.json` and `query/` alone:

```bash
python3 evaluation/structure_analyzer.py
```

**Oracle validation.** `mutation_score.py` mutates each reference program --
deleting a statement, deleting a body atom, swapping a join variable -- and asks
whether some variant's expected output detects the change.

| | |
|---|---|
| Mutants | 1,948 |
| Killed by the evaluation variants | 1,877 (96.4%) |
| Survivors | 71, across 20 tasks |
| Survivors adjudicated equivalent to their reference | 71 of 71 |

The verdicts are in `reviewed_survivors.json`, keyed by the exact rule text each
was judged on, so editing a reference program invalidates its verdict visibly.
`benchmark/qa/mutation_frozen.json` holds the exhaustive run and
`survivor_triage_frozen.json` its adjudication. To recompute them (hours):

```bash
bash docker/run-harness.sh python3 evaluation/mutation_score.py \
    --sample 0 --max-mutants 0 --jobs 8 --out benchmark/qa/mutation_frozen.json
```

Both zeros are required; the defaults give a sampled, truncated run.

---

## Running the Systems

These calls cost inference. Keys are read from the environment, never from the
command line: `EVAL_API_KEY` for Direct prompting (`DS_API_KEY` for the DeepSeek
models), `AGENT_API_KEY` for the coding agents. Endpoints are set in
`evaluation/model_inventory.py` and `docker/{codex,claude}.sh`.

### Direct prompting

6 models × {signature, description} × {0-shot, 1-shot}, all 136 tasks:

```bash
TEMPERATURE=0 NUM_RUNS=1 ./run_llm.sh          # RESUME=1 continues an interrupted grid
```

The model list, output budgets and request timeouts live only in
`evaluation/model_inventory.py`. Resume reuses an inference only when its
response is non-empty and its prompt digest and token budget still match.

### Coding agents

The agents run inside a container: only the per-task scratch directory is
mounted, so the reference programs are absent rather than merely unreadable, and
an egress proxy allows the model endpoint alone. Build the images on a
workstation and load them on the runner, so one pinned image fixes the Soufflé
build and both CLI builds:

```bash
TARGET_PLATFORM=linux/amd64 bash docker/export-images.sh     # workstation
bash docker/load-images.sh dlb-images-linux-amd64.tar.gz      # runner
python3 synthesis/agent_container.py --verify
```

`--verify` measures the boundary rather than inspecting configuration; all six
checks must pass (the key is not on any command line, a reference program is
unreadable by absolute path, a public URL is blocked, each agent's endpoint is
reachable, and Soufflé runs inside and matches the host build). Then:

```bash
TEMPERATURE=0 NUM_RUNS=1 ./run_agent.sh        # AGENTS=claude RESUME=1 continues
```

Per cell, the middle step is mandatory:

```bash
python3 synthesis/synth_coding_agent.py --agent codex --method signature \
    --dataset all --temperature 0 --num_runs 1 --max_iterations 4
python3 synthesis/agent_finalize_measurement.py --agent codex \
    --method signature --dataset all --run 0
python3 evaluation/eval_coding_agent.py --agent codex --method signature \
    --dataset all --num_runs 1
```

Finalizing decides how a task that never produced a program is counted, and the
evaluator refuses a run that has not been through it. Such a task is scored as a
failure and never executed, since a bare schema would compile and collect a
compile pass for free. When a task was retried across separate invocations, pass
every synthesis log oldest first (`--attempt-log ...`); `--dry-run` shows the
decisions first.

### Symbolic baselines

```bash
bash baselines/setup.sh     # fetches each tool at its pinned commit, patches, builds
TOOLS="gensynth egs prosynth" EGS_JAVA=/usr/lib/jvm/java-11-openjdk-amd64/bin/java \
  PROSYNTH_RULE_WIDTH=2 ./run_symbolic.sh
```

- **GenSynth** -- genetic search; stochastic, reported as a 5-run mean.
- **EGS** -- example-guided synthesis; building needs a JDK between 11 and 17.
- **ProSynth** -- provenance-guided rule selection; needs `z3-solver` and
  `autoconf automake libtool mcpp bison flex`.

Each tool learns from a task's first evaluation variant and is graded on all of
that task's variants with the same harness.

---

## Run Matrix

The paper reports 29 cells. A **cell** is one (system, prompt condition) pair
graded on all 136 tasks; its name indexes `benchmark/res_data/<cell>_*`. Model
calls use temperature 0. The Direct
cells come from `run_llm.sh`, the coding-agent cells from `run_agent.sh` (with a
budget of 4 iterations per task), and the symbolic cells from `run_symbolic.sh`
(ProSynth with rule width 2).

| # | Setting | System | Schema | I/O examples in the prompt | Runs | Cell |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Direct | `gpt-5.6-sol` | signature | 0 | 1 | `gpt-5.6-sol_signature_all` |
| 2 | Direct | `gpt-5.6-sol` | description | 0 | 1 | `gpt-5.6-sol_description_all` |
| 3 | Direct | `gpt-5.6-sol` | signature | 1 | 1 | `gpt-5.6-sol_1-shot_signature_all` |
| 4 | Direct | `gpt-5.6-sol` | description | 1 | 1 | `gpt-5.6-sol_1-shot_description_all` |
| 5 | Direct | `claude-opus-5` | signature | 0 | 1 | `claude-opus-5_signature_all` |
| 6 | Direct | `claude-opus-5` | description | 0 | 1 | `claude-opus-5_description_all` |
| 7 | Direct | `claude-opus-5` | signature | 1 | 1 | `claude-opus-5_1-shot_signature_all` |
| 8 | Direct | `claude-opus-5` | description | 1 | 1 | `claude-opus-5_1-shot_description_all` |
| 9 | Direct | `gemini-3.7-flash` | signature | 0 | 1 | `gemini-3.7-flash_signature_all` |
| 10 | Direct | `gemini-3.7-flash` | description | 0 | 1 | `gemini-3.7-flash_description_all` |
| 11 | Direct | `gemini-3.7-flash` | signature | 1 | 1 | `gemini-3.7-flash_1-shot_signature_all` |
| 12 | Direct | `gemini-3.7-flash` | description | 1 | 1 | `gemini-3.7-flash_1-shot_description_all` |
| 13 | Direct | `deepseek-v4-pro` | signature | 0 | 1 | `deepseek-v4-pro_signature_all` |
| 14 | Direct | `deepseek-v4-pro` | description | 0 | 1 | `deepseek-v4-pro_description_all` |
| 15 | Direct | `deepseek-v4-pro` | signature | 1 | 1 | `deepseek-v4-pro_1-shot_signature_all` |
| 16 | Direct | `deepseek-v4-pro` | description | 1 | 1 | `deepseek-v4-pro_1-shot_description_all` |
| 17 | Direct | `deepseek-v4-flash` | signature | 0 | 1 | `deepseek-v4-flash_signature_all` |
| 18 | Direct | `deepseek-v4-flash` | description | 0 | 1 | `deepseek-v4-flash_description_all` |
| 19 | Direct | `deepseek-v4-flash` | signature | 1 | 1 | `deepseek-v4-flash_1-shot_signature_all` |
| 20 | Direct | `deepseek-v4-flash` | description | 1 | 1 | `deepseek-v4-flash_1-shot_description_all` |
| 21 | Direct | `deepseek-v4-flash-non-thinking` | signature | 0 | 1 | `deepseek-v4-flash-non-thinking_signature_all` |
| 22 | Direct | `deepseek-v4-flash-non-thinking` | description | 0 | 1 | `deepseek-v4-flash-non-thinking_description_all` |
| 23 | Direct | `deepseek-v4-flash-non-thinking` | signature | 1 | 1 | `deepseek-v4-flash-non-thinking_1-shot_signature_all` |
| 24 | Direct | `deepseek-v4-flash-non-thinking` | description | 1 | 1 | `deepseek-v4-flash-non-thinking_1-shot_description_all` |
| 25 | Coding agent | `codex` on `gpt-5.6-sol` | signature | 0 (tests its programs on the demonstration instance) | 1 | `gpt-5.6-sol_codex_signature_all` |
| 26 | Coding agent | `claude` on `claude-opus-5` | signature | 0 (tests its programs on the demonstration instance) | 1 | `claude-opus-5_claude_signature_all` |
| 27 | Symbolic | `egs` | -- | learns from the first evaluation variant | 1 | `egs_all` |
| 28 | Symbolic | `gensynth` | -- | learns from the first evaluation variant | 5 | `gensynth_all` |
| 29 | Symbolic | `prosynth` | -- | learns from the first evaluation variant | 1 | `prosynth_all` |

---

## Outputs and Metrics

Each cell of the run matrix writes:

- `infer_data/<model>/<tag>/all/run_<i>/<Case>.dl` (symbolic:
  `infer_data/<tool>/all/run_<i>/`) -- the graded program; for model cells,
  `<Case>.inference.json` holds the raw response and its request contract.
- `res_data/<cell>_summary.csv` -- CP, EX, precision, recall, F1 and negative
  exclusion, each with a 95% interval over tasks; `_details.jsonl` has one row
  per task, including the Soufflé error message when the program failed;
  `_provenance.json` records the grading image and Soufflé build.

Tuple-level precision, recall and F1 are macro-averaged over tasks. Negative
exclusion is the fraction of mined `.undesired` tuples a program avoids deriving,
averaged over programs that compile. A program that times out compiles, is never
exact, and misses all of that variant's expected tuples.

`compare_llm.py` compares two cells with a paired sign-flip permutation test over
shared tasks and a bootstrap 95% CI of the difference, across settings:

```bash
python3 evaluation/compare_llm.py \
    --kind_a coding --agent_a codex --method_a signature \
    --kind_b llm --model_b gpt-5.6-sol --method_b signature --metric pass_at_1
```

The analyses reported in the paper, each reading `res_data/`:

```bash
python3 evaluation/ablation_report.py              # schema and example effects
python3 evaluation/tier_report.py                  # results by specification tier
python3 evaluation/difficulty.py                   # difficulty tiers
python3 evaluation/error_report.py --fine --out <path>.csv   # compile-failure taxonomy
python3 evaluation/decl_repair_probe.py --share    # declaration repair
python3 evaluation/agent_measurement_report.py --out <path>  # decomposition of the agents' gains
python3 evaluation/eval_symbolic.py --tool egs --overlap     # overlap with symbolic synthesis
python3 evaluation/leaked_predicates.py --reported-only --originals <path>  # contamination scan
```

`leaked_predicates.py` takes the upstream sources as a path argument; none is
redistributed here.

---

## Data Sources and Licenses

DatalogBench is assembled from public research artifacts: Soufflé (UPL-1.0), EGS
(MIT), CodeFuse-Query (Apache-2.0) and GenSynth, whose repository declares no
license. The symbolic baselines are referenced as pinned, unmodified submodules
and never vendored. All questions, relation descriptions and annotations were
written by the authors; the benchmark contains only synthetic relational facts
and program-analysis fixtures, with no personal data.

The code and the benchmark files in this repository are released under the
[MIT License](LICENSE).

## Citation

```bibtex
@article{li2026datalogbench,
  title   = {DatalogBench: Evaluating Large Language Models on Text-to-Datalog Synthesis},
  author  = {Yuan Li and Hanyun Jiang and Guowei Tian and Chengpeng Wang and Peisen Yao},
  journal = {arXiv preprint arXiv:2609.37233},
  year    = {2026}
}
```
