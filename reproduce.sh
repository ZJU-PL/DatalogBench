#!/usr/bin/env bash
#
# Check the benchmark, then re-grade whatever generations are present and print
# each cell next to the value the paper reports.
#
# The integrity gates need nothing but this repository: no API key, no network,
# no Docker. The generated programs are not part of the repository -- the
# pipelines write them to benchmark/infer_data/ (README, "Running the Systems") -- so a cell
# is re-graded only if its generations are there, and reported as absent
# otherwise. The comparison reproduces the reported values exactly only on the
# generations they came from; on programs generated anew it shows how that run
# compares.
#
#   bash reproduce.sh              # gates, then every cell that has generations
#   bash reproduce.sh --quick      # gates, then one cell of each kind
#   STAGE=grade bash reproduce.sh  # skip the gates
#
set -uo pipefail
cd "$(dirname "$0")"

QUICK=0
[[ "${1:-}" == "--quick" ]] && QUICK=1
STAGE="${STAGE:-all}"
HARNESS="${HARNESS:-}"      # set to "bash docker/run-harness.sh" to grade in the pinned image
rc=0

REGRADED="$(mktemp)"; ABSENT="$(mktemp)"
trap 'rm -f "$REGRADED" "$ABSENT"' EXIT
run() { echo; echo "\$ $*"; $HARNESS "$@"; }
mark() { echo "$1" >> "$REGRADED"; }          # a cell this run actually re-graded
# present <path under infer_data> <cell>: are this cell's generations here?
present() {
  [[ -e "benchmark/infer_data/$1" ]] && return 0
  echo "  [ABSENT] $2 (no benchmark/infer_data/$1)"; echo "$2" >> "$ABSENT"; return 1
}

echo "############ 0. prerequisites"
python3 --version || { echo "[FATAL] python3 not found"; exit 1; }
if ! command -v souffle >/dev/null 2>&1; then
  echo "[FATAL] souffle not found. It is the grader; see 'Quick Start' in the README."; exit 1
fi
souffle --version 2>&1 | head -2

if [[ "$STAGE" == "all" ]]; then
  echo; echo "############ 1. benchmark integrity (no model, no network)"
  run python3 benchmark/qa_check.py               || rc=1
  run python3 benchmark/regen_expected.py --check || rc=1
fi

echo; echo "############ 2. re-grade the generations that are present"
if [[ ! -d benchmark/infer_data ]]; then
  echo "No benchmark/infer_data/: the generated programs are not part of the"
  echo "repository. Run the pipelines ('Running the Systems' in the README), then run this script"
  echo "again to grade them."
  echo
  if (( rc )); then echo "############ FINISHED WITH FAILURES"; else echo "############ OK (integrity gates only)"; fi
  exit $rc
fi
MODELS=(gpt-5.6-sol claude-opus-5 gemini-3.7-flash deepseek-v4-pro
        deepseek-v4-flash deepseek-v4-flash-non-thinking)
if (( QUICK )); then
  MODELS=(gpt-5.6-sol)
  METHODS=(signature); SHOTS=(0)
else
  METHODS=(signature description); SHOTS=(0 1)
fi

for m in "${MODELS[@]}"; do
  for meth in "${METHODS[@]}"; do
    for n in "${SHOTS[@]}"; do
      tag="$meth"; (( n )) && tag="1-shot_$meth"
      present "$m/$tag/all/run_0/all.json" "${m}_${tag}_all" || continue
      # V4 Pro's generations predate the per-row provider contract, so its
      # artifacts cannot satisfy that check and the evaluator refuses them
      # outright. The generations themselves are fixed; the exception is granted
      # only because the run's own provenance.json records which model the
      # service served.
      extra=(); [[ "$m" == deepseek-v4-pro ]] && extra=(--trust-run-provenance)
      if run python3 evaluation/eval_llm.py --model "$m" --method "$meth" \
          --fewshot "$n" --dataset all --num_runs 1 "${extra[@]+"${extra[@]}"}"; then
        mark "${m}_${tag}_all"
      else
        rc=1
      fi
    done
  done
done

for spec in "codex gpt-5.6-sol" "claude claude-opus-5"; do
  read -r a m <<< "$spec"
  if present "$m/${a}_signature/all/run_0/all.json" "${m}_${a}_signature_all"; then
    if run python3 evaluation/eval_coding_agent.py --agent "$a" --method signature \
        --dataset all --num_runs 1; then
      mark "${m}_${a}_signature_all"
    else
      rc=1
    fi
  fi
  (( QUICK )) && break
done

# GenSynth is stochastic and is reported as the mean of 5 runs; the other two
# are deterministic and run once. Grading gensynth with --num_runs 1 reads run 0
# alone and lands 1.03 points off the reported mean.
for t in egs gensynth prosynth; do
  k=1; [[ "$t" == gensynth ]] && k=5
  if present "$t/all/run_$((k - 1))" "${t}_all"; then
    if run python3 evaluation/eval_symbolic.py --tool "$t" --dataset all --num_runs "$k"; then
      mark "${t}_all"
    else
      rc=1
    fi
  fi
  (( QUICK )) && break
done

if [[ ! -s "$REGRADED" ]] && (( rc == 0 )); then
  echo
  echo "No generations to grade: they are not part of the repository. Run the"
  echo "pipelines ('Running the Systems' in the README), then run this script again to grade them."
  echo; echo "############ OK (integrity gates only)"
  exit 0
fi

echo; echo "############ 3. reported vs re-graded"
python3 - "$REGRADED" "$ABSENT" <<'PY'
import csv, os, sys
# (file stem, label, reported CP, reported EX)
REPORTED = [
    ("gpt-5.6-sol_signature_all",                    "GPT 5.6 Sol / sig",      77.21, 55.88),
    ("gpt-5.6-sol_description_all",                  "GPT 5.6 Sol / desc",     72.06, 51.47),
    ("claude-opus-5_signature_all",                  "Claude Opus 5 / sig",    75.74, 52.21),
    ("claude-opus-5_description_all",                "Claude Opus 5 / desc",   83.82, 60.29),
    ("gemini-3.7-flash_signature_all",               "Gemini 3.7 Flash / sig", 81.62, 61.76),
    ("gemini-3.7-flash_description_all",             "Gemini 3.7 Flash / desc",83.09, 65.44),
    ("deepseek-v4-pro_signature_all",                "V4 Pro / sig",           67.65, 47.79),
    ("deepseek-v4-pro_description_all",              "V4 Pro / desc",          66.91, 50.00),
    ("deepseek-v4-flash_signature_all",              "V4 Flash think / sig",   52.94, 40.44),
    ("deepseek-v4-flash_description_all",            "V4 Flash think / desc",  50.74, 36.76),
    ("deepseek-v4-flash-non-thinking_signature_all", "V4 Flash non-th / sig",  59.56, 36.03),
    ("deepseek-v4-flash-non-thinking_description_all","V4 Flash non-th / desc",65.44, 41.91),
    ("gpt-5.6-sol_1-shot_signature_all",             "GPT 5.6 Sol / sig 1s",   72.06, 61.03),
    ("gpt-5.6-sol_1-shot_description_all",           "GPT 5.6 Sol / desc 1s",  72.06, 59.56),
    ("claude-opus-5_1-shot_signature_all",           "Claude Opus 5 / sig 1s", 71.32, 56.62),
    ("claude-opus-5_1-shot_description_all",         "Claude Opus 5 / desc 1s",82.35, 68.38),
    ("gemini-3.7-flash_1-shot_signature_all",        "Gemini / sig 1s",        72.06, 65.44),
    ("gemini-3.7-flash_1-shot_description_all",      "Gemini / desc 1s",       71.32, 63.24),
    ("deepseek-v4-pro_1-shot_signature_all",         "V4 Pro / sig 1s",        63.97, 52.21),
    ("deepseek-v4-pro_1-shot_description_all",       "V4 Pro / desc 1s",       68.38, 55.88),
    ("deepseek-v4-flash_1-shot_signature_all",       "V4 Flash think / sig 1s",56.62, 44.85),
    ("deepseek-v4-flash_1-shot_description_all",     "V4 Flash think / desc 1s",52.94, 44.12),
    ("deepseek-v4-flash-non-thinking_1-shot_signature_all",  "V4 Flash non-th / sig 1s", 58.82, 35.29),
    ("deepseek-v4-flash-non-thinking_1-shot_description_all","V4 Flash non-th / desc 1s",59.56, 36.76),
    ("gpt-5.6-sol_codex_signature_all",              "Codex agent",            97.79, 82.35),
    ("claude-opus-5_claude_signature_all",           "CC agent",               94.12, 83.82),
    ("egs_all",                                      "EGS",                    100.00, 36.03),
    ("gensynth_all",                                 "GenSynth",               100.00, 25.44),
    ("prosynth_all",                                 "ProSynth",               100.00,  8.09),
]
regraded = set(open(sys.argv[1]).read().split())
absent = set(open(sys.argv[2]).read().split())
print(f"{'cell':26s} {'CP paper':>9s} {'CP here':>8s} {'EX paper':>9s} {'EX here':>8s}  ")
bad = 0
for stem, label, cp_r, ex_r in REPORTED:
    path = f"benchmark/res_data/{stem}_summary.csv"
    if stem not in regraded or not os.path.exists(path):
        why = "no generations" if stem in absent else "not re-graded in this run"
        print(f"{label:26s} {cp_r:9.2f} {'--':>8s} {ex_r:9.2f} {'--':>8s}  {why}")
        continue
    row = next(csv.DictReader(open(path)))
    cp = float(row["compile_pass_rate_mean"]) * 100
    ex = float(row["pass_at_1_mean"]) * 100
    ok = abs(cp - cp_r) < 0.01 and abs(ex - ex_r) < 0.01
    bad += not ok
    print(f"{label:26s} {cp_r:9.2f} {cp:8.2f} {ex_r:9.2f} {ex:8.2f}  {'match' if ok else '*** DIFFERS ***'}")
print()
if bad:
    print(f"{bad} re-graded cell(s) differ from the reported value."); sys.exit(1)
print(f"all {len(regraded)} re-graded cell(s) match the reported value"
      + (f"; {len(absent)} had no generations" if absent else ""))
PY
[[ $? -ne 0 ]] && rc=1

echo
if (( rc )); then echo "############ FINISHED WITH FAILURES"; else echo "############ OK"; fi
exit $rc
