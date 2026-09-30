#!/usr/bin/env bash

# Symbolic-synthesizer baselines (GenSynth / EGS) over the benchmark.
# GenSynth runs locally (Python + Souffle). EGS also runs locally: build its fat
# jar once with a JDK 11+ and sbt (no Docker) --
#   cd baselines/egs/egs && printf 'sbt.version=1.3.13\n' > project/build.properties && sbt assembly
# then point EGS_JAVA at a Java 11+ runtime (the jar uses String.strip; the rest
# of the benchmark is fine on Java 8). Ensure submodules are present first:
#   git submodule update --init --recursive
#
# Example:
#   TOOLS="gensynth egs" EGS_JAVA=/opt/homebrew/opt/openjdk@11/bin/java ./run_symbolic.sh

set -u

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

DATASET="${DATASET:-all}"
SOUFFLE_PATH="${SOUFFLE_PATH:-$(command -v souffle || echo souffle)}"
GENSYNTH_THREADS="${GENSYNTH_THREADS:-8}"
TIMEOUT_SEC="${TIMEOUT_SEC:-300}"
PROSYNTH_RULE_WIDTH="${PROSYNTH_RULE_WIDTH:-2}"
RESUME="${RESUME:-0}"
# Which tools to run (space-separated). Default: gensynth only.
TOOLS="${TOOLS:-gensynth}"

# Synthesis stays on the host: each baseline brings its own toolchain (Python,
# a JDK, z3) and none of that is in the grading image. Grading is the same
# execution-based oracle the other two settings use, so it runs in the image
# like they do -- otherwise the symbolic rows in the results table would be the
# only ones whose numbers came from whatever Souffle this machine has, and the
# table compares them directly against the LLM and agent rows.
source docker/freeze-image.sh || exit 2

grade() {
  if [ "${GRADE_ON_HOST:-0}" = "1" ]; then
    "$@"
  else
    bash docker/run-harness.sh "$@"
  fi
}

total=0; ok=0; failed=0

for tool in $TOOLS; do
  total=$((total + 1))
  echo "============================================================"
  echo "[RUN $total] tool=$tool dataset=$DATASET"

  synth_cmd=(
    python3 synthesis/synth_symbolic.py
    --tool "$tool"
    --dataset "$DATASET"
    --timeout_sec "$TIMEOUT_SEC"
  )
  if [[ "$RESUME" == "1" ]]; then
    synth_cmd+=(--resume)
  fi
  if [[ "$tool" == "gensynth" ]]; then
    synth_cmd+=(--threads "$GENSYNTH_THREADS" --souffle_path "$SOUFFLE_PATH")
  elif [[ "$tool" == "egs" && -n "${EGS_JAVA:-}" ]]; then
    synth_cmd+=(--java_bin "$EGS_JAVA")
  elif [[ "$tool" == "prosynth" ]]; then
    synth_cmd+=(--rule_width "$PROSYNTH_RULE_WIDTH")
  fi

  eval_cmd=(python3 evaluation/eval_symbolic.py --tool "$tool" --dataset "$DATASET")

  echo "[SYNTH] ${synth_cmd[*]}"
  if ! "${synth_cmd[@]}"; then
    echo "[FAIL][SYNTH] tool=$tool"; failed=$((failed + 1)); continue
  fi
  echo "[EVAL ] ${eval_cmd[*]}"
  if ! grade "${eval_cmd[@]}"; then
    echo "[FAIL][EVAL ] tool=$tool"; failed=$((failed + 1)); continue
  fi
  ok=$((ok + 1))
  echo "[OK   ] tool=$tool"
done

echo "============================================================"
echo "[DONE] total=$total ok=$ok failed=$failed"
[[ "$failed" -gt 0 ]] && exit 1
exit 0
