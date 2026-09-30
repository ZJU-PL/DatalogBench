#!/usr/bin/env bash

# Primary coding-agent matrix: {codex, claude} x {signature} x zero-shot.
# Set METHODS="signature description" only for a separately declared
# schema-by-agent interaction study; description is not part of the main grid.
#
# The agents run inside the isolation container, and this script refuses to
# start unless the container is verified first. That order is deliberate: the
# container is what makes the reference programs absent rather than merely
# unread, and a run that quietly fell back to the host would produce numbers
# labelled as isolated while carrying the isolation the container replaced.

set -u

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# Coding-agent inference has a credential independent of Direct prompting.
if [[ -z "${AGENT_API_KEY:-}" ]]; then
  echo "[ERROR] Missing agent inference key. Please export AGENT_API_KEY."
  echo "        See docker/README.md."
  exit 2
fi

read -r -a AGENTS <<< "${AGENTS:-codex claude}"
read -r -a METHODS <<< "${METHODS:-signature}"
DATASETS=("all")

TEMPERATURE="${TEMPERATURE:-0}"
NUM_RUNS="${NUM_RUNS:-1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-4}"
TIMEOUT_SEC="${TIMEOUT_SEC:-600}"
RESUME="${RESUME:-0}"

# Preflight. Checked before any model call, because every failure it guards
# against is silent: a stale allowlist, an image built against a different
# endpoint, or a compiler whose integer width disagrees with the grader's all
# produce a run that looks normal and means something else.
echo "[PREFLIGHT] verifying container isolation"
if ! python3 synthesis/agent_container.py --verify; then
  echo "[ERROR] Container verification failed. Aborting before any model call."
  echo "        Fix the setup (bash docker/setup.sh) rather than passing"
  echo "        --no-container: host runs are not agent results."
  exit 2
fi

# Synthesis runs on the host: it needs the `openai` package and the evaluation
# endpoint. Grading runs in the pinned image, so the number a cell reports comes
# from a fixed compiler rather than from whatever this machine has installed.
# GRADE_ON_HOST=1 falls back, which is only sensible when docker is unavailable
# and `--verify` has shown the two Souffle builds to be identical anyway.
# The image is resolved and pinned by content once, here, rather than re-read
# from the working tree at each cell's evaluation step. See docker/freeze-image.sh
# for what went wrong when it was not.
source docker/freeze-image.sh || exit 2

grade() {
  if [ "${GRADE_ON_HOST:-0}" = "1" ]; then
    "$@"
  else
    bash docker/run-harness.sh "$@"
  fi
}

total=0; ok=0; failed=0

for agent in "${AGENTS[@]}"; do
  for method in "${METHODS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      total=$((total + 1))
      echo "============================================================"
      echo "[RUN $total] agent=$agent method=$method dataset=$dataset"

      synth_cmd=(
        python3 synthesis/synth_coding_agent.py
        --agent "$agent"
        --method "$method"
        --dataset "$dataset"
        --temperature "$TEMPERATURE"
        --num_runs "$NUM_RUNS"
        --max_iterations "$MAX_ITERATIONS"
        --timeout_sec "$TIMEOUT_SEC"
      )
      if [[ "$RESUME" == "1" ]]; then
        synth_cmd+=(--resume)
      fi

      eval_cmd=(
        python3 evaluation/eval_coding_agent.py
        --agent "$agent"
        --method "$method"
        --dataset "$dataset"
        --num_runs "$NUM_RUNS"
      )

      echo "[SYNTH] ${synth_cmd[*]}"
      "${synth_cmd[@]}"
      synth_rc=$?
      if [[ "$synth_rc" -ne 0 ]]; then
        echo "[FAIL][SYNTH] agent=$agent method=$method dataset=$dataset"
        failed=$((failed + 1))
        if [[ "$synth_rc" -eq 3 ]]; then
          echo "[ABORT][QUOTA] Stop the entire grid; replenish AGENT_API_KEY "
          echo "               and restart with RESUME=1."
          exit 3
        fi
        if [[ "$synth_rc" -eq 4 ]]; then
          echo "[ABORT][PROVIDER] The account/organization is disabled; stop the "
          echo "                  entire grid and retry only after a successful smoke."
          exit 4
        fi
        continue
      fi

      echo "[EVAL ] ${eval_cmd[*]}"
      if ! grade "${eval_cmd[@]}"; then
        echo "[FAIL][EVAL ] agent=$agent method=$method dataset=$dataset"
        failed=$((failed + 1)); continue
      fi

      ok=$((ok + 1))
      echo "[OK   ] agent=$agent method=$method dataset=$dataset"
    done
  done
done

echo "============================================================"
echo "[DONE] total=$total ok=$ok failed=$failed"
[[ "$failed" -gt 0 ]] && exit 1
exit 0
