#!/usr/bin/env bash

set -u

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# The grid is derived from evaluation/model_inventory.py, the single place the
# model list lives. Repeating it here is what let the shell script and the
# Python tools drift apart before. MODELS_OVERRIDE permits a scoped recovery
# without starting already-complete cells from other providers.
if [[ -n "${MODELS_OVERRIDE:-}" ]]; then
  read -r -a MODELS <<< "$MODELS_OVERRIDE"
else
  read -r -a MODELS <<< "$(python3 evaluation/model_inventory.py)"
fi
if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "[ERROR] Could not read the model inventory (evaluation/model_inventory.py)."
  exit 2
fi

# Keys come from the environment, never from an argument: a key on a command
# line is readable by every process on the host through `ps`. Resolve the key
# variable per selected model because DeepSeek uses DS_API_KEY while the other
# evaluated providers use EVAL_API_KEY.
if ! REQUIRED_KEY_ENVS="$(python3 - "${MODELS[@]}" <<'PYKEYS'
import sys
sys.path.insert(0, "evaluation")
from model_inventory import MODELS, api_key_env_for

selected = sys.argv[1:]
unknown = [model for model in selected if model not in MODELS]
if unknown:
    print("[ERROR] Model(s) outside the evaluated inventory: " + ", ".join(unknown),
          file=sys.stderr)
    raise SystemExit(2)
print(" ".join(dict.fromkeys(api_key_env_for(model) for model in selected)))
PYKEYS
)"; then
  exit 2
fi
read -r -a REQUIRED_KEY_ENV_LIST <<< "$REQUIRED_KEY_ENVS"
for key_env in "${REQUIRED_KEY_ENV_LIST[@]}"; do
  if [[ -z "${!key_env:-}" ]]; then
    echo "[ERROR] Missing API key. Please export $key_env."
    exit 2
  fi
done

METHODS=("signature" "description")
FEWSHOTS=(0 1)
DATASETS=("all")

# Examples in the prompt are a two-level factor (0 or 1). Each case holds out
# exactly one demonstration input (`<case>/demo/0`), disjoint from the variants
# it is scored on, so a 2-shot cell would need a second held-out input on every
# case. synth_llm.py refuses a shot count the demo pool cannot supply rather
# than quietly sending a shorter prompt.

# A case's knowledge field is always part of its prompt; there is no
# with/without-knowledge arm. Whether domain knowledge matters is answered
# observationally by slicing results on spec_tier (evaluation/tier_report.py).

# pass@1 protocol: greedy decoding. Default to a single run per cell -- the
# grid is the research axis and inference is over the paired cases (across-case
# CI + paired permutation test), so k=1 suffices.
TEMPERATURE="${TEMPERATURE:-0}"
NUM_RUNS="${NUM_RUNS:-1}"
RESUME="${RESUME:-0}"
MAX_PROMPT_CHARS="${MAX_PROMPT_CHARS:-500000}"

# Preflight: the demonstration pool must be able to serve the largest requested
# shot count. Checked before any inference, because the failure this guards
# against is not loud -- a few-shot prompt with no examples is a well-formed
# prompt, and its results read as "few-shot does not help".
MAX_SHOTS=0
for s in "${FEWSHOTS[@]}"; do (( s > MAX_SHOTS )) && MAX_SHOTS=$s; done
if ! python3 - "$MAX_SHOTS" "$MAX_PROMPT_CHARS" <<'PYCHK'
import sys, pathlib
sys.path.insert(0, "evaluation")
sys.path.insert(0, "synthesis")
from common_eval import case_demo_dirs
from synth_llm import load_or_build_context, prepare_prompts
import json
want = int(sys.argv[1])
max_prompt_chars = int(sys.argv[2])
if want == 0:
    sys.exit(0)
short = [c["id"] for c in json.load(open("benchmark/dataset.json"))
         if len(case_demo_dirs(c["id"])) < want]
if short:
    print(f"[ERROR] {len(short)} case(s) cannot supply {want} demonstration example(s): "
          f"{', '.join(short[:5])}{' ...' if len(short) > 5 else ''}")
    print("        Lower FEWSHOTS, or add inputs under <case>/demo/. Never point few-shot")
    print("        at the eval pool: those inputs are scored, and showing one leaks the oracle.")
    sys.exit(1)
print(f"[PREFLIGHT] demonstration pool serves up to {want}-shot for every case")
tasks = load_or_build_context("all")
for method in ("signature", "description"):
    for fewshot in (0, 1):
        prepare_prompts(tasks, method, fewshot, max_prompt_chars)
PYCHK
then
  echo "[ERROR] Aborting before any model call."
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

total=0
ok=0
failed=0

for model in "${MODELS[@]}"; do

  for method in "${METHODS[@]}"; do
    for fewshot in "${FEWSHOTS[@]}"; do
      for dataset in "${DATASETS[@]}"; do
        total=$((total + 1))
        echo "============================================================"
        echo "[RUN $total] model=$model method=$method fewshot=$fewshot dataset=$dataset"

        synth_cmd=(
          python3 synthesis/synth_llm.py
          --model "$model"
          --method "$method"
          --fewshot "$fewshot"
          --dataset "$dataset"
          --temperature "$TEMPERATURE"
          --num_runs "$NUM_RUNS"
          --max_prompt_chars "$MAX_PROMPT_CHARS"
        )
        if [[ "$RESUME" == "1" ]]; then
          synth_cmd+=(--resume)
        fi

        eval_cmd=(
          python3 evaluation/eval_llm.py
          --model "$model"
          --method "$method"
          --fewshot "$fewshot"
          --dataset "$dataset"
          --num_runs "$NUM_RUNS"
        )

        echo "[SYNTH] ${synth_cmd[*]}"
        if ! "${synth_cmd[@]}"; then
          echo "[FAIL][SYNTH] model=$model method=$method fewshot=$fewshot dataset=$dataset"
          failed=$((failed + 1))
          echo "[STOP] fail-fast: fix/resume this cell before starting later cells"
          exit 1
        fi

        echo "[EVAL ] ${eval_cmd[*]}"
        if ! grade "${eval_cmd[@]}"; then
          echo "[FAIL][EVAL ] model=$model method=$method fewshot=$fewshot dataset=$dataset"
          failed=$((failed + 1))
          echo "[STOP] fail-fast: invalid synthesis artifacts must not be scored"
          exit 1
        fi

        ok=$((ok + 1))
        echo "[OK   ] model=$model method=$method fewshot=$fewshot dataset=$dataset"
      done
    done
  done
done

echo "============================================================"
echo "[DONE] total=$total ok=$ok failed=$failed"
if [[ "$failed" -gt 0 ]]; then
  exit 1
fi

exit 0
