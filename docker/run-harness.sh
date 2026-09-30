#!/usr/bin/env bash
# Run a harness command inside the pinned image, against the whole repository.
#
# This exists because Souffle decides every reported number, and until now the
# copy that decided them was whatever the host happened to have installed. That
# is how a 32-bit host and a 64-bit image came to disagree, and how a reference
# program's expected output came to encode integer wraparound. Running the
# harness in the image makes the compiler the same pinned binary the image
# carries, rather than a property of the machine.
#
# Scope: the steps whose only external tool is Souffle -- qa_check,
# regen_expected, the eval_* scripts, mutation_score, negative_examples,
# input_generator, select_variants. Synthesis is NOT in scope and cannot be:
# synth_llm needs the `openai` package and the evaluation endpoint, and
# synth_coding_agent issues `docker` commands of its own, so running it in here
# would mean giving a container the host's Docker socket. Its Souffle is instead
# held to the image's by `agent_container.verify()`, which compares the two full
# version strings and fails on any difference.
#
# The mount is the whole repository -- the opposite of the agent container,
# which mounts only a scratch directory so the reference programs are absent.
# Grading needs them; the agent must not have them.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO="$(pwd -P)"

IMAGE="${HARNESS_IMAGE:-$(python3 -c 'import sys;sys.path.insert(0,"synthesis");import agent_container as a;print(a.IMAGE)')}"

if [ $# -eq 0 ]; then
  echo "usage: bash docker/run-harness.sh <command...>" >&2
  echo "   e.g. bash docker/run-harness.sh python3 benchmark/qa_check.py" >&2
  exit 2
fi

docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "[ERROR] image $IMAGE is not present. Load it first:" >&2
  echo "        bash docker/load-images.sh dlb-images-linux-amd64.tar.gz" >&2
  exit 2
}

# A grid that ran `docker/freeze-image.sh` pinned the image by content, not by
# name. Honour that pin here rather than trusting the tag: a tag is a mutable
# pointer, so `docker load` of a rebuilt image keeps the name identical while
# the compiler underneath changes, and every log line would still read the same.
# Comparing the resolved id catches that, and running by id means the container
# started below is the one the grid started with even if the tag has since
# moved. Drift stops the run: half a grid graded by one compiler and half by
# another is worse than no grid, because nothing in the output would show it.
if [ -n "${HARNESS_IMAGE_ID:-}" ]; then
  now="$(docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null || true)"
  if [ "$now" != "$HARNESS_IMAGE_ID" ]; then
    echo "[ERROR] grading image drifted mid-run." >&2
    echo "        tag:    $IMAGE" >&2
    echo "        frozen: $HARNESS_IMAGE_ID" >&2
    echo "        now:    ${now:-<gone>}" >&2
    echo "        Cells already graded used the frozen image; continuing would mix" >&2
    echo "        two compilers in one grid. Restart the grid with RESUME=1 so every" >&2
    echo "        cell is scored by one image, or re-pin with HARNESS_IMAGE." >&2
    exit 2
  fi
  IMAGE="$HARNESS_IMAGE_ID"
fi

# The evaluators write which image graded a cell into their provenance sidecar,
# and they can only write what reaches them. Resolve the identity here and pass
# it in on every invocation, not just when a grid froze it: the documented way
# to grade one cell is a bare `run-harness.sh` call, and that is precisely the
# path that would leave the field null. A provenance field that is empty in the
# common case records nothing and is worse than none, because its presence
# suggests the question was answered.
RUN_IMAGE_TAG="${HARNESS_IMAGE:-$IMAGE}"
RUN_IMAGE_ID="${HARNESS_IMAGE_ID:-$(docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null || true)}"

# --network none: grading reaches nothing, and cannot quietly acquire a
#   dependency on something remote.
# --user: outputs land in the repository owned by the caller, not by root.
# --entrypoint "": the image's entrypoint configures the agent CLIs and wants a
#   key; none of that applies here.
exec docker run --rm -i \
  --network none \
  -v "$REPO:/repo" \
  -w /repo \
  --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,exec,mode=1777 \
  -e HOME=/tmp \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e "HARNESS_IMAGE=$RUN_IMAGE_TAG" \
  -e "HARNESS_IMAGE_ID=$RUN_IMAGE_ID" \
  --entrypoint "" \
  "$IMAGE" "$@"
