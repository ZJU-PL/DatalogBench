#!/usr/bin/env bash
# Resolve the grading image once, at the start of a grid, and pin it by content.
#
# Reading `agent_container.IMAGE` on every invocation would make the grading
# image whatever the working tree says at the moment each cell reaches its
# evaluation step. If IMAGE changed between cells and both tags existed
# locally, earlier and later cells of one grid would be graded by different
# compilers and nothing would say so.
#
# Pinning the tag is not enough. A tag is a mutable pointer: `docker load` of a
# rebuilt image reuses it, and the name in the log stays identical while the
# bits underneath change. So this resolves the tag to its image ID and every
# later container runs *by ID*. A mid-run rebuild then cannot substitute
# itself, and a mid-run edit to agent_container.py cannot either.
#
# Sourced, not executed:
#     source docker/freeze-image.sh    # exports HARNESS_IMAGE, HARNESS_IMAGE_ID
#
# Honours an already-set HARNESS_IMAGE so an operator can pin a specific tag on
# the command line, which is how the 09-07 recovery run was restarted.
# GRADE_ON_HOST=1 skips the whole thing: there is no image to freeze.

if [ "${GRADE_ON_HOST:-0}" = "1" ]; then
  echo "[IMAGE] GRADE_ON_HOST=1 -- grading on the host, no image to freeze."
  echo "[IMAGE] Only sound when --verify has shown the host and image Souffle"
  echo "        builds to be identical; the run records grading_host=host."
  return 0 2>/dev/null || exit 0
fi

if [ -z "${HARNESS_IMAGE:-}" ]; then
  HARNESS_IMAGE="$(python3 -c 'import sys;sys.path.insert(0,"synthesis");import agent_container as a;print(a.IMAGE)')" || {
    echo "[ERROR] could not read agent_container.IMAGE" >&2
    return 2 2>/dev/null || exit 2
  }
fi

HARNESS_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$HARNESS_IMAGE" 2>/dev/null)" || true
if [ -z "$HARNESS_IMAGE_ID" ]; then
  # Distinguish the three ways this fails, because the fix differs and a wrong
  # hint costs more than no hint. "Load the image" is useless advice when the
  # daemon is down.
  if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] docker is not installed, so the grading image cannot be pinned." >&2
  elif ! docker info >/dev/null 2>&1; then
    echo "[ERROR] the docker daemon is not running, so the grading image cannot be pinned." >&2
  else
    echo "[ERROR] grading image $HARNESS_IMAGE is not present locally." >&2
    echo "        Load it: bash docker/load-images.sh dlb-images-linux-amd64.tar.gz" >&2
  fi
  echo "        Grading must not silently fall back to the host compiler: set" >&2
  echo "        GRADE_ON_HOST=1 deliberately if that is what you want, and only" >&2
  echo "        after --verify has shown the two Souffle builds to be identical." >&2
  return 2 2>/dev/null || exit 2
fi

export HARNESS_IMAGE HARNESS_IMAGE_ID
echo "[IMAGE] frozen for this grid: $HARNESS_IMAGE"
echo "[IMAGE] id: $HARNESS_IMAGE_ID"
echo "[IMAGE] every cell runs by id, so a retag or a rebuild mid-run cannot"
echo "        change the compiler behind the numbers -- it stops the grid instead."
