#!/bin/sh
# Materialise the agent's CLI configuration, then hand over to the real command.
#
# HOME is a fresh tmpfs on every run. The harness mounts only the active CLI's
# session subdirectory from a per-case scratch directory; credentials and
# preferences remain on tmpfs, while session history survives just long enough
# for continuation within that case. No state can carry to the next task.
#
# The key arrives in the environment and is written to files on the tmpfs. It is
# never part of an image layer: the image is meant to be shared, and a
# credential baked into a layer is shared with it.
set -eu

: "${AGENT_API_KEY:?AGENT_API_KEY must be passed into the container}"

/opt/dlb/codex.sh  >/dev/null
/opt/dlb/claude.sh >/dev/null

exec "$@"
