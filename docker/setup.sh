#!/usr/bin/env bash
# One-command bring-up of the agent isolation. Run from the repository root.
#
# It is one script rather than a list of steps in a README because the failure
# this setup already had came from a step being right in one place and stale in
# another. Everything here is idempotent; run it again after changing an endpoint
# or a pinned version.
set -euo pipefail

CODEX_VERSION="${CODEX_VERSION:-0.153.4}"
CLAUDE_VERSION="${CLAUDE_VERSION:-2.1.261}"

cd "$(dirname "$0")/.."

echo "==> networks"
docker network inspect dlb-agent-int >/dev/null 2>&1 \
  || docker network create --internal dlb-agent-int
docker network inspect dlb-agent-out >/dev/null 2>&1 \
  || docker network create dlb-agent-out

AGENT_IMAGE="$(python3 -c 'import sys;sys.path.insert(0,"synthesis");import agent_container as a;print(a.IMAGE)')"
PROXY_IMAGE="$(python3 -c 'import sys;sys.path.insert(0,"synthesis");import agent_container as a;print(a.PROXY_IMAGE)')"

# Images that are already present are left alone. On a run machine they normally
# arrived through docker/load-images.sh, and rebuilding them there would defeat
# the point of shipping them: the image pins a Souffle build and two CLI builds,
# and a second build on another machine at another time is how a reported number
# and the tool that produced it drift apart. Pass REBUILD=1 to force one.
if [ "${REBUILD:-0}" != "1" ] \
   && docker image inspect "$AGENT_IMAGE" >/dev/null 2>&1 \
   && docker image inspect "$PROXY_IMAGE" >/dev/null 2>&1; then
  echo "==> images already present, not rebuilding (REBUILD=1 to force)"
  docker image inspect --format '    {{index .RepoTags 0}}  {{.Os}}/{{.Architecture}}  {{.Id}}' \
    "$AGENT_IMAGE" "$PROXY_IMAGE"
else
  echo "==> building images (codex $CODEX_VERSION, claude $CLAUDE_VERSION)"
  docker build -f docker/egress-proxy.Dockerfile -t "$PROXY_IMAGE" .
  docker build -f docker/agent.Dockerfile -t "$AGENT_IMAGE" \
    --build-arg "CODEX_VERSION=$CODEX_VERSION" \
    --build-arg "CLAUDE_VERSION=$CLAUDE_VERSION" .
fi

echo "==> egress proxy (allowlist generated from CLI configuration)"
python3 synthesis/agent_container.py --start-proxy

echo "==> host compiler"
if ! command -v souffle >/dev/null 2>&1; then
  echo "  souffle is NOT on PATH. The harness grades with it, so install it before"
  echo "  running anything -- the container's copy is the agent's, not the grader's." >&2
  exit 1
fi
souffle --version | sed -n '2,3p' | sed 's/^/  /'

echo "==> verification"
if [ -z "${EVAL_API_KEY:-}" ]; then
  echo "  EVAL_API_KEY is not set; skipping --verify."
  echo "  export EVAL_API_KEY=<zzz key> and re-run this script." >&2
  exit 1
fi
python3 synthesis/agent_container.py --verify
