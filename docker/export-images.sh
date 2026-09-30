#!/usr/bin/env bash
# Build the images for the SERVER's architecture and pack them into one file.
#
# Run this on a workstation; copy the result to the server and load it there.
# The server then needs no image build, which matters beyond convenience: the
# agent image compiles or fetches a pinned Souffle and installs pinned CLI
# builds, and doing that a second time on a different machine at a different
# time is exactly how a result and the tool that produced it come apart.
#
# Two things this handles that a plain `docker build && docker save` does not:
#
#   1. Cross-architecture. A workstation may be arm64 while the server is
#      x86-64, and an image is not portable between them. TARGET_PLATFORM is
#      explicit rather than inherited from whoever runs this.
#   2. Attestation manifests. BuildKit attaches provenance/SBOM by default,
#      which turns the result into a manifest list; `docker load` on the far end
#      can then land you with an image that has no runnable platform. Both are
#      switched off.
set -euo pipefail

cd "$(dirname "$0")/.."

TARGET_PLATFORM="${TARGET_PLATFORM:-linux/amd64}"
CODEX_VERSION="${CODEX_VERSION:-0.153.4}"
CLAUDE_VERSION="${CLAUDE_VERSION:-2.1.261}"
OUT="${OUT:-dlb-images-${TARGET_PLATFORM//\//-}.tar.gz}"

AGENT_IMAGE="$(python3 -c 'import sys;sys.path.insert(0,"synthesis");import agent_container as a;print(a.IMAGE)')"
PROXY_IMAGE="$(python3 -c 'import sys;sys.path.insert(0,"synthesis");import agent_container as a;print(a.PROXY_IMAGE)')"

echo "==> building for $TARGET_PLATFORM"
echo "    $AGENT_IMAGE (codex $CODEX_VERSION, claude $CLAUDE_VERSION)"
echo "    $PROXY_IMAGE"

docker build --platform "$TARGET_PLATFORM" --provenance=false --sbom=false \
  -f docker/egress-proxy.Dockerfile -t "$PROXY_IMAGE" .
docker build --platform "$TARGET_PLATFORM" --provenance=false --sbom=false \
  -f docker/agent.Dockerfile -t "$AGENT_IMAGE" \
  --build-arg "CODEX_VERSION=$CODEX_VERSION" \
  --build-arg "CLAUDE_VERSION=$CLAUDE_VERSION" .

# Refuse to ship an image for the wrong machine. Silent here means a load that
# succeeds on the server and an `exec format error` at the first agent call.
for img in "$AGENT_IMAGE" "$PROXY_IMAGE"; do
  got="$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$img")"
  if [ "$got" != "$TARGET_PLATFORM" ]; then
    echo "[ERROR] $img is $got, expected $TARGET_PLATFORM" >&2
    exit 1
  fi
done

echo "==> saving to $OUT"
docker save "$AGENT_IMAGE" "$PROXY_IMAGE" | gzip -1 > "$OUT"

# A digest for each image, so the server can prove it loaded what was built
# rather than something that happened to carry the same tag.
docker image inspect --format '{{.Id}}  {{index .RepoTags 0}}' \
  "$AGENT_IMAGE" "$PROXY_IMAGE" > "$OUT.digests"

# Souffle also has to exist on the host, because synthesis runs there (it needs
# the openai package, and the coding-agent path issues its own docker commands,
# so it cannot itself live in a container). Ship the very package the image
# installs, so the host build is the image build rather than whatever the
# server's package index happens to offer -- `--verify` compares the two full
# version strings and fails if they differ.
if [ "$TARGET_PLATFORM" = "linux/amd64" ]; then
  SOUFFLE_VERSION="${SOUFFLE_VERSION:-2.5}"
  DEB="souffle-${SOUFFLE_VERSION}-amd64.deb"
  if [ ! -f "$DEB" ]; then
    echo "==> fetching the host Souffle package ($DEB)"
    curl -fsSL -o "$DEB" \
      "https://github.com/souffle-lang/souffle/releases/download/${SOUFFLE_VERSION}/x86_64-ubuntu-2204-souffle-${SOUFFLE_VERSION}-Linux.deb"
  fi
  echo "  $DEB  ($(du -h "$DEB" | cut -f1))  -- install this on the server host"
fi

echo
echo "  $OUT  ($(du -h "$OUT" | cut -f1))"
cat "$OUT.digests"
echo
echo "Copy the tarball, its .digests, and the .deb to the server, then run there:"
echo "  sudo apt-get install -y ./souffle-*-amd64.deb   # the grader, on the host"
echo "  bash docker/load-images.sh $OUT"
