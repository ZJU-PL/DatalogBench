#!/usr/bin/env bash
# Load prebuilt images on the server and bring the isolation up. No image build,
# and nothing pulled: everything the container needs arrived in the tarball.
#
# Usage: bash docker/load-images.sh dlb-images-linux-amd64.tar.gz
set -euo pipefail

cd "$(dirname "$0")/.."
TARBALL="${1:-dlb-images-linux-amd64.tar.gz}"

[ -f "$TARBALL" ] || { echo "[ERROR] $TARBALL not found" >&2; exit 2; }

AGENT_IMAGE="$(python3 -c 'import sys;sys.path.insert(0,"synthesis");import agent_container as a;print(a.IMAGE)')"
PROXY_IMAGE="$(python3 -c 'import sys;sys.path.insert(0,"synthesis");import agent_container as a;print(a.PROXY_IMAGE)')"

echo "==> loading $TARBALL"
gunzip -c "$TARBALL" | docker load

# Same architecture check as the export side, because this is the machine where
# a mismatch actually costs something.
HOST_ARCH="$(docker info --format '{{.OSType}}/{{.Architecture}}' 2>/dev/null || echo unknown)"
case "$HOST_ARCH" in
  linux/x86_64|linux/amd64) WANT=linux/amd64 ;;
  linux/aarch64|linux/arm64) WANT=linux/arm64 ;;
  *) WANT="" ;;
esac
for img in "$AGENT_IMAGE" "$PROXY_IMAGE"; do
  got="$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$img")"
  echo "    $img  $got"
  if [ -n "$WANT" ] && [ "$got" != "$WANT" ]; then
    echo "[ERROR] $img is $got but this host is $WANT." >&2
    echo "        Re-export with TARGET_PLATFORM=$WANT on the build machine." >&2
    exit 1
  fi
done

if [ -f "$TARBALL.digests" ]; then
  echo "==> verifying digests against the build machine's record"
  while read -r want tag; do
    got="$(docker image inspect --format '{{.Id}}' "$tag")"
    if [ "$got" = "$want" ]; then
      echo "    OK   $tag"
    elif python3 - "$TARBALL" "$want" "$tag" "$got" <<'PY'
import hashlib
import json
import sys
import tarfile

archive, want, tag, got = sys.argv[1:]
algorithm, sep, digest = want.partition(":")
if sep != ":" or algorithm != "sha256":
    raise SystemExit(1)

# With Docker's containerd image store, `docker image inspect .Id` on the
# build machine can be the OCI manifest digest; after `docker load` into the
# classic store it is the config digest.  Prove that the recorded manifest is
# present, intact, points at the loaded config, and belongs to the requested
# tag instead of treating those two legitimate IDs as a mismatch.
with tarfile.open(archive, "r:*") as tf:
    member = f"blobs/sha256/{digest}"
    manifest_bytes = tf.extractfile(member).read()
    if hashlib.sha256(manifest_bytes).hexdigest() != digest:
        raise SystemExit(1)
    oci_manifest = json.loads(manifest_bytes)
    if oci_manifest.get("config", {}).get("digest") != got:
        raise SystemExit(1)

    saved = json.load(tf.extractfile("manifest.json"))
    config_path = "blobs/sha256/" + got.removeprefix("sha256:")
    if not any(tag in item.get("RepoTags", []) and item.get("Config") == config_path
               for item in saved):
        raise SystemExit(1)
PY
    then
      echo "    OK   $tag (OCI manifest -> $got)"
    else
      echo "[ERROR] $tag is $got, the build machine recorded $want" >&2
      exit 1
    fi
  done < "$TARBALL.digests"
fi

echo "==> networks"
docker network inspect dlb-agent-int >/dev/null 2>&1 \
  || docker network create --internal dlb-agent-int
docker network inspect dlb-agent-out >/dev/null 2>&1 \
  || docker network create dlb-agent-out

echo "==> egress proxy (allowlist generated from CLI configuration)"
python3 synthesis/agent_container.py --start-proxy

echo "==> host compiler"
command -v souffle >/dev/null 2>&1 || {
  echo "[ERROR] souffle is not on PATH. It grades every reported number, and the" >&2
  echo "        container's copy is the agent's tool, not the grader. Install it" >&2
  echo "        (see 'Quick Start' in the README) before running anything." >&2
  exit 1
}
souffle --version | sed -n '2,3p' | sed 's/^/  /'

echo "==> verification"
[ -n "${EVAL_API_KEY:-}" ] || {
  echo "[ERROR] EVAL_API_KEY is not set." >&2
  exit 1
}
python3 synthesis/agent_container.py --verify
