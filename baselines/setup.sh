#!/usr/bin/env bash
#
# One-shot setup for the symbolic-synthesizer baselines: init the submodules,
# apply the committed patches that make them build on a modern/macOS toolchain,
# then build each tool. Re-runnable (patches are applied idempotently).
#
# The upstream tools are never vendored: GenSynth's repository declares no
# license, so the repository references all three by URL and pinned commit and
# fetches them here. In a git checkout they are submodules; in a copy without
# .git, the pins below are the only record of which commit to build -- keep them
# in step with .gitmodules.
#
# The patches live in baselines/patches/ and are the ONLY source modifications
# we depend on (submodule content itself is never committed to the parent repo):
#   - gensynth-pool-size.patch     : bounds worker pools by requested populations
#   - egs-build.properties.patch   : pins sbt 1.x so sbt-assembly 0.14.10 resolves
#   - prosynth-configure.ac.patch  : drops a duplicate AC_CONFIG_MACRO_DIR that
#                                    modern autoconf rejects
#
# DLB_FETCH_ONLY=1 stops after fetching and patching, which is what the artifact
# build verifies: the upstream trees must NOT be present in the shipped bundle
# (their build output embeds the builder's own home directory).
#
# Build prerequisites (install yourself; see README):
#   EGS      : sbt, and a JDK between 11 and 17 -- JDK 8 lacks String.strip and
#              JDK 18+ rejects the SecurityManager that sbt 1.3.13 installs.
#              The reported runs were built on Java 11.0.32.
#   ProSynth : python z3 (`pip install z3-solver`) and the Souffle build chain
#              (`brew install autoconf automake libtool mcpp bison flex`)

set -u

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BASELINES="$ROOT_DIR/baselines"
PATCHES="$BASELINES/patches"

cd "$ROOT_DIR"

# name|url|pinned commit -- must match .gitmodules and the recorded submodule ids
UPSTREAM=(
  "gensynth|https://github.com/jonomendelson/gensynth|829dc8cceba6d98dc45e7739d314f8456e712c3c"
  "egs|https://github.com/aalok-thakkar/egs-artifact|c117977845b628b09b59d9b77a4a40f53b1da059"
  "prosynth|https://github.com/petablox/popl2020-artifact|f51cda5d7034b0597e8d834443605afea69cc694"
)

# fetch_upstream <name> <url> <sha>
# Three cases: already present (leave it alone), inside a git checkout (use the
# submodule machinery), or a plain directory tree such as the released artifact
# (clone and check out the pin). Failing to obtain a tool is fatal: the old
# behaviour was to warn and carry on, which ended in "== done ==" with nothing
# built.
fetch_upstream() {
  local name="$1" url="$2" sha="$3" dir="$BASELINES/$1"
  if [[ -e "$dir/.git" ]]; then
    echo "[OK]   present: $name ($(git -C "$dir" rev-parse --short HEAD))"
    return 0
  fi
  if git -C "$ROOT_DIR" rev-parse --git-dir >/dev/null 2>&1; then
    git -C "$ROOT_DIR" submodule update --init --recursive "baselines/$name" || {
      echo "[FAIL] submodule init failed: $name"; return 1; }
  else
    echo "[CLONE] $name <- $url @ ${sha:0:12}"
    rm -rf "$dir"
    git clone --quiet "$url" "$dir" || { echo "[FAIL] clone failed: $name"; return 1; }
    git -C "$dir" checkout --quiet "$sha" || { echo "[FAIL] pin not found: $name $sha"; return 1; }
    git -C "$dir" submodule update --init --recursive --quiet || true
  fi
  echo "[OK]   $name at $(git -C "$dir" rev-parse --short HEAD)"
}

echo "== 1. fetch upstream tools at their pinned commits =="
missing=0
for entry in "${UPSTREAM[@]}"; do
  IFS='|' read -r name url sha <<< "$entry"
  fetch_upstream "$name" "$url" "$sha" || missing=1
done
if (( missing )); then
  echo "[FATAL] at least one upstream tool could not be obtained; see the messages above." >&2
  exit 1
fi

# apply_patch <submodule_worktree_dir> <patch_file>
# Idempotent: skips if the patch is already applied (reverse-applies cleanly).
apply_patch() {
  local dir="$1" patch="$2"
  if [[ ! -f "$patch" ]]; then
    echo "[SKIP] patch not found: $patch"; return 0
  fi
  if git -C "$dir" apply --reverse --check "$patch" >/dev/null 2>&1; then
    echo "[OK]   already applied: $(basename "$patch")"; return 0
  fi
  if git -C "$dir" apply --check "$patch" >/dev/null 2>&1; then
    git -C "$dir" apply "$patch" && echo "[APPLY] $(basename "$patch") -> $dir"
  else
    echo "[WARN] cannot apply $(basename "$patch") to $dir (already patched differently?)"
  fi
}

echo "== 2. apply patches =="
apply_patch "$BASELINES/gensynth" "$PATCHES/gensynth-pool-size.patch"
apply_patch "$BASELINES/egs" "$PATCHES/egs-build.properties.patch"
apply_patch "$BASELINES/prosynth" "$PATCHES/prosynth-configure.ac.patch"

if [[ "${DLB_FETCH_ONLY:-0}" == "1" ]]; then
  echo "== fetch-only requested; stopping before the builds =="
  exit 0
fi

echo "== 3. build EGS (needs sbt and a JDK between 11 and 17) =="
if command -v sbt >/dev/null 2>&1; then
  egs_java_major="$(java -version 2>&1 | sed -n '1s/.*version "\([0-9]*\).*/\1/p')"
  if [[ -n "$egs_java_major" ]] && { (( egs_java_major < 11 )) || (( egs_java_major > 17 )); }; then
    echo "[WARN] java $egs_java_major is outside the 11-17 range sbt 1.3.13 builds under;"
    echo "       set JAVA_HOME to a JDK 11-17 before this step (see README)."
  fi
  ( cd "$BASELINES/egs/egs" && sbt assembly ) && echo "[OK] EGS jar built" \
    || echo "[WARN] EGS build failed — check the JDK is in the 11-17 range"
else
  echo "[SKIP] sbt not found; install it and re-run, or build manually (see README)"
fi

echo "== 4. build ProSynth patched Souffle (needs autotools/bison/flex) =="
PROSYNTH_SOUFFLE_DIR="$BASELINES/prosynth/prosynth/souffle"   # finalized by setup; see README
if [[ -d "$PROSYNTH_SOUFFLE_DIR" ]] && command -v autoconf >/dev/null 2>&1; then
  export PATH="/opt/homebrew/opt/bison/bin:/opt/homebrew/opt/flex/bin:/opt/homebrew/opt/libtool/libexec/gnubin:$PATH"
  ( cd "$PROSYNTH_SOUFFLE_DIR" && ./bootstrap && ./configure CXXFLAGS="-std=c++17 -Wno-error" && make -j4 ) \
    && echo "[OK] patched Souffle built" \
    || echo "[WARN] Souffle build failed — see README build notes"
else
  echo "[SKIP] autotools missing or souffle dir absent; see README for the ProSynth build"
fi
python3 -c "import z3" 2>/dev/null && echo "[OK] z3 python present" \
  || echo "[NOTE] ProSynth also needs z3: pip install z3-solver"

echo "== done =="
