# Isolation for the coding-agent setting.
#
# Isolation used to be assembled from whatever switch each CLI happened to
# expose, and they expose different things: codex has a seatbelt sandbox, claude
# has none and relies on a permission rule, and Antigravity exposes no tool
# control at all. The constraints therefore differed per agent, which is why
# tab:agent cannot mark a best cell -- a gap between two rows could be a gap in
# scaffolding or a gap in what we were able to switch off. Moving the boundary
# into a container makes it the same boundary for every agent.
#
# Two channels this closes, both measured open beforehand:
#
#   1. Reading a reference program by absolute path. A scratch working directory
#      only removes the relative path. `workspace-write` restricts writes and
#      network, not reads, so `cat /abs/path/benchmark/query/Path.dl` printed the
#      reference program. The container closes this by construction rather than
#      by permission: benchmark/query and the eval inputs are never mounted, so
#      there is no path to read. Prompts arrive on stdin, so nothing from the
#      repository needs to be inside.
#   2. A shell that can reach the network. claude's Bash is scoped to souffle by
#      a permission rule, and a rule is not a boundary.
#
# The image pins the compiler and the CLI versions, which also answers the
# reproducibility limitation in conclu.tex: the agent row is currently the least
# reproducible of the three settings because it records only what was requested,
# not what was served.

FROM debian:bookworm-slim

ARG SOUFFLE_VERSION=2.5
ARG NODE_MAJOR=22
ARG TARGETARCH

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg git \
    && curl -fsSL https://deb.nodesource.com/setup_${NODE_MAJOR}.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Soufflé ships a prebuilt package for x86_64 only, so arm64 hosts -- Apple
# silicon among them -- have to compile it. Branching on TARGETARCH keeps the
# image buildable on both rather than silently producing an image whose only
# tool is missing.
#
# The source build takes Soufflé's defaults, and in particular does NOT set
# SOUFFLE_DOMAIN_64BIT -- so the arm64 image is 32-bit while the official
# x86-64 .deb is 64-bit, and the two arches genuinely differ in the width of
# `number`. What must match is not a particular width but the agent's compiler
# and the grading compiler: the agent validates its own program before
# submitting it, so if the two disagree a program can pass inside and fail
# outside, and the failure reads as the agent's. That is why the same .deb
# ships to the server as the host grader, and why `agent_container.verify()`
# check (e) compares the whole version identity of the two builds rather than
# assuming they agree.
#
# On amd64 the image installs the official package, which is 64-bit; each run
# records the word size in its provenance. `regen_expected.py --check` passes
# under both widths.
RUN set -eux; \
    apt-get update; \
    if [ "${TARGETARCH}" = "amd64" ]; then \
        curl -fsSL -o /tmp/souffle.deb \
          "https://github.com/souffle-lang/souffle/releases/download/${SOUFFLE_VERSION}/x86_64-ubuntu-2204-souffle-${SOUFFLE_VERSION}-Linux.deb"; \
        apt-get install -y --no-install-recommends /tmp/souffle.deb; \
        rm -f /tmp/souffle.deb; \
    else \
        apt-get install -y --no-install-recommends \
          build-essential cmake bison flex libffi-dev libncurses-dev mcpp \
          zlib1g-dev sqlite3 libsqlite3-dev; \
        git clone --depth 1 --branch ${SOUFFLE_VERSION} \
          https://github.com/souffle-lang/souffle.git /tmp/souffle-src; \
        cmake -S /tmp/souffle-src -B /tmp/souffle-build -DCMAKE_BUILD_TYPE=Release; \
        cmake --build /tmp/souffle-build --target install -j "$(nproc)"; \
        rm -rf /tmp/souffle-src /tmp/souffle-build; \
    fi; \
    souffle --version; \
    rm -rf /var/lib/apt/lists/*

# CLI versions are pinned as build arguments so the image records what actually
# ran. A floating "latest" would let the tool change under a result, so the
# same run could not be reproduced later.
ARG CODEX_VERSION
ARG CLAUDE_VERSION
RUN test -n "$CODEX_VERSION" -a -n "$CLAUDE_VERSION" \
      || (echo "CODEX_VERSION and CLAUDE_VERSION must be pinned at build time" >&2; exit 1) \
    && npm install -g "@openai/codex@${CODEX_VERSION}" "@anthropic-ai/claude-code@${CLAUDE_VERSION}" \
    && npm cache clean --force

# Both CLIs are configured by the same two scripts used to configure them on a
# workstation, rather than by a second description of the same settings written
# for the image. One definition is the point: if the endpoint lived in two
# places they could disagree without anything failing -- the proxy allowlist
# could name one host while the CLIs dial another, and the reachability check
# would pass against a host neither agent uses.
COPY docker/codex.sh docker/claude.sh docker/agent-init.sh /opt/dlb/
RUN chmod +x /opt/dlb/*.sh

# The base URLs are baked because they are not secret and because the proxy
# allowlist is generated from the same constants (synthesis/agent_container.py).
# The key is not baked: it is passed at run time, so it never enters a layer of
# an image that may be shared.
ARG CODEX_BASE_URL=https://api.zhizengzeng.com/v1
ARG ANTHROPIC_BASE_URL=https://api.zhizengzeng.com/anthropic
ARG CODEX_MODEL=gpt-5.6-sol
ENV CODEX_BASE_URL=${CODEX_BASE_URL} \
    ANTHROPIC_BASE_URL=${ANTHROPIC_BASE_URL} \
    CODEX_MODEL=${CODEX_MODEL} \
    ANTHROPIC_MODEL=claude-opus-5 \
    ANTHROPIC_SMALL_FAST_MODEL=claude-opus-5 \
    API_TIMEOUT_MS=600000 \
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1

# The agent writes here. It is the only writable path that matters, and it is
# mounted from the host so the harness can collect what the agent produced.
WORKDIR /work
RUN useradd -m -u 1000 agent && chown agent:agent /work
USER agent

# The entrypoint only writes the CLI configuration into the per-run HOME and
# execs what it was given, so the command that actually ran is still the one
# visible in the harness code rather than something buried in the image.
ENTRYPOINT ["/opt/dlb/agent-init.sh"]
