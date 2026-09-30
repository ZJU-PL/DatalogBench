# Container isolation for the coding-agent setting

Two images and two networks. The shape is dictated by one constraint: the agent
must not reach the internet, but its CLI must reach its own model. A container
with no network cannot run the setting; a container with a normal bridge network
is not isolated at all.

    agent container ── dlb-agent-int (internal, no route off the host)
                          │
                       proxy ── dlb-agent-out (bridge) ── api.zhizengzeng.com
                          └── CONNECT allowlist: that host, port 443, nothing else

The reference programs are not mounted, so they are absent rather than merely
unreadable. Only the per-case scratch directory is mounted.

## Credentials

    export EVAL_API_KEY=...      # shared zzz inference key

Direct and agent settings share the zzz credential. The key is passed into the
container at run time by environment-variable name
(`docker ... -e EVAL_API_KEY`, never `-e EVAL_API_KEY=<value>`) and written
into CLI configuration on the container's tmpfs by `docker/codex.sh` and
`docker/claude.sh`. Its value is therefore absent from process argv and never
persisted in the image or per-case session mount.

Both agent CLIs use a new container for every turn. Only
`~/.claude/projects` or `~/.codex/sessions` is mounted into a per-case scratch
directory so the next turn can resume. HOME, settings and the credential stay
on tmpfs; the session mount is deleted after the case, so state cannot cross
cases.

Configuring Claude Code with a token rather than an interactive login is also
what makes the container possible: the subscription login keeps its credentials
in the macOS Keychain, which a Linux container cannot reach.

## Build

    docker network create --internal dlb-agent-int
    docker network create dlb-agent-out

    docker build -f docker/egress-proxy.Dockerfile -t datalogbench/egress .
    docker build -f docker/agent.Dockerfile -t datalogbench/agent \
      --build-arg CODEX_VERSION=<pinned> --build-arg CLAUDE_VERSION=<pinned> .

The tag is pinned too (`datalogbench/agent:2026-09-zzz`, see `agent_container.IMAGE`).
Soufflé publishes a prebuilt package for x86_64 only, so on arm64 -- Apple
silicon included -- the image compiles it from source and the build takes a
while.

The CLI versions are required build arguments. A floating tag would let the
tool change under a result, so the same run could not be reproduced later.

## Run the proxy

    python3 synthesis/agent_container.py --start-proxy

The allowlist is generated from where the CLIs are actually configured to send
model calls (`--endpoints` prints them) and mounted into the proxy, rather than
baked into the image. The first version hard-coded the direct-prompting endpoint
while the two agents were configured against two different hosts, so the
reachability check passed on a host neither CLI would ever dial. `available()`
now compares the running proxy's allowlist against current configuration and
refuses to run when they drift.

## Verify before trusting it

    python3 synthesis/agent_container.py --verify

The preflight checks credential handling, isolation, both configured endpoints,
and compiler identity. All must pass before a run.

**(a) the credential value is absent from Docker argv.** The generated command
must contain the separate arguments `-e`, `EVAL_API_KEY`, never a combined
argument carrying its value.

**(b) a reference program is unreadable by absolute path** -- the check that
motivated all of this.

    G=$PWD/benchmark/query/Path.dl
    docker run --rm --network dlb-agent-int alpine:3.20 cat "$G"
    # cat: can't open '...': No such file or directory
    ls "$G"    # ... and it does exist on the host

On the host, `codex sandbox -c 'sandbox_mode="workspace-write"' -- cat "$G"`
printed the reference program: `workspace-write` restricts writes and network,
not reads. Under the container the file is absent, not merely denied, because
nothing but the scratch directory is mounted.

**(c) the public internet is unreachable.**

    docker run --rm --network dlb-agent-int -e HTTPS_PROXY=http://dlb-egress:8888 \
      alpine/curl curl -sS -o /dev/null -w '%{http_code}\n' https://github.com
    # curl: (7) CONNECT tunnel failed, response 403

**(d) each model endpoint is reachable.** Without this the setting cannot run at
all, so it is as much a part of the boundary as the two denials.

    docker run --rm --network dlb-agent-int -e HTTPS_PROXY=http://dlb-egress:8888 \
      alpine/curl curl -sS -o /dev/null -w '%{http_code}\n' \
      https://api.zhizengzeng.com/v1/models
    # 200

**(e) Soufflé compiles and runs** -- the agent validates its own program, so a
container without a working compiler silently changes the task.

    docker run --rm --network dlb-agent-int datalogbench/agent:2026-09-zzz \
      sh -c 'printf ".decl A(x:number)\nA(1).\n.output A\n" > /tmp/t.dl && souffle -D- /tmp/t.dl'

**(f) the agent and grading Soufflé builds have the same identity.** Synthesis
executes feedback on the host while final grading uses the pinned image; a
version or integer-width mismatch would otherwise be attributed to the model.

Do not run the setting when `--verify` fails. Falling back to the host would
apply the isolation this replaces while labelling the results as though it had
not, which is worse than not running. `synth_coding_agent.py` therefore raises
`ContainerUnavailable` instead of degrading; `--no-container` exists for
debugging and prints a warning that its numbers are not agent results.

## What the container does not close

The allowlist acts on the CONNECT target and does not decrypt TLS, so anything
reachable *through* the model endpoint is still reachable. That is the reason
Antigravity stays excluded: its retrieval runs on Google's servers behind the
same endpoint family its model calls use, so no client-side boundary can
separate the two. For codex and claude the retrieval tools are client-side and
are switched off by flags that appear in the recorded command.
