"""Single source of truth for the models this project evaluates.

The agent-to-model mapping used to live in three files at once
(`synth_coding_agent`, `eval_coding_agent`, `compare_llm`), one of which
carried the comment "mirrors synth/eval_coding_agent". That triplication is
exactly how the lists drifted: the model inventory was revised and only some
copies followed, leaving tools that would have written and read different
directories for the same run. Everything now imports from here.

`run_llm.sh` derives its grid from `MODELS` below rather than repeating it, so
the shell script cannot drift either.
"""

import shlex

# Providers are split by what each actually serves. Most of the evaluated
# inventory is available from EVAL_BASE_URL; DeepSeek models use DeepSeek's
# official OpenAI-compatible endpoint, and the models that screened the
# specifications during construction come from PROBE_BASE_URL.
#
# Routing is by model rather than by a module-level constant someone edits
# before each run: an endpoint that does not carry the requested model fails in
# ways that look like model behaviour (empty replies, odd errors) rather than
# like a misrouted request.
EVAL_BASE_URL = "https://api.zhizengzeng.com/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
PROBE_BASE_URL = "https://opencode.ai/zen/go/v1/"

# Keys are per provider and are read from the environment, never passed on a
# command line by default: an argument is visible to every process on the host
# through `ps`.
EVAL_API_KEY_ENV = "EVAL_API_KEY"
DEEPSEEK_API_KEY_ENV = "DS_API_KEY"
PROBE_API_KEY_ENV = "PROBE_API_KEY"

DEEPSEEK_FLASH_THINKING = "deepseek-v4-flash"
DEEPSEEK_FLASH_NON_THINKING = "deepseek-v4-flash-non-thinking"

# The evaluated inventory. Two controlled axes are built into this list, so
# dropping or substituting an entry changes what the comparisons measure:
#   reasoning switch -- V4 Flash with thinking explicitly enabled vs disabled
#   scale -- deepseek-v4-pro vs the V4 Flash pair (reasoning mode held fixed)
# Six models across four providers; see EXCLUDED_MODELS for names that are
# deliberately not part of it.
MODELS = [
    "gpt-5.6-sol",          # frontier breadth (OpenAI)
    "claude-opus-5",        # frontier breadth (Anthropic)
    "gemini-3.7-flash",     # frontier breadth (Google; this line is Flash-only upstream)
    "deepseek-v4-pro",      # frontier breadth + scale-pair upper end (thinking explicitly on)
    DEEPSEEK_FLASH_THINKING,
    DEEPSEEK_FLASH_NON_THINKING,
]

# Experiment-arm names stay distinct so the two V4 Flash modes cannot overwrite
# each other's result directories. Both arms deliberately send the same API
# model name; only the explicit `thinking` request field differs.
DEEPSEEK_MODELS = frozenset({
    "deepseek-v4-pro",
    DEEPSEEK_FLASH_THINKING,
    DEEPSEEK_FLASH_NON_THINKING,
})
API_MODEL_BY_ARM = {
    DEEPSEEK_FLASH_THINKING: "deepseek-v4-flash",
    DEEPSEEK_FLASH_NON_THINKING: "deepseek-v4-flash",
}
MODEL_THINKING_MODE = {
    "deepseek-v4-pro": "enabled",
    DEEPSEEK_FLASH_THINKING: "enabled",
    DEEPSEEK_FLASH_NON_THINKING: "disabled",
}

# Models deliberately dropped from the inventory, with the reason, kept as data
# for the same purpose EXCLUDED_AGENTS serves below: "excluded" and "unknown"
# are different states, and here the fall-through is worse than unconstrained.
# `endpoint_for` and `api_key_env_for` route by membership in MODELS, so a name
# merely deleted from the list resolves to the *probe* endpoint and the probe
# key -- a Qwen run would not fail, it would quietly go somewhere else and come
# back with numbers that look like evaluation results. Deleting the entry is
# therefore not enough; it has to be recorded and checked.
EXCLUDED_MODELS = {
    m: ("not part of the evaluated inventory: it belongs to a code-"
        "specialisation pair (general vs code-specialised at one generation) "
        "that is not in the grid, and one model of the pair alone answers no "
        "controlled question.")
    for m in ("qwen3-max", "qwen3-coder-plus")
}
EXCLUDED_MODELS.update({
    "deepseek-reasoner": (
        "aggregation-provider alias not used by this benchmark; use "
        f"{DEEPSEEK_FLASH_THINKING!r}, which directly requests deepseek-v4-flash "
        "with thinking enabled"
    ),
    "deepseek-chat": (
        "aggregation-provider alias not used by this benchmark; use "
        f"{DEEPSEEK_FLASH_NON_THINKING!r}, which directly requests "
        "deepseek-v4-flash with thinking disabled"
    ),
})


def check_model_supported(model: str) -> None:
    """Raise if the model was deliberately dropped from the inventory."""
    name = (model or "").strip()
    if name in EXCLUDED_MODELS:
        raise ValueError(f"model {name!r} is excluded: {EXCLUDED_MODELS[name]}")


# A single 4096-token completion cap is not a controlled budget across plain and
# reasoning models: the latter spend tokens on hidden reasoning before emitting
# content.  In the first full run, DeepSeek V4 Pro exhausted 4096 tokens before
# producing any answer on roughly two thirds of the cases.  Those are truncated
# requests, not model-generated empty programs.  Keep the budget explicit and
# record it with every case so a future inventory change cannot silently inherit
# an unsuitable cap.
DEFAULT_MAX_OUTPUT_TOKENS = 4096
MODEL_MAX_OUTPUT_TOKENS = {
    # The resumed full grid observed gpt-5.6-sol spend the complete 4096-token
    # allowance on hidden reasoning for TicTacToe, then return no answer.  It
    # also needs the reasoning-model contract; keeping the smaller cap would
    # turn a valid model call into an empty synthesized program.
    "gpt-5.6-sol": 32768,
    # Direct Claude exhibited the same hidden-reasoning exhaustion on Earley:
    # finish_reason=length with no answer under the 4096-token default.  This
    # is the OpenAI-compatible direct endpoint, independent of the Claude Code
    # agent's per-turn timeout below in the coding-agent harness.
    "claude-opus-5": 32768,
    "deepseek-v4-pro": 32768,
    DEEPSEEK_FLASH_THINKING: 32768,
    # Giving the two V4-Flash modes the same output allowance avoids a budget
    # confound; the explicit request field controls whether thinking is enabled.
    DEEPSEEK_FLASH_NON_THINKING: 32768,
}


def max_output_tokens_for(model: str) -> int:
    return MODEL_MAX_OUTPUT_TOKENS.get(model, DEFAULT_MAX_OUTPUT_TOKENS)


# The larger reasoning allowance also needs enough wall clock to be produced.
# A 90-second socket timeout merely converted the first V4 smoke from
# ``output_truncated`` into repeated transport timeouts.  Plain models retain the
# shorter bound so an ordinary stalled request still fails quickly.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 90
MODEL_REQUEST_TIMEOUT_SECONDS = {
    "gpt-5.6-sol": 300,
    "claude-opus-5": 300,
    # A formerly failing V4 Pro case took 249s even at low effort, so the 600s
    # outer bound leaves headroom without allowing an unbounded request. Keep
    # the Pro/Flash thinking comparison under one wall-clock contract.
    "deepseek-v4-pro": 600,
    DEEPSEEK_FLASH_THINKING: 600,
    DEEPSEEK_FLASH_NON_THINKING: 600,
}


def request_timeout_for(model: str) -> int:
    return MODEL_REQUEST_TIMEOUT_SECONDS.get(model, DEFAULT_REQUEST_TIMEOUT_SECONDS)


# Transport retries recover a transient gateway failure; they are not extra
# model samples. Five long attempts were counterproductive for V4 Pro: every
# one of its 21 terminal failures used all five attempts, costing roughly
# 8.75 hours without recovering a case. Allow one retry under the longer bound
# and keep the established default for models without that measured pathology.
DEFAULT_INFERENCE_MAX_ATTEMPTS = 5
MODEL_INFERENCE_MAX_ATTEMPTS = {
    "deepseek-v4-pro": 2,
    DEEPSEEK_FLASH_THINKING: 2,
    DEEPSEEK_FLASH_NON_THINKING: 2,
}


def inference_max_attempts_for(model: str) -> int:
    return MODEL_INFERENCE_MAX_ATTEMPTS.get(model, DEFAULT_INFERENCE_MAX_ATTEMPTS)


# DeepSeek V4 thinking is enabled at high effort by default. In the first full
# Pro cell, 21/136 cases exhausted five 300-second calls. A controlled
# OneObject request then spent 511s and returned no visible content under the
# 32,768-token cap. The same prompt at explicit low effort streamed a valid
# answer in 249s after 12,046 reasoning tokens. Pin low (rather than relying on
# a provider default) for both Pro and the V4-Flash thinking arm so the scale
# comparison holds reasoning mode and effort fixed.
MODEL_REASONING_EFFORT = {
    "deepseek-v4-pro": "low",
    DEEPSEEK_FLASH_THINKING: "low",
}


def reasoning_effort_for(model: str):
    return MODEL_REASONING_EFFORT.get(model)


# Streaming does not change the token distribution, but it prevents a long
# response from looking like a dead socket. Stream both V4 Flash modes under
# the same transport contract; only the thinking arm receives
# ``reasoning_effort``.
# Visible content alone is returned to the synthesizer, while aggregate
# reasoning size is retained as provenance.
STREAMING_MODELS = frozenset({
    "deepseek-v4-pro",
    DEEPSEEK_FLASH_THINKING,
    DEEPSEEK_FLASH_NON_THINKING,
})


def stream_response_for(model: str) -> bool:
    return model in STREAMING_MODELS

# Which base model each coding-agent CLI runs on when --model is not given.
# Agent tooling turns over faster than models do, so runs record the CLI version
# they used (`_agent_cli_version` in synth_coding_agent) rather than trusting a
# name to still mean the same program later.
#
# EXCLUDED: Antigravity CLI (`agy`, the Gemini CLI successor). Its built-in web
# tools cannot be switched off from the client: there is no allow/deny flag, the
# entire local config schema it parses is MCP-server scoped (`disabledTools` and
# friends filter an MCP server's tools, and no MCP server provides these), and
# the built-ins are gated by an account-level policy delivered by the service.
# The step budget was unified across agents precisely because a lead bought with
# four times the turns is not the same claim as a lead won in one; by the same
# argument an agent that can retrieve and two that cannot are not measuring the
# same thing -- and here retrieval is a leak channel, since the upstream
# artifacts ship their reference programs. Excluded rather than reported with a
# caveat: the other two restrictions are verifiable from the command line, and
# one condition resting on after-the-fact output inspection would weaken the
# setting as a whole.
AGENT_DEFAULT_MODEL = {
    "codex": "gpt-5.6-sol",
    "claude": "claude-opus-5",
}


# Tool policy for the coding-agent setting.
#
# The agents are given the Souffle toolchain and file access and nothing that
# reaches the network. The reason is not tidiness: the upstream artifacts these
# tasks derive from are public *with their solutions* (the Souffle test suite
# ships the reference programs, EGS ships `sol.dl`), so an agent that can search
# the web can retrieve the answer instead of synthesising it. Disabling
# retrieval is therefore a validity requirement, not a fastidious control -- at
# the cost that these numbers are a lower bound on what the same agent would do
# for a user who lets it browse.
#
# `enforced=False` means the CLI offers no way to constrain its tools; such a
# run is recorded as unconstrained rather than quietly assumed to be clean.
AGENT_TOOL_POLICY = {
    # `--tools` is an allowlist over the built-in set, so WebSearch/WebFetch are
    # excluded by not appearing in it.
    "codex": {
        "enforced": True,
        "args": [],
        "note": "web search is opt-in via --search, which this harness never passes",
    },
    "claude": {
        "enforced": True,
        # Two layers. `--tools` removes the web tools from the built-in set;
        # `--allowedTools` then scopes the shell to the compiler, because a Bash
        # that can run anything can run `curl`, and no web tool needs to be
        # listed for that. Scoping is a permission rule, not a sandbox -- a model
        # determined to get out could chain commands -- but it removes the path a
        # model would take without trying.
        "args": ["--tools", "Bash,Read,Write,Edit",
                 "--allowedTools", "Bash(souffle *) Bash(souffle) Read Write Edit"],
        "note": "--tools drops the web tools; --allowedTools scopes Bash to souffle",
    },
}


# Filesystem and network isolation for the coding-agent setting.
#
# Disabling web tools is not enough on its own, for two reasons discovered by
# checking the harness against the CLIs rather than against its own comments:
#
#   1. The agents were being launched in the harness's own working directory --
#      the repository root, where `benchmark/query/<case>.dl` is the reference
#      program for the very task being posed. `Read` and `Bash` are allowlisted,
#      so the answer was one `cat` away. Every agent turn now runs with `cwd` set
#      to a scratch directory holding nothing.
#   2. `Bash` can reach the network even with no web tool listed. Codex accepts a
#      sandbox mode that blocks it (verified: DNS resolution fails for a `curl`
#      run under `workspace-write`); Claude Code exposes no equivalent flag, so
#      for that agent the shell remains a network path and we say so rather than
#      claiming an isolation we do not have.
#
# What this policy does NOT do, stated so nobody reads more into it later: the
# scratch cwd removes the *relative* path to the references, and `workspace-write`
# restricts writes, but reads outside the workspace are not blocked (verified: an
# absolute path to `benchmark/query/<case>.dl` still resolves under the sandbox).
#
# Those gaps are the reason the reported runs no longer rely on this policy alone.
# Both agents run inside a container (synthesis/agent_container.py), which mounts
# only the scratch directory -- so the reference programs are absent rather than
# merely unread -- and reaches exactly one network host. The per-CLI flags below
# are kept because they cost nothing and remove the tools at the source, but the
# boundary is the container, and it is the same boundary for both agents.
#
# The container also settles two things this policy could not. Codex loads
# whatever `mcp_servers` the host's `~/.codex/config.toml` enables; inside the
# container that file is written fresh at start-up from docker/codex.sh and
# declares none. And the sandbox mechanism is platform-specific -- seatbelt on
# macOS, Landlock/seccomp on Linux -- so the strength of `workspace-write` varied
# with the machine the run happened on. The container does not.
AGENT_ISOLATION = {
    "codex": {
        "cwd_isolated": True,
        "network_blocked": True,
        "args": ["--sandbox", "workspace-write"],
        "note": "runs in a scratch cwd; workspace-write blocks network for shell commands "
                "(the mechanism is platform-specific -- seatbelt on macOS, Landlock/seccomp "
                "on Linux -- which is one reason the container, not this flag, is the boundary)",
    },
    "claude": {
        "cwd_isolated": True,
        "network_blocked": False,
        "args": [],
        "note": "runs in a scratch cwd; the CLI offers no sandbox, so the block is a permission "
                "rule (Bash scoped to souffle) rather than an enforced boundary -- report this "
                "asymmetry with the numbers, and audit the transcripts",
    },
}


def agent_isolation(agent: str) -> dict:
    """Isolation applied to this agent, or an explicit unknown."""
    head = shlex.split(agent)[0] if agent else ""
    return AGENT_ISOLATION.get(head, {
        "cwd_isolated": True,
        "network_blocked": False,
        "args": [],
        "note": "no isolation profile defined for this agent beyond the scratch cwd",
    })


# Agents deliberately dropped from the setting, with the reason. Kept as data so
# selecting one fails loudly: "excluded" and "unknown" are different states, and
# an excluded agent silently falling through to the unconstrained default is the
# exact outcome the exclusion exists to prevent.
# The criterion is not "this agent cannot retrieve" -- no CLI here gives us that
# outright, and claiming it would overstate what the included agents do. It is:
# *every restriction the CLI exposes has been applied, and what it does not expose
# is reported*. Codex exposes both a retrieval switch and a sandbox; Claude Code
# exposes tool and permission allowlists but no sandbox; Antigravity exposes
# neither, so there is nothing to apply and nothing to report but the absence.
# The three are therefore not equally constrained, and the agent numbers are read
# down each column (scaffold increment over the same model) rather than across
# the row.
EXCLUDED_AGENTS = {
    # Measured, not inferred. Tracing the CLI's connections while it answered
    # retrieval queries showed every destination to be Google's own
    # infrastructure and none to be the site whose contents it returned.
    "agy": "Antigravity CLI performs retrieval on the service side, behind the same "
           "endpoint family its model calls use, so no client-side boundary separates "
           "the two -- an isolation that lets the CLI reach its model necessarily lets "
           "its search through, the container the other agents run in included. What "
           "the client does withhold is partial and not ours to rely on (its page-fetch "
           "tool declines in a non-interactive session while its search tool answers "
           "normally) and is set by a service-side policy that can change without "
           "notice, whereas codex and claude are constrained by flags a reader can "
           "check in the recorded command. Retrieval matters on this benchmark because "
           "the upstream artifacts are public together with their reference programs.",
}


def check_agent_supported(agent: str) -> None:
    """Raise if the agent was deliberately excluded from the setting."""
    head = shlex.split(agent)[0] if agent else ""
    if head in EXCLUDED_AGENTS:
        raise ValueError(f"agent {head!r} is excluded: {EXCLUDED_AGENTS[head]}")


def agent_tool_policy(agent: str) -> dict:
    """Tool policy for an agent command line; unknown agents are unconstrained."""
    head = shlex.split(agent)[0] if agent else ""
    return AGENT_TOOL_POLICY.get(
        head, {"enforced": False, "args": [], "note": f"no policy defined for {head!r}"}
    )


def endpoint_for(model: str) -> str:
    """Which provider serves this model.

    DeepSeek evaluation arms resolve to the official DeepSeek endpoint, the
    other evaluated models to the evaluation endpoint, and anything else to
    the probe endpoint. Keys follow the same split -- see `api_key_env_for`.

    Note what this means for a name that is *excluded* rather than unknown: it
    routes to the probe side just like any unrecognised name would. That is why
    `check_model_supported` exists and must be called before this is reached.
    """
    check_model_supported(model)
    if model in DEEPSEEK_MODELS:
        return DEEPSEEK_BASE_URL
    return EVAL_BASE_URL if model in MODELS else PROBE_BASE_URL


def api_key_env_for(model: str) -> str:
    """Environment variable holding the key for this model's provider."""
    check_model_supported(model)
    if model in DEEPSEEK_MODELS:
        return DEEPSEEK_API_KEY_ENV
    return EVAL_API_KEY_ENV if model in MODELS else PROBE_API_KEY_ENV


def requested_model_for(model: str) -> str:
    """API `model` value for an experiment arm."""
    check_model_supported(model)
    return API_MODEL_BY_ARM.get(model, model)


def thinking_mode_for(model: str):
    """Explicit DeepSeek thinking mode, or None for providers without it."""
    check_model_supported(model)
    return MODEL_THINKING_MODE.get(model)


def api_key_for(model: str) -> str:
    """The key for this model's provider, from the environment ("" if unset)."""
    import os
    return os.environ.get(api_key_env_for(model), "").strip()


def resolve_default_model_for_agent(agent: str) -> str:
    """Default base model for an agent command line (e.g. "claude --print")."""
    head = shlex.split(agent)[0]
    try:
        return AGENT_DEFAULT_MODEL[head]
    except KeyError:
        raise ValueError(
            f"No default model mapping for agent '{head}', please pass --model"
        ) from None


def agent_default_help() -> str:
    """Help-string fragment, so CLI text cannot fall out of step with the table."""
    return "Defaults: " + ", ".join(f"{a}->{m}" for a, m in AGENT_DEFAULT_MODEL.items())


if __name__ == "__main__":
    print(" ".join(MODELS))
