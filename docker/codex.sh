#!/bin/sh
# Configure the Codex CLI against the agent endpoint.
#
# The key is read from the environment, not written into the file by hand. It is
# the agent-only AGENT_API_KEY; Direct prompting uses EVAL_API_KEY separately.
#
# This same script runs inside the isolation container (docker/agent.Dockerfile
# copies it in), so the agent's configuration has exactly one definition. The
# earlier split -- one endpoint compiled into the proxy allowlist, another in the
# CLI config -- is how the isolation check came to verify a host nothing dialed.
set -eu

: "${AGENT_API_KEY:?set AGENT_API_KEY}"
CODEX_BASE_URL="${CODEX_BASE_URL:-https://api.zhizengzeng.com/v1}"
CODEX_MODEL="${CODEX_MODEL:-gpt-5.6-sol}"

dir="${CODEX_HOME:-$HOME/.codex}"
mkdir -p "$dir"

cat > "$dir/config.toml" <<EOF
model = "${CODEX_MODEL}"
model_provider = "zzz"

[model_providers.zzz]
name = "zzz"
base_url = "${CODEX_BASE_URL}"
env_key = "AGENT_API_KEY"
wire_api = "responses"
EOF

echo "codex configured: ${CODEX_BASE_URL} (model ${CODEX_MODEL})"
