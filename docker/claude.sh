#!/bin/sh
# Configure the Claude Code CLI against the agent endpoint.
#
# See codex.sh for why the key comes from the environment and why this script is
# also what runs inside the container.
#
# A token in settings.json rather than an interactive login is what makes the
# container possible at all: the subscription login stores its credentials in the
# macOS Keychain, which a Linux container cannot reach.
set -eu

: "${AGENT_API_KEY:?set AGENT_API_KEY}"
ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://api.zhizengzeng.com/anthropic}"
ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-claude-opus-5}"
ANTHROPIC_SMALL_FAST_MODEL="${ANTHROPIC_SMALL_FAST_MODEL:-claude-opus-5}"
API_TIMEOUT_MS="${API_TIMEOUT_MS:-600000}"

dir="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
mkdir -p "$dir"

cat > "$dir/settings.json" <<EOF
{
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "${AGENT_API_KEY}",
    "ANTHROPIC_API_KEY": "",
    "ANTHROPIC_BASE_URL": "${ANTHROPIC_BASE_URL}",
    "ANTHROPIC_MODEL": "${ANTHROPIC_MODEL}",
    "ANTHROPIC_SMALL_FAST_MODEL": "${ANTHROPIC_SMALL_FAST_MODEL}",
    "API_TIMEOUT_MS": "${API_TIMEOUT_MS}",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
  }
}
EOF
chmod 600 "$dir/settings.json"

# Skips the onboarding prompt, which would otherwise block a non-interactive run.
cat > "$HOME/.claude.json" <<'EOF'
{
  "hasCompletedOnboarding": true
}
EOF

echo "claude configured: ${ANTHROPIC_BASE_URL} (model ${ANTHROPIC_MODEL})"
