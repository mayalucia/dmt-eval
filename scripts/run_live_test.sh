#!/usr/bin/env bash
# Run the live LLM agent test.
#
# Usage:
#   ./scripts/run_live_test.sh
#
# The script will prompt you for your Anthropic API key (hidden input),
# run the agent, and print the grade report. The key is never written
# to disk or shell history.

set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT_DIR="./llm_agent_output"

echo "═══════════════════════════════════════════════════════"
echo "  DMT-Eval — Live LLM Agent Test"
echo "═══════════════════════════════════════════════════════"
echo ""

# Prompt for key if not already set
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo -n "Enter your Anthropic API key: "
    read -rs ANTHROPIC_API_KEY
    echo ""
    export ANTHROPIC_API_KEY
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: No API key provided."
    exit 1
fi

echo "Output directory: $OUTPUT_DIR"
echo ""

# Clean previous output
rm -rf "$OUTPUT_DIR"

# Run the live agent
uv run --extra llm --extra dev python scripts/run_live_agent.py "$OUTPUT_DIR"
