#!/usr/bin/env bash
# Run the DMT-Eval tournament (multiple models × multiple briefs).
#
# Usage:
#   ./scripts/run_tournament.sh
#
# Prompts for your Anthropic API key (hidden input).
# The key is never written to disk or shell history.

set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT_DIR="./tournament_output"

echo "═══════════════════════════════════════════════════════"
echo "  DMT-Eval — Tournament"
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

# Run the tournament
uv run --extra llm --extra dev python scripts/run_tournament.py "$OUTPUT_DIR"
