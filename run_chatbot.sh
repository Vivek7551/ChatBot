#!/usr/bin/env bash
set -euo pipefail

# Always run from this script's directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -x ".venv/bin/python" ]]; then
  exec .venv/bin/python chatbot.py
else
  exec python chatbot.py
fi
