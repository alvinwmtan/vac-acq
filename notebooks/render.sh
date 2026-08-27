#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export QUARTO_PYTHON="$(cd .. && pwd)/.venv/bin/python"
quarto render "$@"
