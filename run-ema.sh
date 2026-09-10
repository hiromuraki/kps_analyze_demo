#!/bin/bash
set -euo pipefail

uv sync --frozen && uv run static/ema.py
