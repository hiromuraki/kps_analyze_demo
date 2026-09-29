#!/bin/bash
set -euo pipefail

uv sync --frozen && uv run demo_ema/ema.py
