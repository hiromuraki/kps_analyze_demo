#!/bin/bash
set -e

echo "Rule Builder → http://localhost:2800/static/rule-builder.html"

uv run main.py \
    --camera -1 \
    --analyzer-2d mock \
    --analyzer-3d mock \
    "$@"
