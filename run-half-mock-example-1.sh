#!/bin/bash
set -euo pipefail

uv sync --frozen && uv run main.py \
    --camera -1 \
    --video-path "./sample_data/example-1/video.mp4" \
    --analyzer-2d rtmpose \
    --analyzer-3d mhformer