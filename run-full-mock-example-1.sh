#!/bin/bash
set -euo pipefail

uv sync --frozen && uv run main.py \
    --camera -1 \
    --video-path "./sample_data/example-1/video.mp4" \
    --analyzer-2d mock \
    --mock-kp2d "./sample_data/example-1/2d_coco17_kps.npz" \
    --analyzer-3d mock \
    --mock-kp3d "./sample_data/example-1/3d_kps.npz"