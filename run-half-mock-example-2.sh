#!/bin/bash
set -euo pipefail

uv sync --frozen && uv run main.py \
    --camera -1 \
    --video-path "./sample_data/example-2/video.mp4" \
    --analyzer-2d mock \
    --mock-kp2d "./sample_data/example-2/2d_coco17_kps.npz" \
    --analyzer-3d mock \
    --mock-kp3d "./sample_data/example-2/3d_kps.npz" \
    --width 640 \
    --height 480 \
    --fps 30