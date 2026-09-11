#!/bin/bash

uv sync --frozen && uv run python main.py \
    --analyzer-2d rtmpose \
    --analyzer-3d mhformer \
    --camera 2 \
    --camera-width 640 \
    --camera-height 480 \
    --fps 30
