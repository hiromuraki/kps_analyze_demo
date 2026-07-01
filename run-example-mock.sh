#!/bin/bash

uv sync --frozen && uv run example.py --analyzer-2d mock --analyzer-3d mock --camera -1 --width 640 --height 480 --fps 305