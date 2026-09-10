#!/bin/bash
set -euo pipefail

BOLD="\033[1m"
GREEN="\033[32m"
BLUE="\033[34m"
CYAN="\033[36m"
YELLOW="\033[33m"
RESET="\033[0m"

echo ""
echo -e "${BLUE}${BOLD}╔════════════════════════════════════════════════╗${RESET}"
echo -e "${BLUE}${BOLD}║${RESET}  ${BOLD}Rule Builder — 动作规则可视化编辑器${RESET}${BLUE}${BOLD}        ║${RESET}"
echo -e "${BLUE}${BOLD}╠════════════════════════════════════════════════╣${RESET}"
echo -e "${BLUE}${BOLD}║${RESET}                                                ${BLUE}${BOLD}║${RESET}"
echo -e "${BLUE}${BOLD}║${RESET}  ${GREEN}${BOLD}▶ 打开浏览器访问:${RESET}                               ${BLUE}${BOLD}║${RESET}"
echo -e "${BLUE}${BOLD}║${RESET}  ${CYAN}${BOLD}http://localhost:2800/static/rule-builder.html${RESET}  ${BLUE}${BOLD}║${RESET}"
echo -e "${BLUE}${BOLD}║${RESET}                                                ${BLUE}${BOLD}║${RESET}"
echo -e "${BLUE}${BOLD}║${RESET}  ${BOLD}使用说明:${RESET}                                     ${BLUE}${BOLD}║${RESET}"
echo -e "${BLUE}${BOLD}║${RESET}  ${YELLOW}1.${RESET} 上传视频文件 (🎬)                           ${BLUE}${BOLD}║${RESET}"
echo -e "${BLUE}${BOLD}║${RESET}  ${YELLOW}2.${RESET} 上传 2D/3D 骨骼 .npz 文件                   ${BLUE}${BOLD}║${RESET}"
echo -e "${BLUE}${BOLD}║${RESET}  ${YELLOW}3.${RESET} 在骨骼面板选择要测量的骨骼线                  ${BLUE}${BOLD}║${RESET}"
echo -e "${BLUE}${BOLD}║${RESET}  ${YELLOW}4.${RESET} 定位目标帧 → 点「保存当前规则」              ${BLUE}${BOLD}║${RESET}"
echo -e "${BLUE}${BOLD}║${RESET}  ${YELLOW}5.${RESET} 导出 JSON 规则文件                           ${BLUE}${BOLD}║${RESET}"
echo -e "${BLUE}${BOLD}║${RESET}                                                ${BLUE}${BOLD}║${RESET}"
echo -e "${BLUE}${BOLD}╚════════════════════════════════════════════════╝${RESET}"
echo ""

uv run main.py \
    --camera -1 \
    --analyzer-2d mock \
    --analyzer-3d mock \
    "$@"
