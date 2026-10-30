#!/bin/bash
set -euo pipefail

# 前端构建使用 PATH 中的 pnpm；未安装时回退到已有 dist。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BOLD="\033[1m"
GREEN="\033[32m"
BLUE="\033[34m"
CYAN="\033[36m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

FRONTEND_DIR="tools/rule-builder"
DIST_DIR="$FRONTEND_DIR/dist"

echo ""
echo -e "${BLUE}${BOLD}╔════════════════════════════════════════════════╗${RESET}"
echo -e "${BLUE}${BOLD}║${RESET}  ${BOLD}Rule Builder — 动作规则可视化编辑器${RESET}${BLUE}${BOLD}        ║${RESET}"
echo -e "${BLUE}${BOLD}╠════════════════════════════════════════════════╣${RESET}"
echo -e "${BLUE}${BOLD}║${RESET}                                                ${BLUE}${BOLD}║${RESET}"
echo -e "${BLUE}${BOLD}║${RESET}  ${GREEN}${BOLD}▶ 打开浏览器访问:${RESET}                               ${BLUE}${BOLD}║${RESET}"
echo -e "${BLUE}${BOLD}║${RESET}  ${CYAN}${BOLD}http://localhost:2800/${RESET}                         ${BLUE}${BOLD}║${RESET}"
echo -e "${BLUE}${BOLD}║${RESET}                                                ${BLUE}${BOLD}║${RESET}"
echo -e "${BLUE}${BOLD}║${RESET}  ${BOLD}使用说明:${RESET}                                     ${BLUE}${BOLD}║${RESET}"
echo -e "${BLUE}${BOLD}║${RESET}  ${YELLOW}1.${RESET} 上传视频文件 (🎬)                           ${BLUE}${BOLD}║${RESET}"
echo -e "${BLUE}${BOLD}║${RESET}  ${YELLOW}2.${RESET} 上传 2D/3D 骨骼 .npz 文件                   ${BLUE}${BOLD}║${RESET}"
echo -e "${BLUE}${BOLD}║${RESET}  ${YELLOW}3.${RESET} 在骨骼面板选择要测量的骨骼线                  ${BLUE}${BOLD}║${RESET}"
echo -e "${BLUE}${BOLD}║${RESET}  ${YELLOW}4.${RESET} 定位目标帧 → 点「保存当前规则」              ${BLUE}${BOLD}║${RESET}"
echo -e "${BLUE}${BOLD}║${RESET}  ${YELLOW}5.${RESET} 元数据页可预览 / 导出规则 JSON                 ${BLUE}${BOLD}║${RESET}"
echo -e "${BLUE}${BOLD}║${RESET}                                                ${BLUE}${BOLD}║${RESET}"
echo -e "${BLUE}${BOLD}╚════════════════════════════════════════════════╝${RESET}"
echo -e "${YELLOW}提示: 点击「规则集」中的规则可跳回录制帧并还原骨骼选择。${RESET}"
echo ""

# ---------------------------------------------------------------------------
# 构建前端（产物固定输出到 ./tools/rule-builder/dist，由 main.py 托管）
# ---------------------------------------------------------------------------
if [ "${RULE_BUILDER_SKIP_BUILD:-0}" = "1" ]; then
    echo -e "${YELLOW}跳过前端构建（RULE_BUILDER_SKIP_BUILD=1）${RESET}"
elif command -v pnpm >/dev/null 2>&1; then
    echo -e "${CYAN}▶ 构建前端: $FRONTEND_DIR → $DIST_DIR${RESET}"
    (
        cd "$FRONTEND_DIR"
        pnpm install --frozen-lockfile
        pnpm build
    )
elif [ -f "$DIST_DIR/index.html" ]; then
    echo -e "${YELLOW}⚠ 未找到 pnpm，使用已有构建产物: $DIST_DIR${RESET}"
    echo -e "${YELLOW}  安装 Node.js + pnpm 后重新运行本脚本即可自动构建。${RESET}"
else
    echo -e "${RED}错误: 未找到 pnpm，且 $DIST_DIR 不存在。${RESET}" >&2
    echo -e "${RED}请安装 Node.js 与 pnpm，然后执行: cd $FRONTEND_DIR && pnpm install && pnpm build${RESET}" >&2
    exit 1
fi

uv run main.py \
    --camera -1 \
    --analyzer-2d mock \
    --analyzer-3d mock \
    --static-dir "./tools/rule-builder/dist"
