# 基于 Ubuntu 24.04 (glibc 2.39+, ARM64)
FROM arm64v8/ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    # OpenCV 系统依赖
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    # Qt6 系统依赖 (PyQt6 需要)
    libegl1 \
    libfontconfig1 \
    libxkbcommon0 \
    libdbus-1-3 \
    # 视频编解码
    libavcodec-extra \
    libavformat-extra \
    # 工具
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# 先拷贝依赖清单，利用 Docker 缓存
COPY pyproject.toml ./
RUN uv sync --no-dev

# 拷贝项目代码
COPY core/ ./core/
COPY static/ ./static/
COPY sample_data/ ./sample_data/
COPY data/ ./data/
COPY rtm-det-aidlite/ ./rtm-det-aidlite/
COPY mhformer-aidlite/ ./mhformer-aidlite/
COPY main.py pyqt_test.py ./

# WebSocket 端口
EXPOSE 8000

CMD ["uv", "run", "main.py"]
