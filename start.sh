#!/bin/bash

# 获取脚本所在目录
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# 检查 7860 端口占用情况
PORT=7860
echo "正在检查端口 $PORT..."
PID=$(lsof -ti:$PORT)

if [ -n "$PID" ]; then
    echo "端口 $PORT 被进程 $PID 占用，正在终止该进程..."
    kill -9 $PID
    sleep 1
    echo "进程已终止。"
else
    echo "端口 $PORT 未被占用。"
fi

# 激活虚拟环境
if [ -d "venv" ]; then
    echo "激活虚拟环境..."
    source venv/bin/activate
else
    echo "未找到 venv 目录，尝试直接运行..."
fi

# 启动服务
echo "正在启动 Photo Fusion 服务..."
python fusion-tools/app.py
