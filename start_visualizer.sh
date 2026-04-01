#!/bin/bash

echo "========================================"
echo "   FJU Jetson Orin Nano - 音訊手勢專案"
echo "========================================"

# 1. 檢查是否安裝了 PortAudio (Linux 聲音驅動庫)
echo "🔍 檢查系統音訊驅動..."
dpkg -s portaudio19-dev &> /dev/null

if [ $? -ne 0 ]; then
    echo "⚠️  未偵測到 portaudio19-dev，正在嘗試安裝..."
    echo "   (需要輸入密碼)"
    sudo apt-get update
    sudo apt-get install -y portaudio19-dev libasound-dev
else
    echo "✅ 系統音訊驅動已安裝"
fi

# 2. 啟動 Python 程式
echo "🚀 正在啟動程式..."
python3 gesture_audio_fusion.py