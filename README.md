# 聲音波形 (Audio Waveform)

這個專案展示如何在 Nvidia Jetson Orin Nano 上，利用 **Logitech C270** 同時進行影像識別與聲音視覺化。

## 🎯 功能
1. **聲音視覺化**：即時將 C270 麥克風收到的聲音轉為波形圖，疊加在影像上。
2. **多執行緒處理**：確保錄音與影像處理互不干擾，維持流暢 FPS。

## 🛠️ 硬體需求
* Nvidia Jetson Orin Nano (或其他 Linux 設備)
* Logitech C270 HD Webcam (或其他 USB 麥克風/鏡頭組合)

## 📦 安裝說明

### 1. 安裝系統音訊驅動
在 Jetson 上，必須先安裝 PortAudio 庫：
```bash
sudo apt-get update
sudo apt-get install libasound-dev portaudio19-dev
