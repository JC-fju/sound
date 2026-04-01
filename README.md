```Markdown
# 聲音波形與聲紋辨識系統 (Audio Waveform & Voice Authentication)

這個專案展示如何在 Nvidia Jetson Orin Nano 上，利用 **Logitech C270** 同時進行影像識別、聲音視覺化，並結合最新的 AI 聲紋辨識技術，透過網頁介面進行遠端監控與操作。

## 🎯 核心功能
1. **即時聲音視覺化**：將麥克風收到的聲音轉為波形圖，即時疊加在影像上。
2. **手勢追蹤 (MediaPipe)**：精準偵測畫面中的雙手骨架並繪製節點。
3. **AI 聲紋註冊與辨識 (SpeechBrain)**：
   * 透過 ECAPA-TDNN 模型萃取 192 維度聲紋特徵 (Embeddings)。
   * 使用「餘弦相似度 (Cosine Similarity)」比對，支援 3 秒快速無感錄音辨識。
4. **Web UI 控制面板 (Flask)**：無需外接螢幕與鍵盤，只需透過同網段的瀏覽器，即可遠端觀看即時影像並操作註冊/驗證流程。

## 🛠️ 硬體需求
* Nvidia Jetson Orin Nano (或其他支援 CUDA 的 Linux 設備)
* Logitech C270 HD Webcam (或其他 USB 麥克風/鏡頭組合)
* 確保設備與操作用的筆電/手機連線至同一個區域網路 (Wi-Fi)。

## 📦 安裝說明

### 1. 安裝系統音訊驅動 (Linux 必備)
在 Jetson 上，必須先安裝 PortAudio 庫來處理底層音訊：
```bash
sudo apt-get update
sudo apt-get install libasound-dev portaudio19-dev
2. 安裝 Python 依賴套件
強烈建議在虛擬環境中執行。請安裝以下套件 (包含視覺、音訊、網頁框架與 AI 模型)：

```Bash
pip install opencv-python mediapipe pyaudio numpy matplotlib
pip install flask
pip install torch torchaudio
pip install speechbrain
(註：第一次執行聲紋辨識時，程式會自動從 HuggingFace 下載模型權重，請確保網路連線正常。)

🚀 系統啟動與使用方式
啟動伺服器
在專案根目錄下執行後端主程式：

```Bash
python3 app.py
若啟動成功，終端機將會顯示網頁伺服器的執行 IP 與 Port（預設為 5000）。

網頁遠端操作
開啟你筆電或手機的瀏覽器。

在網址列輸入 Jetson 的 IP 地址，例如：http://192.168.X.X:5000。

註冊新聲紋 (Register)：

在控制面板的 User Name 欄位輸入你的名字。

點擊 Register 按鈕。

對著麥克風講話 3 秒鐘，網頁將顯示 REGISTERED 並亮起綠燈。

辨識身分 (Verify)：

點擊 Verify 按鈕。

對著麥克風講話 3 秒鐘。

系統將自動計算餘弦相似度，若分數達標（預設 > 0.75），畫面上會顯示你的名字並通過驗證。

📂 專案檔案結構
app.py: Flask 網頁伺服器主程式，處理 API 請求、控制錄音與 OpenCV 影像串流。

voice_auth.py: 獨立的聲紋 AI 模組，負責載入 SpeechBrain 模型、特徵萃取與餘弦相似度計算。

templates/index.html: Web UI 介面，使用 HTML/CSS/JS 刻劃控制面板。

speaker_db.pkl: 本地端的特徵資料庫（由程式自動生成），用來持久化儲存註冊者的聲紋 Embedding。

requirements.txt: 專案所需的 Python 套件清單。