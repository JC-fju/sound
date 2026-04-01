import cv2
import mediapipe as mp
import pyaudio
import numpy as np
import threading
import collections
import time
from flask import Flask, Response, render_template_string

# --- 參數設定 ---
WIDTH, HEIGHT = 640, 480
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000                # ✅ 已修正: C270 只能跑 16000 或 48000
WAVE_HEIGHT = 80
WAVE_Y_OFFSET = 400
WAVE_COLOR = (0, 255, 0)

# --- Flask 設定 ---
app = Flask(__name__)

class AudioStream:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.running = False
        self.lock = threading.Lock()
        self.audio_buffer = collections.deque(maxlen=WIDTH)
        self.audio_buffer.extend([0] * WIDTH)
        self.device_index = self.find_c270_index()

    def find_c270_index(self):
        print("\n🔍 搜尋麥克風...")
        cnt = self.p.get_device_count()
        idx = None
        for i in range(cnt):
            try:
                info = self.p.get_device_info_by_index(i)
                name = info.get('name')
                if ("C270" in name or "USB" in name) and info.get('maxInputChannels') > 0:
                    print(f"✅ 找到裝置 Index {i}: {name}")
                    idx = i
                    break
            except:
                continue
        return idx

    def start(self):
        if self.running: return
        try:
            self.stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=CHUNK
            )
            self.running = True
            self.thread = threading.Thread(target=self._record_loop, daemon=True)
            self.thread.start()
            print("🎙️  錄音啟動")
        except Exception as e:
            print(f"❌ 錄音失敗 (請檢查是否有其他程式佔用): {e}")

    def _record_loop(self):
        while self.running:
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                int_data = np.frombuffer(data, dtype=np.int16)
                with self.lock:
                    normalized = (int_data[::2] / 150).astype(int)
                    self.audio_buffer.extend(normalized)
            except:
                pass

    def get_waveform_points(self):
        with self.lock:
            data = list(self.audio_buffer)
        data = data[-WIDTH:]
        points = []
        for x, val in enumerate(data):
            y = WAVE_Y_OFFSET - val
            y = max(WAVE_Y_OFFSET - WAVE_HEIGHT, min(WAVE_Y_OFFSET + WAVE_HEIGHT, y))
            points.append([x, y])
        return np.array(points, np.int32)

# 全域變數
audio = AudioStream()
audio.start()

def generate_frames():
    # ... (前面的設定不變) ...
    
    print("📷 正在嘗試開啟攝影機...")
    cap = cv2.VideoCapture(0) # 或是你改成的 1
    
    if not cap.isOpened():
        print("❌ 攝影機無法開啟！(isOpened 為 False)")
        return

    print("✅ 攝影機已開啟，準備讀取畫面...")
    
    while True:
        success, frame = cap.read()
        if not success:
            print("❌ 讀取畫面失敗 (Frame read failed)")
            break
        
        # ... (後面不變) ...    
    # 嘗試開啟鏡頭 0 或 1
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        # 轉 RGB 處理手勢
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # 畫手勢
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        # 畫聲波
        points = audio.get_waveform_points()
        if len(points) > 0:
            cv2.polylines(frame, [points], isClosed=False, color=WAVE_COLOR, thickness=2)
            cv2.line(frame, (0, WAVE_Y_OFFSET), (WIDTH, WAVE_Y_OFFSET), (100, 100, 100), 1)

        # 編碼成 JPEG 串流
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return """
    <html>
    <head><title>Jetson Audio Visualizer</title></head>
    <body style="background:black; color:white; text-align:center;">
        <h1>FJU Nvidia Jetson - Audio & Hand Tracking</h1>
        <img src="/video_feed" style="border: 2px solid green; width:80%;">
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # 讓網頁伺服器跑在 0.0.0.0，這樣你的筆電才連得到
    print("🚀 網頁伺服器啟動中... 請在瀏覽器輸入 Jetson 的 IP:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)