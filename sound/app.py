import cv2
import mediapipe as mp
import pyaudio
import numpy as np
import threading
import wave
from flask import Flask, Response, render_template, request, jsonify
from voice_auth import SpeakerRecognizer

# --- 參數設定 ---
WIDTH, HEIGHT = 640, 480
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

app = Flask(__name__)

class AudioStream:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.device_index = self.find_c270_index()

    def find_c270_index(self):
        cnt = self.p.get_device_count()
        for i in range(cnt):
            info = self.p.get_device_info_by_index(i)
            if ("C270" in info.get('name') or "USB" in info.get('name')) and info.get('maxInputChannels') > 0:
                return i
        return None

    # 改回傳統的「阻擋式錄音」：呼叫時才錄 3 秒
    def record_clip(self, filename="temp.wav", seconds=3):
        print(f"🔴 開始錄音 {seconds} 秒...")
        try:
            stream = self.p.open(
                format=FORMAT, channels=CHANNELS, rate=RATE,
                input=True, input_device_index=self.device_index,
                frames_per_buffer=CHUNK
            )
            frames = []
            for _ in range(0, int(RATE / CHUNK * seconds)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            
            wf = wave.open(filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            print("✅ 錄音完成！")
            return True
        except Exception as e:
            print(f"❌ 錄音失敗: {e}")
            return False

audio = AudioStream()
recognizer = SpeakerRecognizer()

# --- OpenCV 影像產生器 ---
def generate_frames():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): cap = cv2.VideoCapture(1)

    while True:
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# --- 網頁路由 ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- 給網頁按鈕呼叫的 API ---
@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    name = data.get('name', 'Unknown')
    
    if audio.record_clip("temp.wav", seconds=3):
        recognizer.register_user(name, "temp.wav")
        return jsonify({"status": "success", "message": f"Registered: {name}"})
    return jsonify({"status": "error", "message": "錄音失敗"})

@app.route('/api/verify', methods=['POST'])
def verify():
    if audio.record_clip("temp.wav", seconds=3):
        name, score = recognizer.verify_user("temp.wav")
        return jsonify({"status": "success", "name": name, "score": float(score)})
    return jsonify({"status": "error", "message": "錄音失敗"})

if __name__ == "__main__":
    print("🚀 網頁伺服器啟動中... 請在同網域電腦瀏覽器輸入 Jetson 的 IP:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)