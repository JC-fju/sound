import cv2
import pyaudio
import numpy as np
import threading
import wave
import time
import collections
from flask import Flask, Response, render_template, request, jsonify, session, redirect, url_for
from voice_auth import SpeakerRecognizer

# --- 參數設定 ---
WIDTH, HEIGHT = 640, 480
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
WAVE_HEIGHT = 60
WAVE_Y_OFFSET = 400
WAVE_COLOR = (0, 255, 0)

app = Flask(__name__)
app.secret_key = "fju_math_secret_jerry" #

recognizer = SpeakerRecognizer() #
cooldown_cache = {} # 路線二：失敗次數與冷卻紀錄

class AudioStream:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.lock = threading.Lock()
        self.audio_buffer = collections.deque(maxlen=WIDTH)
        self.audio_buffer.extend([0] * WIDTH)
        self.is_recording_auth = False
        self.auth_frames = []
        self.device_index = self.find_c270_index()
        
        # 啟動背景錄音執行緒，用於即時波形顯示
        self.stream = self.p.open(
            format=FORMAT, channels=CHANNELS, rate=RATE,
            input=True, input_device_index=self.device_index,
            frames_per_buffer=CHUNK
        )
        self.thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.thread.start()

    def find_c270_index(self):
        cnt = self.p.get_device_count()
        for i in range(cnt):
            info = self.p.get_device_info_by_index(i)
            if ("C270" in info.get('name') or "USB" in info.get('name')) and info.get('maxInputChannels') > 0:
                return i
        return None

    def _audio_loop(self):
        while True:
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                int_data = np.frombuffer(data, dtype=np.int16)
                
                # 更新即時波形緩衝區
                with self.lock:
                    normalized = (int_data[::2] / 150).astype(int)
                    self.audio_buffer.extend(normalized)
                    
                    # 路線三：如果是驗證/註冊狀態，同步存下音訊幀
                    if self.is_recording_auth:
                        self.auth_frames.append(data)
            except: pass

    def get_waveform_points(self):
        with self.lock:
            data = list(self.audio_buffer)
        points = []
        for x, val in enumerate(data[-WIDTH:]):
            y = WAVE_Y_OFFSET - val
            y = max(WAVE_Y_OFFSET - WAVE_HEIGHT, min(WAVE_Y_OFFSET + WAVE_HEIGHT, y))
            points.append([x, y])
        return np.array(points, np.int32)

    def record_auth_clip(self, filename, seconds=3):
        self.auth_frames = []
        self.is_recording_auth = True
        time.sleep(seconds) # 錄製指定的秒數
        self.is_recording_auth = False
        
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.auth_frames))
        wf.close()
        return True

audio_manager = AudioStream()

# --- 影像串流：僅保留波形顯示，移除手勢 ---
def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): cap = cv2.VideoCapture(1)

    while True:
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)

        # 繪製即時聲紋波形
        points = audio_manager.get_waveform_points()
        if len(points) > 0:
            cv2.polylines(frame, [points], isClosed=False, color=WAVE_COLOR, thickness=2)
            cv2.line(frame, (0, WAVE_Y_OFFSET), (WIDTH, WAVE_Y_OFFSET), (100, 100, 100), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- 路由邏輯 ---
@app.route('/')
def index():
    if session.get('logged_in'): return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'): return redirect(url_for('index'))
    return "<h1>Hello World! 你已登入</h1><p>這是機密網頁內容。</p><a href='/logout'>登出</a>"

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/verify', methods=['POST'])
def api_verify():
    ip = request.remote_addr
    now = time.time()
    user_record = cooldown_cache.get(ip, {"count": 0, "lock_until": 0})

    if now < user_record["lock_until"]:
        return jsonify({"status": "locked", "message": f"冷卻中，剩餘 {int(user_record['lock_until'] - now)} 秒"})

    name = request.json.get('name')
    if audio_manager.record_auth_clip("temp_verify.wav"):
        res_name, score = recognizer.verify_user("temp_verify.wav")
        if res_name == name and score >= 0.75: # 路線一：成功
            user_record["count"] = 0
            cooldown_cache[ip] = user_record
            session['logged_in'] = True
            return jsonify({"status": "success"})
        else: # 路線二：失敗與冷卻
            user_record["count"] += 1
            if user_record["count"] >= 3:
                user_record["lock_until"] = now + 300
            cooldown_cache[ip] = user_record
            return jsonify({"status": "failed", "count": user_record["count"]})

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.json
    name, step = data.get('name'), data.get('step')
    filename = f"reg_{name}_{step}.wav"
    
    if audio_manager.record_auth_clip(filename):
        if step == 3: # 路線三：完成三次錄音
            recognizer.register_user(name, [f"reg_{name}_1.wav", f"reg_{name}_2.wav", f"reg_{name}_3.wav"])
            return jsonify({"status": "completed"})
        return jsonify({"status": "next", "step": step + 1})
    return jsonify({"status": "error"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)