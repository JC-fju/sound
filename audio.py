import cv2
import mediapipe as mp
import pyaudio
import numpy as np
import threading
import collections
import time

# --- 參數設定 ---
WIDTH, HEIGHT = 640, 480    # 視窗大小
CHUNK = 1024                # 音訊緩衝區
FORMAT = pyaudio.paInt16    # 16-bit 格式
CHANNELS = 1                # C270 是單聲道
RATE = 16000                # 取樣率
WAVE_HEIGHT = 80            # 波形圖高度import cv2
import mediapipe as mp
import pyaudio
import numpy as np
import threading
import collections
import time

# --- 參數設定 ---
WIDTH, HEIGHT = 640, 480    # 視窗大小
CHUNK = 1024                # 音訊緩衝區
FORMAT = pyaudio.paInt16    # 16-bit 格式
CHANNELS = 1                # C270 是單聲道
RATE = 16000                # 取樣率
WAVE_HEIGHT = 80            # 波形圖高度
WAVE_Y_OFFSET = 400         # 波形圖中心點 Y 座標
WAVE_COLOR = (0, 255, 0)    # 綠色 (B, G, R)

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
        print("\n🔍 正在搜尋 Logitech C270 麥克風...")
        cnt = self.p.get_device_count()
        found = False
        idx = None
        for i in range(cnt):
            try:
                info = self.p.get_device_info_by_index(i)
                name = info.get('name')
                if ("C270" in name or "USB" in name) and info.get('maxInputChannels') > 0:
                    print(f"✅ 找到裝置 Index {i}: {name}")
                    idx = i
                    found = True
                    # 找到一個就先設為候選，不要 break，以免有更精確的匹配，或是直接用這個
                    break 
            except Exception:
                continue
        
        if not found:
            print("⚠️ 未找到特定麥克風，將使用系統預設裝置。")
            return None
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
            print("🎙️  背景錄音執行緒已啟動")
        except Exception as e:
            print(f"❌ 開啟音訊失敗: {e}")

    def _record_loop(self):
        while self.running:
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                int_data = np.frombuffer(data, dtype=np.int16)
                with self.lock:
                    normalized = (int_data[::2] / 150).astype(int)
                    self.audio_buffer.extend(normalized)
            except Exception as e:
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

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

def main():
    print("🚀 程式初始化中...")
    audio = AudioStream()
    audio.start()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    # 嘗試開啟鏡頭，如果 0 失敗就試試 1
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("⚠️ 鏡頭 0 無法開啟，嘗試鏡頭 1...")
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("❌ 錯誤：找不到任何攝影機！程式即將結束。")
        return

    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)

    print("\n✅ 系統啟動完成！按 'q' 退出。")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("無法讀取影像幀")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        points = audio.get_waveform_points()
        if len(points) > 0:
            cv2.polylines(frame, [points], isClosed=False, color=WAVE_COLOR, thickness=2)
            cv2.line(frame, (0, WAVE_Y_OFFSET), (WIDTH, WAVE_Y_OFFSET), (100, 100, 100), 1)

        cv2.putText(frame, "C270 Audio Visualizer", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Jetson Orin Nano - Fusion", frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    audio.stop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
WAVE_Y_OFFSET = 400         # 波形圖中心點 Y 座標
WAVE_COLOR = (0, 255, 0)    # 綠色 (B, G, R)

class AudioStream:
    """
    背景錄音類別：負責在獨立執行緒中擷取麥克風數據
    """
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.running = False
        self.lock = threading.Lock()
        # 雙向佇列，只保留最新的數據供繪圖用
        self.audio_buffer = collections.deque(maxlen=WIDTH)
        self.audio_buffer.extend([0] * WIDTH)
        self.device_index = self.find_c270_index()

    def find_c270_index(self):
        print("\n🔍 正在搜尋 Logitech C270 麥克風...")
        cnt = self.p.get_device_count()
        for i in range(cnt):
            info = self.p.get_device_info_by_index(i)
            name = info.get('name')
            # 判斷裝置名稱
            if ("C270" in name or "USB" in name) and info.get('maxInputChannels') > 0:
                print(f"✅ 找到裝置 Index {i}: {name}")
                return i
        print("⚠️ 未找到特定麥克風，將使用系統預設裝置。")
        return None

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
            print("🎙️  背景錄音執行緒已啟動")
        except Exception as e:
            print(f"❌ 開啟音訊失敗: {e}")

    def _record_loop(self):
        while self.running:
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                int_data = np.frombuffer(data, dtype=np.int16)
                
                with self.lock:
                    # 降低解析度以符合畫面寬度，並縮放振幅
                    normalized = (int_data[::2] / 150).astype(int)
                    self.audio_buffer.extend(normalized)
            except Exception as e:
                pass

    def get_waveform_points(self):
        with self.lock:
            data = list(self.audio_buffer)
        
        # 取最後 WIDTH 個點
        data = data[-WIDTH:]
        points = []
        for x, val in enumerate(data):
            y = WAVE_Y_OFFSET - val
            # 限制範圍
            y = max(WAVE_Y_OFFSET - WAVE_HEIGHT, min(WAVE_Y_OFFSET + WAVE_HEIGHT, y))
            points.append([x, y])
        return np.array(points, np.int32)

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

def main():
    # 1. 啟動音訊
    audio = AudioStream()
    audio.start()

    # 2. 設定 MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    # 3. 啟動攝影機
    cap = cv2.VideoCapture(0)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)

    print("\n🚀 程式已啟動！按 'q' 退出。")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 鏡像與轉色
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 手勢偵測
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        # 繪製聲波
        points = audio.get_waveform_points()
        if len(points) > 0:
            cv2.polylines(frame, [points], isClosed=False, color=WAVE_COLOR, thickness=2)
            cv2.line(frame, (0, WAVE_Y_OFFSET), (WIDTH, WAVE_Y_OFFSET), (100, 100, 100), 1)

        # 介面資訊
        cv2.putText(frame, "C270 Audio Visualizer", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Jetson Orin Nano - Fusion", frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    audio.stop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()