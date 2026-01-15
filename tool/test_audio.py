import pyaudio
import wave
import sys

# 設定
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
SECONDS = 5
FILENAME = "test_recording.wav"

p = pyaudio.PyAudio()

# 自動尋找 C270
dev_index = None
for i in range(p.get_device_count()):
    if "C270" in p.get_device_info_by_index(i).get('name'):
        dev_index = i
        break

print(f"🎙️  開始錄音 5 秒... (使用裝置 ID: {dev_index})")

try:
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, 
                    input=True, input_device_index=dev_index, 
                    frames_per_buffer=CHUNK)
except Exception as e:
    print(f"❌ 錯誤: {e}")
    sys.exit()

frames = []
for i in range(0, int(RATE / CHUNK * SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("✅ 錄音結束，正在存檔...")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

print(f"💾 檔案已儲存為: {FILENAME}")