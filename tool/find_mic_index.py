import pyaudio

p = pyaudio.PyAudio()

print("\n--- 系統音訊裝置列表 ---")
print(f"總裝置數: {p.get_device_count()}\n")

for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    name = info.get('name')
    channels = info.get('maxInputChannels')
    
    # 標記出可能是 C270 的裝置
    mark = ""
    if "C270" in name or "USB" in name:
        mark = "⬅️  (推薦使用)"
    
    if channels > 0:
        print(f"ID {i}: {name} (輸入聲道: {channels}) {mark}")

p.terminate()
print("\n-----------------------")