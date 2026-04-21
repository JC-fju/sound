# app.py
import time
from flask import Flask, Response, render_template, request, jsonify, session, redirect, url_for
import config
from voice_auth import SpeakerRecognizer
from audio_core import audio_manager
from video_core import generate_frames

app = Flask(__name__)
app.secret_key = config.SECRET_KEY

recognizer = SpeakerRecognizer()
cooldown_cache = {}

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
        if res_name == name and score >= 0.75:
            user_record["count"] = 0
            cooldown_cache[ip] = user_record
            session['logged_in'] = True
            return jsonify({"status": "success"})
        else:
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
        if step == 3:
            recognizer.register_user(name, [f"reg_{name}_1.wav", f"reg_{name}_2.wav", f"reg_{name}_3.wav"])
            return jsonify({"status": "completed"})
        return jsonify({"status": "next", "step": step + 1})
    return jsonify({"status": "error"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)