import os
import torch
import torchaudio
import pickle
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier

class SpeakerRecognizer:
    def __init__(self, db_path="speaker_db.pkl"):
        self.db_path = db_path
        self.db = self.load_db()
        
        print("\n⏳ [AI 聲紋] 正在載入 SpeechBrain 模型 (初次執行需下載)...")
        # 自動判斷是否有 GPU (Jetson 上會啟用 CUDA)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir="tmp_model",
            run_opts={"device": self.device}
        )
        print(f"✅ [AI 聲紋] 模型載入完成！使用設備: {self.device}")

    def load_db(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                return pickle.load(f)
        return {}

    def save_db(self):
        with open(self.db_path, "wb") as f:
            pickle.dump(self.db, f)

    def extract_embedding(self, wav_file):
        """將音訊檔轉換為 192 維特徵向量"""
        signal, fs = torchaudio.load(wav_file)
        embeddings = self.classifier.encode_batch(signal)
        return embeddings.squeeze().cpu().numpy()

    def register_user(self, name, wav_files):
        """
        註冊新聲紋並存入資料庫
        支援傳入單一檔案(字串)或多個檔案(列表)，若為多個檔案則取特徵平均值
        """
        embeddings = []
        
        # 確保 wav_files 是一個列表，方便後續迭代
        if isinstance(wav_files, str): 
            wav_files = [wav_files]
        
        # 依序萃取每次錄音的特徵向量
        for f in wav_files:
            emb = self.extract_embedding(f)
            embeddings.append(emb)
        
        # 將多次錄音的特徵值取平均，讓聲紋模型更穩定
        avg_embedding = np.mean(embeddings, axis=0)
        self.db[name] = avg_embedding
        self.save_db()
        return True

    def verify_user(self, wav_file, threshold=0.75):
        """計算餘弦相似度並回傳結果"""
        if not self.db:
            return "DB Empty", 0.0

        new_embedding = self.extract_embedding(wav_file)
        best_name = "Unknown"
        best_score = -1.0

        for name, db_embedding in self.db.items():
            # 餘弦相似度計算
            dot_product = np.dot(new_embedding, db_embedding)
            norm_a = np.linalg.norm(new_embedding)
            norm_b = np.linalg.norm(db_embedding)
            score = dot_product / (norm_a * norm_b + 1e-10)

            if score > best_score:
                best_score = score
                best_name = name

        # 判斷最高分的相似度是否大於設定的門檻值 (預設 0.