from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import subprocess
import pandas as pd

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './test'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'videos' not in request.files:
        return jsonify({"error": "동영상이 포함되어 있지 않습니다."}), 400

    # 🔄 업로드 폴더 정리
    for file in os.listdir(UPLOAD_FOLDER):
        try:
            os.remove(os.path.join(UPLOAD_FOLDER, file))
        except Exception as e:
            print("⚠️ 파일 삭제 오류:", e)

    # ⬆️ 동영상 저장
    for file in request.files.getlist('videos'):
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))

    try:
        result = subprocess.run(
            ["python", "main_.py"],
            capture_output=True,
            text=True,
            encoding="utf-8"
        )
        print("✅ STDOUT:\n", result.stdout)
        print("❌ STDERR:\n", result.stderr)

        output_path = "predictions.csv"
        if not os.path.exists(output_path):
            return jsonify({"error": "❌ predictions.csv 파일이 생성되지 않았습니다."}), 500

        df = pd.read_csv(output_path)

        result = {}
        for col in ["Curl", "Damage", "Width"]:
            counts = df[col].value_counts(normalize=True).round(4) * 100
            result[col] = counts.to_dict()

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
