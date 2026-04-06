from fastapi import FastAPI, UploadFile, File
import shutil
import os
import sys

# ----------------------------
# 动态加入 src/ 路径，让 Python 能找到 main_system.py
# ----------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入你的AI系统
from main_system import process_audio

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI()

@app.get("/")
def home():
    return {"message": "AI Server Running"}

@app.post("/detect_sound")
async def detect_sound(file: UploadFile = File(...)):

    # 临时保存上传文件
    temp_path = f"temp_{file.filename}"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 调用 AI 系统处理音频
        label, confidence = process_audio(temp_path)

    finally:
        # 删除临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "label": label,
        "confidence": float(confidence)
    }
