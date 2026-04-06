from fastapi import FastAPI, UploadFile, File
import shutil
import os

# 导入你的AI系统
from main_system import process_audio

app = FastAPI()


@app.get("/")
def home():
    return {"message": "AI Server Running"}


@app.post("/detect_sound")
async def detect_sound(file: UploadFile = File(...)):

    # 临时保存上传文件
    temp_path = f"temp_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 调用 AI
    label, confidence = process_audio(temp_path)

    # 删除临时文件
    os.remove(temp_path)

    return {
        "label": label,
        "confidence": float(confidence)
    }
