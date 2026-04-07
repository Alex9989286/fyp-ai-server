# from fastapi import FastAPI, UploadFile, File
# import shutil
# import os
# import sys

# # ----------------------------
# # 动态加入 src/ 路径，让 Python 能找到 main_system.py
# # ----------------------------
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # 导入你的AI系统
# from main_system import process_audio

# # ----------------------------
# # FastAPI app
# # ----------------------------
# app = FastAPI()

# @app.get("/")
# def home():
#     return {"message": "AI Server Running"}

# @app.post("/detect_sound")
# async def detect_sound(file: UploadFile = File(...)):

#     # 临时保存上传文件
#     temp_path = f"temp_{file.filename}"

#     try:
#         with open(temp_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)

#         # 调用 AI 系统处理音频
#         label, confidence = process_audio(temp_path)

#     finally:
#         # 删除临时文件
#         if os.path.exists(temp_path):
#             os.remove(temp_path)

#     return {
#         "label": label,
#         "confidence": float(confidence)
#     }
from fastapi import FastAPI, UploadFile, File
import shutil
import os
import sys
import uvicorn
import uuid

# =====================================================
# 让 Python 能找到 src/main_system.py
# Render 路径: /opt/render/project/src/src/api/app.py
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# =====================================================
# 导入 AI 系统
# =====================================================
from main_system import process_audio

# =====================================================
# FastAPI App
# =====================================================
app = FastAPI(
    title="AI Sound Detection API",
    description="Sound classification backend",
    version="1.0"
)

# -----------------------------------------------------
# Health Check (Render 会用到)
# -----------------------------------------------------
@app.get("/")
def home():
    return {"message": "AI Server Running ✅"}


# -----------------------------------------------------
# Sound Detection Endpoint
# -----------------------------------------------------
@app.post("/detect_sound")
async def detect_sound(file: UploadFile = File(...)):

    # 使用唯一文件名（避免并发冲突）
    temp_path = f"temp_{uuid.uuid4().hex}_{file.filename}"

    try:
        # 保存上传文件
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # AI 推理
        label, confidence = process_audio(temp_path)

        return {
            "label": label,
            "confidence": float(confidence)
        }

    finally:
        # 删除临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)


# =====================================================
# ⭐ Render 启动入口（最重要部分）
# =====================================================
if __name__ == "__main__":
    # Render 会自动提供 PORT
    port = int(os.environ.get("PORT", 10000))

    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
