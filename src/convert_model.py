# from tensorflow.keras.models import load_model
# import os

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "models", "sound_class_model_mfcc_opt.keras")

# print("🔄 Loading Keras 3 model...")

# model = load_model(MODEL_PATH, compile=False)

# print("✅ Loaded")

# # 保存到同一个 models 文件夹
# SAVE_PATH = os.path.join(BASE_DIR, "models", "fixed_model.h5")

# model.save(SAVE_PATH)

# print("💾 Saved as fixed_model.h5")
from tensorflow.keras.models import load_model
import os

# 原始模型
INPUT_MODEL = "models/sound_class_model_mfcc_opt.keras"

# 输出模型（重点🔥）
OUTPUT_MODEL = "models/fixed_model.h5"

print("🔄 Loading Keras 3 model...")

model = load_model(INPUT_MODEL, compile=False)

print("✅ Model loaded")

model.save(OUTPUT_MODEL)

print("💾 Saved as fixed_model.h5")
