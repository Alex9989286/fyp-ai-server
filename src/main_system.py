# import numpy as np
# import librosa
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import LabelEncoder
# import os

# # ==========================
# # Business Logic Import
# # ==========================
# # 请确保 business/logic.py 中有 decide_and_execute(label, confidence) 函数
# from business.logic import decide_and_execute

# # ==========================
# # Load trained model safely
# # ==========================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "sound_class_model_mfcc_opt.h5")

# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# model = load_model(MODEL_PATH)
# print("Model loaded successfully!")

# # ==========================
# # Label Encoder (same as training)
# # ==========================
# target_classes = [
#     "car_horn", "dog", "door_wood_knock",
#     "clock_alarm", "footsteps", "siren"
# ]
# le = LabelEncoder()
# le.fit(target_classes)

# # ==========================
# # Feature extraction function
# # ==========================
# def extract_features_from_signal(y, sr=22050, n_mfcc=40, max_len=128):
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
#     delta = librosa.feature.delta(mfcc)
#     delta2 = librosa.feature.delta(mfcc, order=2)
#     features = np.concatenate([mfcc, delta, delta2], axis=0)
#     if features.shape[1] < max_len:
#         pad = max_len - features.shape[1]
#         features = np.pad(features, ((0, 0), (0, pad)))
#     else:
#         features = features[:, :max_len]
#     return features

# # ==========================
# # Predict + Business Logic
# # ==========================
# def process_audio(file_path):
#     # Load audio
#     y, sr = librosa.load(file_path, sr=22050)

#     # Sliding window parameters
#     segment_len = sr * 2  # 2 seconds per segment
#     step = segment_len // 2  # 50% overlap

#     preds = []
#     for start in range(0, len(y) - segment_len, step):
#         y_seg = y[start:start + segment_len]
#         features = extract_features_from_signal(y_seg)
#         features = features[np.newaxis, ..., np.newaxis]  # Add batch & channel dims
#         pred = model.predict(features, verbose=0)[0]
#         preds.append(pred)

#     # Average predictions
#     pred_avg = np.mean(preds, axis=0)

#     # Get highest probability label
#     best_index = np.argmax(pred_avg)
#     label = le.inverse_transform([best_index])[0]
#     confidence = float(pred_avg[best_index])

#     print(f"Detected: {label} ({confidence:.2f})")

#     # Call business logic
#     decide_and_execute(label, confidence)

#     return label, confidence

# # ==========================
# # Example usage for testing
# # ==========================
# if __name__ == "__main__":
#     test_files = [
#         os.path.join(BASE_DIR, "../test_audio/dog.wav"),
#         os.path.join(BASE_DIR, "../test_audio/horn.wav"),
#         os.path.join(BASE_DIR, "../test_audio/siren.wav")
#     ]

#     for f in test_files:
#         print(f"\nProcessing {os.path.basename(f)} ...")
#         process_audio(f)
# import numpy as np
# import librosa
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import LabelEncoder
# import os

# # ==========================
# # Business Logic Import
# # ==========================
# from business.logic import decide_and_execute

# # ==========================
# # Paths
# # ==========================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "sound_class_model_mfcc_opt.h5")

# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# # ==========================
# # Lazy Load Model (🔥 FIX)
# # ==========================
# model = None

# def get_model():
#     """
#     Load model only when first needed.
#     Prevents Render startup timeout.
#     """
#     global model

#     if model is None:
#         print("🔄 Loading AI model...")
#         model = load_model(MODEL_PATH)
#         print("✅ Model loaded successfully!")

#     return model


# # ==========================
# # Label Encoder (same as training)
# # ==========================
# target_classes = [
#     "car_horn",
#     "dog",
#     "door_wood_knock",
#     "clock_alarm",
#     "footsteps",
#     "siren"
# ]

# le = LabelEncoder()
# le.fit(target_classes)


# # ==========================
# # Feature extraction
# # ==========================
# def extract_features_from_signal(y, sr=22050, n_mfcc=40, max_len=128):

#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
#     delta = librosa.feature.delta(mfcc)
#     delta2 = librosa.feature.delta(mfcc, order=2)

#     features = np.concatenate([mfcc, delta, delta2], axis=0)

#     if features.shape[1] < max_len:
#         pad = max_len - features.shape[1]
#         features = np.pad(features, ((0, 0), (0, pad)))
#     else:
#         features = features[:, :max_len]

#     return features


# # ==========================
# # Predict + Business Logic
# # ==========================
# def process_audio(file_path):

#     # 🔥 Lazy load model here
#     model = get_model()

#     # Load audio
#     y, sr = librosa.load(file_path, sr=22050)

#     segment_len = sr * 2      # 2 seconds
#     step = segment_len // 2   # 50% overlap

#     preds = []

#     for start in range(0, len(y) - segment_len, step):

#         y_seg = y[start:start + segment_len]

#         features = extract_features_from_signal(y_seg)
#         features = features[np.newaxis, ..., np.newaxis]

#         pred = model.predict(features, verbose=0)[0]
#         preds.append(pred)

#     if len(preds) == 0:
#         raise ValueError("Audio too short for prediction.")

#     # Average predictions
#     pred_avg = np.mean(preds, axis=0)

#     best_index = np.argmax(pred_avg)
#     label = le.inverse_transform([best_index])[0]
#     confidence = float(pred_avg[best_index])

#     print(f"🎧 Detected: {label} ({confidence:.2f})")

#     # Business logic
#     decide_and_execute(label, confidence)

#     return label, confidence


# # ==========================
# # Local testing
# # ==========================
# if __name__ == "__main__":

#     test_files = [
#         os.path.join(BASE_DIR, "../test_audio/dog.wav"),
#         os.path.join(BASE_DIR, "../test_audio/horn.wav"),
#         os.path.join(BASE_DIR, "../test_audio/siren.wav"),
#     ]

#     for f in test_files:
#         print(f"\nProcessing {os.path.basename(f)}...")
#         process_audio(f)
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import os

# ==========================
# Business Logic Import
# ==========================
from business.logic import decide_and_execute

# ==========================
# Paths
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sound_class_model_mfcc_opt.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# ==========================
# ⭐ FIX 1: Lazy Load Model
# ==========================
model = None

def get_model():
    global model

    if model is None:
        print("🔄 Loading AI model...")

        try:
            # ⭐ FIX 2: force low-level TF loader (IMPORTANT)
            model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False,
                custom_objects=None
            )

            print("✅ Model loaded successfully!")

        except Exception as e:
            print("🔥 Model load failed:", str(e))
            raise e

    return model


# ==========================
# Label Encoder
# ==========================
target_classes = [
    "car_horn",
    "dog",
    "door_wood_knock",
    "clock_alarm",
    "footsteps",
    "siren"
]

le = LabelEncoder()
le.fit(target_classes)


# ==========================
# Feature extraction
# ==========================
def extract_features_from_signal(y, sr=22050, n_mfcc=40, max_len=128):

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.concatenate([mfcc, delta, delta2], axis=0)

    if features.shape[1] < max_len:
        pad = max_len - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad)))
    else:
        features = features[:, :max_len]

    return features


# ==========================
# Predict
# ==========================
def process_audio(file_path):

    model = get_model()

    y, sr = librosa.load(file_path, sr=22050)

    segment_len = sr * 2
    step = segment_len // 2

    preds = []

    for start in range(0, len(y) - segment_len, step):

        y_seg = y[start:start + segment_len]

        features = extract_features_from_signal(y_seg)
        features = features[np.newaxis, ..., np.newaxis]

        pred = model.predict(features, verbose=0)[0]
        preds.append(pred)

    if len(preds) == 0:
        raise ValueError("Audio too short (minimum 2 seconds required)")

    pred_avg = np.mean(preds, axis=0)

    best_index = np.argmax(pred_avg)
    label = le.inverse_transform([best_index])[0]
    confidence = float(pred_avg[best_index])

    print(f"🎧 Detected: {label} ({confidence:.2f})")

    decide_and_execute(label, confidence)

    return label, confidence


# ==========================
# Local test
# ==========================
if __name__ == "__main__":

    test_files = [
        os.path.join(BASE_DIR, "../test_audio/dog.wav"),
        os.path.join(BASE_DIR, "../test_audio/horn.wav"),
        os.path.join(BASE_DIR, "../test_audio/siren.wav"),
    ]

    for f in test_files:
        print(f"\nProcessing {os.path.basename(f)}...")
        process_audio(f)
