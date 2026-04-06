import numpy as np
import librosa
import tensorflow as tf
from business.logic import decide_and_execute
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os

# ==========================
# Load trained model
# ==========================
MODEL_PATH = "sound_class_model_mfcc_opt.h5"
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# ==========================
# Label Encoder (same as train)
# ==========================
target_classes = [
    "car_horn", "dog", "door_wood_knock",
    "clock_alarm", "footsteps", "siren"
]
le = LabelEncoder()
le.fit(target_classes)

# ==========================
# Feature extraction function
# ==========================
def extract_features_from_signal(y, sr=22050, n_mfcc=40, max_len=128):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate([mfcc, delta, delta2], axis=0)
    if features.shape[1] < max_len:
        pad = max_len - features.shape[1]
        features = np.pad(features, ((0,0),(0,pad)))
    else:
        features = features[:, :max_len]
    return features

# ==========================
# Predict + Business Logic
# ==========================
def process_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    segment_len = sr * 2
    step = segment_len // 2

    preds = []
    for start in range(0, len(y)-segment_len, step):
        y_seg = y[start:start+segment_len]
        features = extract_features_from_signal(y_seg)
        features = features[np.newaxis, ..., np.newaxis]
        pred = model.predict(features, verbose=0)[0]
        preds.append(pred)

    pred_avg = np.mean(preds, axis=0)

    # 取概率最高的 label
    best_index = np.argmax(pred_avg)
    label = le.inverse_transform([best_index])[0]
    confidence = pred_avg[best_index]

    # print(f"Detected: {label} ({confidence:.2f})")

    # # 调用 Business Logic
    # decide_and_execute(label, confidence)
    print(f"Detected: {label} ({confidence:.2f})")

    decide_and_execute(label, confidence)

    return label, confidence


# ==========================
# Example usage
# ==========================
if __name__ == "__main__":
    test_files = [
        "../test_audio/dog.wav",
        "../test_audio/horn.wav",
        "../test_audio/siren.wav"
    ]

    for f in test_files:
        print(f"\nProcessing {os.path.basename(f)} ...")
        process_audio(f)
