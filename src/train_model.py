
# #  预备版本
# import os
# import numpy as np
# import pandas as pd
# import librosa
# import tensorflow as tf
# from tensorflow.keras import layers, models
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder


# print("所有库导入成功！")

# # -------------------------------
# # 1️⃣ 设置数据集路径
# # -------------------------------
# DATA_PATH = "../data/ESC-50"  # 根据你的实际路径修改

# # 读取 CSV 文件
# meta = pd.read_csv(os.path.join(DATA_PATH, "meta/esc50.csv"))

# print("CSV中所有的类别有：")
# print(meta['category'].unique())
# print(meta['category'].value_counts())


# # 选择目标类别
# target_classes = ["car_horn", "dog", "door_wood_knock", "clock_alarm", "footsteps","siren"]
# # "siren", "clapping"
# meta = meta[meta["category"].isin(target_classes)]

# def extract_features(file_path, n_mels=256, max_len=128):
#     y, sr = librosa.load(file_path, sr=22050)
#     # 数据增强示例（可选）
#     # y = y * np.random.uniform(0.8, 1.2)  # 随机音量
#     mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
#     mel_db = librosa.power_to_db(mel, ref=np.max)
#     # pad 或 truncate
#     if mel_db.shape[1] < max_len:
#         pad_width = max_len - mel_db.shape[1]
#         mel_db = np.pad(mel_db, ((0,0),(0,pad_width)), mode='constant')
#     else:
#         mel_db = mel_db[:, :max_len]
#     return mel_db

# # -------------------------------
# # 3️⃣ 准备数据
# # -------------------------------
# X = []
# y = []

# print("开始提取音频特征...")
# for _, row in meta.iterrows():
#     file_path = os.path.join(DATA_PATH, "audio", row["filename"])
#     features = extract_features(file_path)
#     X.append(features)
#     y.append(row["category"])
# print("音频特征提取完成！")

# X = np.array(X)[..., np.newaxis]  # CNN 输入 (samples, n_mels, time, 1)

# # 标签编码
# le = LabelEncoder()
# y_encoded = le.fit_transform(y)
# y_onehot = tf.keras.utils.to_categorical(y_encoded)

# num_classes = y_onehot.shape[1]
# print(f"检测到实际类别数量: {num_classes}")

# # -------------------------------
# # 4️⃣ 划分训练集 / 测试集
# # -------------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_onehot, test_size=0.2, random_state=42)
# print("训练集 / 测试集划分完成！")

# # -------------------------------
# # 5️⃣ 建立 CNN 模型
# # -------------------------------
# model = models.Sequential([
#     layers.Input(shape=(256, 128, 1)),  # n_mels=256, max_len=128
#     layers.Conv2D(32, (3,3), activation='relu'),
#     layers.MaxPooling2D((2,2)),
#     layers.Conv2D(64, (3,3), activation='relu'),
#     layers.MaxPooling2D((2,2)),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(num_classes, activation='softmax')
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()

# # -------------------------------
# # 6️⃣ 训练模型
# # -------------------------------
# history = model.fit(
#     X_train, y_train,
#     epochs=50,
#     batch_size=16,
#     validation_data=(X_test, y_test)
# )
# print("模型训练完成！")

# # -------------------------------
# # 7️⃣ 保存模型
# # -------------------------------
# model.save("sound_class_model.h5")
# print("模型已保存为 sound_class_model.h5")

# # -------------------------------
# # 8️⃣ 测试单个音频 Top-k
# # -------------------------------
# def predict_audio_target(file_path, top_k=5):
#     feature = extract_features(file_path)
#     feature = feature[np.newaxis, ..., np.newaxis]

#     pred = model.predict(feature)[0]
#     top_indices = np.argsort(pred)[-top_k:][::-1]

#     print(f"\nTop-{top_k} 预测结果 for {file_path}:")
#     for i in top_indices:
#         label = le.inverse_transform([i])[0]
#         print(f"{label}: {pred[i]:.3f}")

# # -------------------------------
# # 9️⃣ 测试音频路径
# # -------------------------------
# test_files = [
#     "../test_audio/dog.wav",
#     "../test_audio/horn.wav",
#     "../test_audio/siren.wav"
# ]

# for f in test_files:
#     predict_audio_target(f)



# # 测试声音
# import os
# import random
# import librosa
# import soundfile as sf
# # import IPython.display as ipd  # 如果你用 Jupyter/Colab 可以用这个播放

# # -------------------------------
# # 1️⃣ 设置路径和参数
# # -------------------------------
# DATA_PATH = "../data/ESC-50"  # ESC-50 数据集路径
# TARGET_CLASS = "car_horn"
# SEGMENT_DURATION = 1.0  # 秒
# SAMPLE_RATE = 22050

# # -------------------------------
# # 2️⃣ 读取 metadata 并筛选目标类别
# # -------------------------------
# import pandas as pd
# meta = pd.read_csv(os.path.join(DATA_PATH, "meta/esc50.csv"))
# siren_files = meta[meta["category"] == TARGET_CLASS]["filename"].tolist()

# if len(siren_files) == 0:
#     raise ValueError(f"在 {DATA_PATH} 中没有找到 {TARGET_CLASS} 类别的音频")

# # -------------------------------
# # 3️⃣ 随机抽取音频文件
# # -------------------------------
# file_choice = random.choice(siren_files)
# file_path = os.path.join(DATA_PATH, "audio", file_choice)
# print(f"随机抽取的 siren 音频: {file_path}")

# # -------------------------------
# # 4️⃣ 读取音频
# # -------------------------------
# y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

# # -------------------------------
# # 5️⃣ 随机切出 1 秒段
# # -------------------------------
# segment_length = int(SEGMENT_DURATION * sr)
# if len(y) > segment_length:
#     start = random.randint(0, len(y) - segment_length)
#     y_segment = y[start:start + segment_length]
# else:
#     # 如果不足 1 秒就 padding
#     y_segment = np.pad(y, (0, segment_length - len(y)))

# # -------------------------------
# # 6️⃣ 播放音频（Jupyter/Colab）或保存
# # -------------------------------
# # 如果在 Notebook，可以用以下播放：
# # ipd.Audio(y_segment, rate=sr)






# # 普通脚本保存播放import os
# import os
# import numpy as np
# import pandas as pd
# import librosa
# import tensorflow as tf
# from tensorflow.keras import layers, models
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# import random

# print("所有库导入成功！")

# # -------------------------------
# # 1️⃣ 设置数据集路径
# # -------------------------------
# DATA_PATH = "../data/ESC-50"

# # 读取 CSV 文件
# meta = pd.read_csv(os.path.join(DATA_PATH, "meta/esc50.csv"))

# # 选择目标类别
# target_classes = ["car_horn", "dog", "door_wood_knock", "clock_alarm", "footsteps", "siren"]
# meta = meta[meta["category"].isin(target_classes)]

# # -------------------------------
# # 2️⃣ 特征提取函数（增强版 MFCC）
# # -------------------------------
# def extract_features(file_path, sr=22050, n_mfcc=40, max_len=128, augment=False):
#     y, sr = librosa.load(file_path, sr=sr)
    
#     if augment:
#         # 随机音量 + 噪声
#         y = y * np.random.uniform(0.9, 1.1)
#         y = y + 0.002 * np.random.randn(len(y))
#         # 时间拉伸
#         rate = np.random.uniform(0.9, 1.1)
#         y = librosa.effects.time_stretch(y, rate=rate)
#         # 音高偏移
#         n_steps = np.random.uniform(-2, 2)
#         y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
    
#     # MFCC + Delta + Delta-Delta
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
#     delta = librosa.feature.delta(mfcc)
#     delta2 = librosa.feature.delta(mfcc, order=2)
#     features = np.concatenate([mfcc, delta, delta2], axis=0)
    
#     # pad 或 truncate
#     if features.shape[1] < max_len:
#         pad_width = max_len - features.shape[1]
#         features = np.pad(features, ((0,0),(0,pad_width)), mode='constant')
#     else:
#         features = features[:, :max_len]
    
#     return features

# # -------------------------------
# # 3️⃣ 直接处理音频数组 y（预测用）
# # -------------------------------
# def extract_features_from_signal(y, sr=22050, n_mfcc=40, max_len=128):
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
#     delta = librosa.feature.delta(mfcc)
#     delta2 = librosa.feature.delta(mfcc, order=2)
#     features = np.concatenate([mfcc, delta, delta2], axis=0)
    
#     if features.shape[1] < max_len:
#         pad_width = max_len - features.shape[1]
#         features = np.pad(features, ((0,0),(0,pad_width)), mode='constant')
#     else:
#         features = features[:, :max_len]
    
#     return features

# # -------------------------------
# # 4️⃣ 准备训练数据
# # -------------------------------
# X, y = [], []

# print("开始提取音频特征...")
# for _, row in meta.iterrows():
#     file_path = os.path.join(DATA_PATH, "audio", row["filename"])
#     features = extract_features(file_path, augment=True)
#     X.append(features)
#     y.append(row["category"])
# print("音频特征提取完成！")

# X = np.array(X)[..., np.newaxis]  # (samples, features, time, 1)
# le = LabelEncoder()
# y_encoded = le.fit_transform(y)
# y_onehot = tf.keras.utils.to_categorical(y_encoded)
# num_classes = y_onehot.shape[1]

# # -------------------------------
# # 5️⃣ 划分训练集 / 测试集
# # -------------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_onehot, test_size=0.2, random_state=42, stratify=y_encoded)

# # -------------------------------
# # 6️⃣ 建立 CNN 模型（更深 + BN + GAP + Dropout）
# # -------------------------------
# model = models.Sequential([
#     layers.Input(shape=X_train.shape[1:]),
#     layers.Conv2D(32, (3,3), activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D((2,2)),
    
#     layers.Conv2D(64, (3,3), activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D((2,2)),
    
#     layers.Conv2D(128, (3,3), activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.GlobalAveragePooling2D(),
    
#     layers.Dropout(0.4),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.4),
#     layers.Dense(num_classes, activation='softmax')
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()

# # -------------------------------
# # 7️⃣ 训练模型（EarlyStopping + ReduceLR）
# # -------------------------------
# callbacks = [
#     tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True),
#     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
# ]

# history = model.fit(
#     X_train, y_train,
#     epochs=100,
#     batch_size=16,
#     validation_data=(X_test, y_test),
#     callbacks=callbacks
# )
# print("模型训练完成！")

# # -------------------------------
# # 8️⃣ 保存模型
# # -------------------------------
# model.save("sound_class_model_mfcc_opt.h5")
# print("模型已保存为 sound_class_model_mfcc_opt.h5")

# # -------------------------------
# # 9️⃣ 多段平均预测函数（segments 增加到10）
# # -------------------------------
# def predict_audio_stable(file_path, top_k=5, segments=10):
#     y, sr = librosa.load(file_path, sr=22050)
#     segment_len = sr * 2  # 2秒每段
#     preds = []
    
#     for _ in range(segments):
#         start = random.randint(0, max(0, len(y)-segment_len))
#         y_seg = y[start:start+segment_len]
#         features = extract_features_from_signal(y_seg)
#         features = features[np.newaxis, ..., np.newaxis]
#         pred = model.predict(features, verbose=0)[0]
#         preds.append(pred)
    
#     pred_avg = np.mean(preds, axis=0)
#     top_indices = np.argsort(pred_avg)[-top_k:][::-1]
    
#     print(f"\nTop-{top_k} 预测结果 for {file_path} (平均概率):")
#     for i in top_indices:
#         label = le.inverse_transform([i])[0]
#         print(f"{label}: {pred_avg[i]:.3f}")

# # -------------------------------
# # 10️⃣ 测试音频
# # -------------------------------
# test_files = [
#     "../test_audio/dog.wav",
#     "../test_audio/horn.wav",
#     "../test_audio/siren.wav",
#     "../test_audio/clock.wav",
#     "../test_audio/footsteps.wav",
#     "../test_audio/footsteps1.wav"
# ]

# for f in test_files:
#     predict_audio_stable(f)

# import os
# import numpy as np
# import pandas as pd
# import librosa
# import tensorflow as tf
# from tensorflow.keras import layers, models
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# import random

# print("所有库导入成功！")

# # ===============================
# # 0️⃣ 固定随机性（超级重要）
# # ===============================
# SEED = 42

# os.environ["PYTHONHASHSEED"] = str(SEED)
# random.seed(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)

# # ===============================
# # 1️⃣ 数据路径
# # ===============================
# DATA_PATH = "../data/ESC-50"

# meta = pd.read_csv(os.path.join(DATA_PATH, "meta/esc50.csv"))

# target_classes = [
#     "car_horn", "dog", "door_wood_knock",
#     "clock_alarm", "footsteps", "siren"
# ]

# meta = meta[meta["category"].isin(target_classes)]

# # ===============================
# # 2️⃣ 特征提取
# # ===============================
# def extract_features(file_path, sr=22050, n_mfcc=40, max_len=128, augment=False):

#     y, sr = librosa.load(file_path, sr=sr)

#     if augment:
#         y = y * np.random.uniform(0.9, 1.1)
#         y = y + 0.002 * np.random.randn(len(y))

#         rate = np.random.uniform(0.9, 1.1)
#         y = librosa.effects.time_stretch(y, rate=rate)

#         n_steps = np.random.uniform(-2, 2)
#         y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
#     delta = librosa.feature.delta(mfcc)
#     delta2 = librosa.feature.delta(mfcc, order=2)

#     features = np.concatenate([mfcc, delta, delta2], axis=0)

#     if features.shape[1] < max_len:
#         pad = max_len - features.shape[1]
#         features = np.pad(features, ((0,0),(0,pad)))
#     else:
#         features = features[:, :max_len]

#     return features


# # ===============================
# # 3️⃣ 预测用特征
# # ===============================
# def extract_features_from_signal(y, sr=22050, n_mfcc=40, max_len=128):

#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
#     delta = librosa.feature.delta(mfcc)
#     delta2 = librosa.feature.delta(mfcc, order=2)

#     features = np.concatenate([mfcc, delta, delta2], axis=0)

#     if features.shape[1] < max_len:
#         pad = max_len - features.shape[1]
#         features = np.pad(features, ((0,0),(0,pad)))
#     else:
#         features = features[:, :max_len]

#     return features


# # ===============================
# # 4️⃣ 构建训练数据（原始 + 增强）
# # ===============================
# X, y = [], []

# print("开始提取音频特征...")

# for _, row in meta.iterrows():

#     file_path = os.path.join(DATA_PATH, "audio", row["filename"])

#     try:
#         # 原始
#         X.append(extract_features(file_path, augment=False))
#         y.append(row["category"])

#         # 增强版本
#         X.append(extract_features(file_path, augment=True))
#         y.append(row["category"])

#     except:
#         continue

# print("音频特征提取完成！")

# X = np.array(X)[..., np.newaxis]

# le = LabelEncoder()
# y_encoded = le.fit_transform(y)
# y_onehot = tf.keras.utils.to_categorical(y_encoded)

# # ===============================
# # 5️⃣ 划分数据
# # ===============================
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_onehot,
#     test_size=0.2,
#     random_state=SEED,
#     stratify=y_encoded
# )

# # ===============================
# # 6️⃣ CNN 模型
# # ===============================
# model = models.Sequential([

#     layers.Input(shape=X_train.shape[1:]),

#     layers.Conv2D(32,(3,3),activation='relu',padding='same'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(2),

#     layers.Conv2D(64,(3,3),activation='relu',padding='same'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(2),

#     layers.Conv2D(128,(3,3),activation='relu',padding='same'),
#     layers.BatchNormalization(),

#     layers.GlobalAveragePooling2D(),

#     layers.Dropout(0.4),
#     layers.Dense(128,activation='relu'),
#     layers.Dropout(0.4),

#     layers.Dense(y_onehot.shape[1],activation='softmax')
# ])

# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# model.summary()

# # ===============================
# # 7️⃣ 训练
# # ===============================
# callbacks = [
#     tf.keras.callbacks.EarlyStopping(
#         monitor='val_loss',
#         patience=15,
#         restore_best_weights=True
#     ),
#     tf.keras.callbacks.ReduceLROnPlateau(
#         monitor='val_loss',
#         factor=0.5,
#         patience=5,
#         min_lr=1e-6
#     )
# ]

# history = model.fit(
#     X_train, y_train,
#     epochs=100,
#     batch_size=16,
#     validation_data=(X_test, y_test),
#     callbacks=callbacks
# )

# print("模型训练完成！")

# # ===============================
# # 8️⃣ 保存模型
# # ===============================
# model.save("sound_class_model_mfcc_opt.h5")

# # ===============================
# # 9️⃣ ⭐稳定预测（固定切段）
# # ===============================
# def predict_audio_stable(file_path, top_k=5):

#     y, sr = librosa.load(file_path, sr=22050)

#     segment_len = sr * 2
#     step = segment_len // 2   # 固定滑动窗口

#     preds = []

#     for start in range(0, len(y)-segment_len, step):

#         y_seg = y[start:start+segment_len]

#         features = extract_features_from_signal(y_seg)
#         features = features[np.newaxis, ..., np.newaxis]

#         pred = model.predict(features, verbose=0)[0]
#         preds.append(pred)

#     pred_avg = np.mean(preds, axis=0)

#     top_indices = np.argsort(pred_avg)[-top_k:][::-1]

#     print(f"\nTop-{top_k} 预测结果 for {file_path} (平均概率):")

#     for i in top_indices:
#         label = le.inverse_transform([i])[0]
#         print(f"{label}: {pred_avg[i]:.3f}")


# # ===============================
# # 🔟 测试
# # ===============================
# test_files = [
#     "../test_audio/dog.wav",
#     "../test_audio/horn.wav",
#     "../test_audio/siren.wav",
#     "../test_audio/clock.wav",
#     "../test_audio/footsteps.wav"
# ]

# for f in test_files:
#     predict_audio_stable(f)






# import os
# import numpy as np
# import pandas as pd
# import librosa
# import tensorflow as tf
# from tensorflow.keras import layers, models
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# import random

# print("All libraries imported successfully!")

# # ===============================
# # 0️⃣ Fix randomness (VERY IMPORTANT)
# # ===============================
# SEED = 42

# os.environ["PYTHONHASHSEED"] = str(SEED)
# random.seed(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)

# # ===============================
# # 1️⃣ Dataset path
# # ===============================
# DATA_PATH = "../data/ESC-50"

# meta = pd.read_csv(os.path.join(DATA_PATH, "meta/esc50.csv"))

# target_classes = [
#     "car_horn", "dog", "door_wood_knock",
#     "clock_alarm", "footsteps", "siren"
# ]

# meta = meta[meta["category"].isin(target_classes)]

# # ===============================
# # 2️⃣ Feature extraction
# # ===============================
# def extract_features(file_path, sr=22050, n_mfcc=40, max_len=128, augment=False):

#     y, sr = librosa.load(file_path, sr=sr)

#     if augment:
#         # Random volume scaling + noise injection
#         y = y * np.random.uniform(0.9, 1.1)
#         y = y + 0.002 * np.random.randn(len(y))

#         # Time stretching
#         rate = np.random.uniform(0.9, 1.1)
#         y = librosa.effects.time_stretch(y, rate=rate)

#         # Pitch shifting
#         n_steps = np.random.uniform(-2, 2)
#         y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

#     # MFCC + Delta + Delta-Delta features
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
#     delta = librosa.feature.delta(mfcc)
#     delta2 = librosa.feature.delta(mfcc, order=2)

#     features = np.concatenate([mfcc, delta, delta2], axis=0)

#     # Pad or truncate to fixed length
#     if features.shape[1] < max_len:
#         pad = max_len - features.shape[1]
#         features = np.pad(features, ((0,0),(0,pad)))
#     else:
#         features = features[:, :max_len]

#     return features


# # ===============================
# # 3️⃣ Feature extraction for prediction
# # ===============================
# def extract_features_from_signal(y, sr=22050, n_mfcc=40, max_len=128):

#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
#     delta = librosa.feature.delta(mfcc)
#     delta2 = librosa.feature.delta(mfcc, order=2)

#     features = np.concatenate([mfcc, delta, delta2], axis=0)

#     # Pad or truncate
#     if features.shape[1] < max_len:
#         pad = max_len - features.shape[1]
#         features = np.pad(features, ((0,0),(0,pad)))
#     else:
#         features = features[:, :max_len]

#     return features


# # ===============================
# # 4️⃣ Build training dataset (original + augmented)
# # ===============================
# X, y = [], []

# print("Starting audio feature extraction...")

# for _, row in meta.iterrows():

#     file_path = os.path.join(DATA_PATH, "audio", row["filename"])

#     try:
#         # Original sample
#         X.append(extract_features(file_path, augment=False))
#         y.append(row["category"])

#         # Augmented sample
#         X.append(extract_features(file_path, augment=True))
#         y.append(row["category"])

#     except:
#         continue

# print("Audio feature extraction completed!")

# X = np.array(X)[..., np.newaxis]

# le = LabelEncoder()
# y_encoded = le.fit_transform(y)
# y_onehot = tf.keras.utils.to_categorical(y_encoded)

# # ===============================
# # 5️⃣ Train/Test split
# # ===============================
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_onehot,
#     test_size=0.2,
#     random_state=SEED,
#     stratify=y_encoded
# )

# # ===============================
# # 6️⃣ CNN model architecture
# # ===============================
# model = models.Sequential([

#     layers.Input(shape=X_train.shape[1:]),

#     layers.Conv2D(32,(3,3),activation='relu',padding='same'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(2),

#     layers.Conv2D(64,(3,3),activation='relu',padding='same'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(2),

#     layers.Conv2D(128,(3,3),activation='relu',padding='same'),
#     layers.BatchNormalization(),

#     layers.GlobalAveragePooling2D(),

#     layers.Dropout(0.4),
#     layers.Dense(128,activation='relu'),
#     layers.Dropout(0.4),

#     layers.Dense(y_onehot.shape[1],activation='softmax')
# ])

# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# model.summary()

# # ===============================
# # 7️⃣ Model training
# # ===============================
# callbacks = [
#     tf.keras.callbacks.EarlyStopping(
#         monitor='val_loss',
#         patience=15,
#         restore_best_weights=True
#     ),
#     tf.keras.callbacks.ReduceLROnPlateau(
#         monitor='val_loss',
#         factor=0.5,
#         patience=5,
#         min_lr=1e-6
#     )
# ]

# history = model.fit(
#     X_train, y_train,
#     epochs=100,
#     batch_size=16,
#     validation_data=(X_test, y_test),
#     callbacks=callbacks
# )

# print("Model training completed!")

# # ===============================
# # 8️⃣ Save model
# # ===============================
# model.save("sound_class_model_mfcc_opt.h5")

# # ===============================
# # 9️⃣ Stable prediction (fixed segmentation)
# # ===============================
# def predict_audio_stable(file_path, top_k=5):

#     y, sr = librosa.load(file_path, sr=22050)

#     segment_len = sr * 2
#     step = segment_len // 2   # fixed sliding window

#     preds = []

#     for start in range(0, len(y)-segment_len, step):

#         y_seg = y[start:start+segment_len]

#         features = extract_features_from_signal(y_seg)
#         features = features[np.newaxis, ..., np.newaxis]

#         pred = model.predict(features, verbose=0)[0]
#         preds.append(pred)

#     pred_avg = np.mean(preds, axis=0)

#     top_indices = np.argsort(pred_avg)[-top_k:][::-1]

#     print(f"\nTop-{top_k} prediction results for {file_path} (average probability):")

#     # 更改部分
#     # for i in top_indices:
#     #     label = le.inverse_transform([i])[0]
#     #     print(f"{label}: {pred_avg[i]:.3f}")
#     from business.logic import decide_and_execute

# # 打包 top-k 结果
#     top_labels_conf = [(le.inverse_transform([i])[0], pred_avg[i]) for i in top_indices]

# # 调用你的 business logic
#     decide_and_execute(top_labels_conf)



# # ===============================
# # 🔟 Testing
# # ===============================
# test_files = [
#     "../test_audio/dog.wav",
#     "../test_audio/horn.wav",
#     "../test_audio/siren.wav",
#     "../test_audio/clock.wav",
#     "../test_audio/footsteps.wav"
# ]

# # for f in test_files:
# #     predict_audio_stable(f)

#     # 🔟 Testing with business logic
# for f in test_files:
#     predict_audio_stable(f)

import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random

print("All libraries imported successfully!")

# ===============================
# 0️⃣ Fix randomness
# ===============================
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ===============================
# 1️⃣ Dataset path
# ===============================
DATA_PATH = "../data/ESC-50"

meta = pd.read_csv(os.path.join(DATA_PATH, "meta/esc50.csv"))

target_classes = [
    "car_horn", "dog", "door_wood_knock",
    "clock_alarm", "footsteps", "siren"
]

meta = meta[meta["category"].isin(target_classes)]

# ===============================
# 2️⃣ Feature extraction
# ===============================
def extract_features(file_path, sr=22050, n_mfcc=40, max_len=128, augment=False):

    y, sr = librosa.load(file_path, sr=sr)

    if augment:
        y = y * np.random.uniform(0.9, 1.1)
        y = y + 0.002 * np.random.randn(len(y))

        rate = np.random.uniform(0.9, 1.1)
        y = librosa.effects.time_stretch(y, rate=rate)

        n_steps = np.random.uniform(-2, 2)
        y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

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


# ===============================
# 3️⃣ Build dataset
# ===============================
X, y = [], []

print("Starting feature extraction...")

for _, row in meta.iterrows():

    file_path = os.path.join(DATA_PATH, "audio", row["filename"])

    try:
        X.append(extract_features(file_path, augment=False))
        y.append(row["category"])

        X.append(extract_features(file_path, augment=True))
        y.append(row["category"])

    except:
        continue

print("Feature extraction done!")

X = np.array(X)[..., np.newaxis]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_onehot = tf.keras.utils.to_categorical(y_encoded)

# ===============================
# 4️⃣ Split dataset
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot,
    test_size=0.2,
    random_state=SEED,
    stratify=y_encoded
)

# ===============================
# 5️⃣ CNN model
# ===============================
model = models.Sequential([

    layers.Input(shape=X_train.shape[1:]),

    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),

    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),

    layers.Dense(y_onehot.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===============================
# 6️⃣ Training
# ===============================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
]

model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

print("Training completed!")

# ===============================
# 7️⃣ ⭐ FIX: SAVE MODEL (IMPORTANT)
# ===============================

# ❌ 不要再用 .h5
# model.save("sound_class_model_mfcc_opt.h5")


# ✅ 正确方式（Render兼容）
model.save("sound_class_model_mfcc_opt.keras")

print("Model saved in .keras format!")
