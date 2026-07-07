# 🧠 FYP AI Server - Sound Classification API

A production-ready sound classification backend service built with **TensorFlow 2.12** and **FastAPI**. This is the AI backend for my Final Year Project, designed to detect and classify environmental sounds such as sirens, car horns, dog barks, and more.

## 🚀 Tech Stack

- **AI Framework**: TensorFlow 2.12.0 / Keras
- **API Framework**: FastAPI 0.110.0
- **Audio Processing**: Librosa 0.10.1, SoundFile
- **ML Libraries**: NumPy 1.23.5, Scikit-learn 1.2.2, SciPy 1.10.1
- **Server**: Uvicorn (ASGI)
- **Deployment**: Render.com
- **Python Version**: 3.11.8

## 📋 Features

- ✅ Sound classification (6 environmental sound classes)
- ✅ RESTful API with automatic OpenAPI documentation
- ✅ File upload via multipart/form-data
- ✅ Sliding window prediction for variable-length audio
- ✅ Rule-based decision engine with priority levels
- ✅ CORS support for frontend integration
- ✅ Lazy model loading for cloud deployment
- ✅ Render ready with render.yaml

## 🎯 Supported Sound Classes

| Class | Action | Priority | Threshold |
|-------|--------|----------|-----------|
| 🚨 Siren | EMERGENCY_ALERT | 3 | 0.70 |
| 📯 Car Horn | WARNING | 2 | 0.65 |
| 🚪 Door Knock | CHECK | 1 | 0.60 |
| ⏰ Clock Alarm | NOTIFY | 1 | 0.60 |
| 👣 Footsteps | MONITOR | 1 | 0.55 |
| 🐕 Dog | IGNORE | 0 | 0.50 |

## 🛠️ Local Setup

### Prerequisites
- [Python 3.11.8](https://www.python.org/downloads/)
- [Git](https://git-scm.com/)

### Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Alex9989286/fyp-ai-server.git

# 2. Navigate to project directory
cd fyp-ai-server

# 3. Create virtual environment
# Windows:
python -m venv venv
venv\Scripts\activate

# macOS/Linux:
python3 -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the server
cd src/api
python app.py
```

Open your browser and navigate to `http://localhost:8000`

### Test the API

```bash
# Health check
curl http://localhost:8000/

# Sound detection
curl -X POST "http://localhost:8000/detect_sound" \
  -F "file=@test_audio/dog.wav"
```

## 📁 Project Structure

```
fyp-ai-server/
├── src/
│   ├── api/
│   │   └── app.py              # FastAPI application entry point
│   ├── business/
│   │   ├── actions.py          # Action handlers
│   │   ├── logic.py            # Decision engine
│   │   └── rules.py            # Sound classification rules
│   ├── main_system.py          # Core AI inference pipeline
│   ├── train_model.py          # Model training script
│   ├── convert_model.py        # Model format converter
│   └── preprocess.py           # Audio preprocessing utilities
├── models/
│   └── sound_class_model_mfcc_opt.h5   # Trained model
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Python version
├── render.yaml                 # Render deployment config
└── .gitignore                  # Git ignore rules
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/detect_sound` | Upload audio for classification |

### POST `/detect_sound`

**Request:**
- `Content-Type`: `multipart/form-data`
- `file`: Audio file (WAV, MP3, etc.)

**Response:**
```json
{
  "label": "dog",
  "confidence": 0.85
}
```

**Error Response:**
```json
{
  "error": "Audio too short (minimum 2 seconds required)"
}
```

### Interactive API Docs

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🧠 AI Model Details

### Feature Extraction
- **MFCC**: 40 coefficients
- **Delta** & **Delta-Delta** features
- **Feature Shape**: (120, 128, 1) for CNN input

### Model Architecture

```
Input (120, 128, 1)
    ↓
Conv2D (32) + BN + MaxPool
    ↓
Conv2D (64) + BN + MaxPool
    ↓
Conv2D (128) + BN + GlobalAvgPool
    ↓
Dropout (0.4) + Dense (128) + Dropout (0.4)
    ↓
Softmax (6 classes)
```

### Training Details
- **Dataset**: ESC-50 subset (6 classes)
- **Data Augmentation**: Volume, noise, time-stretch, pitch-shift
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy
- **Early Stopping**: patience=15

## 🌐 Deployment

### Option 1: Deploy to Render (Free Tier)

1. Push code to GitHub
2. Go to [Render.com](https://render.com)
3. Click **New +** → **Web Service**
4. Connect your GitHub repository
5. Set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `cd src/api && uvicorn app:app --host 0.0.0.0 --port $PORT`
6. Click **Create Web Service**

Your API will be available at: `https://fyp-ai-server.onrender.com/`

### Option 2: Render Blueprint (Auto-Deploy)

1. Push code to GitHub
2. Go to [Render.com](https://render.com)
3. Click **New +** → **Blueprint**
4. Connect your GitHub repository
5. Render auto-detects `render.yaml`
6. Click **Apply**

### Environment Variables (Optional)

| Variable | Value | Description |
|----------|-------|-------------|
| `TF_CPP_MIN_LOG_LEVEL` | `2` | Reduce TensorFlow logging |
| `OMP_NUM_THREADS` | `1` | Limit threads for memory |
| `PORT` | Auto-assigned | Render provides this |

## 📸 Screenshots

### API Health Check
<img width="543" height="258" alt="Screenshot 2026-07-07 181034" src="https://github.com/user-attachments/assets/e627f0a0-237c-43f4-b5d5-5a5680ea9b61" />


### Sound Detection Response
<img width="1918" height="1022" alt="Screenshot 2026-07-07 181157" src="https://github.com/user-attachments/assets/e03e3f8b-775b-4ea9-8850-9957a085cf28" />



## 👨‍💻 Author

- **GitHub**: [Alex9989286](https://github.com/Alex9989286)
- **Email**: jianshengalexng17317@gmail.com
- **University**: Xiamen University Malaysia

## 📄 License

This project is for educational purposes only

## 🙏 Acknowledgments

- [ESC-50 Dataset](https://github.com/karolpiczak/ESC-50)
- [Librosa](https://librosa.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Render](https://render.com)
- [TensorFlow](https://www.tensorflow.org/)

---

## 🎯 FYP Project Context

This server serves as the AI backend for my Final Year Project, an **Intelligent Sound Recognition System**. It integrates with a mobile/frontend application to provide real-time sound detection for safety and monitoring purposes.

**Key Use Cases:**
- 🚨 Emergency siren detection
- 🚗 Vehicle horn warning
- 🚪 Door security monitoring
- ⏰ Alarm notification system
- 👣 Footstep tracking
- 🐕 Pet monitoring
