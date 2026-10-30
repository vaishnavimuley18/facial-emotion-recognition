# AI Facial Emotion Recognition 😊

A deep learning and computer vision web application built with **Streamlit**, **OpenCV**, and **TensorFlow/Keras** to detect and classify facial emotions in real-time from images or a live webcam feed.

---

## 🌟 Features

- **📷 Image Analysis**: Upload images (`.jpg`, `.jpeg`, `.png`) to automatically detect faces, view the cropped face region, and predict facial expressions with confidence scores.
- **🎥 Real-Time Webcam Stream**: Live webcam stream using WebRTC with real-time bounding boxes and emotion classification overlay.
- **⚙️ Multi-Model Architecture**:
  - **Custom CNN (48×48 - Recommended)**: Trained on the FER-2013 dataset with balanced predictions across all 7 emotions.
  - **MobileNetV2 (96×96 - Experimental)**: Transfer learning model architecture.
- **7 Facial Emotion Classes**:
  - 😠 Angry
  - 🤢 Disgust
  - 😨 Fear
  - 😊 Happy
  - 😢 Sad
  - 😲 Surprise
  - 😐 Neutral

---

## 📁 Project Structure

```text
facial-emotion-recognition/
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── emotion_labels.json        # Class mapping for emotions
├── test_image.jpg             # Sample test image
├── test_predictor.py          # Standalone test script
├── model/
│   ├── facial_emotion_cnn.keras            # Custom CNN trained on FER-2013
│   └── final_emotion_mobilenetv2.keras     # MobileNetV2 model
└── src/
    ├── predictor.py           # Face detection & inference pipeline
    └── train_mobilenet.py     # MobileNetV2 training pipeline
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/vaishnavimuley18/facial-emotion-recognition.git
cd facial-emotion-recognition
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🧠 How It Works

1. **Face Detection**: Uses OpenCV's Haar Cascade classifier (`haarcascade_frontalface_default.xml`) to detect and crop face regions.
2. **Preprocessing**: 
   - Custom CNN: Grayscale conversion, resizing to $48 \times 48$, and pixel value normalization ($[0, 1]$).
   - MobileNetV2: RGB conversion, resizing to $96 \times 96$, and normalization.
3. **Inference**: Deep neural network predicts class probability distributions using softmax activation.
4. **Display**: Displays prediction metrics and probability progress bars.

---

## 📜 License
MIT License
