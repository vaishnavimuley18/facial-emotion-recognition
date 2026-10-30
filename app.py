import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from streamlit_webrtc import webrtc_streamer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Facial Emotion Recognition",
    page_icon="😊",
    layout="wide"
)


# ============================================================
# MODEL CONFIGURATION & CACHING
# ============================================================

CNN_MODEL_PATH = "model/facial_emotion_cnn.keras"
MOBILENET_MODEL_PATH = "model/final_emotion_mobilenetv2.keras"

EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]


@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model(CNN_MODEL_PATH)


@st.cache_resource
def load_mobilenet_model():
    return tf.keras.models.load_model(MOBILENET_MODEL_PATH)


cnn_model = load_cnn_model()
mobilenet_model = load_mobilenet_model()


# ============================================================
# FACE DETECTOR
# ============================================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


def detect_faces(gray_img):
    faces = face_detector.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )
    if len(faces) == 0:
        # Fallback to slightly more sensitive detection
        faces = face_detector.detectMultiScale(
            gray_img,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(30, 30)
        )
    return faces


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_face(face, use_cnn=True):
    if use_cnn:
        # Custom CNN: Grayscale (48 x 48 x 1)
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(gray, (48, 48))
        face_norm = face_resized.astype(np.float32) / 255.0
        face_input = np.expand_dims(
            np.expand_dims(face_norm, axis=-1),
            axis=0
        )
        probabilities = cnn_model.predict(face_input, verbose=0)[0]
    else:
        # MobileNetV2: RGB (96 x 96 x 3)
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(rgb, (96, 96))
        face_norm = face_resized.astype(np.float32) / 255.0
        face_input = np.expand_dims(face_norm, axis=0)
        probabilities = mobilenet_model.predict(face_input, verbose=0)[0]

    predicted_class = int(np.argmax(probabilities))
    emotion = EMOTIONS[predicted_class]
    confidence = float(probabilities[predicted_class])

    return (
        emotion,
        confidence,
        probabilities
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Model Configuration")

model_choice = st.sidebar.selectbox(
    "Choose Model Architecture",
    [
        "Custom CNN (48×48 - Recommended)",
        "MobileNetV2 (96×96 - Experimental)"
    ],
    index=0,
    help="Custom CNN is specifically trained on FER-2013 and produces balanced predictions across all 7 emotions."
)

use_cnn_model = "Custom CNN" in model_choice

st.sidebar.markdown("---")
st.sidebar.subheader("Model Status")

if use_cnn_model:
    st.sidebar.success("✅ Custom CNN Active")
    st.sidebar.write("**Input:** 48 × 48 (Grayscale)")
    st.sidebar.write("**Dataset:** FER-2013")
    st.sidebar.write("**Classes:** 7 emotions")
    st.sidebar.write("**Validation Accuracy:** ~58% (Balanced across all emotions)")
else:
    st.sidebar.warning("⚠️ MobileNetV2 Active")
    st.sidebar.write("**Input:** 96 × 96 (RGB)")
    st.sidebar.write("**Dataset:** FER-2013")
    st.sidebar.write("**Classes:** 7 emotions")
    st.sidebar.info("Note: The current MobileNetV2 weights exhibit mode collapse toward 'Angry'.")


# ============================================================
# TITLE
# ============================================================

st.title(
    "😊 AI Facial Emotion Recognition"
)

st.subheader(
    "Deep Learning • Computer Vision • Real-Time Recognition"
)

st.write(
    "Analyze facial emotions from an uploaded image "
    "or directly through your webcam."
)


# ============================================================
# TABS
# ============================================================

image_tab, webcam_tab = st.tabs(
    [
        "📷 Image Analysis",
        "🎥 Live Webcam"
    ]
)


# ============================================================
# IMAGE ANALYSIS
# ============================================================

with image_tab:

    st.header(
        "📷 Emotion Recognition from Image"
    )

    uploaded_file = st.file_uploader(
        "Upload a face image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        with col2:

            if st.button(
                "🔍 Analyze Emotion",
                type="primary"
            ):

                # Convert PIL → OpenCV
                image_array = np.array(
                    image
                )

                image_bgr = cv2.cvtColor(
                    image_array,
                    cv2.COLOR_RGB2BGR
                )

                # Detect face
                gray = cv2.cvtColor(
                    image_bgr,
                    cv2.COLOR_BGR2GRAY
                )

                faces = detect_faces(gray)

                if len(faces) == 0:
                    st.warning(
                        "No face detected by the Haar Cascade detector. "
                        "Analyzing entire image as face region..."
                    )
                    face = image_bgr
                else:
                    # Use largest detected face
                    x, y, w, h = max(
                        faces,
                        key=lambda f: f[2] * f[3]
                    )

                    face = image_bgr[
                        y:y+h,
                        x:x+w
                    ]

                # Show cropped face
                face_preview = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                st.image(
                    face_preview,
                    caption="Extracted Face",
                    width=140
                )

                # Prediction
                emotion, confidence, probabilities = predict_face(
                    face,
                    use_cnn=use_cnn_model
                )

                st.success(
                    f"### Detected Emotion: {emotion}"
                )

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

                st.subheader(
                    "Emotion Probabilities"
                )

                for i, probability in enumerate(
                    probabilities
                ):
                    st.write(
                        f"**{EMOTIONS[i]}** "
                        f"{probability * 100:.2f}%"
                    )

                    st.progress(
                        float(probability)
                    )


# ============================================================
# LIVE WEBCAM
# ============================================================

with webcam_tab:

    st.header(
        "🎥 Real-Time Emotion Recognition"
    )

    st.info(
        "Allow camera access when your browser asks."
    )

    st.write(
        f"Active model: **{'Custom CNN (48×48)' if use_cnn_model else 'MobileNetV2 (96×96)'}**"
    )

    # Closure flag for webcam worker thread
    use_cnn_thread = use_cnn_model

    def video_frame_callback(frame):

        # Convert frame to OpenCV
        img = frame.to_ndarray(
            format="bgr24"
        )

        # Grayscale for face detection
        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        # Detect faces
        faces = detect_faces(gray)

        # Process every face
        for (x, y, w, h) in faces:

            # Crop face
            face = img[
                y:y+h,
                x:x+w
            ]

            # Prediction
            emotion, confidence, _ = predict_face(
                face,
                use_cnn=use_cnn_thread
            )

            # Draw face rectangle
            cv2.rectangle(
                img,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            # Prediction text
            text = (
                f"{emotion}: "
                f"{confidence * 100:.1f}%"
            )

            # Text background
            cv2.rectangle(
                img,
                (x, y - 35),
                (x+w, y),
                (0, 255, 0),
                -1
            )

            # Text
            cv2.putText(
                img,
                text,
                (x + 5, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

        # Return processed frame
        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

    webrtc_streamer(
        key=f"emotion-streamer-{'cnn' if use_cnn_model else 'mobilenet'}",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown("---")

st.subheader(
    "🧠 How the System Works"
)

st.write(
    """
    1. **Face Detection**: Haar Cascade locates faces in the input image or webcam frame.
    2. **Face Extraction**: The face is cropped and preprocessed (grayscale 48×48 for Custom CNN, RGB 96×96 for MobileNetV2).
    3. **Inference**: Neural network calculates probabilities across all 7 FER-2013 emotion categories.
    4. **Visualization**: Displays the bounding box, predicted emotion, and confidence scores.
    """
)

st.caption(
    "FER-2013 Dataset • 7 Emotion Classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral"
)
