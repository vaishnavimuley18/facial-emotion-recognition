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
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = "model/final_emotion_mobilenetv2.keras"

EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]


# ============================================================
# LOAD MOBILENETV2
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


model = load_model()


# ============================================================
# FACE DETECTOR
# ============================================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_face(face):

    # Convert BGR → RGB
    face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    # Resize for MobileNetV2
    face = cv2.resize(
        face,
        (96, 96)
    )

    # Normalize
    face = face.astype(
        np.float32
    ) / 255.0

    # Add batch dimension
    face = np.expand_dims(
        face,
        axis=0
    )

    # Prediction
    probabilities = model.predict(
        face,
        verbose=0
    )[0]

    predicted_class = int(
        np.argmax(probabilities)
    )

    emotion = EMOTIONS[
        predicted_class
    ]

    confidence = float(
        probabilities[predicted_class]
    )

    return (
        emotion,
        confidence,
        probabilities
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Model Information")

st.sidebar.success(
    "MobileNetV2 loaded"
)

st.sidebar.write(
    "**Input:** 96 × 96 × 3"
)

st.sidebar.write(
    "**Dataset:** FER-2013"
)

st.sidebar.write(
    "**Classes:** 7"
)

st.sidebar.write(
    "**Architecture:** MobileNetV2"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "😊 AI Facial Emotion Recognition"
)

st.subheader(
    "MobileNetV2 • Transfer Learning • Real-Time Vision"
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

                faces = face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(60, 60)
                )

                if len(faces) == 0:

                    st.warning(
                        "No face detected. "
                        "Please upload a clear face image."
                    )

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

                    # Prediction
                    emotion, confidence, probabilities = (
                        predict_face(face)
                    )

                    st.success(
                        f"### {emotion}"
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
        "The system detects the face first and "
        "then sends the cropped RGB face to MobileNetV2."
    )


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
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        # Process every face
        for (x, y, w, h) in faces:

            # Crop face
            face = img[
                y:y+h,
                x:x+w
            ]

            # MobileNetV2 prediction
            emotion, confidence, _ = predict_face(
                face
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
        key="mobilenet-emotion-recognition",

        video_frame_callback=
        video_frame_callback,

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
    Image / Webcam
    ↓
    Face Detection
    ↓
    Face Crop
    ↓
    RGB 96×96
    ↓
    MobileNetV2
    ↓
    Emotion Classification
    ↓
    Confidence Score
    """
)

st.caption(
    "FER-2013 • MobileNetV2 Transfer Learning • "
    "Real-Time Facial Emotion Recognition"
)