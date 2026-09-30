import av
import cv2
import numpy as np
import streamlit as st

from streamlit_webrtc import webrtc_streamer

from src.predictor import model, labels, face_detector


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Real-Time Facial Emotion Recognition",
    page_icon="😊",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("😊 Real-Time Facial Emotion Recognition")

st.write(
    "Allow camera access and the CNN will predict "
    "facial emotions in real time."
)

st.info(
    "The system detects the face first and then "
    "classifies the facial expression using the CNN."
)


# --------------------------------------------------
# Emotion names
# --------------------------------------------------

emotions = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]


# --------------------------------------------------
# Real-time video processing
# --------------------------------------------------

def video_frame_callback(frame):

    # Convert incoming frame to OpenCV format
    img = frame.to_ndarray(format="bgr24")

    # Convert to grayscale for face detection
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

    # Process every detected face
    for (x, y, w, h) in faces:

        # Crop face
        face = gray[
            y:y+h,
            x:x+w
        ]

        # Resize to CNN input size
        face = cv2.resize(
            face,
            (48, 48)
        )

        # Normalize
        face = face.astype(
            np.float32
        ) / 255.0

        # Add channel dimension
        face = np.expand_dims(
            face,
            axis=-1
        )

        # Add batch dimension
        face = np.expand_dims(
            face,
            axis=0
        )

        # CNN prediction
        probabilities = model.predict(
            face,
            verbose=0
        )[0]

        predicted_class = int(
            np.argmax(probabilities)
        )

        emotion = labels[predicted_class]

        confidence = float(
            probabilities[predicted_class]
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

        # Draw prediction
        cv2.putText(
            img,
            text,
            (x + 5, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )

    # Return processed frame
    return av.VideoFrame.from_ndarray(
        img,
        format="bgr24"
    )


# --------------------------------------------------
# Start webcam
# --------------------------------------------------

webrtc_streamer(
    key="emotion-recognition",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
)


# --------------------------------------------------
# Information
# --------------------------------------------------

st.markdown("---")

st.subheader("How it works")

st.write(
    """
    Camera → Face Detection → 48×48 Grayscale →
    CNN → Emotion + Confidence
    """
)

st.caption(
    "Model trained on the FER-2013 facial expression dataset."
)