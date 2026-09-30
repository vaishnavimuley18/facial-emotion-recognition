import json
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


MODEL_PATH = "model/facial_emotion_cnn.keras"
LABELS_PATH = "emotion_labels.json"


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)


# --------------------------------------------------
# Load emotion labels
# --------------------------------------------------

with open(LABELS_PATH, "r") as file:
    labels = json.load(file)

labels = {
    int(key): value
    for key, value in labels.items()
}


# --------------------------------------------------
# Face detector
# --------------------------------------------------

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# --------------------------------------------------
# Detect face
# --------------------------------------------------

def detect_face(image):
    """
    Detect the largest face in the image.
    Returns the cropped face as a PIL image.
    """

    image_array = np.array(image)

    # Convert RGB → grayscale
    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) == 0:
        return None

    # Select largest detected face
    x, y, w, h = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    face = image_array[
        y:y+h,
        x:x+w
    ]

    return Image.fromarray(face)


# --------------------------------------------------
# Preprocess face
# --------------------------------------------------

def preprocess_image(image):

    # Convert to grayscale
    image = image.convert("L")

    # Resize to FER-2013 dimensions
    image = image.resize((48, 48))

    # Convert to NumPy
    image = np.array(
        image,
        dtype=np.float32
    )

    # Normalize
    image = image / 255.0

    # Add channel dimension
    image = np.expand_dims(
        image,
        axis=-1
    )

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# --------------------------------------------------
# Predict emotion
# --------------------------------------------------

def predict_emotion(image):

    # Detect face
    face = detect_face(image)

    if face is None:
        return None, None, None, None

    # Preprocess face
    processed_image = preprocess_image(face)

    # Predict
    probabilities = model.predict(
        processed_image,
        verbose=0
    )[0]

    predicted_class = int(
        np.argmax(probabilities)
    )

    predicted_emotion = labels[
        predicted_class
    ]

    confidence = float(
        probabilities[predicted_class]
    )

    return (
        predicted_emotion,
        confidence,
        probabilities,
        face
    )