from PIL import Image
from src.predictor import predict_emotion


image_path = "test_image.jpg"

image = Image.open(image_path)

(
    emotion,
    confidence,
    probabilities,
    face
) = predict_emotion(image)


if emotion is None:

    print("\nNo face detected.")

else:

    print("\nPrediction")
    print("-" * 30)

    print(
        "Emotion:",
        emotion
    )

    print(
        "Confidence:",
        f"{confidence * 100:.2f}%"
    )

    print("\nAll probabilities:")

    emotions = [
        "Angry",
        "Disgust",
        "Fear",
        "Happy",
        "Sad",
        "Surprise",
        "Neutral"
    ]

    for emotion_name, probability in zip(
        emotions,
        probabilities
    ):

        print(
            f"{emotion_name:10s}: "
            f"{probability * 100:.2f}%"
        )

    # Save detected face for inspection
    face.save("detected_face.jpg")

    print(
        "\nDetected face saved as "
        "'detected_face.jpg'"
    )