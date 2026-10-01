import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = "data/fer2013/filtered_fer2013.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "facial_emotion_mobilenetv2.keras")

IMG_SIZE = 96
BATCH_SIZE = 64
INITIAL_EPOCHS = 5
NUM_CLASSES = 7

EMOTION_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 60)
print("MOBILENETV2 FACIAL EMOTION RECOGNITION")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(CSV_PATH)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())


# ============================================================
# 2. SELECT TRAIN / TEST DATA
# ============================================================

train_df = df[df["Usage"] == "Training"].copy()
test_df = df[df["Usage"] == "PublicTest"].copy()

print("\nTraining samples:", len(train_df))
print("Test samples:", len(test_df))


# ============================================================
# 3. CONVERT PIXELS TO IMAGES
# ============================================================

def prepare_images(dataframe):

    images = []

    for pixels in dataframe["pixels"]:

        image = np.fromstring(pixels, dtype=np.uint8, sep=" ")

        image = image.reshape(48, 48)

        # Resize from 48x48 → 96x96
        image = tf.image.resize(
            image[..., np.newaxis],
            (IMG_SIZE, IMG_SIZE)
        ).numpy()

        # Convert grayscale → RGB
        image = np.repeat(image, 3, axis=-1)

        images.append(image)

    return np.array(images, dtype=np.float32)


print("\nPreparing training images...")

X_train = prepare_images(train_df)

print("Preparing test images...")

X_test = prepare_images(test_df)


# ============================================================
# 4. NORMALIZATION
# ============================================================

X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = train_df["emotion"].values
y_test = test_df["emotion"].values


print("\nTraining shape:", X_train.shape)
print("Test shape:", X_test.shape)


# ============================================================
# 5. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
], name="data_augmentation")


# ============================================================
# 6. LOAD PRETRAINED MOBILENETV2
# ============================================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers initially
base_model.trainable = False


# ============================================================
# 7. BUILD MODEL
# ============================================================

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = data_augmentation(inputs)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(0.35)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.25)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = models.Model(
    inputs,
    outputs,
    name="MobileNetV2_Facial_Emotion"
)


# ============================================================
# 8. COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 9. MODEL SUMMARY
# ============================================================

print("\nModel Summary:")

model.summary()

print(
    "\nTrainable parameters:",
    sum(
        np.prod(v.shape)
        for v in model.trainable_variables
    )
)


# ============================================================
# 10. CALLBACKS
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)

callbacks = [

    EarlyStopping(
        monitor="val_accuracy",
        patience=2,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
        verbose=1
    ),

    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]


# ============================================================
# 11. TRAIN
# ============================================================

print("\nStarting transfer-learning training...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=INITIAL_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    shuffle=True
)


# ============================================================
# 12. FINAL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

print("\nTest Loss:", loss)
print("Test Accuracy:", accuracy)

print("\nBest model saved to:")
print(MODEL_PATH)