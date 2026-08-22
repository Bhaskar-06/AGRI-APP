import os
import json
import tensorflow as tf
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "plant_disease_model.h5")
INDICES_PATH = os.path.join(BASE_DIR, "models", "class_indices.json")

model = tf.keras.models.load_model(MODEL_PATH)
with open(INDICES_PATH, "r") as f:
    class_indices = json.load(f)

idx_to_class = {int(k): v for k, v in class_indices.items()}


def predict(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)
    idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][idx]) * 100

    print(f"Disease: {idx_to_class[idx]}  |  Confidence: {confidence:.2f}%")


if __name__ == "__main__":
    predict(os.path.join(BASE_DIR, "test_leaf.jpg"))  # replace with any test image path