import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Load model & class indices
model = tf.keras.models.load_model("models/plant_disease_model.h5")
with open("models/class_indices.json") as f:
    class_indices = json.load(f)

# Reverse map: index → label
idx_to_class = {v: k for k, v in class_indices.items()}

def predict(image_path):
    img = Image.open(image_path).resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)
    idx = np.argmax(preds)
    confidence = preds[0][idx] * 100
    print(f"Disease: {idx_to_class[idx]}  |  Confidence: {confidence:.2f}%")

predict("test_leaf.jpg")  # use any plant image