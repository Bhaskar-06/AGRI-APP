"""
convert_to_tflite.py
Converts the trained Keras model to a much lighter TFLite model for
low-memory deployment (fixes OOM-triggered reconnects on Streamlit Cloud).
"""

import os
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KERAS_MODEL_PATH = os.path.join(BASE_DIR, "models", "plant_disease_model.h5")
TFLITE_MODEL_PATH = os.path.join(BASE_DIR, "models", "plant_disease_model.tflite")

model = tf.keras.models.load_model(KERAS_MODEL_PATH)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # quantizes weights, shrinks file size significantly
tflite_model = converter.convert()

with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)

original_size = os.path.getsize(KERAS_MODEL_PATH) / (1024 * 1024)
new_size = os.path.getsize(TFLITE_MODEL_PATH) / (1024 * 1024)
print(f"✅ Converted. Original: {original_size:.1f} MB → TFLite: {new_size:.1f} MB")