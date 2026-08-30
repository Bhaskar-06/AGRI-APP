"""
train_plant_model.py
model.build() is called before model.save() to prevent the
"(None, 7, 7, 1280)" shape-tuple corruption bug.
Includes a fine-tuning phase for higher accuracy.
"""

import os
import json
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "plantvillage")
MODEL_SAVE = os.path.join(BASE_DIR, "models", "plant_disease_model.h5")
INDICES_SAVE = os.path.join(BASE_DIR, "models", "class_indices.json")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 10
EPOCHS_FINE_TUNE = 5

if not os.path.isdir(DATA_DIR):
    sys.exit(
        f"\n❌ Dataset folder not found: {DATA_DIR}\n"
        f"   Extract your PlantVillage zip so that class folders like "
        f"'Apple___Apple_scab' sit directly inside '{DATA_DIR}'.\n"
    )

os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='training', shuffle=True, seed=42
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation', shuffle=False, seed=42
)

NUM_CLASSES = len(train_generator.class_indices)
print(f"Number of classes: {NUM_CLASSES}")

idx_to_class = {v: k for k, v in train_generator.class_indices.items()}
with open(INDICES_SAVE, "w") as f:
    json.dump(idx_to_class, f, indent=2)
print(f"✅ Saved class indices to {INDICES_SAVE}")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights='imagenet'
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5)
]

print("\n=== Phase 1: training classification head ===")
model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS_HEAD, callbacks=callbacks)

print("\n=== Phase 2: fine-tuning top layers of MobileNetV2 ===")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS_FINE_TUNE, callbacks=callbacks)

model.build((None, 224, 224, 3))
model.save(MODEL_SAVE)
print(f"✅ Model saved to {MODEL_SAVE}")

val_loss, val_acc = model.evaluate(val_generator)
print(f"✅ Final validation accuracy: {val_acc * 100:.2f}%")