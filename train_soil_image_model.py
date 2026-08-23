"""
train_soil_image_model.py
Trains a soil TYPE classifier (Alluvial / Black / Clay / Red) using
MobileNetV2 transfer learning + fine-tuning, same pattern as
train_plant_model.py.
"""

import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "soil_types")
MODEL_SAVE = os.path.join(BASE_DIR, "models", "soil_type_model.h5")
INDICES_SAVE = os.path.join(BASE_DIR, "models", "soil_class_indices.json")
IMG_SIZE = (224, 224)
BATCH_SIZE = 16          # smaller dataset than PlantVillage — smaller batch trains more stably
EPOCHS_HEAD = 15
EPOCHS_FINE_TUNE = 10

os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,
    horizontal_flip=True,
    vertical_flip=True,        # soil textures look valid upside-down too — extra augmentation helps on a small dataset
    zoom_range=0.25,
    rotation_range=25,
    brightness_range=[0.7, 1.3],  # soil color under different lighting is the main real-world variation
    width_shift_range=0.15,
    height_shift_range=0.15,
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
print(f"Number of soil classes: {NUM_CLASSES}")
print(f"Classes: {list(train_generator.class_indices.keys())}")

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
x = layers.Dense(256, activation='relu')(x)   # smaller head — small dataset, avoid overfitting
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5)
]

print("\n=== Phase 1: training classification head ===")
model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS_HEAD, callbacks=callbacks)

print("\n=== Phase 2: fine-tuning top layers of MobileNetV2 ===")
base_model.trainable = True
for layer in base_model.layers[:-20]:
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