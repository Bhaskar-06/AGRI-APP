import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
os.makedirs("models", exist_ok=True)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DATASET_DIR = "C:\\Users\\Bhaskar B\\Downloads\\archive (2)\\plantvillage dataset\\color"  # Change this to your dataset path
MODEL_SAVE   = "models/plant_disease_model.h5"
print("Using dataset dir:", os.path.abspath(DATASET_DIR))
print("Exists:", os.path.exists(DATASET_DIR))
if not os.path.exists(DATASET_DIR):
    raise SystemExit("❌ DATASET_DIR path is wrong. Fix it before training.")
INDICES_SAVE = "models/class_indices.json"
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
EPOCHS       = 15

# ── DATA GENERATORS ────────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

NUM_CLASSES = len(train_generator.class_indices)
print(f"Number of classes: {NUM_CLASSES}")
print(f"Class names: {list(train_generator.class_indices.keys())}")

# ── SAVE CLASS INDICES ─────────────────────────────────────────────────────────
# Save as {index: class_name} format
idx_to_class = {v: k for k, v in train_generator.class_indices.items()}
with open(INDICES_SAVE, "w") as f:
    json.dump(idx_to_class, f, indent=2)
print(f"✅ Saved class indices to {INDICES_SAVE}")

# ── BUILD MODEL (Transfer Learning with MobileNetV2) ───────────────────────────
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model initially
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── CALLBACKS ──────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6, verbose=1),
    ModelCheckpoint(MODEL_SAVE, monitor='val_accuracy', save_best_only=True, verbose=1)
]

# ── PHASE 1: TRAIN TOP LAYERS ─────────────────────────────────────────────────
print("\n🚀 Phase 1: Training top layers...")
history1 = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# ── PHASE 2: FINE-TUNE (Unfreeze top layers of base) ─────────────────────────
print("\n🔧 Phase 2: Fine-tuning top 50 layers of MobileNetV2...")
base_model.trainable = True
# Freeze all layers except top 50
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# ── EVALUATE ─────────────────────────────────────────────────────────────────
print("\n📊 Evaluating model...")
loss, accuracy = model.evaluate(val_generator)
print(f"✅ Validation Accuracy: {accuracy*100:.2f}%")
print(f"✅ Validation Loss: {loss:.4f}")
print(f"✅ Model saved to: {MODEL_SAVE}")
print(f"✅ Class indices saved to: {INDICES_SAVE}")
