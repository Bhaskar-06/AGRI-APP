import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DATA_DIR   = "data/soil_types"
MODEL_SAVE = "models/soil_model.h5"
INDEX_SAVE = "models/soil_class_indices.json"
IMG_SIZE   = (224, 224)
BATCH_SIZE = 16
EPOCHS_INIT     = 10
EPOCHS_FINE_TUNE = 15

# ─────────────────────────────────────────
# VALIDATE DATASET FIRST
# ─────────────────────────────────────────
print("=" * 50)
print("SOIL TYPE IMAGE CLASSIFIER TRAINING")
print("=" * 50)

# Check folder exists
if not os.path.exists(DATA_DIR):
    print(f"\n❌ Dataset folder not found: {os.path.abspath(DATA_DIR)}")
    print(f"   Extract your soil types zip so that class folders")
    print(f"   like 'Alluvial Soil' sit directly inside '{os.path.abspath(DATA_DIR)}'")
    exit(1)

# Check class folders
class_folders = [
    f for f in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, f))
]

if len(class_folders) == 0:
    print(f"\n❌ No class folders found in {DATA_DIR}")
    print(f"   Current contents: {os.listdir(DATA_DIR)}")
    print(f"\n   Expected structure:")
    print(f"   data/soil_types/")
    print(f"     ├── Alluvial Soil/")
    print(f"     ├── Black Soil/")
    print(f"     ├── Clay Soil/")
    print(f"     └── Red Soil/")
    exit(1)

print(f"\n✅ Found {len(class_folders)} soil classes:")
total_images = 0
for cls in sorted(class_folders):
    cls_path = os.path.join(DATA_DIR, cls)
    imgs = [
        f for f in os.listdir(cls_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    total_images += len(imgs)
    print(f"   {cls}: {len(imgs)} images")

print(f"\n✅ Total images: {total_images}")

if total_images == 0:
    print("❌ No images found! Check your dataset.")
    exit(1)

NUM_CLASSES = len(class_folders)
print(f"✅ Number of classes: {NUM_CLASSES}")
print()

# ─────────────────────────────────────────
# DATA GENERATORS
# ─────────────────────────────────────────
print("📁 Setting up data generators...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

NUM_CLASSES = train_generator.num_classes
print(f"✅ Training samples: {train_generator.samples}")
print(f"✅ Validation samples: {val_generator.samples}")
print(f"✅ Classes: {train_generator.class_indices}")

# Save class indices
os.makedirs("models", exist_ok=True)
with open(INDEX_SAVE, "w") as f:
    json.dump(train_generator.class_indices, f, indent=2)
print(f"✅ Class indices saved to {INDEX_SAVE}")

# ─────────────────────────────────────────
# BUILD MODEL
# ─────────────────────────────────────────
print("\n🏗️ Building model...")

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)
base_model.trainable = False

inputs  = tf.keras.Input(shape=(*IMG_SIZE, 3))
x       = base_model(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dense(256, activation='relu')(x)
x       = layers.Dropout(0.4)(x)
x       = layers.Dense(128, activation='relu')(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs, outputs)
print(f"✅ Model built with {NUM_CLASSES} output classes")

# ─────────────────────────────────────────
# PHASE 1 - FEATURE EXTRACTION
# ─────────────────────────────────────────
print("\n🔥 Phase 1: Feature Extraction")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    ModelCheckpoint(
        MODEL_SAVE,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_INIT,
    callbacks=callbacks,
    verbose=1
)

best_acc1 = max(history1.history['val_accuracy'])
print(f"\n✅ Phase 1 Best Accuracy: {best_acc1*100:.2f}%")

# ─────────────────────────────────────────
# PHASE 2 - FINE TUNING
# ─────────────────────────────────────────
print("\n🔧 Phase 2: Fine Tuning")

base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_FINE_TUNE,
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────────────────
# SAVE & EVALUATE
# ─────────────────────────────────────────
model.build((None, 224, 224, 3))
model.save(MODEL_SAVE)
print(f"\n✅ Model saved to {MODEL_SAVE}")

val_loss, val_acc = model.evaluate(val_generator, verbose=0)
print(f"✅ Final validation accuracy: {val_acc * 100:.2f}%")
print(f"✅ Final validation loss: {val_loss:.4f}")

print("\n🎉 Training Complete!")
print(f"   Model: {MODEL_SAVE}")
print(f"   Classes: {INDEX_SAVE}")