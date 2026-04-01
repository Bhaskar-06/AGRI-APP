# train_plant_model.py  — Fixed version
# KEY FIX: Call model.build() before model.save() to prevent the
#           "(None, 7, 7, 1280)" shape tuple corruption bug.

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Model
import json, os

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = "data/plantvillage"   # path to your PlantVillage dataset
MODEL_SAVE   = "models/plant_disease_model.h5"
INDICES_SAVE = "models/class_indices.json"
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
EPOCHS       = 10

os.makedirs("models", exist_ok=True)

# ── Data Generators ───────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=15,
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
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

# ── Save Class Indices ────────────────────────────────────────────────────────
idx_to_class = {v: k for k, v in train_generator.class_indices.items()}
with open(INDICES_SAVE, "w") as f:
    json.dump(idx_to_class, f, indent=2)
print(f"✅ Saved class indices to {INDICES_SAVE}")

# ── Build Model (MobileNetV2 Transfer Learning) ───────────────────────────────
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
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

# ── Train ─────────────────────────────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5)
]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ── CRITICAL FIX: build before save ──────────────────────────────────────────
# This prevents the "(None, 7, 7, 1280)" shape tuple bug in the saved config.
model.build((None, 224, 224, 3))
model.save(MODEL_SAVE)
print(f"✅ Model saved to {MODEL_SAVE}")

val_loss, val_acc = model.evaluate(val_generator)
print(f"✅ Validation accuracy: {val_acc*100:.2f}%")