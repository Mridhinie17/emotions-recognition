# src/train_efficient.py
"""
Train EfficientNetB0 for FER using folder dataset (data/train, data/val, data/test).
This script forces RGB inputs (3 channels) so ImageNet weights can be used even if
the source images are grayscale -- the loader will convert grayscale -> RGB.
It saves best and final models and plots training history.
"""

import os
import math
import json
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt

# ---------- USER CONFIG ----------
DATA_DIR = "data"                 # must contain train/val/test or train/ and validation/ folders
IMG_SIZE = 96                     # EfficientNet-like small size; change if you want
BATCH_SIZE = 64
EPOCHS = 40                       # increase for final training (e.g. 60)
LEARNING_RATE = 2e-4
MODEL_DIR = "saved_models"
BEST_MODEL_NAME = "best_model_efficient.h5"
FINAL_MODEL_NAME = "final_model_efficient.h5"
PLOT_NAME = "training_history_efficient.png"
SEED = 1337
AUTOTUNE = tf.data.AUTOTUNE
# -------------------------------

os.makedirs(MODEL_DIR, exist_ok=True)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Helper: get dataset path layout (image_dataset_from_directory wants separate folders per class)
train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "val")
test_dir = os.path.join(DATA_DIR, "test")

# If user only has 'train' and 'validation' names, try common alternatives
if not os.path.isdir(val_dir) and os.path.isdir(os.path.join(DATA_DIR, "validation")):
    val_dir = os.path.join(DATA_DIR, "validation")
if not os.path.isdir(test_dir) and os.path.isdir(os.path.join(DATA_DIR, "test")):
    test_dir = os.path.join(DATA_DIR, "test")

# If there's no val/test directories but fer2013.csv mode is used, the loader must be changed.
# This script expects folders. If your dataset is only CSV, ask and I'll adapt.

print("Dataset dirs:", train_dir, val_dir, test_dir)

# Build tf.data datasets using Keras helper (it will convert grayscale -> RGB automatically when color_mode='rgb')
def make_datasets(img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    # training dataset with augmentation
    train_ds = image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=True,
        seed=SEED,
        color_mode="rgb"  # IMPORTANT: force 3 channels
    )

    val_ds = image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=False,
        color_mode="rgb"
    )

    test_ds = image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=False,
        color_mode="rgb"
    )

    class_names = train_ds.class_names
    print("Classes:", class_names)

    # Prefetch & caching
    train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=AUTOTUNE)

    # Augmentation pipeline (applied only to training)
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomContrast(0.08),
        layers.RandomTranslation(0.06, 0.06),
    ], name="data_augmentation")

    def train_map(x, y):
        x = data_augmentation(x)
        return x, y

    train_ds = train_ds.map(lambda x, y: train_map(x, y), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names

train_ds, val_ds, test_ds, class_names = make_datasets()

num_classes = len(class_names)
print("Num classes:", num_classes)

# compute class weights to help imbalance
def compute_class_weights(dataset):
    labels = []
    for _, y in dataset.unbatch().map(lambda x, y: (x, y)).take(10**9):
        labels.append(int(y.numpy()))
    counts = Counter(labels)
    total = sum(counts.values())
    class_weights = {i: total/(len(counts)*counts[i]) for i in counts}
    print("Class counts:", counts)
    print("Class weights:", class_weights)
    return class_weights

class_weights = compute_class_weights(train_ds)

# Build model: EfficientNetB0 base + small head
def build_model(img_size=IMG_SIZE, num_classes=num_classes):
    input_shape = (img_size, img_size, 3)  # force 3 channels (RGB)
    base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=input_shape)
    base.trainable = False  # start with frozen base

    inputs = layers.Input(shape=input_shape)
    x = inputs
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="efficientnetb0_fer")
    return model, base

model, base = build_model()
model.summary()

# Compile
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Callbacks
cbks = []
checkpoint_path = os.path.join(MODEL_DIR, BEST_MODEL_NAME)
cbks.append(callbacks.ModelCheckpoint(checkpoint_path, monitor="val_accuracy",
                                      save_best_only=True, save_weights_only=False, verbose=1))
cbks.append(callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1))
cbks.append(callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1))
cbks.append(callbacks.TerminateOnNaN())

# Train (stage 1: head only)
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS//2,
    class_weight=class_weights,
    callbacks=cbks
)

# Save intermediate
model.save(os.path.join(MODEL_DIR, "intermediate_model_efficient_stage1.h5"))
print("Saved intermediate model after stage 1")

# Stage 2: unfreeze top of base for fine-tuning
base.trainable = True
# Unfreeze last N layers if you want finer control:
fine_tune_at = len(base.layers) - 30
for i, layer in enumerate(base.layers):
    layer.trainable = i >= fine_tune_at

# Re-compile with lower LR
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    initial_epoch=history1.epoch[-1] + 1,
    class_weight=class_weights,
    callbacks=cbks
)

# Save final model
final_path = os.path.join(MODEL_DIR, FINAL_MODEL_NAME)
model.save(final_path)
print("Saved final model to", final_path)

# Save best model path already saved by checkpoint. If checkpoint is present, report it.
best_path = checkpoint_path if os.path.exists(checkpoint_path) else final_path
print("Best model path:", best_path)

# Evaluate on test set
print("Evaluating on test set...")
res = model.evaluate(test_ds)
print("Test result:", res)

# Plot history combined
def plot_history(h1, h2, out_path=os.path.join(MODEL_DIR, PLOT_NAME)):
    # merge
    acc = []
    val_acc = []
    loss = []
    val_loss = []
    for h in (h1, h2):
        acc += h.history.get("accuracy", [])
        val_acc += h.history.get("val_accuracy", [])
        loss += h.history.get("loss", [])
        val_loss += h.history.get("val_loss", [])
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(acc, label="train_acc")
    plt.plot(val_acc, label="val_acc")
    plt.legend()
    plt.title("Accuracy")
    plt.subplot(1,2,2)
    plt.plot(loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.legend()
    plt.title("Loss")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("Saved training plot to", out_path)

plot_history(history1, history2)
