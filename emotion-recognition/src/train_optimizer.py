# src/train_optimizer.py
import os
import sys
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

from tensorflow.keras import layers, models, callbacks
from src.data_loader import load_fer2013_from_folders
from src.model import build_transfer_model
from collections import Counter

# CONFIG
BATCH_SIZE = 64
IMG_SIZE = (96, 96)
EPOCHS_STAGE1 = 8
EPOCHS_STAGE2 = 12
OUT_DIR = 'saved_models'
DATA_DIR = 'data'

os.makedirs(OUT_DIR, exist_ok=True)

def compute_class_weights(train_dir=os.path.join(DATA_DIR, 'train')):
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    counts = []
    for cname in class_names:
        p = os.path.join(train_dir, cname)
        n = len(os.listdir(p))
        counts.append(n)
    total = sum(counts)
    class_weights = {i: total / (len(counts) * c) for i, c in enumerate(counts)}
    print("Class counts:", dict(zip(class_names, counts)))
    print("Class weights:", class_weights)
    return class_weights, len(class_names)

def train_backbone(architecture_name, model_save_filename):
    print(f"\n========================================================")
    print(f"[TRAINING OPTIMIZED MODEL]: {architecture_name}")
    print(f"========================================================")

    train_ds, val_ds, test_ds = load_fer2013_from_folders(
        DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE, augment=True
    )
    
    class_weights, num_classes = compute_class_weights()

    # Build model with frozen backbone first
    model = build_transfer_model(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        num_classes=num_classes,
        dropout_rate=0.4,
        train_base=False,
        architecture=architecture_name,
        learning_rate=3e-4
    )

    best_checkpoint_path = os.path.join(OUT_DIR, model_save_filename)
    initial_epoch = 0
    if os.path.exists(best_checkpoint_path):
        print(f"Resuming model training from existing checkpoint: {best_checkpoint_path}")
        try:
            model = tf.keras.models.load_model(best_checkpoint_path)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy']
            )
            initial_epoch = 4  # Resume from Epoch 5
            print(f"Successfully loaded checkpoint! Resuming training from Epoch 5.")
        except Exception as e:
            print(f"Building fresh model architecture: {e}")

    cbks_stage1 = [
        callbacks.ModelCheckpoint(best_checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        callbacks.EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1)
    ]

    print(f"--- Stage 1: Pre-training Classification Head ({architecture_name}) ---")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE1,
        initial_epoch=initial_epoch,
        class_weight=class_weights,
        callbacks=cbks_stage1
    )

    # Stage 2: Unfreeze top layers of base for fine-tuning
    print(f"--- Stage 2: Fine-Tuning Backbone ({architecture_name}) ---")
    base_layer = None
    for l in model.layers:
        if hasattr(l, 'layers') and len(l.layers) > 10:
            base_layer = l
            break
            
    if base_layer is not None:
        base_layer.trainable = True
        fine_tune_at = len(base_layer.layers) - 40
        for i, l in enumerate(base_layer.layers):
            l.trainable = (i >= fine_tune_at)
        print(f"Unfroze top layers from layer {fine_tune_at} of {len(base_layer.layers)}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    cbks_stage2 = [
        callbacks.ModelCheckpoint(best_checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1)
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE2,
        class_weight=class_weights,
        callbacks=cbks_stage2
    )

    print(f"Evaluating {architecture_name} on test set...")
    if os.path.exists(best_checkpoint_path):
        model = tf.keras.models.load_model(best_checkpoint_path)
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"[RESULT] {architecture_name} Test Accuracy: {test_acc*100:.2f}%\n")
    return test_acc

def main():
    acc1 = train_backbone('resnet50v2', 'best_model_resnet.h5')
    acc2 = train_backbone('mobilenetv2', 'best_model_opt_mobilenet.h5')
    
    print("\nRunning ensemble evaluation across all trained models...")
    os.system(r"C:\tf_env\Scripts\python.exe src/ensemble_eval.py")

if __name__ == '__main__':
    main()
