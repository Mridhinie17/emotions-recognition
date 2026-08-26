# src/ensemble_eval.py
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.preprocessing import image_dataset_from_directory

DATA_DIR = "data"
MODELS_DIR = "saved_models"
OUT_FILE = "outputs/paper_tables_results.txt"

def load_test_ds(img_size=(96,96), batch_size=64):
    test_dir = os.path.join(DATA_DIR, "test")
    ds = image_dataset_from_directory(
        test_dir, image_size=img_size, color_mode="rgb", batch_size=batch_size, shuffle=False
    )
    mapped = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).prefetch(tf.data.AUTOTUNE)
    return mapped

def evaluate_models_and_ensemble():
    train_dir = os.path.join(DATA_DIR, "train")
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    print("Class names:", class_names)

    # find available model files
    candidate_files = [
        "best_model_convnext.h5",
        "best_model_efficientv2.h5",
        "best_model_resnet.h5",
        "best_model_efficient.h5",
        "best_model_finetuned.h5",
        "best_model.h5"
    ]
    
    available_models = [m for m in candidate_files if os.path.exists(os.path.join(MODELS_DIR, m))]
    print("Found models for ensemble evaluation:", available_models)

    if not available_models:
        print("No models found in saved_models/")
        return

    all_probs = []
    model_names = []
    y_true = None

    for mfile in available_models:
        mpath = os.path.join(MODELS_DIR, mfile)
        print(f"\n--- Loading {mfile} ---")
        try:
            model = tf.keras.models.load_model(mpath)
        except Exception as e:
            print(f"Skipping {mfile}: {e}")
            continue

        in_shape = model.input_shape
        h = in_shape[1] if (in_shape and len(in_shape) > 1 and in_shape[1] is not None) else 96
        w = in_shape[2] if (in_shape and len(in_shape) > 2 and in_shape[2] is not None) else 96
        c = in_shape[-1] if (in_shape and len(in_shape) > 3 and in_shape[-1] is not None) else 3
        img_size = (h, w)

        test_ds = load_test_ds(img_size=img_size)

        probs_list = []
        labels_list = []
        for x_b, y_b in test_ds:
            if c == 1 and x_b.shape[-1] == 3:
                x_input = tf.image.rgb_to_grayscale(x_b)
            elif c == 3 and x_b.shape[-1] == 1:
                x_input = tf.image.grayscale_to_rgb(x_b)
            else:
                x_input = x_b

            preds = model.predict(x_input, verbose=0)
            probs_list.append(preds)
            if y_true is None:
                labels_list.extend(y_b.numpy())

        probs = np.concatenate(probs_list, axis=0)
        if y_true is None:
            y_true = np.array(labels_list)

        acc = accuracy_score(y_true, np.argmax(probs, axis=1))
        print(f"Model {mfile} Test Accuracy: {acc*100:.2f}%")
        all_probs.append(probs)
        model_names.append(mfile)

    if len(all_probs) > 1:
        # Soft voting ensemble
        ensemble_probs = np.mean(all_probs, axis=0)
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        ensemble_acc = accuracy_score(y_true, ensemble_preds)
        print("\n==================================================")
        print(f"[RESULT] MULTI-MODEL ENSEMBLE TEST ACCURACY: {ensemble_acc*100:.2f}%")
        print("==================================================")
        print("\nClassification Report (Ensemble):")
        print(classification_report(y_true, ensemble_preds, target_names=class_names, digits=4))

if __name__ == '__main__':
    evaluate_models_and_ensemble()
