# src/compare_models.py
# Evaluate multiple saved models (.h5) on FER test set and compare performance.

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

# -------- CONFIG --------
MODELS_DIR = "saved_models"
DATA_DIR = "data/test"       # Folder with subfolders per class
IMG_SIZE = (96, 96)          # For MobileNetV2 models
BATCH_SIZE = 64
OUT_CSV = "outputs/model_comparison.csv"
# -------------------------

os.makedirs("outputs", exist_ok=True)

# Load dataset once
print("📂 Loading test dataset...")
test_ds = image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    color_mode="rgb",
    shuffle=False,
    batch_size=BATCH_SIZE
)
class_names = test_ds.class_names
print("Classes:", class_names)

# Apply preprocessing (MobileNetV2-style)
test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))

# Helper function to evaluate one model
def evaluate_model(model_path):
    print(f"\n🔍 Evaluating {model_path} ...")
    try:
        model = tf.keras.models.load_model(model_path)
        preds, labels = [], []
        for x_batch, y_batch in test_ds:
            p = model.predict(x_batch, verbose=0)
            preds.extend(np.argmax(p, axis=1))
            labels.extend(y_batch.numpy())
        acc = accuracy_score(labels, preds)
        report = classification_report(labels, preds, target_names=class_names, digits=4, output_dict=True)
        f1_macro = np.mean([report[c]["f1-score"] for c in class_names])
        f1_weighted = report["weighted avg"]["f1-score"]
        return {
            "model": os.path.basename(model_path),
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted
        }
    except Exception as e:
        print(f"❌ Failed to evaluate {model_path}: {e}")
        return None

# Evaluate all models in the folder
results = []
for fname in sorted(os.listdir(MODELS_DIR)):
    if fname.endswith(".h5"):
        info = evaluate_model(os.path.join(MODELS_DIR, fname))
        if info:
            results.append(info)

# Summarize
df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
print("\n🏁 Results:")
print(df.to_string(index=False))

# Save CSV
df.to_csv(OUT_CSV, index=False)
print(f"\n✅ Saved comparison results to {OUT_CSV}")
