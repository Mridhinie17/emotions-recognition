# src/predict.py
# Predict facial emotion from a single uploaded image
# (96×96 RGB, values scaled to [0,1])

import os
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from tkinter import Tk, filedialog

# ==== CONFIG ====
MODEL_PATH = "saved_models/best_model_finetuned.h5"
IMG_SIZE = (96, 96)
APPLY_GRAY=True
OUT_DIR = "outputs"
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
# ================


def choose_image():
    """Open file dialog to select one image."""
    Tk().withdraw()
    path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")],
    )
    return Path(path) if path else None


def preprocess_image(img_bgr, img_size=IMG_SIZE):
    """Resize, convert to RGB, and scale to [0,1]."""
    img = cv2.resize(img_bgr, img_size, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Choose image interactively ---
    img_path = choose_image()
    if not img_path:
        print("No image selected. Exiting.")
        return

    # --- Load model ---
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        return
    print(f"✅ Loaded model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    # --- Read and preprocess image ---
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"❌ Could not read image: {img_path}")
        return
    x = preprocess_image(img_bgr)

    # --- Predict ---
    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    label = CLASS_NAMES[idx]
    conf = float(preds[idx])

    # --- Annotate and save ---
    annotated = img_bgr.copy()
    cv2.putText(
        annotated,
        f"{label} ({conf:.2f})",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 255, 0),
        2,
    )
    out_path = Path(OUT_DIR) / f"pred_{img_path.name}"
    cv2.imwrite(str(out_path), annotated)

    print(f"\n🖼️ Image: {img_path.name}")
    print(f"🎯 Prediction: {label} ({conf:.3f})")
    print(f"💾 Saved annotated image to: {out_path}")

    # --- Optional preview ---
    cv2.imshow("Prediction", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
