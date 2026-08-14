# src/test_images_fer.py
import argparse, csv
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

FER_CLASSES = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]

def ensure_gray_48(img, target_size=(48,48)):
    if img is None: return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img = img.astype("float32")/255.0
    img = np.expand_dims(img, -1)  # (48,48,1)
    return img

def load_model_any(path):
    return tf.keras.models.load_model(path, compile=False)

def main():
    ap = argparse.ArgumentParser("Predict emotions on saved images (FER2013 preprocessing).")
    ap.add_argument("--model_path", required=True, help="Path to .h5 or SavedModel dir")
    ap.add_argument("--images_dir", required=True, help="Folder with images (recursively scanned)")
    ap.add_argument("--output_csv", default="predictions.csv")
    ap.add_argument("--classes", nargs="+", default=FER_CLASSES)
    ap.add_argument("--with_gt", action="store_true",
                    help="If images are in subfolders named by label, compute report/confusion matrix")
    args = ap.parse_args()

    model = load_model_any(args.model_path)
    class_names = args.classes

    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp")
    img_paths = []
    for e in exts:
        img_paths += list(Path(args.images_dir).rglob(e))
    img_paths = sorted(img_paths)
    if not img_paths:
        print(f"No images under {args.images_dir}")
        return

    y_true, y_pred = [], []

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath","pred_label","pred_conf"] + [f"p_{c}" for c in class_names])

        for p in img_paths:
            bgr = cv2.imread(str(p))
            if bgr is None:
                print(f"Skip unreadable: {p}")
                continue

            x = ensure_gray_48(bgr)
            x = np.expand_dims(x, 0)  # (1,48,48,1)
            probs = model.predict(x, verbose=0)[0]
            # Be robust in case the model outputs logits:
            if probs.min() < 0 or probs.max() > 1.0 or abs(probs.sum()-1) > 1e-3:
                e = np.exp(probs - np.max(probs))
                probs = e / e.sum()

            idx = int(np.argmax(probs))
            pred_label = class_names[idx]
            pred_conf = float(probs[idx])

            row = [str(p), pred_label, pred_conf] + [float(probs[i]) for i in range(len(class_names))]
            w.writerow(row)

            if args.with_gt:
                y_true.append(p.parent.name)
                y_pred.append(pred_label)

    print(f"\nSaved: {args.output_csv}")
    if args.with_gt and y_true:
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, labels=class_names, zero_division=0))
        print("Confusion matrix (rows=true, cols=pred):")
        print(confusion_matrix(y_true, y_pred, labels=class_names))

if __name__ == "__main__":
    main()
