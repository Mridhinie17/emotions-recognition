# src/predict_live.py
import os
import cv2
import time
import numpy as np
import tensorflow as tf
from collections import deque
from pathlib import Path

# ================== CONFIG ==================
PREFERRED_MODELS = [
    "saved_models/best_model_finetuned.h5",  # MobileNetV2 (96x96x3)
    "saved_models/best_model_transfer.h5",
    "saved_models/final_model.h5"            # old CNN (48x48x1)
]
FACE_PROTO   = "models/deploy.prototxt"
FACE_WEIGHTS = "models/res10_300x300_ssd_iter_140000.caffemodel"

# Detection / cropping
FACE_CONF_THRESH = 0.6
FACE_MARGIN = 0.25         # 25% padding around face box
MIN_FACE = 120             # min face size in px (shorter side) to accept

# Camera
FRONT_CAM_INDEX = 1
BACK_CAM_INDEX  = 0
FRAME_W, FRAME_H = 640, 480

# Smoothing & output
USE_EMA = True
EMA_ALPHA = 0.5          # higher -> more responsive; 0.35–0.55 is good
ROLLING_WINDOW = 7         # used only if USE_EMA=False
CONF_THRESH = 0.30         # below this -> "unsure"
SHOW_TOP3 = True
SHOW_FPS  = True

# Preprocess
APPLY_HISTEQ_FOR_GRAY = False  # set True only if your 48x48 model was trained with hist eq

CLASS_NAMES = ['angry','disgust','fear','happy','neutral','sad','surprise']
# ============================================

def pick_model():
    for p in PREFERRED_MODELS:
        if os.path.exists(p):
            return p
    return None

def infer_input_specs(model):
    try:
        h, w, c = model.inputs[0].shape[1:4]
        return int(h), int(w), int(c)
    except Exception:
        return 96, 96, 3  # safe default

def preprocess_face(face_bgr, target_hw, channels):
    H, W = target_hw
    fh, fw = face_bgr.shape[:2]
    upscale = (fh < H) or (fw < W)
    interp = cv2.INTER_CUBIC if upscale else cv2.INTER_AREA

    if channels == 1:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        if APPLY_HISTEQ_FOR_GRAY:
            gray = cv2.equalizeHist(gray)
        resized = cv2.resize(gray, (W, H), interpolation=interp)
        x = resized.astype("float32") / 255.0
        x = x[..., None]  # (H,W,1)
    else:
        resized = cv2.resize(face_bgr, (W, H), interpolation=interp)
        x = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    return np.expand_dims(x, axis=0)  # (1,H,W,C)

def load_face_detector():
    if not (os.path.exists(FACE_PROTO) and os.path.exists(FACE_WEIGHTS)):
        return None
    try:
        net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_WEIGHTS)
        return net
    except Exception:
        return None

def open_camera(idx):
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # helps on some webcams
    time.sleep(0.2)
    if not cap.isOpened():
        return None
    return cap

def draw_text(img, text, y=28, color=(0,255,0)):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

def expand_with_margin(x1, y1, x2, y2, margin, w, h):
    fw, fh = x2 - x1, y2 - y1
    mx, my = int(margin * fw), int(margin * fh)
    x1m, y1m = max(0, x1 - mx), max(0, y1 - my)
    x2m, y2m = min(w - 1, x2 + mx), min(h - 1, y2 + my)
    return x1m, y1m, x2m, y2m

def main():
    # --- Load model ---
    model_path = pick_model()
    if not model_path:
        print("❌ No model found in saved_models/. Place a .h5 file there.")
        return
    print(f"✅ Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    H, W, C = infer_input_specs(model)
    print(f"ℹ️  Model expects: {H}x{W}x{C} (HxWxC)")
    if len(CLASS_NAMES) != model.output_shape[-1]:
        print(f"⚠️ CLASS_NAMES length {len(CLASS_NAMES)} != model outputs {model.output_shape[-1]}. Check order!")

    # --- Load detector ---
    face_net = load_face_detector()
    if face_net is None:
        print("❌ Face detector not found. Put deploy.prototxt and res10_*.caffemodel in models/")
        return
    print("✅ Loaded DNN face detector")

    # --- Open camera (front -> back fallback) ---
    cam_idx = FRONT_CAM_INDEX
    cap = open_camera(cam_idx) or open_camera(BACK_CAM_INDEX)
    if cap is None:
        print("❌ Could not open any camera. Check device permissions.")
        return
    print(f"🎥 Using camera index {cam_idx}. Press 'q' to quit, '0' to switch.")

    ema = None
    pred_buffer = deque(maxlen=ROLLING_WINDOW)
    last_t = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("⚠️ Camera read failed; retrying...")
                time.sleep(0.05)
                continue

            if cam_idx == FRONT_CAM_INDEX:
                frame = cv2.flip(frame, 1)

            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104,117,123], False, False)
            face_net.setInput(blob)
            detections = face_net.forward()

            # Best detection
            best = None; best_conf = 0.0
            for i in range(detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                if conf > FACE_CONF_THRESH and conf > best_conf:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype("int")
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w-1, x2), min(h-1, y2)
                    best = (x1, y1, x2, y2); best_conf = conf

            if best is None:
                draw_text(frame, "No face detected", 28, (0,0,255))
                ema = None  # reset smoothing when face lost
            else:
                # margin + min-size gate
                x1, y1, x2, y2 = best
                x1m, y1m, x2m, y2m = expand_with_margin(x1, y1, x2, y2, FACE_MARGIN, w, h)
                fw, fh = x2m - x1m, y2m - y1m
                if min(fw, fh) < MIN_FACE:
                    draw_text(frame, "Face too small—move closer", 28, (0,165,255))
                else:
                    face = frame[y1m:y2m, x1m:x2m]
                    cv2.rectangle(frame, (x1m, y1m), (x2m, y2m), (0,255,0), 2)

                    try:
                        x_in = preprocess_face(face, (H, W), C)
                        preds = model.predict(x_in, verbose=0)[0]

                        if USE_EMA:
                            if ema is None:
                                ema = preds.copy()
                            else:
                                ema = (1 - EMA_ALPHA) * ema + EMA_ALPHA * preds
                            avg = ema
                        else:
                            pred_buffer.append(preds)
                            avg = np.mean(pred_buffer, axis=0)

                        idx = int(np.argmax(avg))
                        conf = float(np.max(avg))

                        if conf < CONF_THRESH:
                            emotion = "unsure"
                            color = (0,165,255)
                        else:
                            emotion = CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else str(idx)
                            color = (0,255,0)

                        draw_text(frame, f"{emotion} ({conf:.2f})", 28, color)

                        if SHOW_TOP3:
                            top3 = np.argsort(avg)[-3:][::-1]
                            msg = " | ".join([f"{CLASS_NAMES[i]}:{avg[i]:.2f}" for i in top3])
                            draw_text(frame, msg, 58, (255,255,255))

                    except Exception as e:
                        draw_text(frame, "Prediction error (see console)", 28, (0,0,255))
                        print("❌ Prediction error:", repr(e))

            # FPS
            if SHOW_FPS:
                now = time.time()
                dt = now - last_t
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
                last_t = now
                cv2.putText(frame, f"{fps:.1f} fps", (FRAME_W - 120, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2, cv2.LINE_AA)

            cv2.imshow("Emotion Recognition (Live)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('0'):
                # switch cameras
                cap.release()
                cam_idx = BACK_CAM_INDEX if cam_idx == FRONT_CAM_INDEX else FRONT_CAM_INDEX
                cap = open_camera(cam_idx)
                if cap is None:
                    print("❌ Could not switch camera. Exiting.")
                    break
                ema = None; pred_buffer.clear()
                print(f"🔁 Switched to camera index {cam_idx}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
