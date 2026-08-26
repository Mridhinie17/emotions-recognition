import os
import sys
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score
from src.data_loader import load_fer2013_from_folders

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = "data"
MODELS_DIR = "saved_models"
OUT_FILE = "outputs/paper_tables_results.txt"
os.makedirs("outputs", exist_ok=True)

log_file = open(OUT_FILE, "w", encoding="utf-8")
def log_print(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=log_file, flush=True)

log_print("="*80)
log_print("EMOTION RECOGNITION - PAPER TABLES FAST GENERATOR")
log_print("="*80)

train_dir = os.path.join(DATA_DIR, "train")
test_dir = os.path.join(DATA_DIR, "test")
class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

model_files = ['best_model_efficientv2.h5', 'best_model_resnet.h5', 'best_model_convnext.h5', 'best_model_finetuned.h5', 'best_model.h5', 'final_model.h5']
available_models = [m for m in model_files if os.path.exists(os.path.join(MODELS_DIR, m))]

results = {}

for mfile in available_models:
    mpath = os.path.join(MODELS_DIR, mfile)
    log_print(f"\nEvaluating Model: {mfile} ...")
    try:
        model = tf.keras.models.load_model(mpath)
    except Exception as e:
        log_print(f"Failed to load {mfile}: {e}")
        continue
    
    in_shape = model.input_shape
    h = in_shape[1] if (in_shape and len(in_shape) > 1 and in_shape[1] is not None) else 96
    w = in_shape[2] if (in_shape and len(in_shape) > 2 and in_shape[2] is not None) else 96
    c = in_shape[3] if (in_shape and len(in_shape) > 3 and in_shape[3] is not None) else 3
    img_size = (h, w)
    
    total_params = model.count_params()
    model_size_mb = os.path.getsize(mpath) / (1024 * 1024)
    
    # Load dataset correctly using the same data loader
    train_ds, val_ds, test_ds = load_fer2013_from_folders(
        DATA_DIR, img_size=img_size, batch_size=128, augment=False
    )
    
    # Helper to convert channels if needed
    def prepare_batch(x_b):
        if c == 1 and x_b.shape[-1] == 3:
            return tf.image.rgb_to_grayscale(x_b)
        elif c == 3 and x_b.shape[-1] == 1:
            return tf.image.grayscale_to_rgb(x_b)
        return x_b

    # 1. Test set evaluation
    y_true, y_probs = [], []
    start_time = time.time()
    for x_b, y_b in test_ds:
        x_input = prepare_batch(x_b)
        preds = model.predict(x_input, verbose=0)
        y_probs.append(preds)
        y_true.extend(y_b.numpy())
    end_time = time.time()
    
    y_probs = np.concatenate(y_probs, axis=0)
    y_pred = np.argmax(y_probs, axis=1)
    y_true = np.array(y_true)
    
    total_images = len(y_true)
    total_time = end_time - start_time
    fps = total_images / total_time if total_time > 0 else 0
    ms_per_image = (total_time / total_images) * 1000 if total_images > 0 else 0
    test_acc = np.mean(y_true == y_pred)
    
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
    
    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=len(class_names))
    try:
        auc_macro = roc_auc_score(y_true_onehot, y_probs, average="macro", multi_class="ovr")
        auc_weighted = roc_auc_score(y_true_onehot, y_probs, average="weighted", multi_class="ovr")
    except Exception:
        auc_macro, auc_weighted = None, None

    # 2. Validation set evaluation
    val_correct, val_total = 0, 0
    for x_b, y_b in val_ds:
        x_input = prepare_batch(x_b)
        preds = model.predict(x_input, verbose=0)
        val_correct += np.sum(np.argmax(preds, axis=1) == y_b.numpy())
        val_total += len(y_b)
    val_acc = val_correct / val_total if val_total > 0 else 0.0

    # 3. Train set evaluation (sampled for speed, but correctly shuffled split)
    tr_correct, tr_total = 0, 0
    # Take first 30 batches (3840 samples)
    for x_b, y_b in train_ds.take(30):
        x_input = prepare_batch(x_b)
        preds = model.predict(x_input, verbose=0)
        tr_correct += np.sum(np.argmax(preds, axis=1) == y_b.numpy())
        tr_total += len(y_b)
    train_acc = tr_correct / tr_total if tr_total > 0 else 0.0

    results[mfile] = {
        "model_path": mpath,
        "input_shape": in_shape,
        "total_params": total_params,
        "model_size_mb": model_size_mb,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "fps": fps,
        "ms_per_image": ms_per_image,
        "report": report,
        "auc_macro": auc_macro,
        "auc_weighted": auc_weighted
    }
    
    # Print Table 2 immediately for this model!
    log_print(f"\nTable 2. Class-Wise Performance of Model ({mfile}):")
    log_print(f"{'Emotion':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-score':<10} | {'Support':<10}")
    log_print("-" * 65)
    for cname in class_names:
        row = report[cname]
        log_print(f"{cname.capitalize():<15} | {row['precision']:<10.4f} | {row['recall']:<10.4f} | {row['f1-score']:<10.4f} | {int(row['support']):<10}")
    log_print("-" * 65)
    macro = report['macro avg']
    weighted = report['weighted avg']
    log_print(f"{'Macro Average':<15} | {macro['precision']:<10.4f} | {macro['recall']:<10.4f} | {macro['f1-score']:<10.4f} | {'--':<10}")
    log_print(f"{'Weighted Average':<15} | {weighted['precision']:<10.4f} | {weighted['recall']:<10.4f} | {weighted['f1-score']:<10.4f} | {'--':<10}")

log_print("\n" + "="*80)
log_print("TABLE 1 METRICS COMPARISON SUMMARY")
log_print("="*80)

for mfile, res in results.items():
    log_print(f"\nModel: {mfile}")
    log_print(f"Training Accuracy:      {res['train_acc']:.4f}")
    log_print(f"Validation Accuracy:    {res['val_acc']:.4f}")
    log_print(f"Test Accuracy:          {res['test_acc']:.4f} ({res['test_acc']*100:.2f}%)")
    log_print(f"Precision (Macro/Wtd):  {res['report']['macro avg']['precision']:.4f} / {res['report']['weighted avg']['precision']:.4f}")
    log_print(f"Recall (Macro/Wtd):     {res['report']['macro avg']['recall']:.4f} / {res['report']['weighted avg']['recall']:.4f}")
    log_print(f"F1-score (Macro/Wtd):   {res['report']['macro avg']['f1-score']:.4f} / {res['report']['weighted avg']['f1-score']:.4f}")
    if res['auc_macro'] is not None:
        log_print(f"AUC (Macro/Wtd):        {res['auc_macro']:.4f} / {res['auc_weighted']:.4f}")
    log_print(f"Parameters:             {res['total_params']:,} (~{res['total_params']/1e6:.2f} million)")
    log_print(f"Model Size:             {res['model_size_mb']:.2f} MB")
    log_print(f"Webcam Inference Speed: {res['fps']:.1f} FPS ({res['ms_per_image']:.2f} ms/frame)")

log_file.close()
log_print("\nALL COMPUTATIONS COMPLETE SUCCESSFULLY!")
