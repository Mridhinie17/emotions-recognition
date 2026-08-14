# src/train.py
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from src.data_loader import load_fer2013_from_folders
from src.model import build_transfer_model
from collections import Counter

# CONFIG
BATCH_SIZE = 64
EPOCHS = 25   # good default for transfer learning; increase if you want
IMG_SIZE = (96, 96)
OUT_DIR = 'saved_models'

def get_class_names(train_dir='data/train'):
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    return class_names

def compute_class_weights(train_dir='data/train'):
    class_names = get_class_names(train_dir)
    counts = []
    for cname in class_names:
        p = os.path.join(train_dir, cname)
        n = sum(1 for _ in tf.io.gfile.glob(os.path.join(p, '*')))
        counts.append(n)
    total = sum(counts)
    class_weights = {i: total / (len(counts) * c) for i, c in enumerate(counts)}
    print("Class counts:", dict(zip(class_names, counts)))
    print("Class weights:", class_weights)
    return class_weights

def plot_history(history, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history.get('accuracy', []), label='train_acc')
    plt.plot(history.history.get('val_accuracy', []), label='val_acc')
    plt.legend(); plt.title('Accuracy')
    plt.subplot(1,2,2)
    plt.plot(history.history.get('loss', []), label='train_loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.legend(); plt.title('Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_plots_transfer.png'))
    plt.close()

def main():
    print("Loading data (RGB, 96x96) with augmentation...")
    train_ds, val_ds, test_ds = load_fer2013_from_folders('data', img_size=IMG_SIZE, batch_size=BATCH_SIZE, augment=True)
    class_names = get_class_names('data/train')
    num_classes = len(class_names)
    print("Classes:", class_names)

    model = build_transfer_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=num_classes, train_base=False)
    model.summary()

    os.makedirs(OUT_DIR, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(OUT_DIR, 'best_model_transfer.h5'), monitor='val_accuracy', save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]

    class_weights = compute_class_weights('data/train')

    print("🚀 Starting transfer-learning training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights
    )

    model.save(os.path.join(OUT_DIR, 'final_model_transfer.h5'))
    plot_history(history, out_dir=OUT_DIR)

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.4f}, loss: {test_loss:.4f}")

if __name__ == '__main__':
    main()
