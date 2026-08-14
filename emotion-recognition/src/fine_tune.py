
# src/fine_tune.py  (safer fine-tune: fewer layers, smaller LR, more epochs)
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from src.data_loader import load_fer2013_from_folders

# CONFIG
IMG_SIZE = (96,96)
BATCH_SIZE = 64
FINE_TUNE_EPOCHS = 15
UNFREEZE_TOP_N = 10
FINETUNE_LR = 5e-6
OUT_DIR = 'saved_models'
BASE_CHECKPOINT = os.path.join(OUT_DIR, 'best_model_transfer.h5')

def main():
    print("Loading datasets...")
    train_ds, val_ds, test_ds = load_fer2013_from_folders('data', img_size=IMG_SIZE, batch_size=BATCH_SIZE, augment=True)

    print(f"Loading model from: {BASE_CHECKPOINT}")
    model = load_model(BASE_CHECKPOINT)

    # locate MobileNet base
    base = None
    for layer in model.layers:
        if 'mobilenet' in layer.name:
            base = layer
            break
    if base is None:
        for layer in model.layers:
            if hasattr(layer, 'layers'):
                for sub in layer.layers:
                    if 'mobilenet' in sub.name:
                        base = sub
                        break
                if base:
                    break
    if base is None:
        raise RuntimeError("Could not find MobileNet base.")

    total_layers = len(base.layers)
    print(f"Base: {base.name}, total layers = {total_layers}. Unfreezing top {UNFREEZE_TOP_N} layers.")
    for i, layer in enumerate(base.layers):
        layer.trainable = (i >= total_layers - UNFREEZE_TOP_N)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=FINETUNE_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(OUT_DIR, 'best_model_finetuned.h5'), monitor='val_accuracy', save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-8)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks
    )

    model.save(os.path.join(OUT_DIR, 'final_model_finetuned.h5'))
    loss, acc = model.evaluate(test_ds)
    print(f"Test accuracy after fine-tune: {acc:.4f}, loss: {loss:.4f}")

if __name__ == '__main__':
    main()

