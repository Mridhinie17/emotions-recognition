# src/evaluate.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import load_model
from data_loader import load_fer2013_from_folders


def get_class_names(train_dir='data/train'):
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    return class_names

def main():
    # load best model
    model = load_model('saved_models/best_model_finetuned.h5')


    # load evaluation datasets (no augmentation)
    train_ds, val_ds, test_ds = load_fer2013_from_folders('data', img_size=(96,96), batch_size=64, augment=False)

    class_names = get_class_names('data/train')

    # gather predictions and true labels for the test set
    y_true = []
    y_pred = []
    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch)
        y_pred.extend(np.argmax(preds, axis=1).tolist())
        y_true.extend(y_batch.numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion matrix')
    plt.colorbar()
    ticks = range(len(class_names))
    plt.xticks(ticks, class_names, rotation=45)
    plt.yticks(ticks, class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    os.makedirs('saved_models', exist_ok=True)
    plt.savefig('saved_models/confusion_matrix.png')
    print("Saved confusion matrix to saved_models/confusion_matrix.png")

if __name__ == '__main__':
    main()
