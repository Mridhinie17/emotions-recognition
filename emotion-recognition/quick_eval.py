# quick_eval.py
import tensorflow as tf, numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory

model = tf.keras.models.load_model("saved_models/best_model_finetuned.h5")

test_ds = image_dataset_from_directory(
    "data/test", image_size=(96,96), batch_size=64, color_mode="rgb", shuffle=False
).map(lambda x,y: (tf.cast(x, tf.float32)/255.0, y))

loss, acc = model.evaluate(test_ds, verbose=1)
print("Accuracy:", acc)
