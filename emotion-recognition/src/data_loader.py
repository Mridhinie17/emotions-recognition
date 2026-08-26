# src/data_loader.py
import tensorflow as tf
import os

AUTOTUNE = tf.data.AUTOTUNE

def build_augmentation_layer(img_size=(96,96)):
    """Keras augmentation layer for training (runs on GPU if available)."""
    return tf.keras.Sequential([
        tf.keras.layers.Resizing(img_size[0], img_size[1]),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.12),
        tf.keras.layers.RandomTranslation(0.08, 0.08),
        tf.keras.layers.RandomContrast(0.15),
        tf.keras.layers.RandomBrightness(0.15),
        tf.keras.layers.RandomZoom(0.1),
    ])

def to_rgb(images):
    # images: [B,H,W,1] or [B,H,W,3] -> ensure 3 channels
    if images.shape[-1] == 1:
        images = tf.image.grayscale_to_rgb(images)
    return images

def load_fer2013_from_folders(base_dir='data', img_size=(96,96), batch_size=64, seed=42, augment=True):
    """
    Loads dataset from:
      data/train/<class>/*.jpg
      data/test/<class>/*.jpg
    Returns: train_ds, val_ds, test_ds - tf.data.Dataset of (images, labels)
    Images returned are float32 in range [0,1], shape (H,W,3).
    """
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

    train_raw = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        color_mode='grayscale',   # read grayscale then convert to RGB
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed
    )

    val_raw = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        color_mode='grayscale',
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=seed
    )

    test_raw = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        color_mode='grayscale',
        batch_size=batch_size,
        shuffle=False
    )

    augmentation = build_augmentation_layer(img_size)
    rescale = tf.keras.layers.Rescaling(1./255)

    def prep_train(ds):
        def _map(x, y):
            # x: uint8 [B,H,W,1] -> convert to float and RGB
            x = tf.cast(x, tf.float32)
            x = to_rgb(x)
            x = augmentation(x)
            x = rescale(x)
            return x, y
        ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
        ds = ds.shuffle(500)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    def prep_eval(ds):
        def _map(x, y):
            x = tf.cast(x, tf.float32)
            x = to_rgb(x)
            x = rescale(x)
            return x, y
        ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    train_ds = prep_train(train_raw) if augment else prep_eval(train_raw)
    val_ds = prep_eval(val_raw)
    test_ds = prep_eval(test_raw)

    return train_ds, val_ds, test_ds
