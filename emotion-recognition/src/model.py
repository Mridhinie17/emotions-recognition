# src/model.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_transfer_model(input_shape=(96,96,3), num_classes=7, dropout_rate=0.5, train_base=False, architecture='mobilenetv2', learning_rate=2e-4):
    """
    Transfer learning model builder supporting MobileNetV2, EfficientNetV2, ResNet50V2, etc.
    - input_shape: (H,W,3)
    - train_base: if True, unfreeze base for fine-tuning
    - architecture: 'mobilenetv2', 'efficientnetv2', 'resnet50v2', 'convnext'
    """
    arch_lower = architecture.lower()
    if 'efficientnet' in arch_lower:
        base = tf.keras.applications.EfficientNetV2S(
            input_shape=input_shape, include_top=False, weights='imagenet'
        )
        preprocess = layers.Rescaling(scale=255.0)
    elif 'resnet' in arch_lower:
        base = tf.keras.applications.ResNet50V2(
            input_shape=input_shape, include_top=False, weights='imagenet'
        )
        preprocess = layers.Rescaling(scale=2.0, offset=-1.0)
    elif 'convnext' in arch_lower and hasattr(tf.keras.applications, 'ConvNeXtTiny'):
        base = tf.keras.applications.ConvNeXtTiny(
            input_shape=input_shape, include_top=False, weights='imagenet'
        )
        preprocess = layers.Rescaling(scale=255.0)
    else:
        base = tf.keras.applications.MobileNetV2(
            input_shape=input_shape, include_top=False, weights='imagenet'
        )
        preprocess = layers.Rescaling(scale=2.0, offset=-1.0)

    base.trainable = bool(train_base)  # freeze by default

    inputs = layers.Input(shape=input_shape)
    x = preprocess(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate / 2.0)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name=f'{arch_lower}_fer')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model

