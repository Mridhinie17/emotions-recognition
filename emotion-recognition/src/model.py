# src/model.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_transfer_model(input_shape=(96,96,3), num_classes=7, dropout_rate=0.5, train_base=False):
    """
    MobileNetV2 transfer learning model.
    - input_shape: (H,W,3), we use 96x96 by default.
    - train_base: if True, unfreeze the base for fine-tuning (do this later).
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling=None
    )
    base.trainable = bool(train_base)  # freeze by default

    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name='mobilenetv2_fer')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
