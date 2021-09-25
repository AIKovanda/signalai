import keras
from keras import layers
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def build_model(out_size):
    inputs = layers.Input(shape=(224, 224, 3))
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")
    # Freeze the pretrained weights
    # model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    if out_size == 1:
        outputs = layers.Dense(1, activation="sigmoid", name="pred")(x)
        model = keras.Model(inputs, outputs, name="EfficientNet")
        optimizer = keras.optimizers.Adam(learning_rate=1e-2)
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    else:
        outputs = layers.Dense(out_size, activation="softmax", name="pred")(x)
        model = keras.Model(inputs, outputs, name="EfficientNet")
        optimizer = keras.optimizers.Adam(learning_rate=1e-2)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def unfreeze_model(model, out_size):
    # We unfreeze the top 2 layers while leaving BatchNorm layers frozen
    for layer in model.layers[:-2]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    optimizer = keras.optimizers.Adam(learning_rate=1e-2)
    if out_size == 1:
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    else:
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])


def freeze_model(model, out_size):
    # We unfreeze the top 2 layers while leaving BatchNorm layers frozen
    for layer in model.layers[:-2]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    optimizer = keras.optimizers.Adam(learning_rate=1e-2)
    if (out_size == 1):
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    else:
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
