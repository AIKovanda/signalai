import json
import time
import numpy as np
import keras
from keras_preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras import layers
from tensorflow.keras.applications.efficientnet import *
from config import *
import os
from keras import backend as K
from tensorflow.keras.activations import swish

from models.inception_time import shortcut_layer, inception_time_module
from signal_tools.SignalGenerator import SignalGenerator
from tools.json_tools import get_experiment
from tools.utils import save_report, name_from_network

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def load_neuron(name, path=SAVE_PATH):
    path = Path(path) / "neuron"
    neuron_name = path / f"{name}.json"
    assert neuron_name.exists(), "this neuron does not exist"
    data = json.load(open(neuron_name, "r"))
    return Neuron(data["network"], data["chosen_experiment"], classes=data["classes"], folder=data["folder"], build=False, from_path=path, name=name)


class Neuron:
    def __init__(self, network, chosen_experiment, classes=None, folder=None, build=True, from_path=SAVE_PATH, name=None):
        self.chosen_experiment = chosen_experiment
        self.experiment_config = get_experiment(chosen_experiment)
        self.classes = classes
        self.sub_folder = "binary" if len(self.classes) == 1 or self.classes is None else "categorical"
        self.folder = folder
        self.model = None
        self.network = network.copy()
        self.sub_folder = (
            "multi_label" if network["multi_label"] else ("binary" if network["binary"] else "categorical"))
        self.name = name if name is not None else name_from_network(self.network)
        if build:
            self.build_model()
        else:
            self.load_model(Path(from_path))

    def load_model(self, from_path=SAVE_PATH):
        self.model = keras.models.load_model(from_path / f'{self.name}.h5', custom_objects={'f1_m': f1_m, 'recall_m': recall_m,
                                                                                 'precision_m': precision_m})
        self.compile_model()

    def save(self, chosen_experiment, path=SAVE_PATH):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        json.dump({
            'network': self.network,
            'classes': self.classes,
            'name': self.name,
            'chosen_experiment': self.chosen_experiment,
            'folder': self.folder},
                  open(path / f'{chosen_experiment} {self.name}.json', 'w'), indent=2)
        self.model.save(path / f'{chosen_experiment} {self.name}.h5')
        print(f'{chosen_experiment} {self.name}.h5')

    def build_model(self):
        input_shape = self.network.get("input_shape")
        if self.network.get('is_2D', False):
            input_shape = input_shape[1:]
        x = keras.Input(shape=input_shape)
        inputs = x
        for layer in self.network["layers"]:
            if layer["type"] == "Dense":
                x = layers.Dense(units=layer["units"], kernel_initializer='he_uniform', bias_initializer='zeros',
                                 activation=layer.get("activation", 'relu'),
                                 kernel_regularizer=regularizers.l2(layer.get("L2", 0.001)))(x)

            elif layer["type"] == "Dropout":
                x = layers.Dropout(rate=layer.get("rate", 0.5))(x)

            elif layer["type"] == "Conv2D":
                x = layers.Convolution2D(filters=layer["filters"], kernel_size=layer.get("kernel_size", (3, 3)),
                                         activation=layer.get("activation", swish),
                                         kernel_regularizer=regularizers.l2(layer.get("L2", 0.001)))(x)

            elif layer["type"] == "Pooling":
                x = layers.MaxPooling2D(pool_size=layer.get("pool_size", (2, 2)))(x)

            elif layer["type"] == "Flatten":
                x = layers.Flatten()(x)

            elif layer["type"] == "Reshape":
                x = layers.Reshape(layer["shape"])(x)

            elif layer["type"] == "BatchNorm":
                x = layers.BatchNormalization()(x)

            elif layer["type"] == "EfficientNet":
                model = eval(
                    f"EfficientNetB{layer['version']}(include_top=False, input_tensor=x, weights='imagenet')")
                x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
                x = layers.BatchNormalization(name="end_of_EFNET")(x)
                x = layers.Dropout(0.2, name="top_dropout")(x)

            elif layer["type"] == "InceptionTime":
                x = x
                input_res = x

                for d in range(layer.get("depth", 6)):
                    x = inception_time_module(
                        x,
                        stride=layer.get("stride", 1),
                        bottleneck_size=layer.get("bottleneck_size", 32),
                        kernel_size=layer.get("kernel_size", 40),
                        nb_filters=layer.get("nb_filters", 32))

                    if layer.get("use_residual", True) and d % 3 == 2:
                        x = shortcut_layer(input_res, x)
                        input_res = x

                x = keras.layers.GlobalAveragePooling1D()(x)

            else:
                raise ValueError("unidentified layer type")

        if self.network["multi_label"]:
            x = layers.Dense(units=len(self.classes), activation='sigmoid')(x)
        else:
            if self.network["binary"]:
                x = layers.Dense(units=1, activation='sigmoid')(x)
            else:
                x = layers.Dense(units=len(self.classes), activation='softmax')(x)

        self.model = keras.Model(inputs, x)
        self.compile_model()

    def compile_model(self):
        if self.network["multi_label"]:
            used_loss = "binary_crossentropy"
            used_metrics = "accuracy"
        else:
            if self.network["binary"]:
                used_loss = "binary_crossentropy"
                used_metrics = "accuracy"
            else:
                used_loss = "categorical_crossentropy"
                used_metrics = "categorical_accuracy"

        self.model.compile(
            loss=used_loss,
            optimizer=keras.optimizers.Adam(learning_rate=self.network.get("lr", 1e-2)),
            metrics=[used_metrics] + ([
                                          f1_m,
                                          recall_m,
                                          precision_m] if self.network["binary"] else []))

    def set_trainable_model(self, trainable=True):
        end_of_efnet = -1
        for i, layer in enumerate(self.model.layers):
            if layer.name == "end_of_EFNET":
                end_of_efnet = i
                break
        print(end_of_efnet)
        for layer in self.model.layers[:end_of_efnet - 2]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = trainable
        optimizer = keras.optimizers.Adam(learning_rate=1e-2)
        self.model.compile(
            optimizer=optimizer,
            loss=("binary" if len(self.classes) == 1 else "categorical") + "_crossentropy",
            metrics=[("" if len(self.classes) == 1 else "categorical_") + 'accuracy'] + ([
                                                                                             f1_m,
                                                                                             recall_m,
                                                                                             precision_m] if len(
                self.classes) == 1 else []))

    def fit(self, train_data=None, validation_data=None, network=None, batch_size=16, epochs=None, csv_file="", decay_rate=0.85,
            decay_step=1, trainable=None, shuffle=True):
        if network is None:
            network = self.network

        on_generator = train_data is None or validation_data is None

        train_params = network.get("train_params",
                                   {"epochs": epochs, "decay_rate": decay_rate, "decay_step": decay_step})
        if epochs is not None:
            train_params["epochs"] = epochs

        self.decay_rate = train_params["decay_rate"]
        self.decay_step = train_params["decay_step"]
        assert self.model is not None
        if trainable is not None:
            self.set_trainable_model(trainable)

        time.ctime()
        current_time = time.strftime('%Y%m%d-%H:%M')
        filename = MODEL_PATH / self.folder / self.sub_folder / f'{network["name"]}.{network["version"]}.id{network["network_id"]}.h5'
        model_params = {"filename": str(filename),
                        "trained_on": str(csv_file),
                        "date": current_time}
        save_report(annotations_file=None,
                    model_name=str(Path(filename).stem),
                    metrics=None,
                    sub_folder=self.sub_folder,
                    model_params={**model_params, **network},
                    other={"classes": self.classes})

        model_path = MODEL_PATH / 'models' / self.folder / self.sub_folder
        model_path.mkdir(parents=True, exist_ok=True)
        my_callbacks = [
            keras.callbacks.LearningRateScheduler(self.lr_scheduler, verbose=1),
            keras.callbacks.ModelCheckpoint(
                filepath=model_path / f'{network["name"]}.{network["version"]}.epoch{{epoch}}.id{network["network_id"]}.h5'),
            keras.callbacks.TensorBoard(
                log_dir=f'.log/epochs/{self.sub_folder}/{network["name"]}.{network["version"]}',
                histogram_freq=1
            ),
        ]
        if on_generator:
            history = self.model.fit(
                self.train_generator,
                steps_per_epoch=len(self.train_generator),
                callbacks=my_callbacks,
                validation_data=self.valid_generator,
                validation_steps=len(self.valid_generator),
                shuffle=shuffle,
                epochs=train_params["epochs"])
        else:
            def batch_generator(X, Y, batch_size=64):
                indices = np.arange(len(X))
                batch = []
                while True:  # it might be a good idea to shuffle your data before each epoch
                    np.random.shuffle(indices)
                    for i in indices:
                        batch.append(i)
                        if len(batch) == batch_size:
                            yield X[batch], Y[batch]
                            batch = []

            train_generator = batch_generator(train_data[0], train_data[1], batch_size=batch_size)
            history = self.model.fit(
                train_generator,
                steps_per_epoch=len(train_data[0]) // batch_size,
                validation_data=validation_data,
                callbacks=my_callbacks,
                epochs=train_params["epochs"])
        return history

    def lr_scheduler(self, epoch, lr):
        """
        learning rate decay in time
        """
        if epoch % self.decay_step == 0 and epoch:
            return lr * pow(self.decay_rate, np.floor(epoch / self.decay_step))
        return lr
