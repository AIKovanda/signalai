import json
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import models
import matplotlib.pyplot as plt
from IPython.display import display
from config import *
import h5py
from sklearn import metrics
import pickle

from models.neuron import Neuron, load_neuron
from signal_tools.SignalGenerator import SignalGenerator
from signal_tools.signal_tools import gauss_convolve, gauss_filter
from tools.json_tools import get_experiment
from tools.img_tools import plot_graph
from tools.utils import load_file, name_from_network
import seaborn as sns


class ExperimentDataInterface:
    def __init__(self, chosen_experiment, data_2D=None, data_3D=None):
        self.loaded_data = {}
        self.chosen_experiment = chosen_experiment
        self.experiment_config = get_experiment(chosen_experiment)
        self.data_2D = data_2D
        self.data_3D = data_3D
        self._load_files(data_2D, data_3D)

    def _load_file(self, dataset_names):
        if type(dataset_names) != list:
            dataset_names = [dataset_names]

        loaded_data = []
        for dataset_name in dataset_names:
            if dataset_name not in self.experiment_config["files"]:
                raise KeyError(f"unknown dataset name: {dataset_name}")
            loaded_data.append(load_file(self.experiment_config["files"][dataset_name]))

        print("dataset loaded successfully")
        if len(loaded_data) == 1:
            return loaded_data[0]
        else:
            raise NotImplementedError("multiple input files are not implemented yet")

    def _load_files(self, data_2D, data_3D):
        if data_2D is None and data_3D is None:
            raise ValueError("at least one data must be provided")
        if data_2D is not None:
            self.loaded_data['data_2D'] = self._load_file(data_2D)
        if data_3D is not None:
            self.loaded_data['data_3D'] = self._load_file(data_3D)

    def load_dataset(self, data_type):
        if data_type not in self.loaded_data:
            raise ValueError(f"data type '{data_type}' is not loaded yet")
        return self.loaded_data[data_type]


def load_neuron_interface(name, path=SAVE_PATH, data_interface=None):
    path = Path(path)
    interface_name = path / f"{name}.json"
    assert interface_name.exists(), "this interface does not exist"
    data = json.load(open(interface_name, "r"))
    if data_interface is None:
        raise ValueError(f"data_interface missing, chosen_experiment: {data['chosen_experiment']}, data_2D: {data['data_2D']}, data_3D: {data['data_3D']}")
    return NeuronInterface(data_interface, network_names=data["network_names"], classes=data["classes"], name=data["name"])


def current_time():
    time.ctime()
    return time.strftime('%Y-%m-%d-%H-%M-%S')


class NeuronInterface:
    def __init__(self, data_interface, networks=None, network_names=None, classes=None, name=None, neuron_path=SAVE_PATH, categories="default_categories"):
        """
        :param networks: list of network dicts or one dict
        :param classes: list of classes for multiclass models
        :param data_interface: instance of data_interface for data loading
        """
        self.neurons = []
        self.classes = classes if classes is not None else ['']
        self.data_interface = data_interface
        self.chosen_experiment = self.data_interface.chosen_experiment
        self.categories = categories
        if networks is None:
            assert network_names is not None, "network_names cannot be None if network is None"
            self.network_names = network_names
            self.load_models(self.network_names, neuron_path, categories)

        else:
            networks = [networks] if type(networks) != list else networks
            self.network_names = []
            for network_id, network in enumerate(networks):
                network['network_id'] = network_id
                self.neurons.append(self.build_neuron(network, categories))
                self.network_names.append(name_from_network(network))

        self.name = name if name is not None else f'{current_time()} interface'

        if not all([self.neurons[0].network[key] == i.network[key] for i in self.neurons for key in
                    ["binary", "multi_label"]]):
            raise NotImplementedError("all data must have the same type")

        self.validation_generator = None

    def load_models(self, network_names, path=SAVE_PATH, categories="default_categories"):
        self.neurons = [load_neuron(name, path) for name in network_names]
        for neuron in self.neurons:
            data_type = "data_2D" if neuron.network.get("is_2D", False) else "data_3D"
            loaded_data = self.data_interface.load_dataset(data_type)
            neuron.build_signal_train_generator(
                loaded_data, neuron.network['crop'],
                neuron.network['input_shape'],
                categories=categories,
                chosen_experiment=self.chosen_experiment,
                batch_size=neuron.network['batch_size'],
                total_time=self.data_interface.experiment_config["total_time"],
                is_2D=neuron.network.get("is_2D", False)
            )

    def save(self, name=None, path=SAVE_PATH):
        path = Path(path) / 'neuron'
        path.mkdir(parents=True, exist_ok=True)
        if name is None:
            name = self.name
        json.dump({
            'network_names': list(map(lambda x: f'{self.chosen_experiment} {x}', self.network_names)),
            'classes': self.classes,
            'name': name,
            'chosen_experiment': self.chosen_experiment,
            'data_2D': self.data_interface.data_2D,
            'data_3D': self.data_interface.data_3D,
        }, open(path.parent / f'{name}.json', 'w'), indent=2)
        for neuron in self.neurons:
            print(self.chosen_experiment)
            neuron.save(self.chosen_experiment, path)

    def build_neuron(self, network, categories="default_categories"):
        neuron = Neuron(network,
                        chosen_experiment=self.chosen_experiment,
                        classes=self.classes,
                        folder=self.chosen_experiment)
        data_type = "data_2D" if network.get("is_2D", False) else "data_3D"
        loaded_data = self.data_interface.load_dataset(data_type)
        neuron.build_signal_train_generator(loaded_data, network['crop'], network['input_shape'], categories=categories,
                                            chosen_experiment=self.chosen_experiment,
                                            total_time=self.data_interface.experiment_config["total_time"],
                                            batch_size=network['batch_size'], is_2D=network.get("is_2D", False), stride=network.get("stride", None))
        return neuron

    def train(self, batch_size=None, epochs=None):
        history = []
        for i, neuron in enumerate(self.neurons):
            print(f"\n\nTraining model {i}...\n")
            history.append(neuron.fit(batch_size=batch_size, epochs=epochs))

        return history

    def freeze(self):
        pass  # TODO

    def get_prediction(self, generator_name="all", force_predict=False):
        prediction_h5_path = DATA_DIR / 'model_results' / 'predictions' / f"{self.chosen_experiment}.h5"
        prediction_h5_path.parent.mkdir(parents=True, exist_ok=True)

        h5_predictions = h5py.File(prediction_h5_path, "a")
        predictions = []
        for neuron in self.neurons:
            prediction_name = f"{neuron.sub_folder}/{neuron.network['name']}/{neuron.network['version']}/id{neuron.network['network_id']}"

            if generator_name != "all":
                predictions.append(neuron.predict(generator_name=generator_name))
                continue

            if force_predict or prediction_name not in h5_predictions:
                prediction = neuron.predict(generator_name=generator_name)
                if prediction_name in h5_predictions:
                    del h5_predictions[prediction_name]

                h5_predictions.create_dataset(prediction_name, data=prediction)

            else:
                prediction = np.array(h5_predictions.get(prediction_name))

            predictions.append(prediction)

        h5_predictions.close()
        generator = self.choose_generator(generator_name)
        generator.reset_index()
        return predictions, generator.get_time_positions()

    def evaluate_model(self, data_interface=None, force_predict=False, bandpass_params=None, add_name=""):
        if data_interface is None:
            data_interface = self.data_interface

        prediction_h5_path = DATA_DIR / 'model_results' / 'predictions' / f"{self.chosen_experiment}_evaluation.h5"
        prediction_h5_path.parent.mkdir(parents=True, exist_ok=True)

        h5_predictions = h5py.File(prediction_h5_path, "a")
        predictions = []
        for neuron in self.neurons:
            is_2D = neuron.network.get("is_2D", False)
            data_type = "data_2D" if is_2D else "data_3D"
            loaded_data = data_interface.load_dataset(data_type)
            network = neuron.network
            prediction_name = f"{neuron.sub_folder}/{neuron.network['name']}/{neuron.network['version']}/id{neuron.network['network_id']}{add_name}"
            all_generator = SignalGenerator(
                loaded_data,
                crop=network['crop'],
                result_shape=network['input_shape'],
                categories={'0': [0, data_interface.experiment_config["total_time"]]},
                chosen_experiment=self.chosen_experiment,
                batch_size=network['batch_size'],
                bandpass_params=bandpass_params,
                generate_labels=True,
                shuffle=False,
                is_2D=is_2D,
                stride=network.get("stride")[2]
            )
            if force_predict or prediction_name not in h5_predictions:
                prediction = neuron.model.predict(all_generator, verbose=1)
                if prediction_name in h5_predictions:
                    del h5_predictions[prediction_name]

                h5_predictions.create_dataset(prediction_name, data=prediction)

            else:
                prediction = np.array(h5_predictions.get(prediction_name))

            predictions.append(prediction)

        h5_predictions.close()
        return predictions, all_generator.get_time_positions()

    def validate_models(self):
        for i, neuron in self.neurons:
            print(f"Validating neuron {i}...")
            neuron.model.evaluate(neuron.valid_generator, verbose=1, steps=1000)

    def validate_model(self, bandpass_params=None, categories="default_categories", steps=1000, verbose=0):
        neuron = self.neurons[0]
        if self.validation_generator is None:
            is_2D = neuron.network.get("is_2D", False)
            data_type = "data_2D" if is_2D else "data_3D"
            loaded_data = self.data_interface.load_dataset(data_type)
            network = neuron.network
            self.validation_generator = SignalGenerator(
                loaded_data,
                crop=network['crop'],
                result_shape=network['input_shape'],
                categories=categories,
                bandpass_params=bandpass_params,
                chosen_experiment=self.chosen_experiment,
                batch_size=network['batch_size'],
                generate_labels=True,
                validation_rate=0.1,
                is_training=False,
                shuffle=True,
                is_2D=is_2D,
                stride=network.get("stride")[1],
                verbose=verbose
            )
        else:
            self.validation_generator.bandpass_params = bandpass_params
            self.validation_generator.reset_index()
        return neuron.model.evaluate(self.validation_generator, verbose=verbose, steps=steps)

    def choose_generator(self, generator_name="all"):
        if generator_name == "all":
            return self.neurons[0].all_generator
        elif generator_name == "train":
            return self.neurons[0].train_generator
        else:
            return self.neurons[0].valid_generator

    def signal_generator_example(self, generator_name="all", predict=False):
        generator = self.choose_generator(generator_name)
        print(generator.get_time_positions()[:200])
        i = 0
        for a, b in generator:
            print(f'batch shape: {a.shape}')
            print(f'prediction labels: {b}')
            if predict:
                pred = [f"{i: .01f}" for i in self.neurons[0].model.predict(a)[:, 0]]
                print(f'predicted by model: {pred}')
            for j in [0]:  # range(a.shape[0]):
                print(a.shape, a.dtype, b.shape, b.dtype)
                fig, ax = plt.subplots(figsize=(18, 5))
                if len(a.shape) == 4:
                    ax.imshow(a[j, ..., 0].tolist(), cmap='Greys_r')
                elif len(a.shape) == 3 and a.shape[-1] < 4:  # max 4 channels
                    sns.lineplot(x=range(a.shape[1]), y=a[j, :, 0])
                    plt.figure()
                    sns.lineplot(x=range(200), y=a[j, :200, 0])
                else:
                    ax.imshow(a[j].tolist(), cmap='Greys_r')
                plt.show()
            i += 1
            if i == 1:
                break
        generator.reset_index()

    def summary(self):
        self.neurons[0].model.summary()

    def generator_benchmark(self, generator_name):
        generator = self.choose_generator(generator_name)
        generator.reset_index()
        i = 0
        for a, b in tqdm(generator):
            if i >= len(generator):
                break
            i += 1

    def visualize(self, generator_name="all", force_predict=False, show=True, svg_name=None, categories=None, data_interface=None, text_coordinates=None, add_exp=None, bandpass_params=None, bandpass_name=""):
        if generator_name == "eval":
            result, time_positions = self.evaluate_model(data_interface, force_predict=force_predict, bandpass_params=bandpass_params, add_name=bandpass_name)
        else:
            result, time_positions = self.get_prediction(generator_name=generator_name, force_predict=force_predict)
        # if len(result) == 1:
        #     average = result[0]
        # else:
        #     average = np.expand_dims(np.average(np.concatenate(result, axis=1), axis=1), axis=1)
        #
        X = result  # + [average]
        X = [1 - i for i in X]
        labels = list(range(len(result)))  # + ['average']

        def plot(X, function, svg_file, labels=None):
            plot_graph(
                time_positions,
                X,
                function,
                labels=labels,
                categories=self.data_interface.experiment_config["categories"][categories] if categories is not None else None,
                x_lim=self.data_interface.experiment_config["total_time"],
                validation_split=0.1,
                add_exp=add_exp,
                svg_file=svg_file,
                show=show,
                text_coordinates=text_coordinates)

        if svg_name is None:
            svg_name = ""
        else:
            svg_name = " " + svg_name
        svg_file = f"{self.name}{svg_name}"

        # plot([1-average], lambda x: x, f"{svg_file} non-edited.svg")
        # plot(lambda x: gauss_convolve(x, 500, 0.7))
        # plot(lambda x: gauss_filter(x, 30))
        plot(X, lambda x: gauss_filter(x, 100), f"{svg_file}", labels)
