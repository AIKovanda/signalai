from abc import abstractmethod

import numpy as np
import pandas as pd
from signalai.signal_tools.signal_tools import bandpass_ifft
from signalai.tools.utils import double_sort


def uniform_indices(input_dim, target_dim):
    return np.linspace(0, input_dim - 1, target_dim, dtype=np.int)


def generate_starting_positions(category_names, category_positions, x_length, shuffle=False, stride=None):
    """
    :param category_names: list of category names
    :param category_positions: numpy array of category starting and ending position
    :param x_length: the length of items
    :param shuffle: shuffle
    :param stride: None if no overlapping, int if stride generating
    :return: tuple of positions, tuple of labels
    """

    if stride is None:
        stride = x_length

    category_lengths = category_positions[:, 1] - category_positions[:, 0]
    items_per_category = (np.min(category_lengths) - x_length) // stride

    start_position = []
    labels = []
    for i, category in enumerate(category_positions):
        start_position += (uniform_indices(category_lengths[i] - x_length, items_per_category) + category[0]).tolist()
        labels += [category_names[i]] * items_per_category
    return map(np.array, double_sort(start_position, labels, shuffle=shuffle))


def train_split(categories, validation_ratio, train=True):
    categories = np.array(categories)
    if train:
        categories[:, 2] = categories[:, 2] - (categories[:, 2] - categories[:, 1]) * validation_ratio
    else:
        categories[:, 1] = categories[:, 1] + (categories[:, 2] - categories[:, 1]) * (1 - validation_ratio)
    return categories.tolist()


class SignalGeneratorOld:
    def __init__(self, input_data, crop, result_shape, categories, chosen_experiment, batch_size=128, shuffle=True,
                 generate_labels=True, bandpass_params=None, validation_rate=None, is_training=True, is_2D=False, stride=None, verbose=1):
        """
        input_data.shape = (224, 8380125, 1) for image, (1, 1072656250, 1) for raw signal
        crop=[[0,128], [0,128]]  # y, x
        result_shape=[64,64,1]  # y, x, channels
        bandpass_params=(lowcut, highcut, fs)
        categories={'0': [10.0, 40.0], '1': [618.7, 646.5]}  # in second or string of name
        """
        self.bandpass_params = bandpass_params
        self.stride = stride
        self.is_2D = is_2D
        self.input_data = input_data
        self.is_image = len(input_data.shape) == 3
        self.crop = crop
        self.result_shape = result_shape
        self.generate_labels = generate_labels
        self.total_experiment_time = self.experiment_config["total_time"]
        if type(categories) == str:
            categories = self.experiment_config["categories"][categories]
        self.category_names = list(categories.keys())
        self.binary = len(self.category_names) <= 2  # TODO: 1 means all is the rest
        self.category_positions = np.array([[i * input_data.shape[1] / self.total_experiment_time for i in categories[key]] for key in self.category_names], dtype=np.int)
        if validation_rate is not None:
            if is_training:
                self.category_positions[:, 1] = self.category_positions[:, 0] + (self.category_positions[:, 1] - self.category_positions[:, 0]) * (1 - validation_rate)
            else:
                self.category_positions[:, 0] = self.category_positions[:, 0] + (self.category_positions[:, 1] - self.category_positions[:, 0]) * (1 - validation_rate)

        self.batch_size = batch_size

        self.x_len = crop[1][1] - crop[1][0]
        self.y_len = crop[0][1] - crop[0][0]
        self.starting_indices, self.labels = generate_starting_positions(
            self.category_names,
            self.category_positions,
            self.x_len,
            shuffle=shuffle,
            stride=stride
        )

        self.labels = pd.get_dummies(self.labels, columns=self.category_names).to_numpy(dtype=np.float32)

        self.n = len(self.starting_indices)
        self.i = 0  # current index
        self.starting_indices_time = [self.total_experiment_time * i / self.input_data.shape[-1] for i in self.starting_indices][
                                     :(self.n // self.batch_size) * self.batch_size]

        if verbose == 1:
            print(f"{self.n} pictures will be generated in {len(self.category_names)} categories. Each category has {int(self.n / len(self.category_names))} pictures.")

    def __getitem__(self, index):
        batch_indices = np.arange(self.i, self.i + self.batch_size)  # item indices
        batch_indices[batch_indices >= self.n] = batch_indices[batch_indices >= self.n] % self.n
        self.i = (self.i + self.batch_size) % self.n
        batch_positions = self.starting_indices[batch_indices]

        X = np.zeros([self.batch_size, self.y_len, self.x_len] + self.result_shape[2:], dtype=np.float32)

        for batch_id, start in enumerate(batch_positions):
            temp = self.input_data[self.crop[0][0]:self.crop[0][1], start:start + self.x_len]
            if self.bandpass_params is not None:
                temp = bandpass_ifft(temp, self.bandpass_params)
            X[batch_id] = temp

        X = X[:, uniform_indices(X.shape[1], self.result_shape[0]), ...]
        X = X[:, :, uniform_indices(X.shape[2], self.result_shape[1]), ...]

        if self.is_2D:
            X = X.squeeze(axis=1)

        if self.generate_labels:
            if self.binary:
                return X, self.labels[batch_indices, 0]
            else:
                return X, self.labels[batch_indices]

        return X

    def reset_index(self):
        self.i = 0

    def __len__(self):
        return self.n // self.batch_size

    def get_time_positions(self):
        return self.starting_indices * self.total_experiment_time / self.input_data.shape[1]


# def on_epoch_end(self):
#     pass