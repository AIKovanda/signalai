import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np


def set_intersection(*sets):
    union = sets[0]
    for i in range(1, len(sets)):
        union &= sets[i]
    return union


def load_file(file_paths, dtype="float32"):
    if type(file_paths) != list:
        file_paths = [file_paths]
    all_loaded = []
    for file_path in file_paths:
        file_path = Path(file_path)
        file_type = file_path.suffix
        if file_type == "npy":
            loaded = np.load(file_path)
            if len(loaded.shape) == 1:
                loaded = loaded.reshape((1, len(loaded), 1))
            all_loaded.append(loaded)
        elif file_type == "bin":
            loaded = np.fromfile(open(file_path, 'rb'), dtype=dtype)
            all_loaded.append(loaded.reshape((1, len(loaded), 1)))
        else:
            raise TypeError("Unknown file format")

    if len(all_loaded) == 1:
        return all_loaded[0]
    else:
        raise NotImplemented("multiple input is not implemented yet")


def timefunc(func):
    def _wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        print(f"Function '{func.__name__}' took {(time.time() - start_time)} seconds ---")
        return res

    return _wrapper


def join_dicts(*args):
    if all([i == args[0] for i in args]):
        return args[0]
    else:
        new_info = {}
        for key, value in args[0].items():
            if all([key in i for i in args]):
                if all([value == i[key] for i in args]):
                    new_info[key] = value
        return new_info


def original_length(target_length, transforms=(), fs=None):
    if len(transforms) == 0:
        return target_length
    for t in transforms[::-1]:
        target_length = t.original_signal_length(target_length, fs=fs)
    assert target_length is not None and target_length > 0, "Output of chosen transformations does not make sense."
    return target_length


def by_channel(transform: Callable):
    def wrapper(self, *args, **kwargs):
        arg_len = set([len(arg) for arg in args])
        assert len(arg_len) == 1, f"Inputs must be the same length."
        processed_channels = []
        for i in range(list(arg_len)[0]):
            processed_channels.append(
                transform(self, *[arg[i: i+1] for arg in args], **kwargs)
            )
        return np.concatenate(processed_channels, axis=0)

    return wrapper
