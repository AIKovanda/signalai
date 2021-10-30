from pathlib import Path
import numpy as np


def double_sort(x, y, shuffle=False):
    temp = list(zip(x, y))
    if shuffle:
        np.random.seed(42)
        np.random.shuffle(temp)
    else:
        temp = sorted(temp)
    return zip(*temp)


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


def get_instance(instance_class, params):
    instance_class = instance_class
    instance_from = ".".join(instance_class.split(".")[:-1])
    instance_class_name = instance_class.split(".")[-1]
    exec(f"from {instance_from} import {instance_class_name}")

    instance = eval(f"{instance_class_name}(**params)")
    return instance
