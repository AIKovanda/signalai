import time
from datetime import datetime
from pathlib import Path
import numpy as np


def set_union(*sets):
    union = sets[0]
    for i in range(1, len(sets)):
        union |= sets[i]
    return union


def set_intersection(*sets):
    union = sets[0]
    for i in range(1, len(sets)):
        union &= sets[i]
    return union


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


def time_now(millisecond=False):
    # datetime object containing current date and time
    now = datetime.now()
    if millisecond:
        return str(now)
    return now.strftime("%Y-%m-%d %H:%M:%S")


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
