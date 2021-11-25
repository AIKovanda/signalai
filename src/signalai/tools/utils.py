import pathlib
from pathlib import Path
import numpy as np
import pydub
from pydub import AudioSegment


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


def pydub2numpy(audio: pydub.AudioSegment) -> (np.ndarray, int):
    """
    Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels],
    where each value is in range [-1.0, 1.0].
    Returns tuple (audio_np_array, sample_rate).
    """
    return np.array(audio.get_array_of_samples(), dtype=np.float32).reshape((-1, audio.channels)) / (
            1 << (8 * audio.sample_width - 1)), audio.frame_rate


def audio_file2numpy(file):
    file = Path(file)
    suffix = file.suffix
    file = str(file.absolute())
    if suffix == '.wav':
        audio = AudioSegment.from_wav(file)
    elif suffix == '.mp3':
        audio = AudioSegment.from_mp3(file)
    elif suffix == '.aac':
        audio = AudioSegment.from_file(file, "aac")
    else:
        raise TypeError(f"Suffix '{suffix}' is not supported yet!")

    return pydub2numpy(audio)[0].T
