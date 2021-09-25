import pyximport; pyximport.install()
from signalai.hfd_spectrogram.hfd_spectrogram import *

chosen_experiment = '201103_AE'
experiment_config = get_experiment(chosen_experiment)
source_name = "ch0"

if __name__ == '__main__':
    freq_count = 224
    window_length = 128
    stride = 128

    windows_amount = (experiment_config["float_length"] - window_length) // stride  # approximation of possible window amount

    th = 100000  # how long will be the transformation at once
    generator = TransformerGenerator(freq_count, window_length, stride, dtype=np.float16)
    transformed = [generator.generate_next(source_name, how_many((windows_amount - j * th), th)) for j in range(windows_amount // th + 1)]
    transformed = np.concatenate(transformed, axis=1)
    if len(transformed.shape) == 2:
        transformed = transformed.reshape(list(transformed.shape) + [1])

    array_name = f"{chosen_experiment}-{source_name}-224_128_128.npy"
    np.save(ARRAY_FOLDER / array_name, transformed)
