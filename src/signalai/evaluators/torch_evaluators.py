from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from signalai.config import DEVICE, BASE_DATA_DIR
from signalai.tools.filters import gauss_convolve
import seaborn as sns
from signalai.signal import Signal
from signalai.tools.utils import audio_file2numpy


class TahovkaEvaluator:
    def __init__(self, gen_gen):
        self.gen_gen = gen_gen

    def evaluate(self, signal_model, output_img):
        signal_model.eval()
        with torch.no_grad():
            all_gen = self.gen_gen.get_generator(split=None, log=0, batch_size=32, x_name="X_all", y_name="Y_all")
            result = []
            for i in range(1072656250 // 32 // 32734):
                x, _ = next(all_gen)
                inputs = torch.from_numpy(np.array(x)).to(DEVICE)
                result += list(signal_model(inputs).cpu())

            np_results = np.array([i.numpy() for i in result])[:, 0]
            plt.figure(figsize=(16, 9))
            sns.lineplot(y=gauss_convolve(np.expand_dims(np_results, 0), 30, 0.8)[0], x=range(len(np_results)))
            plt.savefig(output_img)

        signal_model.train()


class DecompositionEvaluator:
    def __init__(self, name):
        self.name = name

    def evaluate(self, signal_model, output_dir):
        for song_name in ['Dua Lipa - Levitating Featuring DaBaby (Official Music Video).mp3',
                          'MB14 - Road to Zion.mp3', 'A_Classic_Education_-_NightOwl_-_0.aac']:

            mp3 = audio_file2numpy(BASE_DATA_DIR / song_name)
            signal_parts = []
            for i in range(mp3.shape[-1] // 10000 + 1):
                signal_part = mp3[:, i * 10000:(i + 1) * 10000]
                new_signal = signal_model.eval_on_batch(np.expand_dims(signal_part, 0))[0]
                signal_parts.append(new_signal)

            predicted_all = np.hstack(signal_parts)

            new_dir = Path(output_dir) / self.name
            new_dir.mkdir()
            for j in range(4):
                new_s = Signal(predicted_all[j * 2:(j + 1) * 2, :])
                new_mp3_file = new_dir / f'{song_name[:-4]}_{j}.mp3'
                new_s.to_mp3(new_mp3_file)
                print(f"{new_mp3_file} created successfully.")
