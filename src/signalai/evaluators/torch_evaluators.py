import numpy as np
import torch
from matplotlib import pyplot as plt
from signalai.config import DEVICE
from signalai.signal.tools import gauss_convolve
import seaborn as sns


class Tahovka:
    def __init__(self, gen_gen):
        self.gen_gen = gen_gen

    def evaluate(self, model, output_path):
        model.eval()
        with torch.no_grad():
            all_gen = self.gen_gen.get_generator(split=None, log=0, batch_size=32, x_name="X_all", y_name="Y_all")
            result = []
            for i in range(1072656250 // 32 // 32734):
                x, _ = next(all_gen)
                inputs = torch.from_numpy(np.array(x)).to(DEVICE)
                result += list(model(inputs).cpu())

            np_results = np.array([i.numpy() for i in result])[:, 0]
            plt.figure(figsize=(16, 9))
            sns.lineplot(y=gauss_convolve(np.expand_dims(np_results, 0), 30, 0.8)[0], x=range(len(np_results)))
            plt.savefig(output_path)

        model.train()
