from pathlib import Path

from matplotlib import pyplot as plt
import seaborn as sns

from config import EVALUATION_DIR
from signalai import config, read_bin
from taskchain.task import Config
from tools.filters import gauss_convolve


def run(config_path, output_img):
    conf = Config(
        config.TASKS_DIR,  # where should be data stored
        config_path,
        global_vars=config,  # set global variables
    )

    chain = conf.chain()
    chain.set_log_level('CRITICAL')

    signal_model = chain.trained_model.value

    bin_file = '/mnt/AXAGO/Datasets/AE stream/AE-DATA-conti-7380295766912963-ch2.bin'
    signal = read_bin(bin_file)

    np_result = signal_model(signal, split_by=32768)

    plt.figure(figsize=(16, 9))
    sns.lineplot(y=gauss_convolve(np_result[0], 30, 0.8)[0], x=range(np_result.shape[-1]))
    plt.savefig(EVALUATION_DIR / output_img)


if __name__ == '__main__':
    chosen_config_path = config.CONFIGS_DIR / 'models' / 'tahovka' / 'basic_inceptiontime.yaml'
    run(chosen_config_path, output_img=f"{Path(__file__).stem}.svg")
