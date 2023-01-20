from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from taskchain.task import Config
from tqdm import trange

from signalai import config
from signalai import read_bin
from signalai.time_series_gen import TimeSeriesHolder
from signalai.tools.visualization import plot_binary_map


OUTPUT_DIR = 'test'
X_MAX = 1000  # time in ms of the experiment
BIN_FILE = config.DATA_DIR / 'burst_detection' / 'hum_and_cracks' / 'AE-DATA-conti-7382946824004051-ch4.bin'


def run(config_path):
    print("Loading config...")
    conf = Config(
        config.TASKS_DIR,  # where Taskchain data should be stored
        config_path,
        global_vars=config,  # set global variables
    )
    chain = conf.chain()
    chain.set_log_level('CRITICAL')

    print("Loading model...")
    # chain.train_model.force()
    signal_model = chain.trained_model.value
    print("Model loaded successfully.")

    print("Loading signal...")
    signal = read_bin(BIN_FILE)
    holder = TimeSeriesHolder(timeseries=[signal])
    holder.set_taken_length(signal_model.taken_length)
    print('Shape of input signal:', signal.data_arr.shape)
    print("Signal loaded successfully.")

    np_result = [signal_model.predict_ts(holder.getitem(i))[0][0].detach().cpu().numpy() for i in trange(len(holder))]
    np_result = np.concatenate(np_result, axis=1)

    print('Shape of output tensor:', np_result.shape)
    print("Signal evaluated successfully.")
    print("Saving results...")
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    cracks_list = ['crack 0', 'crack 1']
    plot_binary_map(np_result > .5, all_object_labels=cracks_list, savefigs=[output_dir / f'binary_map_full.pdf'],
                    figsize=(14, 1), xshift=.7, yshift=-0.5, x_max=X_MAX, show=False, object_name='Crack')

    print('Plotting graphs...')
    plt.figure(figsize=(28, 9))
    sns.lineplot(x=range(np_result.shape[1]), y=np_result[0, :])
    sns.lineplot(x=range(np_result.shape[1]), y=np_result[1, :])
    plt.savefig(str(output_dir / f'bursts.png'), bbox_inches='tight', pad_inches=0.)


if __name__ == '__main__':
    chosen_config_path = config.CONFIGS_DIR / 'models' / 'burst_detection' / 'inceptiontime' / '2layer_x16.yaml'
    chosen_config_path = config.CONFIGS_DIR / 'models' / 'burst_detection' / 'spec2map' / '1x2d-layer_2x1d-layer.yaml'
    run(chosen_config_path)
