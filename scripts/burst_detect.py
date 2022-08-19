import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from taskchain.task import Config

from signalai import config
from signalai import read_bin
from signalai.tools import plot_binary_map


def run(config_path):
    print("Loading config...")
    conf = Config(
        config.TASKS_DIR,  # where Taskchain data should be stored
        config_path,
        global_vars=config,  # set global variables
    )
    chain = conf.chain()
    chain.set_log_level('CRITICAL')

    length, _ = chain.echo_info.value
    fs = 1562500

    x_max = 1000 * length / fs
    print("Config loaded successfully.")

    print("Loading model...")
    # chain.train_model.force()
    signal_model = chain.trained_model.value
    print("Model loaded successfully.")

    bin_file_to_predict = sys.argv[1]

    print("Loading signal...")
    signal = read_bin(bin_file_to_predict)  # .crop([0, 50000000])
    print('Shape of input signal:', signal.data_arr.shape)
    print("Signal loaded successfully.")

    np_result = signal_model(signal, target_length=32768, residual_end=False).data_arr

    if len(np_result.shape) == 1:
        np_result = np.expand_dims(np_result, 0)

    print('Shape of output tensor:', np_result.shape)
    print("Signal evaluated successfully.")
    print("Saving results...")
    file_stem = Path(sys.argv[1]).stem
    output_dir = Path(file_stem)
    output_dir.mkdir(exist_ok=True)

    np.save(str(output_dir / f'{file_stem}.npy'), np_result)
    pd.DataFrame(np_result).T.to_csv(str(output_dir / f'{file_stem}.csv.zip'), index=False)

    cracks_list = ['crack 0', 'crack 1']
    plot_binary_map(np_result > .5, all_object_labels=cracks_list, savefigs=[output_dir / f'binary_map_full.pdf'],
                    figsize=(14, 1), xshift=.7, yshift=-0.5, x_max=x_max, show=False, object_name='Tone')

    print('Plotting graphs...')
    plt.figure(figsize=(28, 9))
    sns.lineplot(x=range(np_result.shape[1]), y=np_result[0, :])
    sns.lineplot(x=range(np_result.shape[1]), y=np_result[1, :])
    plt.savefig(str(output_dir / f'{file_stem}.png'), bbox_inches='tight', pad_inches=0.)
    plt.savefig(str(output_dir / f'{file_stem}.svg'), bbox_inches='tight', pad_inches=0.)


if __name__ == '__main__':
    chosen_config_path = config.CONFIGS_DIR / 'models' / 'burst_detection' / 'inceptiontime' / '0_noscale.yaml'
    run(chosen_config_path)
