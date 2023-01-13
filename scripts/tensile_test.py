from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from taskchain.task import Config
from tqdm import trange

from signalai import config, read_bin
from signalai.time_series_gen import TimeSeriesHolder
from signalai.tools.filters import gauss_convolve

OUTPUT_DIR = 'test'
X_MAX = 1000  # time in ms of the experiment
BIN_FILE = '/mnt/AXAGO/Datasets/tensile_tests/200915/200915_AE.bin'


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

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.figure(figsize=(16, 9))
    
    for model_id in range(signal_model.model_count):  # how many models we have
        print(f"Evaluating signal by model {model_id}...")
        signal_model.load(model_id=model_id)
        
        np_result = [signal_model.predict_ts(holder.getitem(i))[0].detach().cpu().numpy() for i in trange(len(holder))]
        np_result = np.concatenate(np_result, axis=-1)

        print('Shape of output tensor:', np_result.shape)
        print("Signal evaluated successfully.")

        y = gauss_convolve(np_result, 30, 0.8)[0]
        sns.lineplot(y=y, x=range(len(y)), label=model_id)

    print("Saving results...")
    plt.savefig(output_dir / (Path(BIN_FILE).stem + ".png"), dpi=400)
    plt.close()


if __name__ == '__main__':
    chosen_config_path = config.CONFIGS_DIR / 'models' / 'tensile_test' / 'inceptiontime.yaml'
    run(chosen_config_path)
