import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from signalai import config, read_bin
from taskchain.task import Config
from signalai.tools.filters import gauss_convolve


def run(config_path, output_img):
    print("Loading config...")
    conf = Config(
        config.TASKS_DIR,  # where Taskchain data should be stored
        config_path,
        global_vars=config,  # set global variables
    )
    chain = conf.chain()
    chain.set_log_level('CRITICAL')
    print("Config loaded successfully.")

    print("Loading model...")
    signal_model = chain.trained_model.value
    print("Model loaded successfully.")

    print("Loading signal...")
    bin_file = '/mnt/AXAGO/Datasets/AE stream/AE-DATA-conti-7380295766912963-ch2.bin'
    signal = read_bin(bin_file)
    print("Signal loaded successfully.")

    print("Evaluating signal...")
    np_result = signal_model(signal, split_by=32768)
    print("Signal evaluated successfully.")

    print("Saving results...")
    plt.figure(figsize=(16, 9))
    sns.lineplot(y=gauss_convolve(np.expand_dims(np_result, 0), 30, 0.8)[0], x=range(len(np_result)))
    plt.savefig(config.EVALUATION_DIR / output_img)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model_config', '-m', type=str)
    p.add_argument('--eval_dir', '-e', default=None, type=str)
    p.add_argument('--count_step', '-c', default=6000, type=int)
    p.add_argument('--test', '-t', default=False, action='store_true')
    p.add_argument('--processing_fs', '-p', default=None, type=int)
    args = p.parse_args()
    run(
        model_config=args.model_config,
        eval_dir=args.eval_dir,
        count_step=args.count_step,
        test=args.test,
        processing_fs=args.processing_fs,
    )

    chosen_config_path = config.CONFIGS_DIR / 'models' / 'tone_identification' / 'piano_inceptiontime.yaml'
    run(chosen_config_path, output_img=f"{Path(__file__).stem}.svg")
