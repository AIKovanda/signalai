import argparse
import pathlib
from itertools import count
from typing import Union
from signalai.core import SignalModel

from signalai import config, read_audio, Signal
from taskchain.task import Config
import numpy as np


SUFFIXES = [".mp3", ".aac", ".wav"]


def predict_from_batch(
        signal_model: SignalModel, batch: Union[str, int], model_config: str,
        eval_dir: pathlib.PosixPath):

    print(f"Predicting with model from batch '{batch}'.")
    signal_model.load(batch=batch)
    song_names = [f for suffix in SUFFIXES for f in eval_dir.glob(f"*{suffix}")]

    output_dir = config.EVALUATION_DIR / ('.'.join(model_config.split(".")[:-1]) + f"_{batch}")
    output_dir.mkdir(exist_ok=True, parents=True)
    for song_name in song_names:
        song_signal = read_audio(song_name)
        pred_channels = [signal_model(np.expand_dims(song_signal_part, 0), split_by=2 ** 15).data_arr
                         for song_signal_part in song_signal.data_arr]

        for j in range(pred_channels[0].shape[0]):
            one_instrument_signal = Signal(np.vstack([pred_channel[j, :] for pred_channel in pred_channels]))
            out_filename = output_dir / f'{song_name.stem}_{j}.mp3'
            one_instrument_signal.to_mp3(out_filename)
            print(f"{out_filename} created successfully.")


def run(model_config: str, eval_dir: pathlib.PosixPath, count_step=6000):
    config_path = config.CONFIGS_DIR / 'models' / model_config
    conf = Config(
        config.TASKS_DIR,  # where should be data stored
        config_path,
        global_vars=config,  # set global variables
    )

    chain = conf.chain()
    chain.set_log_level('CRITICAL')

    signal_model = chain.trained_model.value

    predict_from_batch(signal_model, 'last', model_config, eval_dir)
    for batch in count(count_step, count_step):
        try:
            predict_from_batch(signal_model, batch, model_config, eval_dir)
        except FileNotFoundError:
            break


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model_config', '-m', type=str)
    p.add_argument('--eval_dir', '-e', default=str(config.EVALUATION_DIR / 'predict'), type=str)
    p.add_argument('--count_step', '-c', default=6000, type=int)
    args = p.parse_args()

    run(
        model_config=args.model_config,
        eval_dir=pathlib.PosixPath(args.eval_dir),
        count_step=args.count_step,
    )
