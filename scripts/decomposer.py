import json
import sys

import numpy as np
from taskchain.task import Config

from signalai import config
from signalai.time_series import read_audio, Signal
from signalai.time_series_gen import TimeSeriesHolder

SUFFIXES = [".mp3", ".aac", ".wav"]
TO_PREDICT_DIR = config.DATA_DIR / 'eval' / f'predict'
RESULT_DIR = config.DATA_DIR / 'eval' / f'predict' / 'result'


def run(config_path, architecture_name, model_name):
    conf = Config(
        config.TASKS_DIR,  # where Taskchain data should be stored
        config_path,
        global_vars=config,  # set global variables
    )
    chain = conf.chain()
    chain.set_log_level('CRITICAL')

    signal_model = chain.trained_model.value

    song_names = [f for suffix in SUFFIXES for f in TO_PREDICT_DIR.glob(f"*{suffix}")]

    ARCHITECTURE_RESULT_DIR = RESULT_DIR / f'{architecture_name}'
    ARCHITECTURE_RESULT_DIR.mkdir(exist_ok=True, parents=True)

    json_file = ARCHITECTURE_RESULT_DIR / f'{model_name}.json'

    dc = chain.evaluate_model.force().value
    with json_file.open('w') as f:
        json.dump(dc, f)

    audio_dir = ARCHITECTURE_RESULT_DIR / 'audio' / f'{model_name}'
    audio_dir.mkdir(exist_ok=True, parents=True)

    for song_name in song_names:
        song_signal = read_audio(song_name)

        holder = TimeSeriesHolder(timeseries=[song_signal])
        holder.set_taken_length(signal_model.taken_length)

        channel_results = []
        for j in range(song_signal.channels_count):
            channel_results.append(np.concatenate(
                [signal_model.predict_ts(holder.getitem(i).take_channels([j]))[0]
                 for i in range(len(holder))], axis=1))

        for j, (instrument_channels) in enumerate(zip(*channel_results)):
            one_instrument_signal = Signal(data_arr=np.vstack(instrument_channels), fs=song_signal.fs)
            out_filename = audio_dir / f'{song_name.stem}_{j}.mp3'
            one_instrument_signal.to_mp3(out_filename)
            print(f"{out_filename} created successfully.")


if __name__ == '__main__':
    architecture = sys.argv[1]
    model = sys.argv[2]
    chosen_config_path = config.CONFIGS_DIR / 'models' / 'decomposer' / f'{architecture}' / f'{model}.yaml'
    run(chosen_config_path, architecture, model)
