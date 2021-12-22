from pathlib import Path

from signalai import config, read_audio, Signal
from taskchain.task import Config
import numpy as np


def run(config_name):
    config_path = config.CONFIGS_DIR / 'models' / 'decomposer' / config_name[0] / config_name[1]
    conf = Config(
        config.TASKS_DIR,  # where should be data stored
        config_path,
        global_vars=config,  # set global variables
    )

    chain = conf.chain()
    chain.set_log_level('CRITICAL')

    signal_model = chain.trained_model.value
    song_names = map(lambda x: Path('/home/martin/Music') / x, [
        'Dua Lipa - Levitating Featuring DaBaby (Official Music Video).mp3',
        'MB14 - Road to Zion.mp3',
        'A_Classic_Education_-_NightOwl_-_0.aac',
    ])
    new_dir = Path(config.EVALUATION_DIR) / config_name[0] / config_name[1].split(".")[0]
    new_dir.mkdir(exist_ok=True, parents=True)
    for song_name in song_names:
        song_signal = read_audio(song_name)
        pred_channels = [signal_model(np.expand_dims(song_signal_part, 0), split_by=2**16)
                         for song_signal_part in song_signal.data_arr]

        for j in range(4):
            one_instrument_signal = Signal(np.vstack([pred_channel[j, :] for pred_channel in pred_channels]))
            out_filename = new_dir / f'{song_name.name[:-4]}_{j}.mp3'
            one_instrument_signal.to_mp3(out_filename)
            print(f"{out_filename} created successfully.")


if __name__ == '__main__':
    # config_tuple = ('augment', 'decomposer1L255_nores.yaml')
    config_tuple = ('augment', 'se_decomposer.yaml')

    # config_tuple = ('base', 'decomposer1L.yaml')
    # config_tuple = ('base', 'decomposer1L255_nores.yaml')
    # config_tuple = ('base', 'decomposer1L255_nores_mish.yaml')
    # config_tuple = ('base', 'decomposer1L255_nores_tanh.yaml')
    # config_tuple = ('base', 'decomposer3L.yaml')
    # config_tuple = ('base', 'decomposer3L255_nores.yaml')
    # config_tuple = ('base', 'decomposer3L_nores.yaml')

    run(config_name=config_tuple)
