from pathlib import Path

from config import EVALUATION_DIR
from signalai import config, read_audio, Signal
from taskchain.task import Config


def run(config_path, project_name="base"):
    conf = Config(
        config.TASKS_DIR,  # where should be data stored
        config_path,
        global_vars=config,  # set global variables
    )

    chain = conf.chain()
    chain.set_log_level('CRITICAL')

    signal_model = chain.trained_model.value
    song_names = [
        'Dua Lipa - Levitating Featuring DaBaby (Official Music Video).mp3',
        'MB14 - Road to Zion.mp3',
        'A_Classic_Education_-_NightOwl_-_0.aac',
    ]
    new_dir = Path(EVALUATION_DIR) / project_name
    new_dir.mkdir()
    for song_name in song_names:
        song_signal = read_audio(song_name)
        predicted_song = signal_model(song_signal, split_by=10000)

        for j in range(4):
            one_instrument_signal = Signal(predicted_song[j * 2:(j + 1) * 2, :])
            out_filename = new_dir / f'{song_name[:-4]}_{j}.mp3'
            one_instrument_signal.to_mp3(out_filename)
            print(f"{out_filename} created successfully.")


if __name__ == '__main__':
    # config_name = 'decomposer1L.yaml'
    config_name = 'decomposer1L255_nores.yaml'
    # config_name = 'decomposer1L255_nores_mish.yaml'
    # config_name = 'decomposer1L255_nores_tanh.yaml'
    # config_name = 'decomposer3L.yaml'
    # config_name = 'decomposer3L255_nores.yaml'
    # config_name = 'decomposer3L_nores.yaml'

    chosen_config_path = config.CONFIGS_DIR / 'models' / 'decomposer' / config_name
    run(chosen_config_path, project_name=config_name)
