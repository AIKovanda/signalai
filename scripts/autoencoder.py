import json
import sys

from taskchain.task import Config

from signalai import config, read_audio
from signalai.timeseries import MultiSeries

SUFFIXES = [".mp3", ".aac", ".wav"]


config_path = config.CONFIGS_DIR / 'models' / 'autoencoder' / f'{sys.argv[1]}.yaml'

conf = Config(
    config.TASKS_DIR,  # where Taskchain data should be stored
    config_path,
    global_vars=config,  # set global variables
)
chain = conf.chain()
chain.set_log_level('CRITICAL')


# chain.train_model.force()
signal_model = chain.trained_model.value


eval_dir = config.EVALUATION_DIR / f'predict_tracks'


signal_model.load(batch='last')
song_names = [f for suffix in SUFFIXES for f in eval_dir.glob(f"*{suffix}")]


nex_dir = config.EVALUATION_DIR / 'autoencoder_results'

json_file = nex_dir / 'json' / f'{sys.argv[1]}.json'

dc = chain.evaluate_model.force().value
with json_file.open('w') as f:
    json.dump(dc, f)


audio_dir = nex_dir / 'audio' / f'{sys.argv[1]}'
audio_dir.mkdir(exist_ok=True, parents=True)


for song_name in song_names:
    song_signal = read_audio(song_name)

    pred_channels = [signal_model(signal_channel) for signal_channel in song_signal]

    for j in range(pred_channels[0].data_arr.shape[0]):
        one_instrument_signal = MultiSeries(
            [pred_channel.take_channels([j]) for pred_channel in pred_channels]
        ).stack_series()
        out_filename = audio_dir / f'{song_name.stem}_{j}.mp3'
        one_instrument_signal.to_mp3(out_filename)
        print(f"{out_filename} created successfully.")
