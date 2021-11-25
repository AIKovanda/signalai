import pandas as pd
from signalai.config import CONFIG_DIR, PIPELINE_SAVE_PATH
from taskorganizer.pipeline import Pipeline
from signalai.tools.utils import audio_file2numpy
from signalai.signal.signal import Signal
import numpy as np


config_path = CONFIG_DIR / "pipeline" / "processing.yaml"

data_config_path = CONFIG_DIR / "data_preparation" / "decomposer.yaml"
model_config_path = CONFIG_DIR / "models" / "decomposer" / "decomposer1L255_nores_mish.yaml"

pipeline = Pipeline(
    config_path,
    config_dir=CONFIG_DIR,
    save_folder=PIPELINE_SAVE_PATH,
    parameter_yamls=[data_config_path, model_config_path]
)



signal_model = pipeline.run("trained_model")


for song_name in ['Dua Lipa - Levitating Featuring DaBaby (Official Music Video).mp3', 'MB14 - Road to Zion.mp3', 'A_Classic_Education_-_NightOwl_-_0.aac']:

    mp3 = audio_file2numpy(f'/home/martin/Music/{song_name}')
    print(mp3.shape)
    signal_parts = []
    for i in range(mp3.shape[-1] // 100000 + 1):
        signal_part = mp3[:, i*100000:(i+1)*100000]
        new_signal = signal_model.eval_on_batch(np.expand_dims(signal_part, 0))[0]
        signal_parts.append(new_signal)

    predicted_all = np.hstack(signal_parts)
    print(predicted_all.shape)
    for j in range(4):
        new_s = Signal(predicted_all[j*2:(j+1)*2, :])
        new_s.to_mp3(f'/home/martin/Music/decomposer/1L255_mish_{song_name[:-4]}_{j}.mp3')
