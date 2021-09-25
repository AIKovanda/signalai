import pandas as pd
from signalai.signal_tools.generator2 import DataGenerator2D
from signalai.config import CONFIG_DIR, PIPELINE_SAVE_PATH
from taskorganizer.pipeline import Pipeline


config_path = CONFIG_DIR / "processing" / "pipeline.yaml"
params_config_path = CONFIG_DIR / "data_preparation" / "diamond_noise.yaml"
pip = Pipeline(
    config_path,
    config_dir=CONFIG_DIR,
    save_folder=PIPELINE_SAVE_PATH,
    parameters_config_yaml=params_config_path
)


signal_cfg = {}
df = pip.run("DatasetLoader")
transform_cfg = {}
gen = DataGenerator2D(signal_cfg, df, transform_cfg)
