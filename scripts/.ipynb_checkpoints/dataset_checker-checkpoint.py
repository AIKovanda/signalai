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

print(pip.run("DatasetLoader", force_task=True, force_next_tasks=True))
# print(pip.run("b", force_task=True, force_next_tasks=True))

print(pip.show_pipeline())
