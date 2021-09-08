from signalai.config import CONFIG_DIR, PIPELINE_SAVE_PATH
from taskorganizer.pipeline import Pipeline
print(CONFIG_DIR)
config_path = CONFIG_DIR / "pipelines" / "pipeline1.yaml"
pip = Pipeline(config_path, config_dir=CONFIG_DIR, save_folder=PIPELINE_SAVE_PATH)
print(pip.run("a"))
print(pip.run("b"))
# print(pip.run("b", force_task=True, force_next_tasks=True))

print(pip.show_pipeline())
pip.delete_irrelevant_data()
