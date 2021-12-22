from signalai import config
from taskchain.task import Config

# config_name = ('test', 'decomposer.yaml')
config_name = ('test', 'se_decomposer.yaml')

config_path = config.CONFIGS_DIR / 'models' / 'decomposer' / config_name[0] / config_name[1]
conf = Config(
    config.TASKS_DIR,  # where should be data stored
    config_path,
    global_vars=config,  # set global variables
)

chain = conf.chain()
chain.set_log_level('CRITICAL')

signal_model = chain.train_model.force().value

print("Script done!")
