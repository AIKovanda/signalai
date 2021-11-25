from pathlib import Path

from signalai import config
from signalai.evaluators.torch_evaluators import DecompositionEvaluator
from taskchain.task import Config


def run(config_path):
    conf = Config(
        config.TASKS_DIR,  # where should be data stored
        config_path,
        global_vars=config,  # set global variables
    )

    chain = conf.chain()
    chain.set_log_level('CRITICAL')

    signal_model = chain.trained_model.value

    evaluator = DecompositionEvaluator(Path(__file__).stem)
    evaluator.evaluate(signal_model=signal_model, output_dir="/home/martin/Music/decomposer")


if __name__ == '__main__':
    # chosen_config_path = config.CONFIGS_DIR / 'models' / 'decomposer' / 'decomposer1L.yaml'
    chosen_config_path = config.CONFIGS_DIR / 'models' / 'decomposer' / 'decomposer1L255_nores.yaml'
    # chosen_config_path = config.CONFIGS_DIR / 'models' / 'decomposer' / 'decomposer1L255_nores_mish.yaml'
    # chosen_config_path = config.CONFIGS_DIR / 'models' / 'decomposer' / 'decomposer1L255_nores_tanh.yaml'
    # chosen_config_path = config.CONFIGS_DIR / 'models' / 'decomposer' / 'decomposer3L.yaml'
    # chosen_config_path = config.CONFIGS_DIR / 'models' / 'decomposer' / 'decomposer3L255_nores.yaml'
    # chosen_config_path = config.CONFIGS_DIR / 'models' / 'decomposer' / 'decomposer3L_nores.yaml'
    run(chosen_config_path)
