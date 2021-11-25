from pathlib import Path

from signalai import config
from signalai.evaluators.torch_evaluators import TahovkaEvaluator
from taskchain.task import Config


def run(config_path):
    conf = Config(
        config.TASKS_DIR,  # where should be data stored
        config_path,
        global_vars=config,  # set global variables
    )

    chain = conf.chain()
    chain.set_log_level('CRITICAL')

    gen_gen = chain.trained_model.value
    signal_model = chain.trained_model.value

    evaluator = TahovkaEvaluator(gen_gen=gen_gen)
    evaluator.evaluate(signal_model=signal_model, output_img=f"/home/martin/Documents/{Path(__file__).stem}.svg")


if __name__ == '__main__':
    chosen_config_path = config.CONFIGS_DIR / 'models' / 'tahovka' / 'basic_inceptiontime.yaml'
    run(chosen_config_path)
