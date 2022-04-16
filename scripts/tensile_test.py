import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from taskchain.task import Config

from signalai import config, read_bin
from signalai.config import DATA_DIR
from signalai.tools.filters import gauss_convolve


def run(config_path):
    print("Loading config...")
    conf = Config(
        config.TASKS_DIR,  # where Taskchain data should be stored
        config_path,
        global_vars=config,  # set global variables
    )
    chain = conf.chain()
    chain.set_log_level('CRITICAL')
    print("Config loaded successfully.")

    print("Loading model...")
    # chain.train_model.force()
    signal_model = chain.trained_model.value
    print("Model loaded successfully.")

    bin_files_to_predict = [
        # DATA_DIR / 'tensile_tests' / '201103' / '201103_AE_ch0.bin',
        DATA_DIR / 'tensile_tests' / '200915' / '200915_AE.bin',
    ]
    for bin_file_to_predict in bin_files_to_predict:
        print("Loading signal...")
        signal = read_bin(bin_file_to_predict)  # .crop([0, 50000000])
        print("Signal loaded successfully.")
        plt.figure(figsize=(16, 9))

        for model_id in range(5):  # how many models we have
            print(f"Evaluating signal by model {model_id}...")
            signal_model.load(model_id=model_id)
            np_result = signal_model(signal, target_length=32768, residual_end=False).data_arr

            if len(np_result.shape) == 1:
                np_result = np.expand_dims(np_result, 0)

            print("Signal evaluated successfully.")

            print("Saving results...")
            y = gauss_convolve(np_result, 30, 0.8)[0]
            sns.lineplot(y=y, x=range(len(y)), label=model_id)

        plt.savefig(config.EVALUATION_DIR / (bin_file_to_predict.stem + ".png"), dpi=400)
        plt.close()


if __name__ == '__main__':
    chosen_config_path = config.CONFIGS_DIR / 'models' / 'tensile_test' / 'base' / 'inceptiontime.yaml'
    run(chosen_config_path)
