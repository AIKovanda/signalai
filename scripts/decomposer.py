import argparse
import pathlib
from itertools import count
from typing import Optional, Union

from taskchain.task import Config

from signalai import config, read_audio
from signalai.core import SignalModel
from signalai.timeseries import MultiSeries

SUFFIXES = [".mp3", ".aac", ".wav"]


def predict_from_batch(
        signal_model: SignalModel, batch: Union[str, int], model_config: str,
        eval_dir: pathlib.PosixPath, test: bool):

    signal_model.load(batch=batch)
    print(f"Predicting with model from batch '{batch}'.")
    song_names = [f for suffix in SUFFIXES for f in eval_dir.glob(f"*{suffix}")]

    test_str = "_test" if test else ""
    output_dir = config.EVALUATION_DIR / ('.'.join(model_config.split(".")[:-1]) + f"_{batch}")
    output_dir = output_dir.parent.parent / f'{output_dir.parent.name}_{signal_model.processing_fs}{test_str}' / output_dir.name
    output_dir.mkdir(exist_ok=True, parents=True)
    for song_name in song_names:
        song_signal = read_audio(song_name)
        pred_channels = [signal_model(signal_channel)
                         for signal_channel in song_signal]  # todo: directly

        for j in range(pred_channels[0].data_arr.shape[0]):
            one_instrument_signal = MultiSeries([pred_channel.take_channels([j]) for pred_channel in pred_channels]
                                                ).stack_series()
            out_filename = output_dir / f'{song_name.stem}_{j}.mp3'
            one_instrument_signal.to_mp3(out_filename)
            print(f"{out_filename} created successfully.")


def run(model_config: str, eval_dir: Optional[str], count_step=6000, test=False, processing_fs: int = None):

    config_path = config.CONFIGS_DIR / 'models' / model_config
    context = {'test': test}
    if processing_fs is not None:
        context.update(processing_fs=processing_fs)

    conf = Config(
        config.TASKS_DIR,  # where should be data stored
        config_path,
        global_vars=config,  # set global variables
        context=context,
    )

    chain = conf.chain()
    chain.set_log_level('CRITICAL')

    # chain.train_model.force()
    signal_model = chain.trained_model.value

    if eval_dir is None:
        eval_dir = str(config.EVALUATION_DIR / f'predict')
    eval_dir = pathlib.PosixPath(eval_dir)

    predict_from_batch(signal_model, 'last', model_config, eval_dir, test)
    for batch in count(count_step, count_step):
        try:
            predict_from_batch(signal_model, batch, model_config, eval_dir, test)
        except FileNotFoundError:
            break


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model_config', '-m', type=str)
    p.add_argument('--eval_dir', '-e', default=None, type=str)
    p.add_argument('--count_step', '-c', default=6000, type=int)
    p.add_argument('--test', '-t', default=False, action='store_true')
    p.add_argument('--processing_fs', '-p', default=None, type=int)
    args = p.parse_args()
    run(
        model_config=args.model_config,
        eval_dir=args.eval_dir,
        count_step=args.count_step,
        test=args.test,
        processing_fs=args.processing_fs,
    )
