from typing import Union
import numpy as np
from signalai.signal import SignalDatasetsKeeper, SignalProcessor, Logger
from taskchain import Parameter, InMemoryData, Task


class TrainSignalGenerator(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = []
        parameters = [
            Parameter("datasets"),
            Parameter("processors"),
            Parameter("load_to_ram", default=False, ignore_persistence=True),
            Parameter("split", default=[.8, .1, .1]),
        ]

    def run(self, datasets, processors, split: Union[tuple, list], load_to_ram: bool) -> SignalProcessor:
        assert np.abs(np.sum(split) - 1) < 1e-8, "Split must sum to 1."
        split_name = "train"
        split_range = (0., split[0])  # todo: not needed for some datasets
        logger = Logger(name=f"{split_name.capitalize()}SignalGenerator", verbose=0)
        keeper = SignalDatasetsKeeper(datasets_config=datasets, split_range=split_range, logger=logger)
        if load_to_ram:
            keeper.load_to_ram()
        return SignalProcessor(processor_config=processors[split_name], keeper=keeper, logger=logger)
