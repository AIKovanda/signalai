from pathlib import Path

from signalai.config import DEVICE
from signalai.torch_core import TorchModel
from taskorganizer.pipeline import PipelineTask


def get_torch_model(model_info, model_params):
    model = get_model(model_info, model_params)
    return model.to(DEVICE)


def get_model(model_info, model_params):
    model_class = model_info["class"]
    model_from = ".".join(model_class.split(".")[:-1])
    model_class_name = model_class.split(".")[-1]
    exec(f"from {model_from} import {model_class_name}")

    model = eval(f"{model_class_name}(**model_params)")
    return model


def create_model(model_info, model_params, save_dir, training_params=None, gen_generator=None):
    if model_info["type"] == "torch":
        model = get_torch_model(model_info, model_params)
        signal_model = TorchModel(model=model,
                                  gen_generator=gen_generator,
                                  model_type='torch',
                                  training_params=training_params,
                                  save_dir=save_dir)
    else:
        raise NotImplementedError(f"Model of type {model_info['type']} is not implemented yet!")
    return signal_model


class TrainModel(PipelineTask):

    def run(self, gen_generator, model_info, model_params, training_params):
        signal_model = create_model(model_info, model_params, self.save_dir, training_params, gen_generator)
        signal_model.train_on_generator()
        return str(self.save_dir)


class TrainedModel(PipelineTask):

    def run(self, train_model, model_info, model_params):
        signal_model = create_model(model_info, model_params, train_model)
        model_path = Path(train_model) / "saved_model" / "last.pth"
        signal_model.load(model_path)
        return signal_model
