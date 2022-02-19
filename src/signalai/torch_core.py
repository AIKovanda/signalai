import os
from typing import Generator, Tuple

from signalai import SeriesProcessor
from signalai.core import SignalModel
from torch import nn, optim
import torch
from tqdm import trange
import numpy as np
from signalai.config import DEVICE


class TorchSignalModel(SignalModel):

    def _train_on_generator(self, series_processor, training_params: dict, models: list = None):
        if models is None:
            models = range(self.model_count)
        else:
            models = filter(lambda z: z < self.model_count, models)

        for i, model_id in enumerate(models):
            if i > 0:
                self.model = self.model.weight_reset().to(DEVICE)
                self.new_optimizer()
                # torch.cuda.empty_cache()

            print(f"Training model {model_id}...")
            batch_indices_generator = trange(training_params["batches"])
            losses = []

            batch_id = 0

            for batch_id in batch_indices_generator:
                x, y = series_processor.next_batch(self.target_signal_length, training_params.get("batch_size", 1))
                new_loss = self.train_on_batch(x, y, training_params)
                losses.append(new_loss)
                mean_loss = np.mean(losses[-training_params["average_losses_to_print"]:])

                if 'stopping_rule' in training_params:
                    if mean_loss < training_params["stopping_rule"]:
                        break

                # progress bar update and printing
                batch_indices_generator.set_description(f"Loss: {mean_loss: .08f}")
                if batch_id % training_params["echo_step"] == 0 and batch_id != 0:
                    print()

                if batch_id % training_params["save_step"] == 0 and batch_id != 0:
                    self.save(model_id=model_id, batch_id=batch_id)

            self.save(model_id=model_id, batch_id=batch_id)

    def eval_on_generator(self, series_processor: SeriesProcessor, evaluation_params: dict
                          ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:

        for _ in range(evaluation_params["batches"]):
            x, y_true = series_processor.next_batch(self.target_signal_length, evaluation_params.get("batch_size", 1))
            y_hat = self.predict_batch(x)
            yield y_true, y_hat

    def save(self, model_id=0, batch_id=0):
        model_dir = self.save_dir / "saved_model" / f"{model_id}"
        model_dir.mkdir(exist_ok=True, parents=True)
        latest_file = (model_dir / f"epoch_last.pth").absolute()
        output_file = (model_dir / f"epoch_{batch_id}.pth").absolute()

        if not str(output_file).endswith(".pth"):
            output_file = str(output_file) + ".pth"

        torch.save(self.model.state_dict(), output_file)

        os.system(f'ln -f "{output_file}" "{latest_file}"')
        self.logger.log(f"Model saved at {output_file}.", priority=2)

    def train_on_batch(self, x: np.ndarray, y: np.ndarray, training_params=None):
        if training_params is None:
            training_params = {}
            
        self.model.train()
        x_batch = torch.from_numpy(x).type(torch.float32).to(DEVICE)
        if self.output_type == "label":
            y_batch = torch.from_numpy(y).type(torch.float32).unsqueeze(1).to(DEVICE)
        else:
            y_batch = torch.from_numpy(y).type(torch.float32).to(DEVICE)

        self.optimizer.zero_grad()
        y_hat = self.model(x_batch)
        loss_lambda = eval(training_params.get('loss_lambda', 'lambda _x, _y, crit: crit(_x, _y)'))
        loss = loss_lambda(y_hat, y_batch, self.criterion)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict_batch(self, x: np.ndarray):
        with torch.no_grad():
            self.model.eval()
            inputs = torch.from_numpy(x).type(torch.float32).to(DEVICE)

            y_hat = self.model(inputs).detach().cpu().numpy()
            self.model.train()
            return y_hat

    def load(self, path=None, batch=None, model_id=0):
        if path is None:
            if batch is not None:
                path = self.save_dir / "saved_model" / f"{model_id}" / f"epoch_{batch}.pth"
            else:
                path = self.save_dir / "saved_model" / f"{model_id}" / "epoch_last.pth"

        self.model.load_state_dict(torch.load(path))
        self.logger.log(f"Weights from {path} loaded successfully.", priority=4)

    def set_criterion(self, criterion_info: dict):
        criterion_name = criterion_info["name"]
        self.logger.log(f"Generating criterion '{criterion_name}'.")

        kwargs = criterion_info.get("kwargs", {})
        if criterion_name == "BCELoss":
            self.criterion = nn.BCELoss(**kwargs)
        elif criterion_name == "MSELoss":
            self.criterion = nn.MSELoss(**kwargs)
        elif criterion_name == "L1Loss":
            self.criterion = nn.L1Loss(**kwargs)
        else:
            raise NotImplementedError(f"Criterion '{criterion_name}' not implemented yet!!")

    def set_optimizer(self, optimizer_info: dict):
        self.optimizer_info = optimizer_info
        self.new_optimizer()

    def new_optimizer(self):
        optimizer_name = self.optimizer_info["name"]
        self.logger.log(f"Generating optimizer '{optimizer_name}'.")

        kwargs = self.optimizer_info.get("kwargs", {})
        if optimizer_name == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), **kwargs)
        else:
            raise NotImplementedError(f"Optimizer '{optimizer_name}' not implemented yet!!")
