import os
from typing import Generator, Tuple

import numpy as np
import torch
from signalai.timeseries import TorchDataset
from sklearn.linear_model import LinearRegression
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from signalai.config import DEVICE
from signalai.core import SignalModel
from signalai.tools.utils import apply_transforms


class TorchSignalModel(SignalModel):

    def _train_on_generator(self, series_processor: TorchDataset, training_params: dict, models: list = None,
                            early_stopping_at=None, early_stopping_min=0, early_stopping_regression=None):
        if models is None:
            models = range(self.model_count)
        else:
            models = filter(lambda z: z < self.model_count, models)

        batch_size = training_params.get("batch_size", 1)

        series_processor.set_target_length(self.target_signal_length)
        series_processor.set_len(training_params["batches"] * batch_size)

        train_loader = DataLoader(series_processor, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())

        for i, model_id in enumerate(models):
            if i > 0:
                self.model = self.model.weight_reset().to(DEVICE)
                self.new_optimizer()

            print(f"Training model {model_id}...")
            batch_indices_generator = tqdm(train_loader, ncols=150)
            losses = []
            trues = []

            batch_id = 0

            for batch_id, (x, y) in enumerate(batch_indices_generator):
                # print(x.shape, y.shape)
                new_loss, y_hat = self.train_on_batch(x, y, training_params)
                x = x.detach()
                y_hat = y_hat.detach()
                losses.append(new_loss)
                if isinstance(y_hat, tuple):
                    trues.append(int(torch.sum(y_hat[0] > .5) / batch_size))
                else:
                    trues.append(int(torch.sum(y_hat > .5) / batch_size))

                mean_loss = np.mean(losses[-training_params["average_losses_to_print"]:])
                mean_true = np.mean(trues[-training_params["average_losses_to_print"]:])

                if early_stopping_at is not None:
                    if mean_loss < early_stopping_at:
                        break

                if (early_stopping_regression is not None and
                        batch_id >= max(early_stopping_regression, early_stopping_min) and
                        batch_id % int(early_stopping_regression / 3) == 0):
                    model = LinearRegression().fit(np.arange(early_stopping_regression).reshape(-1, 1),
                                                   np.array(losses[-early_stopping_regression:]).reshape(-1, 1))
                    if model.coef_[0][0] >= 0:
                        break

                # progress bar update and printing
                batch_indices_generator.set_description(
                    f"Loss: {mean_loss: .06f}, total_true: {int(mean_true)} | tr_mean: {torch.mean(x):.6f} | tr_std: {torch.std(x):.4f} | mean: {torch.mean(y_hat):.6f} | std: {torch.std(y_hat):.4f}"
                )
                if batch_id % training_params["echo_step"] == 0 and batch_id != 0:
                    print()

                if batch_id % training_params["save_step"] == 0 and batch_id != 0:
                    self.save(model_id=model_id, batch_id=batch_id)

            self.save(model_id=model_id, batch_id=batch_id)

    def eval_on_generator(self, series_processor: TorchDataset, evaluation_params: dict, post_transform: bool,
                          ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        np.random.seed(13)

        batch_size = evaluation_params.get("batch_size", 1)
        series_processor.set_target_length(self.target_signal_length)
        series_processor.set_len(evaluation_params["batches"] * batch_size)

        eval_loader = DataLoader(series_processor, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())

        for x, y_true in eval_loader:
            y_hat = self.predict_batch(x)
            y_true = y_true.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()

            for one_y_true, one_y_hat in zip(y_true, y_hat):
                if post_transform and len(self.post_transform.get('predict', [])) > 0:
                    one_y_hat = apply_transforms(one_y_hat, self.post_transform.get('predict')).data_arr
                    one_y_true = apply_transforms(one_y_true, self.post_transform.get('predict')).data_arr
                yield one_y_true, one_y_hat

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

    def train_on_batch(self, x: torch.Tensor, y: torch.Tensor, training_params=None):
        if training_params is None:
            training_params = {}

        self.model.train()
        x_batch = x.type(torch.float32).to(DEVICE)
        if self.output_type == "label":
            y_batch = y.type(torch.float32).unsqueeze(1).to(DEVICE)
        else:
            y_batch = y.type(torch.float32).to(DEVICE)

        assert x_batch is not y_batch
        self.optimizer.zero_grad()
        y_hat = self.model(x_batch)
        loss_lambda = eval(training_params.get('loss_lambda', 'lambda _x, _y, crit: crit(_x, _y)'))
        loss = loss_lambda(y_hat, y_batch, self.criterion)
        loss.backward()
        self.optimizer.step()
        return loss.item(), y_hat

    def predict_batch(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.model.eval()
            inputs = x.type(torch.float32).to(DEVICE)

            y_hat = self.model(inputs)
            self.model.train()
            return y_hat

    def predict_numpy_batch(self, x: np.ndarray) -> np.ndarray:
        return self.predict_batch(torch.from_numpy(x)).detach().cpu().numpy()

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
