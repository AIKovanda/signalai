import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from signalai.core import TimeSeriesModel
from signalai.evaluators import TorchEvaluator
from signalai.torch_dataset import TorchDataset


class TorchTimeSeriesModel(TimeSeriesModel):

    def init_model(self, model_definition: dict | nn.Module) -> None:
        model = model_definition['model']
        if isinstance(model, nn.Module):
            self.model = model
            if (pretrained_weights := model_definition.get('pretrained_weights')) is not None:
                self.load(weights_path=pretrained_weights)
        else:
            raise TypeError(f'Model type of {type(model)} is not supported!')

        if self.device.startswith('cuda'):
            self.model = self.model.to(self.device)

    def train_on_batch(self, x: tuple[torch.Tensor], y: tuple[torch.Tensor]) -> tuple[float, tuple[torch.Tensor]]:
        self.model.train()
        self.optimizer.zero_grad()
        y_hat = self.model(*x)
        if not isinstance(y_hat, tuple):
            y_hat = (y_hat,)
        loss = self._apply_loss(y, y_hat)
        loss.backward()
        self.optimizer.step()
        return loss.item(), y_hat

    def _apply_loss(self, y_batch: tuple, y_hat: tuple):
        return eval(self.training_params.get('loss_eval', 'self.criterion(y_hat[0], y_batch[0])'))

    def train_on_generator(self, time_series_gen: TorchDataset, valid_time_series_gen: TorchDataset = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        batch_history = []
        eval_history = []
        losses = []
        batch_size = self.training_params.get('dataloader_kwargs', {}).get("batch_size", 1)
        save_evaluation_dir = self.training_echo.get('save_evaluation_dir')
        if self.taken_length is not None:
            time_series_gen.set_taken_length(self.taken_length)
        if self.training_params.get("max_batches") is not None:
            time_series_gen.config['max_length'] = self.training_params.get("max_batches")*batch_size

        train_loader = DataLoader(time_series_gen, **self.training_params.get('dataloader_kwargs', {}))
        current_batch = 0
        broken = False
        for model_id in range(self.model_count):
            if model_id > 0:
                self.model = self.model.weight_reset().to(self.device)

            if self.model_count > 1:
                print(f"Training model #{model_id}...")
            if valid_time_series_gen is not None and len(self.training_echo.get('evaluators', [])) != 0 and self.training_echo.get('init_eval', False):
                eval_history.append(self._eval_in_training(valid_time_series_gen))

            for epoch_id in range(self.training_params.get('epochs', 1)):
                if epoch_id in (set_ := self.training_params.get('set')):
                    if 'criterion' in set_[epoch_id]:
                        self.set_criterion(set_[epoch_id].get('criterion'))
                    if 'optimizer' in set_[epoch_id]:
                        self.set_optimizer(set_[epoch_id].get('optimizer'))

                tqdm_train_loader = tqdm(train_loader, ncols=150)
                for batch_id, (x, y) in enumerate(tqdm_train_loader):
                    current_batch += 1
                    x = tuple(val.type(torch.float32).to(self.device) for val in x)
                    y = tuple(val.type(torch.float32).to(self.device) for val in y)

                    new_loss, y_hat = self.train_on_batch(x, y)

                    x_detached = tuple(val.detach() for val in x)
                    y_hat_detached = tuple(val.detach() for val in y_hat)

                    batch_history.append({
                        key: metric(x_detached, y_hat_detached, batch_size=batch_size)
                        for key, metric in self.training_echo.get('metrics', {}).items()})
                    losses.append(new_loss)

                    if (early_stopping_at := self.training_params.get("early_stopping_at")) is not None:
                        mean_loss = np.mean(losses[-self.training_echo["early_stopping_average_losses"]:])
                        if mean_loss < early_stopping_at:
                            broken = True
                            break

                    if ((esr := self.training_params.get("early_stopping_regression")) is not None and
                            current_batch >= max(esr, self.training_params.get("early_stopping_min", esr)) and
                            current_batch % int(esr / 3) == 0):

                        model = LinearRegression().fit(np.arange(esr).reshape(-1, 1),
                                                       np.array(losses[-esr:]).reshape(-1, 1))
                        if model.coef_[0][0] >= 0:
                            broken = True
                            break

                    # progress bar update and printing
                    tqdm_train_loader.set_description(
                        f'E{epoch_id}: ' + " | ".join([f"{key}: {val}" for key, val in {'loss': losses[-1], **batch_history[-1]}.items()])
                    )
                    if "echo_step" in self.training_echo and (current_batch + 1) % self.training_echo["echo_step"] == 0:
                        print()

                    if ('save_batch_step' in self.training_params
                            and (current_batch + 1) % self.training_params["save_batch_step"] == 0):
                        self.save(model_id=model_id, batch_id=current_batch)

                if ('save_epoch_step' in self.training_params
                        and (epoch_id + 1) % self.training_params["save_epoch_step"] == 0):
                    self.save(model_id=model_id, batch_id=current_batch)
                if broken:
                    break
                if valid_time_series_gen is not None and len(self.training_echo.get('evaluators', [])) != 0:
                    if save_evaluation_dir is not None:
                        eval_history.append(self._eval_in_training(
                            valid_time_series_gen,
                            save_evaluation_dir=Path(save_evaluation_dir) / f'epoch{epoch_id}'))
                    else:
                        eval_history.append(self._eval_in_training(valid_time_series_gen))

            self.save(model_id=model_id, batch_id=current_batch)

        batch_history = pd.DataFrame(batch_history)
        batch_history['loss'] = losses
        return batch_history, pd.DataFrame(eval_history)

    def _eval_in_training(self, valid_time_series_gen, save_evaluation_dir: Path = None):
        eval_dict = self.eval_on_generator(
            valid_time_series_gen,
            evaluators=self.training_echo.get('evaluators'),
            evaluation_params=self.training_echo.get("evaluation_params", {}),
            use_tqdm=True,
            save_evaluation_dir=save_evaluation_dir,
        )
        for key, val in eval_dict.items():
            print(f'{key:<12}: {val}')
        return eval_dict

    def predict_batch(self, x: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
        with torch.no_grad():
            self.model.eval()
            y_hat = self.model(*x)
        if not isinstance(y_hat, tuple):
            y_hat = (y_hat,)
        return y_hat

    def predict_numpy(self, *arr: tuple[np.ndarray]):
        x = tuple(torch.from_numpy(np.stack(arr_)).type(torch.float32).to(self.device) for arr_ in zip(*arr))
        return self.predict_batch(x)

    def eval_on_generator(self, time_series_gen: TorchDataset, evaluators: list[TorchEvaluator], evaluation_params: dict,
                          use_tqdm=False, save_evaluation_dir: Path = None) -> dict:
        np.random.seed(13)
        eval_dict = {}
        if save_evaluation_dir is not None:
            save_evaluation_dir = Path(save_evaluation_dir)
            save_evaluation_dir.mkdir(parents=True, exist_ok=True)

        if self.taken_length is not None:
            time_series_gen.set_taken_length(self.taken_length)
        if evaluation_params.get("max_batches") is not None:
            valid_batch_size = evaluation_params.get('dataloader_kwargs', {}).get("batch_size", 1)
            time_series_gen.config['max_length'] = evaluation_params.get("max_batches") * valid_batch_size

        for evaluator in evaluators:
            evaluator.reset_items()

        data_loader = DataLoader(time_series_gen, **evaluation_params.get('dataloader_kwargs', {}))
        tqdm_train_loader = tqdm(data_loader, ncols=150, desc='Evaluating model...') if use_tqdm else data_loader
        for i, (x, y) in enumerate(tqdm_train_loader):
            x = tuple(val.type(torch.float32).to(self.device) for val in x)
            y_true = tuple(val.type(torch.float32).to(self.device) for val in y)
            y_pred = self.predict_batch(x)
            if self.post_transform is not None:
                y_true = tuple(self.post_transform.process_torch(i) for i in y_true)
                y_pred = tuple(self.post_transform.process_torch(i) for i in y_pred)
            if save_evaluation_dir is not None:
                Path(save_evaluation_dir).mkdir(parents=True, exist_ok=True)
                np.save(str(save_evaluation_dir / f'{i}-x.npy'), x[0].detach().cpu().numpy())
                np.save(str(save_evaluation_dir / f'{i}-y_true.npy'), y_true[0].detach().cpu().numpy())
                np.save(str(save_evaluation_dir / f'{i}-y_pred.npy'), y_pred[0].detach().cpu().numpy())

            for evaluator in evaluators:
                evaluator.process_batch(y_true, y_pred)

        for evaluator in evaluators:
            eval_dict.update(evaluator.metric_value)

        return eval_dict

    def save(self, model_id=0, batch_id=0):
        model_dir = self.save_dir / "saved_model" / f"{model_id}"
        model_dir.mkdir(exist_ok=True, parents=True)
        latest_file = (model_dir / f"epoch_last.pth").absolute()
        output_file = (model_dir / f"epoch_{batch_id}.pth").absolute()

        if not str(output_file).endswith(".pth"):
            output_file = str(output_file) + ".pth"

        torch.save(self.model.state_dict(), output_file)

        os.system(f'ln -f "{output_file}" "{latest_file}"')

    def load(self, weights_path=None, batch=None, model_id=0):
        if weights_path is None:
            if batch is not None:
                weights_path = self.save_dir / "saved_model" / f"{model_id}" / f"epoch_{batch}.pth"
            else:
                weights_path = self.save_dir / "saved_model" / f"{model_id}" / "epoch_last.pth"

        self.model.load_state_dict(torch.load(weights_path))

    def set_criterion(self, criterion_info: dict):
        criterion_name = criterion_info["name"]

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
        optimizer_name = optimizer_info["name"]
        kwargs = optimizer_info.get("kwargs", {})
        if optimizer_name == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), **kwargs)
        elif optimizer_name.upper() == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), **kwargs)
        elif optimizer_name.lower() == "rmsprop":
            self.optimizer = optim.RMSprop(self.model.parameters(), **kwargs)
        else:
            raise NotImplementedError(f"Optimizer '{optimizer_name}' not implemented yet!!")
        print('Optimized initialized with parameters ', kwargs)
