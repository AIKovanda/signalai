import os
from signalai.core import SignalModel
from torch import nn, optim
import torch
from tqdm import trange
import numpy as np
from signalai.config import DEVICE


class TorchSignalModel(SignalModel):

    def _train_on_generator(self):
        batch_indices_generator = trange(self.training_params["batches"])
        losses = []
        output_dir = self.save_dir / "saved_model"
        output_dir.mkdir(parents=True, exist_ok=True)

        batch_id = 0

        for batch_id in batch_indices_generator:
            x, y = self.signal_generator.next_batch(self.training_params.get("batch_size", 1))
            new_loss = self.train_on_batch(x, y)
            losses.append(new_loss)
            mean_loss = np.mean(losses[-self.training_params["average_losses_to_print"]:])

            # stopping rule
            if mean_loss < 1e-5:
                break

            # progress bar update and printing
            batch_indices_generator.set_description(f"Loss: {mean_loss: .08f}")
            if batch_id % self.training_params["echo_step"] == 0 and batch_id % self.training_params["save_step"] != 0:
                print()

            if batch_id % self.training_params["save_step"] == 0 and batch_id != 0:
                self.save(output_dir, batch_id=batch_id)
                self.evaluate(output_dir=output_dir, batch_id=batch_id)

        self.save(output_dir=output_dir, batch_id=batch_id)
        self.evaluate(output_dir=output_dir, batch_id=batch_id)

    def save(self, output_dir, batch_id):
        latest_file = (output_dir / f"last.pth").absolute()
        output_file = (output_dir / f"epoch_{batch_id}.pth").absolute()

        if not str(output_file).endswith(".pth"):
            output_file = str(output_file) + ".pth"

        torch.save(self.model.state_dict(), output_file)

        os.system(f'ln -f "{output_file}" "{latest_file}"')
        self.logger.log(f"Model saved at {output_file}.", priority=2)

    def evaluate(self, output_dir, batch_id):
        if self.evaluator is not None:
            with torch.no_grad():
                self.model.eval()
                self.logger.log(f"Evaluation run at batch {batch_id}.")
                self.evaluator.evaluate(output_dir=output_dir)
                self.model.train()

    def train_on_batch(self, x: np.ndarray, y: np.ndarray):
        self.model.train()
        x_batch = torch.from_numpy(x).type(torch.float32).to(DEVICE)
        if self.training_params.get("output_type", "label") == "label":
            y_batch = torch.from_numpy(y).type(torch.float32).unsqueeze(1).to(DEVICE)
        else:
            y_batch = torch.from_numpy(y).type(torch.float32).to(DEVICE)

        self.optimizer.zero_grad()
        y_hat = self.model(x_batch)
        loss = self.criterion(y_hat, y_batch)
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

    def load(self, path=None, epoch=None):
        if epoch is not None:
            path = self.save_dir / "saved_model" / f"epoch_{epoch}.pth"
        if path is None:
            path = self.save_dir / "saved_model" / "last.pth"

        self.model.load_state_dict(torch.load(path))

    def get_criterion(self):
        criterion_info = self.training_params["criterion"]
        criterion_name = criterion_info["name"]
        self.logger.log(f"Generating criterion '{criterion_name}'.")

        kwargs = criterion_info.get("kwargs", {})
        if criterion_name == "BCELoss":
            return nn.BCELoss(**kwargs)
        elif criterion_name == "MSELoss":
            return nn.MSELoss(**kwargs)
        elif criterion_name == "L1Loss":
            return nn.L1Loss(**kwargs)
        else:
            raise NotImplementedError(f"Criterion '{criterion_name}' not implemented yet!!")

    def get_optimizer(self):
        optimizer_info = self.training_params["optimizer"]
        optimizer_name = optimizer_info["name"]
        self.logger.log(f"Generating optimizer '{optimizer_name}'.")

        kwargs = optimizer_info.get("kwargs", {})
        if optimizer_name == "Adam":
            return optim.Adam(self.model.parameters(), **kwargs)
        else:
            raise NotImplementedError(f"Optimizer '{optimizer_name}' not implemented yet!!")
