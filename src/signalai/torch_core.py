import os

from signalai.core import SignalModel
from signalai.models.inceptiontime import InceptionBlock
from torch import nn, optim

import torch
from torch.nn import ModuleList
from tqdm import trange
import numpy as np

from signalai.config import DEVICE


class TorchModel(SignalModel):

    def _train_on_generator(self, verbose=1):
        print("Training params:")
        for key, val in self.training_params.items():
            print(f"{key:>24} [{str(type(val)):<14}]: {val}")

        batches_id = trange(self.training_params["batches"])
        losses = []
        output_dir = self.save_dir / "saved_model"
        output_dir.mkdir(parents=True, exist_ok=True)
        batch_id = 0

        for batch_id in batches_id:
            x, y = next(self.train_gen)
            new_loss = self.train_on_batch(x, y)
            losses.append(new_loss)

            mean_loss = np.mean(losses[-self.training_params["average_losses_to_print"]:])
            if mean_loss < 1e-5:
                break

            batches_id.set_description(f"Loss: {mean_loss: .08f}")
            if batch_id % self.training_params["echo_step"] == 0 and batch_id % self.training_params["save_step"] != 0:
                print()

            if batch_id % self.training_params["save_step"] == 0 and batch_id != 0:
                output_stem = str(self.training_params["output_name"].format(batch_id=batch_id))
                output_file = (output_dir / f"{output_stem}.pth").absolute()
                last_file = (output_dir / f"last.pth").absolute()
                self.save(output_file, verbose=verbose)

                os.system(f'ln -f "{output_file}" "{last_file}"')
                if self.evaluator:
                    output_svg = output_dir / f"{output_stem}.svg"
                    self.evaluate(output_svg)

        if self.evaluator:
            self.evaluator.evaluate(self.model, "oo")

        output_file = (output_dir / f"{batch_id}.pth").absolute()
        self.save(output_file, verbose=verbose)

    def evaluate(self, output_svg, verbose=1):
        self.evaluator.evaluate(self.model, output_svg)
        if verbose > 0:
            print(f"Signal visualization is at {output_svg}.")

    def train_on_batch(self, x, y):
        if isinstance(x, np.ndarray) or isinstance(x, list):
            inputs = torch.from_numpy(np.array(x)).to(DEVICE)
            labels = torch.from_numpy(np.array(y)).type(torch.float32).unsqueeze(1).to(DEVICE)
        elif isinstance(x, torch.Tensor):
            inputs = x.to(DEVICE)
            labels = y.type(torch.float32).unsqueeze(1).to(DEVICE)
        else:
            raise TypeError(f"x of type {type(x)} is not supported yet!!")

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_on_batch(self, x):
        self.model.eval()
        if isinstance(x, np.ndarray):
            inputs = torch.from_numpy(np.array(x)).to(DEVICE)
        else:
            inputs = x.to(DEVICE)

        y_hat = self.model(inputs).cpu()
        self.model.train()
        return y_hat

    def save(self, output_file, verbose=1):
        if not str(output_file).endswith(".pth"):
            raise ValueError(f"Enter a valid path. Filename must end with '.pth'.")
        torch.save(self.model.state_dict(), output_file)
        if verbose > 0:
            print(f"Model saved at {output_file}.")

    def load(self, path=None):
        if path is None:
            path = self.save_dir / "saved_model" / "last.pth"

        self.model.load_state_dict(torch.load(path))

    def get_criterion(self):
        criterion_info = self.training_params["criterion"]
        if criterion_info["name"] == "BCELoss":
            return nn.BCELoss(**criterion_info.get("kwargs", {}))
        else:
            raise NotImplementedError(f"Criterion '{criterion_info['name']}' not implemented yet!!")

    def get_optimizer(self):
        optimizer_info = self.training_params["optimizer"]
        if optimizer_info["name"] == "Adam":
            return optim.Adam(self.model.parameters(), **optimizer_info.get("kwargs", {}))
        else:
            raise NotImplementedError(f"Optimizer '{optimizer_info['name']}' not implemented yet!!")


class InceptionTime(nn.Module):

    def __init__(self, build_config, in_channels=1, linear=None):
        """
        InceptionTime network
        :param build_config: list of dicts
        :param in_channels: integer
        :param linear: None or integer as a number of output classes
        """
        super().__init__()
        n_filters = [in_channels] + [node.get("n_filters", 32) for node in build_config]
        kernel_sizes = [node.get("kernel_sizes", [11, 21, 41]) for node in build_config]
        bottleneck_channels = [node.get("bottleneck_channels", 32) for node in build_config]
        num_of_nodes = len(kernel_sizes)
        self.linear = linear
        self.inception_blocks = ModuleList([InceptionBlock(
            in_channels=n_filters[i],
            n_filters=n_filters[i+1],
            kernel_sizes=kernel_sizes[i],
            bottleneck_channels=bottleneck_channels[i],
            use_residual=True,
            activation=nn.SELU()
        ) for i in range(num_of_nodes)])
        self.in_features = (1 + len(kernel_sizes[-1])) * n_filters[-1] * 1
        if self.linear is not None:
            self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=1)
            assert self.linear > 0, f"Linear must be either None or a positive integer, not {linear}."
            self.linear1 = nn.Linear(
                in_features=self.in_features,
                out_features=linear)
            if self.linear in [1, 2]:
                self.out_activation = nn.Sigmoid()
            else:
                self.out_activation = nn.Softmax()

    def forward(self, x):
        for block in self.inception_blocks:
            x = block(x)
        if self.linear:
            x = self.adaptive_pool(x)
            x = x.view(-1, self.in_features)
            x = self.linear1(x)
            x = self.out_activation(x)

        return x
