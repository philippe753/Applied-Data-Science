import os.path
import time

import file_operations as file_op

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.nn.functional import one_hot
from torch.nn import Sigmoid
from torch.utils.data import TensorDataset, DataLoader

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

from model_library import HyperParams, History, AdditionalInformation, Checkpoint, ModelLockedError
from prettytable import PrettyTable


CONTAINER_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + "/"


def method_print_decorator(func: callable, symbol='-', number_of_symbol_per_line: int = 40) -> callable:
    def wrapper(*args, **kwargs):
        print(symbol * number_of_symbol_per_line)
        func(*args, **kwargs)
        print(symbol * number_of_symbol_per_line)

    return wrapper


def load_samples(data: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    return data.sample(n_samples)


@variational_estimator
class BayesianNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_width: int=64, **kwargs):
        super().__init__()
        self.bayes_linear1 = BayesianLinear(input_dim, hidden_width)
        self.bayes_linear2 = BayesianLinear(hidden_width, 2)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x_ = self.bayes_linear1(x)
        return self.sigmoid(self.bayes_linear2(x_))


class Model:
    def __init__(self, network: nn.Module, optimizer: optim.Optimizer, file_path: str, *,
                 hyper_params: HyperParams=HyperParams(),
                 input_shape: int=30):
        # Model info
        self.net = network
        self.optimizer = optimizer
        self.hyper_params = hyper_params

        # Creating save directory
        self._file_dir_path = file_op.file_path_without_extension(file_path) + '/'
        self.model_save_path = self._file_dir_path + "model.pth"
        self.__make_dir()

        self.history_save_path = self._file_dir_path + "history.csv"
        self.history = History(self.history_save_path, label="Training")

        self.info_file_path = self._file_dir_path + "info.json"
        self.info = AdditionalInformation(self.info_file_path)

        self.locked = bool(file_op.is_file(self.model_save_path))
        self.net, self.optimizer = self.__construct_model(network, optimizer, input_shape)

        if file_op.is_file(self.model_save_path):
            self.load()

    def train(self, data, labels, number_of_epochs, *, batch_size: int=16, criterion=torch.nn.BCELoss(),
              shuffle: bool=True, sampling_number: int=3):
        if self.locked:
            raise ModelLockedError(self.model_save_path)

        self.net.train()
        x = torch.tensor(data.values, dtype=torch.float32)
        y = torch.tensor(labels.values, dtype=torch.float32)

        train_data = TensorDataset(x, y)
        data_loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)

        number_of_mini_batches = len(train_data) // batch_size + 1

        for epoch_num in range(number_of_epochs):
            epoch_start_time = time.perf_counter()
            running_loss = 0.0
            running_accuracy = 0.0
            for i, (_data_points, _labels) in enumerate(data_loader_train):
                loss, accuracy = self.__train_mini_batch(_data_points, _labels,
                                                         criterion=criterion, sampling_number=sampling_number)
                running_accuracy += accuracy[0]
                running_loss += loss
                print('-'*30)
                print(f"Trained batch {i + 1} of {len(data_loader_train)}")
                print(f"Loss: {loss}\tAccuracy: {accuracy}")
                print('-' * 30)

            epoch_end_time = time.perf_counter()

            self.info.add_runtime(epoch_end_time - epoch_start_time)
            running_loss /= number_of_mini_batches
            running_accuracy /= number_of_mini_batches
            self.history.step(float(running_loss), float(running_accuracy))

            self.save()
            self.plot_loss()
            self.plot_accuracy()

        return None

    def __train_mini_batch(self, _data_points, _labels, criterion=torch.nn.BCELoss(),
                           sampling_number: int=3) -> (float, (float, float, float)):
        self.optimizer.zero_grad()
        one_hot_labels = one_hot(_labels.type(torch.long), 2).type(torch.float32)
        loss = self.net.sample_elbo(inputs=_data_points, labels=one_hot_labels, criterion=criterion,
                                    sample_nbr=sampling_number)

        acc = self.test(data=_data_points, labels=one_hot_labels, samples=sampling_number)
        loss.backward()
        self.optimizer.step()
        return loss, acc

    def test(self, data, labels, samples: int=30, std_multiplier: float = 1.96) -> (float, float, float):
        was_training = self.net.training
        self.net.eval()

        predictions = [self.net(data) for _ in range(samples)]
        predictions = torch.stack(predictions)

        means = predictions.mean(axis=0)
        stds = predictions.std(axis=0)

        ci_upper = means + (std_multiplier * stds)
        ci_lower = means - (std_multiplier * stds)

        ic_acc = (ci_lower <= labels) * (ci_upper >= labels)
        ic_acc = ic_acc.float().mean()

        mean_ci_upper = (ci_upper >= labels).float().mean()
        mean_ci_lower = (ci_lower <= labels).float().mean()

        if was_training:
            self.net.train()

        return ic_acc, mean_ci_lower, mean_ci_upper

    @method_print_decorator
    def save(self) -> None:
        self.save_checkpoint()
        self.save_model_history()
        self.save_info()
        return None

    @method_print_decorator
    def load(self) -> None:
        print("Loading model...")
        if not file_op.is_file(self.model_save_path):
            raise FileNotFoundError

        checkpoint = Checkpoint.load_checkpoint(self.model_save_path, device=self.hyper_params.device)
        checkpoint.apply(self.net, self.optimizer)

        self.net.eval()
        self.net = self.net.to(self.hyper_params.device)
        print("Model loaded!")
        return None

    def unlock(self) -> None:
        self.locked = False
        return None

    def save_checkpoint(self) -> None:
        print('\033[96mSaving model checkpoint...\033[0m')  # CYAN TEXT
        checkpoint = Checkpoint(self.net, self.optimizer)
        checkpoint.save(self.model_save_path)
        return None

    def save_model_history(self) -> None:
        print('\033[94mSaving model history...\033[0m')  # BLUE TEXT
        self.history.save()
        return None

    def save_info(self) -> None:
        print("Saving model info...")
        self.info.save()
        return None

    def count_parameters(self):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.net.named_parameters():
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        print("+---------------+------------+")
        return total_params

    def plot_accuracy(self, title="Model accuracy over time") -> None:
        self.history.plot_accuracy(title=title)
        plt.savefig(self._file_dir_path + title.strip().lower())
        plt.close()
        return None

    def plot_loss(self, title="Model loss over time") -> None:
        self.history.plot_loss(title=title)
        plt.savefig(self._file_dir_path + title.strip().lower())
        plt.close()
        return None

    def __construct_model(self, network: nn.Module, optimizer, input_shape: int=30):
        net_ = network(input_shape, hidden_width=self.hyper_params.hidden_width)
        optimizer_ = optimizer(net_.parameters(), lr=self.hyper_params.learning_rate)
        return net_, optimizer_

    def __make_dir(self) -> None:
        if file_op.is_dir(self._file_dir_path):
            return None

        file_op.make_dir(self._file_dir_path)
        return None


def main():
    hyper_params = HyperParams(hidden_width=128)

    train_data = pd.read_csv(CONTAINER_DIR + "creditcard.csv").astype(np.float32)

    x = train_data.drop("Class", axis=1)
    y = train_data["Class"]

    bayesian_model = Model(BayesianNeuralNetwork, optim.Adam, "data/large_model.pth", hyper_params=hyper_params)
    bayesian_model.count_parameters()

    bayesian_model.unlock()
    bayesian_model.train(x, y, number_of_epochs=300, batch_size=256)


if __name__ == "__main__":
    main()
