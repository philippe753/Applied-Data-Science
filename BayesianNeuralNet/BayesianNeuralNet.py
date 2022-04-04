import os.path
import time

from typing import Type

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


CONTAINER_DIR = os.path.dirname(os.path.dirname(__file__)) + "/"


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
    def __init__(self, input_dim, hidden_width: int = 64, **kwargs):
        super().__init__()
        self.bayes_linear1 = BayesianLinear(input_dim, hidden_width)
        self.bayes_linear2 = BayesianLinear(hidden_width, 2)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x_ = self.bayes_linear1(x)
        return self.sigmoid(self.bayes_linear2(x_))


class Model:
    def __init__(self, network: Type[nn.Module], optimizer: Type[optim.Optimizer], file_path: str, *,
                 hyper_params: HyperParams = HyperParams(),
                 input_shape: int = 30):
        # Model info
        self.net = network
        self.optimizer = optimizer
        self.hyper_params = hyper_params

        # Creating save directory
        self._file_dir_path = file_op.file_path_without_extension(file_path) + '/'
        self.model_save_path = self._file_dir_path + "model.pth"
        self.__make_dir()

        self.validation_save_path = self._file_dir_path + "validation_history.csv"
        self.validation_history = History(self.validation_save_path, label="validation")

        self.history_save_path = self._file_dir_path + "history.csv"
        self.history = History(self.history_save_path, label="Training")

        self.info_file_path = self._file_dir_path + "info.json"
        self.info = AdditionalInformation(self.info_file_path)

        self.locked = bool(file_op.is_file(self.model_save_path))
        self.net, self.optimizer = self.__construct_model(network, optimizer, input_shape)

        if file_op.is_file(self.model_save_path):
            self.load()

    def train(self, data, labels, number_of_epochs, *, batch_size: int = 16, criterion=torch.nn.BCELoss(),
              shuffle: bool = True, sampling_number: int = 3, k_fold_samples: int = 3):
        if self.locked:
            raise ModelLockedError(self.model_save_path)

        class_labels = list(set(labels))
        self.net.train()

        x_split_by_labels = [data.where(labels == num).dropna() for num in class_labels]
        x_split_by_labels = [torch.tensor(x.values, dtype=torch.float32) for x in x_split_by_labels]
        y_split_by_labels = [labels.where(labels == num).dropna() for num in class_labels]
        y_split_by_labels = [torch.tensor(y.values, dtype=torch.float32) for y in y_split_by_labels]

        y_split_by_labels = [one_hot(y.type(torch.long), 2).type(torch.float32) for y in y_split_by_labels]

        minimum_num_batches = min(map(len, y_split_by_labels)) // batch_size + 1

        while minimum_num_batches < k_fold_samples:
            batch_size = int(batch_size * 0.8)
            minimum_num_batches = min(map(len, y_split_by_labels)) // batch_size + 1
        print(f"batch size: {batch_size}")

        train_data_by_label = [TensorDataset(x_split_by_labels[i], y_split_by_labels[i])
                               for i in range(len(class_labels))]
        data_loaders_by_label = [DataLoader(train_data_by_label[i], batch_size=batch_size, shuffle=shuffle)
                                 for i in range(len(class_labels))]

        num_to_sample_by_label = [int((1 / k_fold_samples) * len(data_loaders_by_label[i]))
                                  for i in range(len(class_labels))]

        loss_scaling_factor = k_fold_samples * len(class_labels) * sum(num_to_sample_by_label)

        for epoch_num in range(number_of_epochs):
            epoch_start_time = time.perf_counter()

            train_losses = [[0.0 for _ in class_labels] for _ in range(k_fold_samples)]
            train_accuracies = [[0.0 for _ in class_labels] for _ in range(k_fold_samples)]
            validation_losses = [[0.0 for _ in class_labels] for _ in range(k_fold_samples)]
            validation_accuracies = [[0.0 for _ in class_labels] for _ in range(k_fold_samples)]

            for k_step in range(k_fold_samples):
                print(f"Evaluation k-fold iter: {k_step + 1} of {k_fold_samples}")
                for i, data_loader in enumerate(data_loaders_by_label):
                    iterable_dataloader = iter(data_loader)

                    for j in range(num_to_sample_by_label[i]):
                        x_val, y_val = next(iterable_dataloader)
                        self.net.eval()
                        loss_val, acc_val = self.__test_mini_batch(x_val, y_val,
                                                                   criterion=criterion, sampling_number=sampling_number)
                        self.net.train()
                        validation_losses[k_step][i] += loss_val
                        validation_accuracies[k_step][i] += acc_val[0]

                    for j in range(num_to_sample_by_label[i] * (k_fold_samples - 1)):
                        x_train, y_train = next(iterable_dataloader)
                        loss_train, acc_train = self.__train_mini_batch(x_train, y_train,
                                                                        criterion=criterion,
                                                                        sampling_number=sampling_number)
                        train_losses[k_step][i] += loss_train
                        train_accuracies[k_step][i] += acc_train[0]

            train_loss = sum(sum(losses) for losses in train_losses) / loss_scaling_factor
            train_accuracy = sum(sum(accuracies) for accuracies in train_accuracies) / loss_scaling_factor

            validation_loss = (sum(sum(losses) for losses in validation_losses) * (k_fold_samples - 1)) / loss_scaling_factor
            validation_accuracy = (sum(sum(accuracies) for accuracies in validation_accuracies) * (k_fold_samples - 1)) / loss_scaling_factor

            print('-' * 30)
            print(f"Sampling # batches per label: {num_to_sample_by_label}")
            print(f"Trained epoch {epoch_num + 1} of {number_of_epochs}")
            print(f"Train | Loss: {train_loss}\tAccuracy: {train_accuracy}")
            print(f"Valid | Loss: {validation_loss}\tAccuracy: {validation_accuracy}")
            print('-' * 30)
            epoch_end_time = time.perf_counter()

            self.info.add_runtime(epoch_end_time - epoch_start_time)
            self.history.step(float(train_loss), float(train_accuracy))
            self.validation_history.step(float(validation_loss), float(validation_accuracy))

            self.save()
            self.plot_loss()
            self.plot_accuracy()

        return None

    def __train_mini_batch(self, _data_points, _labels, criterion=torch.nn.BCELoss(),
                           sampling_number: int = 3) -> (float, (float, float, float)):
        self.optimizer.zero_grad()

        loss = self.net.sample_elbo(inputs=_data_points, labels=_labels, criterion=criterion,
                                    sample_nbr=sampling_number)

        acc = self.test(data=_data_points, labels=_labels, samples=sampling_number)
        loss.backward()
        self.optimizer.step()
        return loss, acc

    def __test_mini_batch(self, _data_points, _labels, criterion=torch.nn.BCELoss(),
                          sampling_number: int = 3) -> (float, (float, float, float)):
        loss = self.net.sample_elbo(inputs=_data_points, labels=_labels, criterion=criterion,
                                    sample_nbr=sampling_number)
        acc = self.test(data=_data_points, labels=_labels, samples=sampling_number)

        return loss, acc

    def test(self, data, labels, samples: int = 3, std_multiplier: float = 2, criterion=nn.BCELoss) -> (float, float, float):
        was_training = self.net.training
        self.net.eval()

        predictions = [self.net(data) for _ in range(samples)]
        predictions = torch.stack(predictions)

        means = predictions.mean(dim=0)
        stds = predictions.std(dim=0)

        ci_upper = means + (std_multiplier * stds)
        ci_lower = means - (std_multiplier * stds)

        ic_acc = (ci_lower <= labels) * (ci_upper >= labels)
        ic_acc = ic_acc.float().mean()

        mean_ci_upper = (ci_upper >= labels).float().mean()
        mean_ci_lower = (ci_lower <= labels).float().mean()

        if was_training:
            self.net.train()

        return ic_acc, mean_ci_upper, mean_ci_lower

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
        self.validation_history.save()
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
        ax = self.history.plot_accuracy(title=title)
        self.validation_history.plot_accuracy(axes=ax, title=title)
        plt.savefig(self._file_dir_path + title.strip().lower())
        plt.close()
        return None

    def plot_loss(self, title="Model loss over time") -> None:
        ax = self.history.plot_loss(title=title)
        self.validation_history.plot_loss(axes=ax, title=title)
        plt.savefig(self._file_dir_path + title.strip().lower())
        plt.close()
        return None

    def __construct_model(self, network: Type[nn.Module], optimizer: Type[optim.Optimizer],
                          input_shape: int = 30):
        net_ = network(input_shape, hidden_width=self.hyper_params.hidden_width)  # noqa
        optimizer_ = optimizer(net_.parameters(), lr=self.hyper_params.learning_rate)  # noqa
        return net_, optimizer_

    def __make_dir(self) -> None:
        if file_op.is_dir(self._file_dir_path):
            return None

        file_op.make_dir(self._file_dir_path)
        return None


def main():
    hyper_params = HyperParams(hidden_width=128, learning_rate=0.001)

    bayesian_model = Model(BayesianNeuralNetwork, optim.Adam, "data/medium_model_with_scv.pth", hyper_params=hyper_params)
    bayesian_model.count_parameters()

    data_path = CONTAINER_DIR + "/Datasets/creditcard.csv"

    train_data = pd.read_csv(data_path).astype(np.float32)

    x = train_data.drop("Class", axis=1)
    y = train_data["Class"]

    bayesian_model.unlock()
    bayesian_model.train(x, y, number_of_epochs=300, batch_size=256)


if __name__ == "__main__":
    main()
