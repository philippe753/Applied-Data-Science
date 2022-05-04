import os.path
import time

from typing import Type

from collections import Counter

import file_operations as file_op

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.nn import Sigmoid

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

from model_library import HyperParams, History, AdditionalInformation, Checkpoint, ModelLockedError
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split, StratifiedKFold


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
        print("INPUT DIM:", input_dim)
        self.bayes_linear1 = BayesianLinear(input_dim, hidden_width)
        self.bayes_linear2 = BayesianLinear(hidden_width, 1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x_ = self.bayes_linear1(x)
        return self.sigmoid(self.bayes_linear2(x_)).squeeze(dim=1)


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

        num_points = data.shape[0]

        self.net.train()

        skf = StratifiedKFold(n_splits=k_fold_samples, shuffle=shuffle)

        loss_scaling_factor = len(set(labels)) * k_fold_samples * num_points

        for epoch_num in range(number_of_epochs):
            epoch_start_time = time.perf_counter()

            train_losses = [0.0 for _ in range(k_fold_samples)]
            train_accuracies = [0.0 for _ in range(k_fold_samples)]
            validation_losses = [0.0 for _ in range(k_fold_samples)]
            validation_accuracies = [0.0 for _ in range(k_fold_samples)]
            k_step = 0
            for train_index, test_index in skf.split(data, labels):
                print(f"Evaluation k-fold iter: {k_step + 1} of {k_fold_samples}")
                x_train_fold, x_test_fold = data[train_index], data[test_index]
                y_train_fold, y_test_fold = labels[train_index], labels[test_index]

                x_train_fold = torch.Tensor(x_train_fold).to(torch.float)
                x_test_fold = torch.Tensor(x_test_fold).to(torch.float)

                y_train_fold = torch.as_tensor(y_train_fold, dtype=torch.float32)
                y_test_fold = torch.as_tensor(y_test_fold, dtype=torch.float32)

                self.net.train()
                loss_train, acc_train = self.__train_mini_batch(x_train_fold, y_train_fold,
                                                                criterion=criterion, sampling_number=sampling_number)

                self.net.eval()
                loss_val, acc_val = self.__test_mini_batch(x_test_fold, y_test_fold,
                                                           criterion=criterion, sampling_number=sampling_number)

                validation_losses[k_step] += loss_val
                validation_accuracies[k_step] += acc_val[0]
                train_losses[k_step] += loss_train
                train_accuracies[k_step] += acc_train[0]

                k_step += 1

            train_loss = sum(train_losses) / loss_scaling_factor
            train_accuracy = 100 * sum(train_accuracies) / len(train_accuracies)

            validation_loss = sum(validation_losses) / loss_scaling_factor
            validation_accuracy = 100 * sum(validation_accuracies) / len(train_accuracies)

            print('-' * 30)
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

    def train_subsampled(self, data, labels, number_of_epochs, *, batch_size: int=16, criterion=torch.nn.BCELoss(),
                         shuffle: bool = True, sampling_number: int = 3, k_fold_samples: int = 3):
        if self.locked:
            raise ModelLockedError(self.model_save_path)

        classes = list(set(labels))

        self.net.train()

        class_count = Counter(labels)
        minority_label = min(class_count, key=class_count.get)
        majority_label = max(class_count, key=class_count.get)

        minority_data = data[labels == minority_label]
        minority_labels = labels[labels == minority_label]

        majority_data = data[labels == majority_label]
        majority_labels = labels[labels == majority_label]

        minority_data = np.column_stack((minority_data, minority_labels))
        majority_data = np.column_stack((majority_data, majority_labels))

        end_of_array = min(minority_data.shape[0], batch_size // 2)

        train_percent = int(0.8 * end_of_array)

        for epoch_num in range(number_of_epochs):
            epoch_start_time = time.perf_counter()

            np.random.shuffle(minority_data)
            np.random.shuffle(majority_data)

            x_train = np.vstack((minority_data[:train_percent, :-1], majority_data[:train_percent, :-1]))
            y_train = np.hstack((minority_data[:train_percent, -1], majority_data[:train_percent, -1]))

            x_test = np.vstack((minority_data[train_percent:end_of_array, :-1],
                                     majority_data[train_percent:end_of_array, :-1]))
            y_test = np.hstack((minority_data[train_percent:end_of_array, -1],
                                     majority_data[train_percent:end_of_array, -1]))

            x_train = torch.Tensor(x_train).to(torch.float)
            x_test = torch.Tensor(x_test).to(torch.float)

            y_train = torch.as_tensor(y_train, dtype=torch.float32)
            y_test = torch.as_tensor(y_test, dtype=torch.float32)

            self.net.train()
            loss_train, acc_train = self.__train_mini_batch(x_train, y_train,
                                                            criterion=criterion, sampling_number=sampling_number)

            self.net.eval()
            loss_val, acc_val = self.__test_mini_batch(x_test, y_test,
                                                       criterion=criterion, sampling_number=sampling_number)

            print('-' * 30)
            print(f"Trained epoch {epoch_num + 1} of {number_of_epochs}")
            print(f"Train | Loss: {loss_train}\tAccuracy: {acc_train[0]}")
            print(f"Valid | Loss: {loss_val}\tAccuracy: {acc_val[0]}")
            print('-' * 30)
            epoch_end_time = time.perf_counter()

            self.info.add_runtime(epoch_end_time - epoch_start_time)
            self.history.step(float(loss_train), float(acc_train[0]))
            self.validation_history.step(float(loss_val), float(acc_val[0]))

            self.save()
            self.plot_loss()
            self.plot_accuracy()

        return None

    def __train_mini_batch(self, _data_points, _labels, criterion=torch.nn.BCELoss(),
                           sampling_number: int = 3) -> (float, (float, float, float)):
        self.optimizer.zero_grad()

        loss = self.net.sample_elbo(inputs=_data_points, labels=_labels, criterion=criterion, sample_nbr=sampling_number)
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

    def test(self, data, labels, samples: int = 3, std_multiplier: float = 2) -> (float, float, float):
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
    hyper_params = HyperParams(hidden_width=200, learning_rate=0.000_1)

    bayesian_model = Model(BayesianNeuralNetwork, optim.Adam, "data/undersampled_medium_model_with_scv.pth", hyper_params=hyper_params)
    bayesian_model.count_parameters()

    data_path = r"Q:\MichaelsStuff\EngMaths\Year4\AppliedDataScience\creditcard.csv"

    train_data = pd.read_csv(data_path).astype(np.float32)

    x = train_data.drop("Class", axis=1)
    y = train_data["Class"]

    train, test = train_test_split(x, test_size=0.2)

    bayesian_model.unlock()
    bayesian_model.train_subsampled(x.to_numpy(), y.to_numpy(), number_of_epochs=300, batch_size=256)

    print("test scores:", bayesian_model.test(test))


if __name__ == "__main__":
    main()
