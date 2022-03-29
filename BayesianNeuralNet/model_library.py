from __future__ import annotations
from typing import Callable, Union, Dict, Any

import file_operations as file_op
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import optim
from sklearn.metrics import confusion_matrix, accuracy_score


class Checkpoint:
    def __init__(self, model, optimizer: optim.Optimizer):
        self.state_dict = model.state_dict()
        self.optimizer = optimizer.state_dict()

    def __getstate__(self):
        return {"state_dict": self.state_dict, "optimizer": self.optimizer}

    def __setstate__(self, state):
        self.optimizer = state["optimizer"]
        self.state_dict = state["state_dict"]

    def __repr__(self):
        return str(self.__getstate__())

    def save(self, file_path: str) -> None:
        state = self.__getstate__()
        torch.save(state, file_path)
        return None

    def safe_save(self, file_path: str) -> None:
        if file_op.is_file(file_path):
            raise FileExistsError
        self.save(file_path)
        return None

    def apply(self, model, optimizer: optim.Optimizer) -> (Any, optim.Optimizer):
        """ Apply the checkpoint saved values to instances of models or optimizers """
        model.load_state_dict(self.state_dict)

        if self.is_legacy_checkpoint:
            self.optimizer = optimizer.state_dict()
        else:
            optimizer.load_state_dict(self.optimizer)
        return model, optimizer

    @classmethod
    def load_checkpoint(cls, file_path: str, device: torch.device=torch.device("cpu")) -> Checkpoint:
        if not file_op.is_file(file_path):
            raise FileNotFoundError
        state = torch.load(file_path, map_location=device)
        checkpoint = cls.__new__(cls)
        if not isinstance(state, dict) or "state_dict" not in state or "optimizer" not in state:
            checkpoint.state_dict = state
            checkpoint.optimizer = None
            return checkpoint

        checkpoint.state_dict = state["state_dict"]
        checkpoint.optimizer = state["optimizer"]
        return checkpoint

    @property
    def is_legacy_checkpoint(self):
        """Old checkpoints were stored purely as state dicts, not {state_dict, optimizer}"""
        return self.optimizer is None


class HyperParams:
    def __init__(self, num_layers: int = 2, dropout: float = 0,
                 device="cuda", learning_rate: float = 0.01, optimizer=optim.Adam,
                 l2_regularisation=0, l1_regularisation=0,
                 patience=5, early_stopping_mode="moving_average"):
        # General parameters
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.l1_regularisation = l1_regularisation
        self.l2_regularisation = l2_regularisation
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Validation parameters
        self.patience = patience
        self.early_stopping_mode = early_stopping_mode

        # Read Only Fields
        # self.__optimizer = optimizer
        self.__optimizer = optimizer
        self.__num_layers = num_layers

    @property
    def optimizer(self):
        return self.__optimizer

    @property
    def num_layers(self):
        return self.__num_layers


class History:
    """ Uses a csv file to write its loss and accuracy"""

    def __init__(self, file_path: str, decimal_places: int = -1, label="training"):
        self.__file_path = file_path
        self.__decimal_places = decimal_places

        self.__loss = []
        self.__accuracy = []

        self.__additional_metrics = None

        if self.is_file():
            self.load()

        self.label = label

    def __len__(self):
        return len(self.loss)

    @property
    def file_path(self):
        return self.__file_path

    @property
    def decimal_places(self):
        return self.__decimal_places

    @property
    def loss(self):
        return self.__loss

    @property
    def tracking_additional_metrics(self):
        return self.__additional_metrics is not None

    @property
    def accuracy(self):
        return self.__accuracy

    def is_file(self) -> bool:
        return file_op.is_file(self.file_path)

    def assert_not_empty(self) -> None:
        if len(self) == 0:
            print('No history has been provided! Loss/Accuracy empty!')
            raise ValueError
        return None

    def step(self, loss, accuracy, additional_metrics: Dict[str, float] = None) -> None:
        rounded_loss = round(loss, self.decimal_places) if self.decimal_places != -1 else loss
        rounded_accuracy = round(accuracy, self.decimal_places) if self.decimal_places != -1 else accuracy
        self.loss.append(rounded_loss)
        self.accuracy.append(rounded_accuracy)

        if additional_metrics is not None:
            for key, value in additional_metrics.items():
                self.__additional_metrics[key].append(value)
        return None

    def track_metrics(self, metric_evaluator: MetricEvaluator) -> None:
        self.__additional_metrics = {metric: [np.nan for _ in range(len(self.loss))]
                                     for metric in metric_evaluator.metrics_to_track}
        return None

    def save(self) -> None:
        self.assert_not_empty()
        print('\033[94mSaving Model History...\033[0m')  # BLUE TEXT
        data_to_write = pd.DataFrame({'loss': self.loss, 'accuracy': self.accuracy})

        if self.tracking_additional_metrics:
            for key, value in self.__additional_metrics.items():
                data_to_write[key] = value

        # KNOWN INSPECTION BUG FOR TO_CSV()
        # https://stackoverflow.com/questions/68787744/
        #   pycharm-type-checker-expected-type-none-got-str-instead-when-using-pandas-d
        # noinspection PyTypeChecker
        data_to_write.to_csv(path_or_buf=self.file_path)
        return None

    def load(self) -> None:
        assert file_op.is_file(self.file_path), FileNotFoundError
        print('Loading Model History...')
        data_from_file = pd.read_csv(self.file_path)
        self.__loss = data_from_file['loss'].tolist()
        self.__accuracy = data_from_file['accuracy'].tolist()

        if self.tracking_additional_metrics:
            data_from_file.pop('loss')
            data_from_file.pop('accuracy')

            for key, value in data_from_file.items():
                self.__additional_metrics[key] = data_from_file[key].tolist()

        return None

    def plot_loss(self, axes=None, title='') -> plt.axes:
        assert self.is_file(), FileNotFoundError
        epoch_steps = range(len(self))

        figure_loss = self.loss.copy()

        for i, value in enumerate(self.loss):
            figure_loss[i] = round(value, 3)

        ax = axes
        if axes is None:
            fig, ax = plt.subplots()

        ax.plot(epoch_steps, figure_loss, label=self.label)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')

        ax.legend()

        return ax

    def plot_accuracy(self, axes=None, title='') -> plt.axes:
        assert self.is_file(), FileNotFoundError
        epoch_steps = range(len(self))

        figure_acc = self.accuracy.copy()

        for i, value in enumerate(self.accuracy):
            figure_acc[i] = round(value, 3)

        ax = axes
        if axes is None:
            fig, ax = plt.subplots()

        ax.plot(epoch_steps, figure_acc, label=self.label)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')

        ax.legend()

        return ax


class EarlyStoppingTraining:
    """ https://clay-atlas.com/us/blog/2021/08/25/pytorch-en-early-stopping/ """
    modes = ("strict", "moving_average", "minimum", "none")

    def __init__(self, save_checkpoint: Callable, file_path: str,
                 patience: int = 5, validation_history: Union[None, History] = None, mode: str = "minimum"):
        self.step = self.__select_measure(mode)

        self.__assert_valid_patience(patience)

        self.patience = patience
        self.triggers_file_writer = file_op.TextWriterSingleLine(file_path)

        val_history = validation_history if len(validation_history) != 0 else None
        self.loss_comparison = self.__calculate_loss_comparison(mode, val_history)

        self.trigger_times = 0
        if self.triggers_file_writer.file_exists:
            self.trigger_times = int(self.triggers_file_writer.load_safe())

        # For saving checkpoints during each __call__
        self.save_checkpoint = save_checkpoint

    def __call__(self, loss: float) -> bool:
        out = self.step(loss)
        self.save_trigger_times()
        return out

    def reset_validation_trigger(self) -> None:
        self.trigger_times = 0
        self.save_checkpoint()
        return None

    def save_trigger_times(self) -> None:
        self.triggers_file_writer.save(self.trigger_times)
        return None

    def __select_measure(self, mode) -> Callable[[float], bool]:
        self.assert_valid_mode(mode)

        if mode == "moving_average":
            return self.__moving_average

        if mode == "minimum":
            return self.__minimum

        if mode == "none":
            return self.__none

        if mode == "strict":
            return self.__strict

        return self.__minimum

    def __calculate_loss_comparison(self, mode, validation_history: Union[None, History] = None):
        self.assert_valid_mode(mode)

        if validation_history is None:
            return np.inf

        if mode == "moving_average":
            average = 0
            for val_loss in validation_history.loss:
                average = (val_loss + average) / 2
            return average

        if mode == "minimum":
            return min(validation_history.loss)

        if mode == "strict":
            return validation_history.loss[-1]

        return np.inf

    def __strict(self, loss) -> bool:
        """ loss comparison = previous loss"""
        if loss >= self.loss_comparison:
            self.trigger_times += 1
            print('Trigger times:', self.trigger_times)

            if self.trigger_times >= self.patience:
                print('Training Stopped Early!')
                return True
        else:
            self.reset_validation_trigger()

        self.loss_comparison = loss

        return False

    def __moving_average(self, loss) -> bool:
        """ loss comparison = (loss comparison + loss) / 2"""
        if loss >= self.loss_comparison:
            self.trigger_times += 1
            print('Trigger times:', self.trigger_times)

            if self.trigger_times >= self.patience:
                print('Training Stopped Early!')
                return True
        else:
            self.reset_validation_trigger()

        self.loss_comparison = (self.loss_comparison + loss) / 2

        return False

    def __minimum(self, loss) -> bool:
        """ loss comparison = global minimum loss"""
        print(f'LOSS: {loss}, LC: {self.loss_comparison}')
        if loss >= self.loss_comparison:
            self.trigger_times += 1
            print('Trigger times:', self.trigger_times)

            if self.trigger_times >= self.patience:
                print('Training Stopped Early!')
                return True
        else:
            self.reset_validation_trigger()
            self.loss_comparison = loss

        return False

    def __none(self, loss) -> bool:
        self.reset_validation_trigger()
        return False

    @staticmethod
    def assert_valid_mode(mode) -> None:
        if mode not in EarlyStoppingTraining.modes:
            raise ValueError
        return None

    @staticmethod
    def __assert_valid_patience(patience: int) -> None:
        if type(patience) != int:
            raise TypeError
        if patience < 0:
            raise ValueError
        return None


class Metrics:
    def __init__(self, predictions: torch.Tensor, labels: torch.Tensor):
        self.confusion_mtrx = confusion_matrix(predictions, labels)
        self.__accuracy = accuracy_score(predictions, labels)

    def recall(self) -> float:
        recall = np.diag(self.confusion_mtrx) / np.sum(self.confusion_mtrx, axis=1)
        recall_mean = float(np.mean(recall))
        return recall_mean

    def precision(self) -> float:
        precision = np.diag(self.confusion_mtrx) / np.sum(self.confusion_mtrx, axis=0)
        precision_mean = float(np.mean(precision))
        return precision_mean

    def f1_score(self) -> float:
        precision = self.precision()
        recall = self.recall()

        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score

    def accuracy(self) -> float:
        return self.__accuracy


class MetricEvaluator:
    metric_scores = ("recall", "precision", "f1_score", "accuracy")

    class InvalidMetricError(Exception):
        def __init__(self, metric_name: str):
            message = f"The given metric: {metric_name} is invalid. Try one of {MetricEvaluator.metric_scores}."
            super().__init__(message)

    def __init__(self, *metric_names):
        for metric in metric_names:
            self.__assert_valid_metric(metric)

        self.metrics_to_track = metric_names

    def __len__(self):
        return len(self.metrics_to_track)

    def __str__(self):
        return str(self.metrics_to_track)

    def __call__(self, predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        return self.evaluate(predictions, labels)

    def evaluate(self, predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        metrics = Metrics(predictions, labels)

        metric_functions = [self.__get_metric(metric, metrics) for metric in self.metrics_to_track]
        metrics_evaluated = {metric: metric_function()
                             for metric_function, metric in zip(metric_functions, self.metrics_to_track)}
        return metrics_evaluated

    @staticmethod
    def __get_metric(metric_func_name: str, metrics_instance: Metrics) -> Callable:
        metric_function = getattr(metrics_instance, metric_func_name)
        return metric_function

    @staticmethod
    def __assert_valid_metric(metric_name: str) -> None:
        assert metric_name in MetricEvaluator.metric_scores, MetricEvaluator.InvalidMetricError
        return None

    @staticmethod
    def print_all_metric_types() -> None:
        print(MetricEvaluator.metric_scores)
        return None


class AdditionalInformation:
    def __init__(self, file_path: str):
        self.file_path = file_path

        self.__dict_writer = file_op.DictWriter(self.file_path)

        self.data = {'total_runtime': 0}

        if self.__dict_writer.file_exists:
            self.data = self.__dict_writer.load()

    def __setitem__(self, key: str, value):
        self.data[key] = value
        return None

    def __getitem__(self, key):
        return self.data[key]

    def __delitem__(self, key):
        if key in self.protected_keys:
            raise KeyError
        del self.data[key]
        return None

    @property
    def protected_keys(self):
        """These cannot be deleted"""
        return ["total_runtime"]

    @property
    def is_file(self):
        return self.__dict_writer.file_exists

    def add_runtime(self, runtime: float):
        self.data['total_runtime'] += runtime

    def reset_runtime(self):
        self.data['total_runtime'] = 0

    @property
    def runtime(self):
        return self.data['total_runtime']

    def save(self) -> None:
        self.__dict_writer.save(self.data)
        return None

    def load(self) -> dict:
        return self.__dict_writer.load()