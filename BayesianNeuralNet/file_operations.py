import json
import csv
import os
import pickle
import warnings

from typing import List, Iterable, Any

from pathlib import Path
import pandas as pd


class InvalidPathError(Exception):
    """ When a path could never point to a file, e.g missing .extension"""
    pass


class FileEmptyError(Exception):
    def __init__(self, file_path: str):
        self.message = f"The file {file_path} is empty!"
        super(FileEmptyError, self).__init__(self.message)


def list_files_in_directory(directory_path: str, extension_type: str = '.txt') -> list:
    child_file_paths = os.listdir(directory_path)
    return [file_path for file_path in child_file_paths
            if file_path_is_of_extension(file_path, extension=extension_type)]


def file_path_is_of_extension(file_path: str, extension: str= '.txt') -> bool:
    """ Only checks the string, not that file exists"""
    if len(file_path) <= len(extension):
        return False

    if '.' not in extension:
        raise InvalidPathError

    if file_path[-len(extension):] != extension:
        return False
    return True


def file_path_extension(file_path: str) -> str:
    dot_index = -1
    reversed_index_iterator = range(len(file_path) - 1, -1, -1)
    for index in reversed_index_iterator:
        if file_path[index] == '.':
            dot_index = index
            break
    if dot_index != -1:
        return file_path[dot_index:]
    raise InvalidPathError


def count_file_lines(file_path: str) -> int:
    with open(file_path, "r") as file:
        data = file.read()

    if len(data) == 0:
        return 1

    number_of_lines = sum(1 for line in data if line[-1] == '\n') + 1

    return number_of_lines


def is_file(file_path: str) -> bool:
    """ Assumes that the file exists """
    if len(file_path) == 0:
        return False
    if not os.path.isfile(file_path):
        return False
    if len(file_path) < 2:
        return False
    return True


def is_dir(dir_path: str) -> bool:
    if len(dir_path) == 0:
        return False
    return os.path.isdir(dir_path)


def make_dir(dir_path: str, parents=True) -> None:
    if dir_path[-1] != '/':
        raise InvalidPathError

    Path(dir_path).mkdir(parents=parents, exist_ok=True)
    return None


def dirname(file_path: str) -> str:
    return os.path.dirname(file_path)


def file_path_without_extension(file_path: str) -> str:
    """ Does not assume the file exists """

    if '.' not in file_path:
        warnings.warn(f"File path {file_path} does not contain \'.\' extension of any type!")
        return file_path

    stop_index = 0
    for i, character in enumerate(reversed(file_path)):
        if character == '.':
            stop_index = i + 1
            break

    return file_path[:-stop_index]


def child_path(path: str, child_num: int=1) -> str:
    # TODO swap to a heap and while loop rather than recursion.
    child_path_start = 0
    if len(path) <= 1:
        raise InvalidPathError
    for i, char in enumerate(path):
        if char == '/' or char == '\\':
            child_path_start = i + 1
            break

    child_path_str = path
    if child_path_start != 0:
        child_path_str = path[child_path_start:]

    if child_num > 1:
        return child_path(child_path_str, child_num - 1)

    return child_path_str


def parent_path(path: str, parent_num: int=1) -> str:
    if len(path) <= 1:
        raise InvalidPathError

    parent_path_start = 0
    if len(path) <= 1:
        raise InvalidPathError
    for i, char in enumerate(reversed(path)):
        if char == '/' or char == '\\':
            parent_path_start = i + 1
            break

    parent_path_str = path
    if parent_path_start != 0:
        parent_path_str = path[:-parent_path_start]

    if parent_num > 1:
        return parent_path(parent_path_str, parent_num - 1)

    return parent_path_str


def root_dir(path: str) -> str:
    root_end = 0
    if len(path) <= 1:
        raise InvalidPathError

    for i, char in enumerate(path):
        if char == '\\' or char == '/':
            root_end = i
            break
    return path[:root_end]


def trim_end_of_file_blank_line(file_path: str) -> None:
    with open(file_path, 'r') as in_file:
        data = in_file.read()

    with open(file_path, 'w') as out_file:
        out_file.write(data.rstrip('\n'))

    return None


def make_empty_file(file_path: str) -> None:
    """Use make_empty_file_safe if you don't want to overwrite data"""

    with open(file_path, "w") as outfile:
        outfile.write('')
    return None


def make_empty_file_safe(file_path: str) -> None:
    if os.path.isfile(file_path):
        raise FileExistsError
    make_empty_file(file_path)
    return None


def load_print_decorator(func: callable) -> callable:
    type_return = Any

    try:
        type_return = func.__annotations__["return"]
    except KeyError:
        pass

    def wrapper(*args, **kwargs) -> type_return:
        file_path = args[0].file_path
        print('-'*80)
        print(f"Loading file: {file_path}")
        data = func(*args, **kwargs)
        print(f"Finished loading file: {file_path}")
        print('-'*80)
        return data
    return wrapper


def save_print_decorator(func: callable) -> callable:
    type_return = Any

    try:
        type_return = func.__annotations__["return"]
    except KeyError:
        pass

    def wrapper(*args, **kwargs) -> type_return:
        file_path = args[0].file_path
        print('-' * 80)
        print(f"Saving to file: {file_path}")
        data = func(*args, **kwargs)
        print(f"Finished saving to file: {file_path}")
        print('-' * 80)
        return data
    return wrapper


class DictWriter:
    supported_file_extensions = ('.p', '.json')

    def __init__(self, file_path: str):
        assert file_path_extension(file_path) in DictWriter.supported_file_extensions, InvalidPathError
        self.file_path = file_path

    @property
    def file_exists(self):
        return is_file(self.file_path)

    @property
    def file_empty(self):
        return self.file_exists and os.stat(self.file_path).st_size == 0

    @property
    def keys(self) -> list:
        if not self.file_exists:
            raise KeyError
        return list(self.load().keys())

    @load_print_decorator
    def load(self) -> dict:
        file_extension = file_path_extension(self.file_path)
        if file_extension == ".p":
            return self.__load_pickle()
        if file_extension == ".json":
            return self.__load_json()
        raise InvalidPathError

    @save_print_decorator
    def save(self, data: dict) -> None:
        assert type(data) == dict, TypeError
        file_extension = file_path_extension(self.file_path)
        if file_extension == ".p":
            return self.__save_pickle(data)
        if file_extension == ".json":
            return self.__save_json(data)
        print(f"Finished saving to file: {self.file_path}")
        return None

    def __load_pickle(self) -> dict:
        with open(self.file_path, "rb") as pickle_file:
            data = pickle.load(pickle_file)
        return data

    def __save_pickle(self, data: dict) -> None:
        with open(self.file_path, "wb") as pickle_file:
            pickle.dump(data, pickle_file)
        return None

    def __load_json(self) -> dict:
        with open(self.file_path, 'r') as json_file:
            data = json.load(json_file)
        return data

    def __save_json(self, data: dict) -> None:
        with open(self.file_path, 'w') as json_file:
            json.dump(data, json_file)
        return None


class TextWriterSingleLine:
    def __init__(self, file_path: str):
        self.file_path = file_path

    @property
    def file_exists(self):
        return is_file(self.file_path)

    @property
    def file_empty(self):
        return self.file_exists and os.stat(self.file_path).st_size == 0

    @load_print_decorator
    def load(self) -> str:
        file_extension = file_path_extension(self.file_path)
        if file_extension == ".txt":
            return self.__load_text()
        raise InvalidPathError

    @load_print_decorator
    def load_safe(self):
        if not self.file_exists:
            return None

        return self.load()

    @save_print_decorator
    def save(self, data: str) -> None:
        assert hasattr(data, "__repr__")
        file_extension = file_path_extension(self.file_path)
        if file_extension == ".txt":
            self.__save_text(str(data))
        return None

    def __load_text(self) -> str:
        with open(self.file_path, "r") as text_file:
            data = text_file.read()
        return data

    def __save_text(self, data) -> None:
        with open(self.file_path, "w") as text_file:
            text_file.write(data)
        return None


class TextLogger:
    def __init__(self, file_path: str):
        assert file_path_extension(file_path) == ".txt", InvalidPathError
        self.file_path = file_path

    @property
    def file_exists(self):
        return is_file(self.file_path)

    @property
    def file_empty(self):
        return self.file_exists and os.stat(self.file_path).st_size == 0

    @load_print_decorator
    def load(self) -> List[str]:
        file_extension = file_path_extension(self.file_path)
        if file_extension == ".txt":
            return self.__load_text()
        raise InvalidPathError

    @load_print_decorator
    def load_safe(self):
        if not self.file_exists:
            return None

        return self.load()

    @save_print_decorator
    def save(self, data: list) -> None:
        assert type(data) == list, TypeError
        file_extension = file_path_extension(self.file_path)
        if file_extension == ".txt":
            self.__save_text(data)
        return None

    def append_lines(self, lines: list) -> None:
        if not self.file_exists:
            self.__save_text(lines)
        else:
            with open(self.file_path, 'a') as text_file:
                text_file.writelines('\n'.join(lines))
                text_file.write('\n')
        return None

    def remove_last_line(self) -> None:
        with open(self.file_path, 'r') as text_file:
            content = text_file.readlines()

        with open(self.file_path, 'w') as text_file:
            text_file.writelines(content[:-1])

    def __load_text(self) -> list:
        with open(self.file_path, "r") as text_file:
            data = text_file.readlines()
        return data

    def __save_text(self, lines: list) -> None:
        with open(self.file_path, "w") as text_file:
            text_file.writelines('\n'.join(lines))
            text_file.write('\n')
        return None


class CSV_Writer:
    def __init__(self, file_path: str, header: Iterable=None, delimiter='|'):
        self.__assert_is_csv(file_path)
        self.file_path = file_path
        self.delimiter = delimiter
        self.header = header

        if header == '$auto':
            if len(self) <= 1:
                raise FileEmptyError(self.file_path)
            self.header = None
            self.header = self.load_line(0)[0]

        self._batch_index = 0

    def __len__(self):
        # Remove 1, due to the empty line @ EOF.
        return count_file_lines(self.file_path) - 1

    @property
    def file_exists(self):
        return is_file(self.file_path)

    @property
    def file_empty(self):
        return self.file_exists and os.stat(self.file_path).st_size == 0

    @load_print_decorator
    def load_line(self, line_number: int) -> list:
        """ For efficiently loading a specific line number """

        self.__assert_valid_line_number(line_number)

        line_increment = 1 if self.header is not None else 0

        with open(self.file_path, "r") as file:
            content = [self.__parse_row(x) for i, x in enumerate(file) if i == line_number + line_increment]

        return content

    @load_print_decorator
    def load_range(self, start_index: int, end_index: int) -> List[list]:
        if start_index == end_index:
            return self.load_line(start_index)

        line_increment = 1 if self.header is not None else 0

        self.__assert_valid_line_number(start_index, end_index - 1)
        # Makes use of early stopping AND known list memory allocation.
        with open(self.file_path, "r") as file:
            lower_limit = start_index + line_increment
            upper_limit = end_index + line_increment
            content = [[] for _ in range(start_index, end_index)]
            for i, x in enumerate(file):
                if i >= upper_limit:
                    break
                if i >= lower_limit:
                    content[i - line_increment - start_index] = self.__parse_row(x)

        return content

    @load_print_decorator
    def load_sequential(self, batch_size: int, from_start=False) -> List[List]:
        if from_start:
            self._batch_index = 0

        line_increment = 1 if self.header is not None else 0

        if batch_size > len(self) - line_increment:
            raise ValueError

        start_index = self._batch_index
        end_index = self._batch_index + batch_size

        overlap = False
        if end_index > len(self) - line_increment:
            overlap = True
            end_index = len(self) - line_increment

        content1 = self.load_range(start_index, end_index)

        if overlap:
            end_index_wrapped = batch_size - (len(self) - line_increment - start_index)

            content2 = self.load_range(0, end_index_wrapped)

            self._batch_index = end_index_wrapped
            return content1 + content2

        if end_index >= len(self) - line_increment:
            self._batch_index = 0
        else:
            self._batch_index = end_index

        return content1

    @load_print_decorator
    def load_as_dataframe(self, safe=True) -> pd.DataFrame:
        if safe and not self.file_exists:
            raise FileExistsError

        return self.__load()

    @load_print_decorator
    def load_all(self, safe=True, as_float=False) -> list:
        if safe and not self.file_exists:
            raise FileNotFoundError

        return self.__load_as_list(as_float=as_float)

    @save_print_decorator
    def write(self, data: Iterable) -> None:
        assert isinstance(data, Iterable), TypeError
        file_extension = file_path_extension(self.file_path)
        if file_extension == ".csv":
            self.__save(data)
        return None

    def write_dataframe(self, data: pd.DataFrame):
        data_header = data.columns.values.tolist()
        self.header = data_header
        data.to_csv(self.file_path, sep=self.delimiter, index=False, header=True)
        return None

    def append_lines(self, lines: list) -> None:
        if not self.file_exists or self.file_empty:
            self.__save(lines)
        else:
            if self.__lines_are_empty(lines):
                return None

            with open(self.file_path, 'a') as file:
                for line in lines:
                    file.write(self.delimiter.join(line) + '\n')
        return None

    def remove_last_line(self) -> None:
        if len(self) == 0:
            warnings.warn("File is empty, cannot remove empty line")
            return None

        with open(self.file_path, 'r') as text_file:
            content = text_file.readlines()

        with open(self.file_path, 'w') as text_file:
            text_file.writelines(content[:-1])

        return None

    def __load(self) -> pd.DataFrame:
        data = pd.read_csv(self.file_path, sep=self.delimiter)
        return data

    def __load_as_list(self, as_float=False) -> list:
        with open(self.file_path, 'r') as file:
            csv_reader = csv.reader(file, delimiter=self.delimiter)
            if self.header is not None:
                next(csv_reader)

            if as_float:
                data = [[float(element) for element in row] for row in csv_reader]
            else:
                data = [row for row in csv_reader]
        return data

    def __save(self, lines: list) -> None:
        if self.__lines_are_empty(lines):
            return None

        rows_to_write = [*lines]
        if self.header is not None:
            rows_to_write = [list(self.header)] + rows_to_write

        with open(self.file_path, 'w', newline='', encoding="utf-8") as file:
            csv_writer = csv.writer(file, delimiter=self.delimiter)
            csv_writer.writerows(rows_to_write)
        return None

    @staticmethod
    def __assert_is_csv(file_path) -> None:
        assert file_path_is_of_extension(file_path, '.csv'), InvalidPathError
        return None

    @staticmethod
    def __lines_are_empty(lines: List[list]) -> bool:
        if sum([1 for line in lines if not line]) > 0:
            warnings.warn("Do not try to write an empty list.")
            return True
        return False

    def __assert_valid_line_number(self, *line_numbers: int) -> None:
        assert not self.file_empty and self.file_exists, FileNotFoundError
        maximum_line_number = len(self) - 1 if self.header is None else len(self) - 2
        for line_number in line_numbers:
            if line_number < 0 or line_number > maximum_line_number:
                print(f'Invalid line {line_number}')
                raise ValueError
        return None

    def __parse_row(self, row: str) -> list:
        return row[:-1].split(self.delimiter)


if __name__ == "__main__":
    pass
