"""A collection of commonly used functions/classes for Ergo."""
from enum import Enum
import logging as l
import numpy as np
import os
import sys
import typing
import datetime


class ConstantsDict(typing.TypedDict):
    sensors: dict[int, str]
    n_sensors: int
    sensor_bounds: dict[str, int]
    baud_rate: int


def read_constants(path: str = "src/constants.yaml") -> ConstantsDict:
    """Reads a YAML file at the given path and returns its contents as a dictionary.

    Args:
        path (str, optional): The path to the YAML file to be read. Defaults to
        "src/constants.yaml".

    Returns:
        ConstantsDict: A dictionary containing the key-value pairs read from
        the YAML file.
    """

    import yaml

    with open(path, "r") as f:
        constants = yaml.safe_load(f)
    return constants


def check_X_y(X: np.ndarray, y: np.ndarray):
    assert X is not None
    assert y is not None
    assert type(X) is np.ndarray
    assert type(y) is np.ndarray
    assert X.shape[0] == y.shape[0]
    return X, y


def make_gestures_and_indices(y_str):
    """Create two vectorized functions to convert gesture names to integer
    indices and vice versa.

    Includes a special case for the `gesture0255` gesture, since mapping
    `gesture0255` => `255` causes all sorts of problems in tensorflow.

    Args:
    - y_str: A 1-dimensional ndarray containing the gesture names.

    Returns:
    A tuple containing two vectorized functions:
    - to_index: A function that takes a gesture name and returns an integer index.
    - to_gesture: A function that takes an integer index and returns a gesture name.
    """

    # Determine the maximum integer value, which is one less than the number of
    # unique gestures.
    maximum = len(np.unique(y_str)) - 1

    # Define two vectorized functions to convert gestures to indices and vice
    # versa.
    to_index = np.vectorize(lambda g: int(g[-4:]) if g != "gesture0255" else maximum)
    to_gesture = np.vectorize(
        lambda i: f"gesture{i:0>4}" if i != maximum else "gesture0255"
    )
    return (to_index, to_gesture)


class ControlFlow(Enum):
    """
    An enumeration of control flow options for handler sequences.

    Options:
    - BREAK: Stop the handler sequence immediately.
    - CONTINUE: Skip the rest of the handler sequence and start a new one.
    """

    BREAK = 0
    CONTINUE = 1


class AbstractHandler:
    """An abstract base class for serial port data handlers.

    Attributes:
    - control_flow: A ControlFlow enum representing the control flow for the
      loop.

    Methods:
    - execute: A method that handles the serial port data.
    """

    def __init__(self):
        self.const: ConstantsDict = read_constants()
        self.control_flow = ControlFlow.CONTINUE

    def execute(
        self,
        past_handlers,
    ):
        """Handle the serial port data.

        :param past_handlers: The list of previous handlers.
        :type past_handlers: List[AbstractHandler]

        :returns: None.
        :raises NotImplementedError: This is an abstract method and must be
        implemented in derived classes.
        """
        raise NotImplementedError


def init_logs():
    """Initiate logging to a file with the current timestamp as the filename.

    This function configures the logging module to write log messages to a file
    with a name that includes the current timestamp. The log file is created in
    a "logs" directory in the current working directory. The logging level is
    set to INFO, which means that messages with a severity level of INFO or
    higher will be logged.

    Returns:
        None
    """
    if not os.path.exists("logs"):
        os.mkdir("logs")

    log = l.getLogger("logger")
    log.setLevel(l.DEBUG)

    formatter = l.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    file_handler = l.FileHandler(
        f"logs/{str(datetime.datetime.now()).replace(' ', 'T')}.log",
        mode="w",
        encoding="utf-8",
    )
    file_handler.setLevel(l.DEBUG)
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    stream_handler = l.StreamHandler(sys.stdout)
    stream_handler.setLevel(l.INFO)
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)