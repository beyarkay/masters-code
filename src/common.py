"""A collection of commonly used functions/classes for Ergo."""
import tqdm
import sklearn
from enum import Enum
import logging as l
import numpy as np
import os
import sys
from typing import List, Optional, TypedDict, TypeVar
import datetime
import pandas as pd

T = TypeVar("T")


class ConstantsDict(TypedDict):
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


def make_gestures_and_indices(
    y_str, not_g255=None, to_i=None, to_g=None, g255="gesture0255"
):
    """Create two vectorized functions to convert gesture names to integer
    indices and vice versa.

    Includes a special case for the `gesture0255` gesture, since mapping
    `gesture0255` => `255` causes all sorts of problems in tensorflow.

    Args:
    - y_str: A 1-dimensional ndarray containing the gesture names.
    - not_g255: A one-parameter function that returns a bool and determines if
      an element in y_str is the g255 element.
    - to_i: a one-parameter function that converts an element of y_str into an
      integer.
    - to_g: a one-parameter function that converts an integer into a gesture.
    - g255: the representation of gesture0255 in y_str.

    Returns:
    A tuple containing two vectorized functions:
    - to_index: A function that takes a gesture name and returns an integer index.
    - to_gesture: A function that takes an integer index and returns a gesture name.
    """
    if to_g is None:
        def to_g(i): return f"gesture{i:0>4}"  # noqa: E731
    if to_i is None:
        def to_i(g): return int(g[-4:])  # noqa: E731
    if not_g255 is None:
        def not_g255(g): return g != g255  # noqa: E731

    # Determine the maximum integer value, which is one less than the number of
    # unique gestures.
    maximum = len(np.unique(y_str)) - 1

    # Define two vectorized functions to convert gestures to indices and vice
    # versa.
    to_index = np.vectorize(lambda g: to_i(g) if not_g255(g) else maximum)
    to_gesture = np.vectorize(lambda i: to_g(i) if i != maximum else g255)
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
        self.stdout: str = ""

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


def make_windows(
    data: pd.DataFrame, window_size: int, pbar=None
) -> (np.ndarray, np.ndarray):
    """Process data into a windowed format for machine learning.

    Args:
    - data: A pandas DataFrame containing the data to be processed.
    - window_size: An integer representing the size of the rolling window to use.

    Returns:
    A tuple containing two numpy ndarrays:
    - X: A 3-dimensional ndarray with shape (size, window_size, 30).
    - y: A 1-dimensional ndarray with shape (size,).
    """

    # Group data by file and apply rolling window of size window_size
    rolling = data.groupby("file").rolling(
        window=window_size, min_periods=window_size)

    # Calculate unique number of files
    uniq = len(data.value_counts("file"))

    # Calculate number of windows
    size = len(data) - uniq * (window_size - 1) + 1

    # Read finger constants from a file
    const: ConstantsDict = read_constants()
    sensors = const["sensors"].values()

    # Loop over the windows and populate X and y
    Xs = []
    ys = []
    dts = []
    for i, window in enumerate(rolling):
        if pbar is not None:
            pbar.update(1)
        if len(window) < window_size:
            continue
        Xs.append(window[sensors].values)
        ys.append(window.gesture.values[-1])
        dts.append(window.datetime.values[-1])

    # Return X and y as a tuple
    return (np.stack(Xs), np.array(ys), np.array(dts))


def save_as_windowed_npz(df, n_timesteps=25):
    """Given a DataFrame of gestures, split them into windows of n_timesteps
    long and then save that windowed data as .npz files.

    There will be one file `./gesture_data/trn_40.npz` which contains the training &
    validation data, and one file `./gesture_data/tst.npz` which contains the
    testing data."""
    X, y_str, dt = make_windows(
        df,
        n_timesteps,
        pbar=tqdm.tqdm(total=len(df), desc="Making windows"),
    )
    g2i, i2g = make_gestures_and_indices(y_str)
    y = g2i(y_str)

    # fmt: off
    X_trn, X_tst, y_trn, y_tst, dt_trn, dt_tst = sklearn.model_selection.train_test_split(    # noqa: E501
        X, y, dt, stratify=y,
    )
    # fmt: on
    np.savez(
        f"./gesture_data/trn_{n_timesteps}.npz", X_trn=X_trn, y_trn=y_trn, dt_trn=dt_trn
    )
    np.savez(
        f"./gesture_data/tst_{n_timesteps}.npz", X_tst=X_tst, y_tst=y_tst, dt_tst=dt_tst
    )


def first_or_fail(arr: List[T], msg: Optional[str] = None) -> T:
    if len(arr) == 1:
        return arr[0]
    else:
        raise Exception(
            msg if msg else f"Provided array had {len(arr)} (!= 1) elements: {arr}"
        )
