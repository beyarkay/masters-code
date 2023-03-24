"""Takes care of reading in data from the serial port stream and converts it to
numpy arrays. Also reads data in from disk.
"""
import logging as l
import serial
import pandas as pd
import typing
import os
import datetime
import common
import pickle
import models
import numpy as np


def read_model(directory: str) -> models.TemplateClassifier:
    with open(f"{directory}/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


def read_data(directory: str = "./gesture_data/train/") -> pd.DataFrame:
    """Reads in CSV files from a directory and returns a concatenated Pandas DataFrame.

    Args:
        directory (str, optional): Path to the directory containing the CSV
        files. Defaults to "./gesture_data/train/".

    Returns:
        pd.DataFrame: A concatenated Pandas DataFrame with columns: "datetime",
        "gesture", "file", "finger", "orientation" and additional columns
        representing finger data.
    """
    # Load finger data constants
    sensors: list[str] = list(common.read_constants()["sensors"].values())
    # Initialize empty list to store DataFrames
    dfs = []
    # Get list of paths to CSV files in directory and sort them
    paths = sorted([p for p in os.listdir(directory) if p.endswith(".csv")])
    # Iterate through paths and load each CSV file into a DataFrame
    for i, path in enumerate(paths):
        df = pd.read_csv(
            directory + path,
            names=["datetime", "gesture"] + sensors,
            parse_dates=["datetime"],
        )
        # Add a column with the filename for reference
        df["file"] = path
        dfs.append(df)
    # Concatenate DataFrames into one
    together = pd.concat(dfs)
    # Extract finger and orientation data from the gesture column and add to DataFrame
    together["finger"] = together["gesture"].apply(
        lambda g: None if g == "gesture0255" else int(g[-3:]) % 10
    )
    together["orientation"] = together["gesture"].apply(
        lambda g: None if g == "gesture0255" else int(g[-3:]) // 10
    )
    # Select relevant columns and reorder them
    together = together[
        ["datetime", "gesture", "file", "finger", "orientation"] + sensors
    ]
    # Sort DataFrame by datetime and reset index
    return together.sort_values("datetime").reset_index(drop=True)


def window_data(data: pd.DataFrame, window_size: int) -> tuple[np.ndarray]:
    """
    Process data into a windowed format for machine learning.

    Args:
    - data: A pandas DataFrame containing the data to be processed.
    - window_size: An integer representing the size of the rolling window to use.

    Returns:
    A tuple containing two numpy ndarrays:
    - X: A 3-dimensional ndarray with shape (size, window_size, 30).
    - y: A 1-dimensional ndarray with shape (size,).
    """

    # Group data by file and apply rolling window of size window_size
    rolling = data.groupby("file").rolling(window=window_size, min_periods=window_size)

    # Calculate unique number of files
    uniq = len(data.value_counts("file"))

    # Calculate number of windows
    size = len(data) - uniq * (window_size - 1)

    # Initialize empty ndarrays for X and y
    X = np.empty((size, window_size, 30))
    y = np.empty((size,))

    # Read finger constants from a file
    const: common.ConstantsDict = common.read_constants()
    sensors = const["sensors"].values()

    # Loop over the windows and populate X and y
    count = 0
    for i, window in enumerate(rolling):
        if len(window) < window_size:
            continue
        X[i] = window[sensors].values
        y[i] = window.gesture.values[-1]
        count += 1

    # Ensure that the expected number of windows was processed
    assert count == size, f"{count} != {size}"

    # Return X and y as a tuple
    return (X, y)


# TODO this should be a rewrite of `ml_utils.get_serial_port`.
def find_port() -> str:
    raise NotImplementedError


class ReadLineHandler(common.AbstractHandler):
    def __init__(self, port_name=None, baud_rate=None, mock=None):
        common.AbstractHandler.__init__(self)
        # keep track of how many timesteps have passed
        self.timesteps = 0

        # If we should mock the serial data
        if port_name is None and baud_rate is None and mock is not None:
            self.should_mock: bool = True
            # There are 3 ways to mock:
            if type(mock) is bool and mock:
                # Just always return a constant array of 500s
                self.mock_fn: typing.Callable = (
                    lambda _t: ["500"] * self.const["n_sensors"]
                )
                l.info("Mocking data using 500s")
            elif callable(mock):
                # The returned values are dictated by a provided callback
                self.mock_fn: typing.Callable = mock
                l.info("Mocking data using custom callable")
            elif type(mock) is str and os.path.exists(mock):
                # The returned values are dictated by the contents of a file on
                # disk
                self._data: pd.DataFrame = pd.read_csv(mock)

                def mock_fn(timestep: int):
                    if timestep > len(self._data):
                        return None
                    else:
                        values = self._data.iloc[timestep].values
                        l.info(f"Returning mock data for timestep {timestep}: {values}")
                        return ",".join([str(i) for i in values[2:]])

                self.mock_fn: typing.Callable = mock_fn
                l.info(f"Mocking data using '{mock}' ({len(self._data)} lines)")
        elif port_name is not None and mock is None:
            self.should_mock: bool = False
            self.port_name: str = port_name
            self.baud_rate: int = (
                common.read_constants()["baud_rate"] if baud_rate is None else baud_rate
            )
            self.port: serial.serialposix.Serial = serial.Serial(
                port=self.port, baudrate=self.baud_rate, timeout=1
            )

            def not_implemented(_t: int):
                raise NotImplementedError

            self.mock_fn: typing.Callable = not_implemented

        else:
            raise ValueError(
                "Either `port_name` should be None OR `mock` should be None"
            )

    def __del__(self):
        if hasattr(self, "port") and self.port.isOpen():
            l.info("Closing ReadLineHandler's port")
            self.port.close()

    def execute(
        self,
        past_handlers: list[common.AbstractHandler],
    ):
        l.info("Executing ReadLineHandler")
        if self.should_mock:
            result = self.mock_fn(self.timesteps)
            if result is None:
                self.control_flow = common.ControlFlow.BREAK
                return
            self.line = "0,0," + result + ","
        else:
            self.line = self.port.readline().decode("utf-8")
        self.timesteps += 1


class ParseLineHandler(common.AbstractHandler):
    """A handler for parsing and validating a line of serial data.

    This handler reads a line of data from a serial port and validates the data
    by checking if the number of values is correct and if the values can be
    converted to integers. The values are clamped between an upper and lower
    bound to eliminate hardware-induced spikes.

    :ivar const: A dictionary containing constants used by the handler.
    :vartype const: ConstantsDict

    :ivar line: The last line of data that was read from the serial port.
    :vartype line: str

    :ivar raw_values: The raw values from the last line of data that was read
    from the serial port.
    :vartype raw_values: list[str]
    """

    def __init__(self):
        common.AbstractHandler.__init__(self)
        self.samples: pd.DataFrame = pd.DataFrame(
            columns=["datetime"] + list(self.const["sensors"].values())
        )

    def execute(
        self,
        past_handlers: list[common.AbstractHandler],
    ):
        """Validates and parses a line of data from the serial port.

        This method reads a line of data from a serial port and validates the
        data by checking if the number of values is correct and if the values
        can be converted to integers. The values are clamped between an upper
        and lower bound to eliminate hardware-induced spikes.

        :param past_handlers: A list of the previous handlers that have
        executed.
        :type past_handlers: list[common.AbstractHandler]
        """
        l.info("Executing ParseLineHandler")
        read_line_handler: ReadLineHandler = next(
            h for h in past_handlers if type(h) is ReadLineHandler
        )
        self.line = read_line_handler.line
        # Comments starting with `#` act as heartbeats
        if self.line.startswith("#"):
            l.info(f"Found comment in serial stream: {self.line}")
            self.control_flow = common.ControlFlow.BREAK
            return
        self.raw_values: list[str] = self.line.strip().split(",")[:-1]

        # If there are no values, then break. This is a common hardware bug
        if len(self.raw_values) == 0:
            l.warn("No values found from serial connection. Try restarting the device.")
            self.control_flow = common.ControlFlow.BREAK
            return

        # If there aren't the correct number of values, then break and retry.
        if len(self.raw_values) != self.const["n_sensors"] + 2:
            self.control_flow = common.ControlFlow.BREAK
            return

        # Try to convert all the values into integers
        try:
            sample = [int(val) for val in self.raw_values[2:]]
        except ValueError as e:
            l.warn(f"Value Error: {e}, {self.raw_values=}")
            self.control_flow = common.ControlFlow.BREAK
            return

        # Clamp the values to be between an upper and lower bound. This
        # eliminates some hardware-induced spikes
        upper = self.const["sensor_bounds"]["upper"]
        lower = self.const["sensor_bounds"]["lower"]
        # Calculate the latest sample, and append it onto the list of all
        # previous samples
        sample = [min(upper, max(lower, val)) for val in sample]
        sensors = list(self.const["sensors"].values())
        self.samples = pd.concat(
            (
                self.samples,
                pd.DataFrame(
                    {"datetime": [datetime.datetime.now()]}
                    | {f: [s] for f, s in zip(sensors, sample)}
                ),
            ),
            ignore_index=True,
        )


def execute_handlers(handlers: list[common.AbstractHandler]):
    """Stream data from a serial port and execute a list of handlers on the data.

    This function opens a serial port with the specified port and baudrate, and
    continuously reads data from it. For each line of data that is read, the
    list of handlers is executed in order. Each handler is provided with the
    data and the list of previously executed handlers, which can be used to
    modify the behavior of subsequent handlers. If a handler encounters an
    error and needs to break the control flow of the handler chain, it can set
    its control_flow attribute to common.ControlFlow.BREAK.

    Args:
        handlers (list[common.AbstractHandler]): A list of handler objects to execute
            on the data.
        port (str): The name of the serial port to connect to.
        baudrate (int): The baudrate to use for the serial connection. Defaults
            to the value defined in the `constants.yaml` file.

    Returns:
        None
    """

    while True:
        for i in range(len(handlers)):
            # Execute the handler, and provide context by passing in
            # all previously executed handlers
            handlers[i].execute(handlers[:i])
            # If the handler encountered an error, then break this
            # iteration (to start the next iteration)
            if handlers[i].control_flow == common.ControlFlow.BREAK:
                handlers[i].control_flow = common.ControlFlow.CONTINUE
                break
