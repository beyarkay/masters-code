"""Takes care of reading in data from the serial port stream and converts it to
numpy arrays. Also reads data in from disk.
"""
# TODO these handlers should probably be in the correct files. ie SaveHandler -> save.py

import logging as l
import serial
import pandas as pd
import typing
import os
import datetime
import common


def read_data(directory: str = "./gesture_data/train/") -> pd.DataFrame:
    """Concatenates the CSVs in `directory` into one pandas DataFrame.

    WARN: Used to be called parse_csvs

    :param directory: The directory to search for CSV files. Defaults to
        "../gesture_data/train".
    :type directory: str

    :returns: A concatenated Pandas DataFrame containing the data read from all
        the CSV files in the directory.
    :rtype: any

    :raises FileNotFoundError: If the specified directory does not exist.
    """
    fingers: dict[int, str] = common.read_constants().get("fingers")

    dfs = []
    for path in [p for p in os.listdir(directory) if p.endswith(".csv")]:
        dfs.append(
            pd.read_csv(
                directory + path,
                names=["datetime", "gesture"] + list(fingers.values()),
                parse_dates=["datetime"],
            )
        )
    return pd.concat(dfs)


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
                self.mock_fn: typing.Callable = (
                    lambda t: None
                    if t > len(self._data)
                    else ",".join([str(i) for i in self._data.iloc[t].values[2:]])
                )
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
            columns=["datetime"] + list(self.const["fingers"].values())
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
        fingers = list(self.const["fingers"].values())
        self.samples = pd.concat(
            (
                self.samples,
                pd.DataFrame(
                    {"datetime": [datetime.datetime.now()]}
                    | {f: [s] for f, s in zip(fingers, sample)}
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
