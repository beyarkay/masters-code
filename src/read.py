"""Takes care of reading in data from the serial port stream and converts it to
numpy arrays. Also reads data in from disk.
"""
import datetime
import logging as l
import os
import pickle
import sys
from typing import Optional, NewType, Callable

import common
import numpy as np
import pandas as pd
import serial
from serial.tools.list_ports import comports


def read_model(directory: str):
    with open(f"{directory}/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


def read_data(
    directory: str = "./gesture_data/train/", offsets: Optional[str] = None
) -> pd.DataFrame:
    """Reads in CSV files from a directory and returns a concatenated Pandas DataFrame.

    Args:
        directory (str, optional): Path to the directory containing the CSV
        files. Defaults to "./gesture_data/train/".

    Returns:
        pd.DataFrame: A concatenated Pandas DataFrame with columns: "datetime",
        "gesture", "file", "finger", "orientation" and additional columns
        representing finger data.
    """
    if offsets is not None:
        offsets = pd.read_csv(offsets)
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
    # Select relevant columns and reorder them
    together = together[["datetime", "gesture", "file"] + sensors]
    df = together.sort_values("datetime").reset_index(drop=True)

    # If offsets have been specified, then read in the offsets and apply them
    # to the DF
    if offsets is not None:
        offsets = pd.read_csv("offsets.csv")
        # Set the new index to be old index + offset
        offsets["new_idx"] = offsets["idx"] + offsets["offset"]
        # Set the new gesture to be the re-indexed old gestures
        offsets["new_gesture"] = df["gesture"].iloc[offsets["idx"].values].values
        offsets = offsets.set_index("new_idx")
        # Keep track of the unaligned gesture, just in case
        df["unaligned_gesture"] = np.NaN
        df["unaligned_gesture"] = df["gesture"].values
        # Set all the gestures to be 0255 by default
        df["gesture"] = "gesture0255"
        # And set only the correct new gestures
        df["gesture"].iloc[offsets.index] = offsets["new_gesture"]
        column_order = ["datetime", "gesture"] + \
            sensors + ["file", "unaligned_gesture"]
    else:
        column_order = ["datetime", "gesture"] + sensors + ["file"]
    return df[column_order]


SerialPort = NewType("SerialPort", str)
BaudRate = NewType("BaudRate", int)

# TODO this should be a rewrite of `ml_utils.get_serial_port`.


def find_port() -> Optional[tuple[SerialPort, BaudRate]]:
    """Look at the open serial ports and return one if it's correctly
    formatted, otherwise exit with status code 1.

    Only considers serial ports starting with `/dev/cu.usbmodem` and will offer
    the user a choice if more than one port is available."""
    # Read in all the available ports starting with `/dev/cu.usbmodem`
    ports = [p.device for p in comports(
    ) if p.device.startswith("/dev/cu.usbmodem")]
    port = ""
    print(f"Looking through ports {ports}")
    if len(ports) >= 1:
        # check that the available ports are actually communicating
        filtered_ports = []
        for i, port in enumerate(ports):
            try:
                with serial.Serial(
                    port=port, baudrate=19_200, timeout=1
                ) as serial_port:
                    if serial_port.isOpen():
                        # flush the port so the line we read will definitely
                        # start from the beginning of the line
                        line = serial_port.readline().decode("utf-8").strip()
                        if line and "#" not in line:
                            print(
                                f"Port {port} is returning non empty non-comment lines"
                            )
                            filtered_ports.append(port)
            except Exception as e:
                continue
        if len(filtered_ports) != 1:
            for i, port in filtered_ports:
                print(f"[{i}]: {port}")
            idx = int(
                input(f"Please choose a port index [0..{len(ports)-1}]: "))
            port = ports[idx]
        elif len(filtered_ports) == 0:
            print("No ports beginning with `/dev/cu.usbmodem` found")
            return None
        else:
            port = filtered_ports[0]
    elif len(ports) == 0:
        # If there are no ports available, exit with status code 1
        print("No ports beginning with `/dev/cu.usbmodem` found")
        return None
    # Finally, return the port
    return SerialPort(port), BaudRate(19_200)


class ReadLineHandler(common.AbstractHandler):
    def __init__(
        self,
        port_name: Optional[SerialPort] = None,
        baud_rate: Optional[BaudRate] = None,
        mock=None,
    ):
        common.AbstractHandler.__init__(self)
        # keep track of how many timesteps have passed
        self.timesteps = 0

        # If we should mock the serial data
        if port_name is None and baud_rate is None and mock is not None:
            self.should_mock: bool = True
            # There are 3 ways to mock:
            if type(mock) is bool and mock:
                # Just always return a constant array of 500s
                self.mock_fn: Callable = lambda _t: (
                    ["500"] * self.const["n_sensors"],
                    None,
                )
                l.info("Mocking data using 500s")
            elif callable(mock):
                # The returned values are dictated by a provided callback
                self.mock_fn: Callable = mock
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
                        truth: str = values[1]
                        l.info(
                            f"Returning mock data for timestep {timestep}: {values}")
                        return (",".join([str(i) for i in values[2:]]), truth)

                self.mock_fn: Callable = mock_fn
                l.info(
                    f"Mocking data using '{mock}' ({len(self._data)} lines)")
        elif port_name is not None and mock is None:
            # Read the serial data live from the serial port
            self.should_mock: bool = False
            self.port_name: str = port_name
            self.baud_rate: int = (
                common.read_constants()[
                    "baud_rate"] if baud_rate is None else baud_rate
            )
            self.port: serial.serialposix.Serial = serial.Serial(
                port=self.port_name, baudrate=self.baud_rate, timeout=1
            )

            def not_implemented(_t: int):
                # TODO this should read data from the serial port
                # TODO also have a checker that ensures a decent signal is
                # coming from each sensor, and errors helpfully if that's not
                # the case
                raise NotImplementedError

            self.mock_fn: Callable = not_implemented

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
        self.truth: Optional[str] = None
        if self.should_mock:
            result, truth = self.mock_fn(self.timesteps)
            if result is None:
                self.control_flow = common.ControlFlow.BREAK
                return
            self.line = "0,0," + result + ","
            self.truth = truth
        else:
            self.line = self.port.readline().decode("utf-8")
            self.truth = None
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
            # print(f"Executing handlers[{i}]: {handlers[i]}")
            # Execute the handler, and provide context by passing in
            # all previously executed handlers
            handlers[i].execute(handlers[:i])
            # If the handler encountered an error, then break this
            # iteration (to start the next iteration)
            if handlers[i].control_flow == common.ControlFlow.BREAK:
                handlers[i].control_flow = common.ControlFlow.CONTINUE
                break
