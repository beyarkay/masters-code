from enum import Enum
import typing


class ConstantsDict(typing.TypedDict):
    fingers: dict[int, str]
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
