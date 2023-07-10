"""Takes care of saving numpy arrays of data to disk.
"""

import datetime
import os
import read
import common


class SaveHandler(common.AbstractHandler):
    """Save the latest sample to disc."""

    def __init__(self, path: str, fail_if_exists=True):
        super().__init__()
        self.path = path
        self.fail_if_exists = fail_if_exists
        assert not (
            (self.fail_if_exists) and os.path.exists(self.path)
        ), f"Failing to save data because path `{self.path}` already exists"

    def execute(
        self,
        past_handlers: list[common.AbstractHandler],
    ):
        parse_line_handler: read.ParseLineHandler = common.first_or_fail(
            [h for h in past_handlers if type(h) is read.ParseLineHandler]
        )
        now = datetime.datetime.now().isoformat(sep="T")

        line_to_save = f"{now},None,{','.join(parse_line_handler.raw_values[2:])}\n"
        with open(self.path, "a") as f:
            f.writelines([line_to_save])
