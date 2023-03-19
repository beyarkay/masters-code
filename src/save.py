"""Takes care of saving numpy arrays of data to disk.
"""

import common


class SaveHandler(common.AbstractHandler):
    """Save the latest sample to disc."""

    def execute(
        self,
        past_handlers: list[common.AbstractHandler],
    ):
        raise NotImplementedError
