"""Visualises raw numpy data and model predictions.

This should include functions for visualising results via a live GUI, as well
as for visualising results via the CLI.
"""

import logging as l
import common
import pred


class StdOutHandler(common.AbstractHandler):
    """Output various information to stdout for debugging purposes."""

    def execute(
        self,
        past_handlers: list[common.AbstractHandler],
    ):
        pred_handler: pred.PredictGestureHandler = next(
            h for h in past_handlers if type(h) is pred.PredictGestureHandler
        )
        l.info("prediction: " + pred_handler.prediction)
