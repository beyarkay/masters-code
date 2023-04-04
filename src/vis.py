"""Visualises raw numpy data and model predictions.

This should include functions for visualising results via a live GUI, as well
as for visualising results via the CLI.
"""

import logging as l
import common
import pred
import read


class StdOutHandler(common.AbstractHandler):
    """Output various information to stdout for debugging purposes."""

    def execute(
        self,
        past_handlers: list[common.AbstractHandler],
    ):
        read_line_handler: read.ReadLineHandler = next(
            h for h in past_handlers if type(h) is read.ReadLineHandler
        )
        truth: str = read_line_handler.truth
        pred_handler: pred.PredictGestureHandler = next(
            h for h in past_handlers if type(h) is pred.PredictGestureHandler
        )
        l.info(f"Prediction: {str(pred_handler.prediction): <20} Truth: {truth: <20}")
        print(f"Prediction: {str(pred_handler.prediction): <20} Truth: {truth: <20}")
