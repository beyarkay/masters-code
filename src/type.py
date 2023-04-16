"""Takes care of converting gesture predictions to keystrokes.
"""
import common
import logging as l


class TypeGestureHandler(common.AbstractHandler):
    """Take in a prediction and convert it to a keystroke

    This uses the predicting model's config.
    """

    def execute(
        self,
        past_handlers: list[common.AbstractHandler],
    ):
        l.info("Executing ", self.__module__)
        # TODO implement this
        raise NotImplementedError

        # predict_handler: pred.PredictGestureHandler = next(
        #     h for h in past_handlers if type(h) is pred.PredictGestureHandler
        # )
        # _prediction = predict_handler.prediction
