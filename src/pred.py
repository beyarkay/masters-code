"""Takes care of making predictions, given a model that's been trained

This module should also take care of pre-processing the given data into a
format that's appropriate for the model.
"""
import read
import common
import logging as l
import numpy as np
from autocorrect import Speller
import yaml


class PredictGestureHandler(common.AbstractHandler):
    """Use a model to predict the gesture from the latest observation."""

    def __init__(self, clf):
        common.AbstractHandler.__init__(self)
        self.clf = clf

    def execute(
        self,
        past_handlers: list[common.AbstractHandler],
    ):
        assert self.clf.config is not None

        l.info("Executing PredictGestureHandler")
        parse_line_handler: read.ParseLineHandler = next(
            h for h in past_handlers if type(h) is read.ParseLineHandler
        )
        n_timesteps = self.clf.config['preprocessing']['n_timesteps']
        if len(parse_line_handler.samples) < n_timesteps:
            if len(parse_line_handler.samples) == 1:
                print("Waiting for enough history before making prediction...")
            self.control_flow = common.ControlFlow.BREAK
            n_samples = len(parse_line_handler.samples)
            return
        sample = parse_line_handler.samples.tail(n_timesteps)
        const: common.ConstantsDict = common.read_constants()
        sample = sample[const["sensors"].values()].values
        processed_sample = sample.astype(
            np.float32).reshape((1, *sample.shape))

        self.prediction = self.clf.predict(processed_sample)[0]
        self.stdout = f" Prediction: {self.prediction}"


class MapToKeystrokeHandler(common.AbstractHandler):
    """Use the gesture to keystroke mapping to convert from predictions to
    keystrokes."""

    def __init__(self) -> None:
        common.AbstractHandler.__init__(self)
        with open("gesture_data/gesture_info.yaml", 'r') as f:
            self.g2k = yaml.safe_load(f)['gestures']

    def execute(
        self,
        past_handlers: list[common.AbstractHandler],
    ) -> None:
        print("Executing MapToKeystrokeHandler")
        prediction_handler: PredictGestureHandler = next(
            h for h in past_handlers if type(h) is PredictGestureHandler
        )
        if not hasattr(prediction_handler, "prediction"):
            print("WARN: prediction_handler doesn't have any predictions")
            return

        gesture = 'unknown'
        keystroke = 'unknown'
        print(
            f'prediction: {prediction_handler.prediction}, gesture: {gesture}, keystroke: {keystroke}')  # noqa: E501

        raise NotImplementedError(
            "MapToKeystrokeHandler has not been implemented yet")


class SpellCheckHandler(common.AbstractHandler):
    """Use a spell checker to go back and correct typos"""

    def __init__(self) -> None:
        common.AbstractHandler.__init__(self)
        self.speller = Speller(fast=True)

    def execute(
        self,
        past_handlers: list[common.AbstractHandler],
    ) -> None:
        print("Executing SpellCheckHandler")
        # TODO: Keep a log of the words which have been emitted so that we can
        # run the spell checker on them.
        raise NotImplementedError(
            "Spell Check handler has not been implemented yet")
