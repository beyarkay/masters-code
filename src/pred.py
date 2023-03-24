"""Takes care of making predictions, given a model that's been trained

This module should also take care of pre-processing the given data into a
format that's appropriate for the model.
"""
import models
import read
import common
import logging as l


class PredictGestureHandler(common.AbstractHandler):
    """Use a model to predict the gesture from the latest observation."""

    def __init__(self, model: models.TemplateClassifier):
        common.AbstractHandler.__init__(self)
        self.model: models.TemplateClassifier = model

    def execute(
        self,
        past_handlers: list[common.AbstractHandler],
    ):

        l.info("Executing PredictGestureHandler")
        parse_line_handler: read.ParseLineHandler = next(
            h for h in past_handlers if type(h) is read.ParseLineHandler
        )
        if len(parse_line_handler.samples) < self.model.config["n_timesteps"]:
            self.control_flow = common.ControlFlow.BREAK
            n_samples = len(parse_line_handler.samples)
            n_timesteps = self.model.config["n_timesteps"]
            l.warn(f"Not enough samples recorded ({n_samples} < {n_timesteps})")
            return
        sample = parse_line_handler.samples.tail(self.model.config["n_timesteps"])
        const: common.ConstantsDict = common.read_constants()
        sample = sample[const["sensors"].values()].values
        self.prediction = self.model.predict(sample)
