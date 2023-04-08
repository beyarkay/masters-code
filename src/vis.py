"""Visualises raw numpy data and model predictions.

This should include functions for visualising results via a live GUI, as well
as for visualising results via the CLI.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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


def plot_conf_mats(model, Xs, ys, titles=None):
    """Plots the confusion matrices for the given model with the given data.

    The confusion matrices are log10 transformed to highlight low-occuring
    misclassifications."""
    conf_mats = []
    for X, y in zip(Xs, ys):
        conf_mats.append(model.confusion_matrix(y, X_to_pred=X).numpy())
    vmax = np.log10(np.max(np.array(conf_mats).flatten()))
    fig, axs = plt.subplots(1, len(conf_mats), figsize=(5 * len(conf_mats), 4))

    for cm, ax, title in zip(conf_mats, axs, titles):
        sns.heatmap(np.log10(cm), ax=ax, vmin=0, vmax=vmax, square=True)
        ax.set(
            xlabel="Predicted Gesture",
            ylabel="Actual Gesture",
        )
        if titles is not None:
            ax.set_title(title)
    return fig, axs
