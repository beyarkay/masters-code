"""Visualises raw numpy data and model predictions.

This should include functions for visualising results via a live GUI, as well
as for visualising results via the CLI.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging as l
import common
import pred
import read
from matplotlib.colors import LogNorm, Normalize


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
        conf_mats.append(model.confusion_matrix(y, X_to_pred=X))
    fig, axs = plt.subplots(1, len(conf_mats), figsize=(5 * len(conf_mats), 4))

    for cm, ax, title in zip(conf_mats, axs, titles):
        sns.heatmap(
            cm,
            ax=ax,
            vmin=0,
            vmax=cm.max(),
            square=True,
            norm=LogNorm(),
        )
        ax.set(
            xlabel="Predicted Gesture",
            ylabel="Actual Gesture",
        )
        if titles is not None:
            ax.set_title(title)
    return fig, axs


def plot_distributions(y_val, y_pred):
    conf_mat = tf.math.confusion_matrix(y_val, y_pred, num_classes=51).numpy()

    cols = conf_mat.sum(axis=0)
    rows = conf_mat.sum(axis=1)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].barh(range(len(rows)), np.log10(rows))
    axs[0].set_title("Distribution of true labels")

    axs[1].barh(range(len(cols)), np.log10(cols))
    axs[1].set_title("Distribution of predicted labels")

    for ax in axs[:-1]:
        ax.set(
            yticks=range(0, 51, 5),
            yticklabels=range(0, 51, 5),
            ylabel="Gesture",
            xlabel="$\log_{10}$(distribution)",
        )

    for x, y, t in zip(np.log10(rows), np.log10(cols), [str(i) for i in range(51)]):
        if np.isfinite(x) and np.isfinite(y):
            axs[2].text(x, y, t)

    axs[2].plot([0, 10], [0, 10])
    axs[2].set(
        xlim=(0, np.max(np.log10(rows))),
        ylim=(0, np.max(np.log10(cols))),
        title="$\log_{10}$(predicted) vs $\log_{10}$(actual)",
        xlabel="$\log_{10}$(predicted)",
        ylabel="$\log_{10}$(actual)",
    )
    plt.tight_layout()
    return fig, axs


def plot_cusum(dictionary, axs=None):
    """Expects a dataframe like the one output by `_cusum`"""
    index = np.linspace(0, len(dictionary["x"]) - 1, len(dictionary["x"]))
    if axs is None:
        fig, axs = plt.subplots(2, 1, figsize=(10, 7))
    axs[0].plot(dictionary["x"], color="tab:grey")
    axs[0].plot(dictionary["target"], color="tab:grey")

    std_devs = dictionary["allowed_std_devs"]
    axs[0].fill_between(
        index,
        dictionary["lower_limit"],
        dictionary["upper_limit"],
        alpha=0.1,
        color="grey",
        label=f"Expected region ({std_devs} std devs)",
    )
    axs[0].fill_between(
        x=index,
        y1=dictionary["lower_limit"],
        y2=dictionary["x"],
        where=(dictionary["lower_limit"] > dictionary["x"]),
        alpha=0.1,
        color="tab:orange",
        label="Negative contributions",
    )

    axs[0].fill_between(
        x=index,
        y1=dictionary["upper_limit"],
        y2=dictionary["x"],
        where=(dictionary["upper_limit"] < dictionary["x"]),
        alpha=0.1,
        color="tab:blue",
        label="Positive contributions",
    )

    axs[0].legend()

    axs[1].plot(dictionary["cusum_pos"], label="CuSUM statistic (positive)")
    axs[1].plot(dictionary["cusum_neg"], label="CuSUM statistic (negative)")

    axs[1].scatter(
        index[dictionary["too_low"] > 0],
        dictionary["cusum_neg"][dictionary["too_low"] > 0],
        color="tab:orange",
        s=15,
        zorder=10,
        label="Out of bounds datapoints (too low)",
    )

    axs[1].scatter(
        index[dictionary["too_high"] > 0],
        dictionary["cusum_pos"][dictionary["too_high"] > 0],
        color="tab:blue",
        s=15,
        zorder=10,
        label="Out of bounds datapoints (too high)",
    )
    axs[1].legend()
    plt.suptitle("CuSUM diagnostics")
    axs[0].set_title("Raw data")
    axs[1].set_title("+'ve and -'ve CuSUM statistics")
    plt.tight_layout()

    return axs
