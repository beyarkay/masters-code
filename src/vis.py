"""Visualises raw numpy data and model predictions.

This should include functions for visualising results via a live GUI, as well
as for visualising results via the CLI.
"""

from typing import Optional
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import common
import read
from matplotlib.colors import LogNorm
import colorama as C
import datetime


class InsertLabelHandler(common.AbstractHandler):
    """Output various information to stdout for debugging purposes."""

    def __init__(self, labels=None):
        super().__init__()
        self.labels = [] if labels is None else labels
        self.labels_index = 0
        self.can_increment = True

    def execute(
        self,
        past_handlers: list[common.AbstractHandler],
    ):
        read_line_handler: read.ReadLineHandler = common.first_or_fail(
            [h for h in past_handlers if type(h) is read.ReadLineHandler]
        )
        # Check that we're not reading data in from anywhere but serial input
        assert not read_line_handler.should_mock

        # Get the current time
        ms = datetime.datetime.now().microsecond / 1000
        s = datetime.datetime.now().second
        # We want to repeat the label printing every `period_ms`
        period_ms = 500
        div, mod = divmod(s * 1000 + ms, period_ms)
        remaining_ms = int(period_ms - mod)
        # If it's time to repeat
        if remaining_ms < 35:
            # Only actually take any action on the first iteration of each cycle
            if self.can_increment:
                self.can_increment = False
                # Print out a special message
                label = self.labels[self.labels_index]
                self.truth = label.replace("g0", "gesture0")
                self.stdout = f"{C.Fore.BLACK}{C.Back.RED}Label: {label}"
                self.labels_index = (self.labels_index + 1) % len(self.labels)
        else:
            self.can_increment = True
            self.truth = "gesture0255"
            pbar = "#" * (remaining_ms//50)
            self.stdout = f"{C.Style.BRIGHT}{self.labels[self.labels_index]} {C.Style.RESET_ALL}{C.Style.DIM}{remaining_ms}{C.Style.RESET_ALL}: {pbar: <20}"


class StdOutHandler(common.AbstractHandler):
    """Output various information to stdout for debugging purposes."""

    def execute(
        self,
        past_handlers: list[common.AbstractHandler],
    ):
        read_line_handler: read.ReadLineHandler = common.first_or_fail(
            [h for h in past_handlers if type(h) is read.ReadLineHandler]
        )
        truth: Optional[str] = read_line_handler.truth

        parse_line_handler: read.ParseLineHandler = common.first_or_fail(
            [h for h in past_handlers if type(h) is read.ParseLineHandler]
        )

        stdouts = ';'.join(
            str(h.stdout) + C.Style.RESET_ALL for h in past_handlers if h.stdout)

        def colour_map(low, high):
            """Given a lower bound and an upper bound, return a colour-mapping
            function for values in that range."""
            colours = [
                C.Fore.YELLOW,
                C.Fore.RED,
                C.Fore.MAGENTA,
                C.Fore.BLUE,
                C.Fore.CYAN,
                C.Fore.GREEN,
            ]

            def colour_mapper(x: int):
                # Clamp `x` to be in the range (low, high)
                oob_str = ""
                if x > high:
                    oob_str = C.Style.DIM
                    x = high
                elif x < low:
                    oob_str = C.Style.DIM
                    x = low
                # Normalise the provided value
                normed = (x - low) / (high - low)
                # Return the ASCII string required to colour the output
                return oob_str + colours[int(normed * (len(colours) - 1))]

            return colour_mapper

        bars = "▁▂▃▄▅▆▇█"
        low = 400
        high = 800
        coloured_values = []
        coloured_bars = []
        for value in parse_line_handler.raw_values:
            clamped = min(max(int(value), low), high)
            bar = bars[int(((clamped - low) / (high - low)) * (len(bars)-1))]
            colour = colour_map(low, high)(int(value))
            coloured_bars.append(f"{colour}{bar}")
            coloured_values.append(f"{colour}{value}")

        now = str(datetime.datetime.now())[:-3]
        chunked = ' '.join(
            ''.join(coloured_bars[i:i + 3]) for i in range(0, len(coloured_bars), 3)
        )

        print(
            f"[{now}] {chunked}{C.Style.RESET_ALL} | {stdouts}{C.Style.RESET_ALL}"
        )


def conf_mat(cm, ax=None, norm=0):
    cm_normed = cm if norm is None else cm / cm.sum(axis=norm)
    p = sns.heatmap(
        cm_normed,
        annot=cm if cm.shape[0] <= 5 else False,
        fmt='d',
        square=True,
        mask=(cm == 0),
        cmap='viridis',
        ax=ax,
        vmax=1 if np.all(cm_normed <= 1) else None,
        vmin=0 if np.all(cm_normed <= 1) else None,
        xticklabels=1 if cm.shape[0] <= 10 else 5,
        yticklabels=1 if cm.shape[0] <= 10 else 5,
    )
    if ax is not None:
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Ground Truth')
    else:
        plt.xlabel('Predicted')
        plt.ylabel('Ground Truth')
    return p


def plot_conf_mats(model, Xs, ys, titles):
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
            cmap="turbo_r",
        )
        ax.set(
            xlabel="Predicted Gesture",
            ylabel="Actual Gesture",
        )
        if titles is not None:
            ax.set_title(title)
    return fig, axs


def plot_observations(model, index_slice):
    max(0, 1)
    assert model.is_fitted_, "Model must be fitted before observations are plotted"
    raise NotImplementedError("Plot observations is not implemented.")


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

    # NOTE: this is called a CalibrationDisplay
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


def cmp_ts(time_series, span=None, axs=None, color='grey', lw=0.5, ylim=(None, None)):
    """Compare a list of time series datasets.
    Each of the time series in `time_series` must have the same shape,
    and that shape must be (_, 30).
    """
    assert all(ts.shape == time_series[0].shape for ts in time_series)
    assert all(len(ts.shape) == 2 for ts in time_series)
    assert all(ts.shape[1] == 30 for ts in time_series)
    const = common.read_constants('../src/constants.yaml')
    if axs is None:
        _fig, axs = plt.subplots(
            5,
            6,
            figsize=(10, 9)
        )
    assert axs.shape == (5, 6)
    ylim = (
        ylim[0] if ylim[0] is not None else np.array(time_series).min() * 0.9,
        ylim[1] if ylim[1] is not None else np.array(time_series).max() * 1.1
    )

    for i, ax in enumerate(axs.flatten()):
        ax.set(
            ylim=ylim,
        )
        ax.set_title(list(const['sensors'].values())[i], y=1.0, pad=-14)

        if span is not None:
            ax.fill_between(
                range(*span),
                ylim[0],
                ylim[1],
                alpha=0.1,
                color='grey'
            )
        if i % 6 != 0:
            ax.set_yticks([])
        if i < 24:
            ax.set_xticks([])

        for ts in time_series:
            ax.plot(
                ts[:, i],
                lw=lw,
                c=color,
                alpha=.5,
            )
    plt.subplots_adjust(wspace=0, hspace=0)
    return axs
