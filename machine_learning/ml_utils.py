from keras import layers
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from typing import Callable, List, Any, Tuple
from wandb.keras import WandbCallback
import datetime
import keras.backend as K
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import tensorflow as tf
import yaml

keras.utils.set_random_seed(42)


FINGERS = [
    "left-5-x",
    "left-5-y",
    "left-5-z",
    "left-4-x",
    "left-4-y",
    "left-4-z",
    "left-3-x",
    "left-3-y",
    "left-3-z",
    "left-2-x",
    "left-2-y",
    "left-2-z",
    "left-1-x",
    "left-1-y",
    "left-1-z",
    "right-1-x",
    "right-1-y",
    "right-1-z",
    "right-2-x",
    "right-2-y",
    "right-2-z",
    "right-3-x",
    "right-3-y",
    "right-3-z",
    "right-4-x",
    "right-4-y",
    "right-4-z",
    "right-5-x",
    "right-5-y",
    "right-5-z",
]


def make_batches(X, y, t, window_size=10, window_skip=1):
    assert window_skip == 1, "window_skip is not supported for values other than 1"
    ends = np.array(range(window_size, len(y)))
    starts = ends - window_size
    batched_X = np.empty((ends.shape[0], window_size, X.shape[1]))
    batched_y = np.empty((ends.shape[0],), dtype="object")
    for i, (start, end) in enumerate(zip(starts, ends)):
        # Don't add the X,y pair if it would go over a time boundary
        if any(np.diff(t[start:end]) > np.timedelta64(5, "s")):
            continue
        batched_X[i] = X[start:end]
        batched_y[i] = y[end]
    batched_X = batched_X[pd.notna(batched_y)]
    batched_y = batched_y[pd.notna(batched_y)]
    return batched_X, batched_y


def gestures_and_indices(y):
    labels = sorted(np.unique(y))
    g2i_dict = {g: i for i, g in enumerate(labels)}
    i2g_dict = {i: g for i, g in enumerate(labels)}

    def g2i(g):
        not_list = type(g) not in [list, np.ndarray]
        if not_list:
            g = [g]
        result = np.array([g2i_dict.get(gi, gi) for gi in g])
        return result[0] if not_list else result

    def i2g(i):
        not_list = type(i) not in [list, np.ndarray]
        if not_list:
            i = [i]
        result = np.array([i2g_dict.get(ii, ii) for ii in i])
        return result[0] if not_list else result

    return g2i, i2g


def one_hot_and_back(y_all):
    return (
        lambda y: tf.one_hot(y, len(np.unique(y_all))),
        lambda onehot: tf.argmax(one_hot, axis=1),
    )


def conf_mat(y_true, y_pred, i2g, perc=None, hide_zeros=True, ax=None, cbar=True):
    assert perc in ["cols", "rows", "both", None]
    #     y_pred = np.argmax(model.predict(X, verbose=0), axis=1)
    #     y_true = y
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred).numpy()

    axis = None
    if perc == "cols":
        axis = 0
        confusion_mtx = confusion_mtx / confusion_mtx.sum(axis=axis) * 100
    elif perc == "rows":
        axis = 1
        confusion_mtx = (confusion_mtx.T / confusion_mtx.sum(axis=axis) * 100).T
    elif perc == "both":
        axis = (0, 1)
        confusion_mtx = confusion_mtx / confusion_mtx.sum() * 100
    elif perc is None:
        axis = None

    conf_mat_is_zero = np.where(confusion_mtx == 0)
    conf_mat_not_zero = np.where(confusion_mtx != 0)
    # confusion_mtx = np.round(confusion_mtx)

    to_print = np.empty(confusion_mtx.shape, dtype="object")
    to_print[conf_mat_is_zero] = ""
    to_print[conf_mat_not_zero] = (
        confusion_mtx[conf_mat_not_zero].astype(int).astype(str)
    )

    labels = [i2g(i).replace("gesture0", "g") for i in range(confusion_mtx.shape[0])]

    sns.heatmap(
        confusion_mtx,
        annot=to_print,
        fmt="",
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        cbar=cbar,
        vmin=confusion_mtx.min(),
        vmax=confusion_mtx[:-1, :-1].max(),
        ax=ax,
    )
    if ax is None:
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
    else:
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
    return confusion_mtx


def plot_timeseries(X, y, t=None, per="dimension", axs=None, draw_text=True):
    # Make sure the given dataset is correctly formatted
    FINGERS = [
        "left-little-finger-x",
        "left-little-finger-y",
        "left-little-finger-z",
        "left-ring-finger-x",
        "left-ring-finger-y",
        "left-ring-finger-z",
        "left-middle-finger-x",
        "left-middle-finger-y",
        "left-middle-finger-z",
        "left-index-finger-x",
        "left-index-finger-y",
        "left-index-finger-z",
        "left-thumb-x",
        "left-thumb-y",
        "left-thumb-z",
        "right-thumb-x",
        "right-thumb-y",
        "right-thumb-z",
        "right-index-finger-x",
        "right-index-finger-y",
        "right-index-finger-z",
        "right-middle-finger-x",
        "right-middle-finger-y",
        "right-middle-finger-z",
        "right-ring-finger-x",
        "right-ring-finger-y",
        "right-ring-finger-z",
        "right-little-finger-x",
        "right-little-finger-y",
        "right-little-finger-z",
    ]

    assert (
        X.shape[0] == y.shape[0]
    ), f"There must be one y value for each X value, but got {X.shape[0]} y values and {y.shape[0]} X values"
    assert X.shape[1] == len(
        FINGERS
    ), f"{X.shape[1]=} doesn't equal the number of finger labels ({len(FINGERS)})"
    assert not np.isnan(
        X
    ).any(), f"Input dataset has {np.isnan(X).sum()} NaN values. Should have 0"

    # If we've got many many points, only show an abridged version of the plot
    abridged = X.shape[0] > 4000 or not draw_text
    # Only create new axs if we're not plotting on existing axs
    if axs is None:
        if per == "dimension":
            nrows, ncols = (3, 1)
        elif per == "finger":
            nrows, ncols = (5, 2)
        _fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 6))
        if len(axs.shape) > 1:
            axs = axs.T.flatten()
    else:
        assert axs.shape in [
            (3,),
            (10,),
        ], f"Given axs shape is {axs.shape}, but must only be (3,) or (10,))"
        per = "dimension" if axs.shape == (3,) else "finger"

    ymin = float("inf")
    ymax = float("-inf")

    max_std = X.std(axis=0).max()
    for d in range(X.shape[1]):
        if per == "dimension":
            ax_idx = d % 3
        elif per == "finger":
            ax_idx = d // 3

        ax = axs[ax_idx]
        data_to_plot = X[:, d]
        ax.plot(
            data_to_plot,
            alpha=np.clip(data_to_plot.std() / max_std, 0.05, 1.0),
            label=FINGERS[d],
            c=None
            if per == "dimension"
            else ("tab:red", "tab:green", "tab:blue")[d % 3],
        )

        # Set the title of each plot
        if per == "dimension":
            ax.set_title(f"{FINGERS[d][-1].replace('-', ' ').title()}")
        elif per == "finger":
            ax.set_title(f"{FINGERS[d][:-2].replace('-', ' ').title()}")

        ymax = max(ymax, X[:, d].max())
        ymin = min(ymin, X[:, d].min())

    # Plot the ticks and legend for each axis
    NUM_LABELS = 30 if per == "dimension" else 30
    TICKS_PER_LABEL = max(1, X.shape[0] // NUM_LABELS)
    for i, ax in enumerate(axs):
        if abridged:
            ax.set_xticks([])
            ax.set_xticklabels([])
        else:
            ax.set_xticks(range(0, X.shape[0], TICKS_PER_LABEL))
            if (per == "dimension" and i != len(axs) - 1) or (
                per == "finger" and i % 5 != 4
            ):
                ax.set_xticklabels([])
            elif t is not None:
                ax.set_xticklabels(t[::TICKS_PER_LABEL], rotation=90)
        if per == "dimension" and draw_text:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))
    # Plot the labels for each timestep and axis
    for dim_idx, ax in enumerate(axs):
        backtrack = 0
        for time in range(X.shape[0]):
            #             if abridged:
            #                 continue
            if y[time] not in ["gesture0255", "g255"] and time != X.shape[0] - 1:
                backtrack += 1
                continue
            elif y[time] in ["gesture0255", "g255"] and backtrack == 0:
                continue
            else:
                ax.fill_betweenx(
                    y=[ymin * 0.9, ymax * 1.1],
                    x1=[time - backtrack - 0.5, time - backtrack - 0.5],
                    x2=[time - 0.5, time - 0.5],
                    color="grey",
                    alpha=0.3,
                )

                txt = y[time - backtrack].replace("gesture0", "g")
                ax.text(
                    time - backtrack / 2 - 0.5,
                    (ymax - ymin) / 2 + ymin,
                    txt,
                    va="baseline",
                    ha="center",
                    rotation=90,
                )
                backtrack = 0
        ax.set_ylim((ymin * 0.9, ymax * 1.1))

    plt.tight_layout()
    return axs


def plot_means(Xs, per="finger"):
    assert Xs.shape[-1] == len(
        FINGERS
    ), f"Xs is of shape {Xs.shape}, not (None, None, {len(FINGERS)})"
    assert len(Xs.shape) == 3, f"Xs should have 3 dimensions, not {len(Xs.shape)}"
    X_mean = Xs.mean(axis=0)
    X_std = Xs.std(axis=0)

    blank_labels = np.array(["g255"] * X_mean.shape[0])

    axs = plot_timeseries(X_mean, blank_labels, per=per)

    ymin = float("inf")
    ymax = float("-inf")
    max_std = X_mean.std(axis=0).max()

    for d in range(X_mean.shape[1]):
        if per == "dimension":
            ax_idx = d % 3
        elif per == "finger":
            ax_idx = d // 3

        ax = axs[ax_idx]

        high = X_mean[:, d] + X_std[:, d]
        low = X_mean[:, d] - X_std[:, d]
        ymin = min(ymin, min(low))
        ymax = max(ymax, max(high))

        kwargs = (
            {}
            if per == "dimension"
            else {"color": ("tab:red", "tab:green", "tab:blue")[d % 3]}
        )
        ax.fill_between(
            range(len(X_mean[:, d])),
            low,
            high,
            alpha=np.clip(X_mean[:, d].std() / (4 * max_std), 0.05, 1.0),
            **kwargs,
        )

    for ax in axs:
        ax.set_ylim((ymin * 0.9, ymax * 1.1))

    plt.tight_layout()
    return axs


def plot_mean_gesture(gesture, window_size=15, per="finger"):
    y_orig = df["gesture"].to_numpy()
    X_orig = df.drop(["datetime", "gesture"], axis=1).to_numpy()
    t_orig = df["datetime"].to_numpy()
    # Get a series which is y_orig, but shifted backwards by one
    y_offset = np.concatenate((["gesture0255"], y_orig[:-1]))
    # Get all the indices where the gesture goes [..., !=gesture, ==gesture, ...]
    indices = np.nonzero(y_orig == gesture)[0]
    # Filter out those indices too close to the starts/finishes for it to be viable
    indices = indices[
        (indices > window_size) & (indices + window_size + 1 < X_orig.shape[0])
    ]

    Xs = np.empty((len(indices), window_size * 2 + 1, X_orig.shape[-1]))

    for i, idx in enumerate(indices):
        window_start = idx - window_size
        window_finsh = idx + window_size + 1
        Xs[i] = X_orig[window_start:window_finsh]

    return plot_means(Xs, per=per)


def parse_csvs(root="../gesture_data/train/"):
    dfs = []
    for path in [p for p in os.listdir(root) if p.endswith(".csv")]:
        dfs.append(
            pd.read_csv(
                root + path,
                names=["datetime", "gesture"] + FINGERS,
                parse_dates=["datetime"],
            )
        )
    df = pd.concat(dfs)
    #     df.datetime = df.datetime.apply(pd.Timestamp)
    return df


class PerClassCallback(keras.callbacks.Callback):
    """A basic wrapper to calculate the sklearn `classification_report` at
    the end of each epoch. Made trickier because `validation_data` isn't
    directly available from Keras."""

    def __init__(self, val_data, i2g):
        super().__init__()
        self.validation_data = val_data
        self.i2g = i2g

    def on_train_begin(self, logs={}):
        self.model.reports = []

    def on_epoch_end(self, batch, logs={}):
        # Only do expensive logging every ~10 epochs
        if len(self.model.reports) > 10 and np.random.random() > 0.1:
            self.model.reports.append(self.model.reports[-1])
            return
        y_pred = np.argmax(
            np.asarray(self.model.predict(self.validation_data[0], verbose=0)), axis=1
        )
        y_true = self.validation_data[1]
        self.model.reports.append(
            classification_report(
                y_true,
                y_pred,
                target_names=self.i2g(np.unique(y_true)),
                output_dict=True,
                zero_division=0,
            )
        )
        return


def compile_and_fit(X_train, y_train, X_valid, y_valid, config, i2g, verbose=1):
    # Normalise against each of the sensor values
    normalizer = layers.Normalization(axis=-2)
    normalizer.adapt(X_train)
    print(X_train.shape)

    dense_layers = []

    for layer_number, num_units in config.get("n_hidden_units").items():
        dense_layers.append(
            layers.Dense(
                units=num_units,
                activation=config["activation_fn"],
            )
        )
        if config.get("dropout_frac", 1.0) < 1.0:
            dense_layers.append(layers.Dropout(config["dropout_frac"]))

    def init_biases(shape, dtype=None):
        assert shape == [
            len(config["class_weight"])
        ], f"Shape {shape} isn't ({len(config['class_weight'])},)"
        inv_freqs = np.array([1 / v for v in config["class_weight"].values()])
        return np.log(inv_freqs)

    dense_kwargs = (
        {"bias_initializer": init_biases}
        if config.get("bias_final_layer", False)
        else {}
    )

    model = tf.keras.Sequential(
        [
            layers.Input(shape=X_train.shape[1:]),
            normalizer,
            layers.Flatten(),
            *dense_layers,
            layers.Dense(
                len(np.unique(y_train)),
                activation=config.get("activation"),
                **dense_kwargs,
            ),
        ]
    )

    # Define an object that calculates per-class metrics
    per_class_callback = PerClassCallback((X_valid, y_valid), i2g)
    # Reduce the learning rate if validation loss doesn't improve
    patience = 5
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=patience
    )
    # Stop training if validation loss doesn't ever improve
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience * 3,
        mode="min",
        restore_best_weights=True,
        verbose=0,
    )

    # Instantiate and compile the model
    if config["optimiser"] == "adam":
        optimiser = keras.optimizers.Adam(learning_rate=config["lr"])
    elif config["optimiser"] == "sgd":
        optimiser = keras.optimizers.SGD(learning_rate=config["lr"])
    elif config["optimiser"] == "rmsprop":
        optimiser = keras.optimizers.RMSprop(learning_rate=config["lr"])
    else:
        print(f"Optimiser {config['optimiser']} unknown")

    model.compile(
        optimizer=optimiser,
        loss=config["loss_fn"],
    )
    kwargs = (
        {"class_weight": config["class_weight"]} if config["use_class_weights"] else {}
    )
    callbacks = [
        per_class_callback,
        early_stopping,
        # reduce_lr,
    ]
    if config.get("use_wandb", False):
        callbacks.append(WandbCallback())

    # Fit the model
    history = model.fit(
        X_train,
        y_train,
        verbose=verbose,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        validation_data=(X_valid, y_valid),
        callbacks=callbacks,
        **kwargs,
    )
    return history, model


def plot_losses(history, show_figs, d, results, trimmed_config, y_true=None):
    _fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(history.history["val_loss"])
    ax.plot(history.history["loss"])

    most_frequent, _count = sorted(
        list(zip(*np.unique(y_true, return_counts=True))), key=lambda x: -x[1]
    )[0]

    y_pred = tf.one_hot(np.full(y_true.shape, most_frequent), len(np.unique(y_true)))
    baseline_scce = (
        keras.losses.sparse_categorical_crossentropy(
            y_true,
            y_pred,
        )
        .numpy()
        .mean()
    )

    ax.plot(
        [baseline_scce] * len(history.history["loss"]),
        c="tab:green",
        label="Dumb",
    )
    ax.scatter([len(history.history["loss"]) - 1], [baseline_scce], s=10, c="tab:green")
    ax.text(
        len(history.history["loss"]) - 1,
        baseline_scce,
        str(np.round(baseline_scce, 4)),
        horizontalalignment="right",
        verticalalignment="bottom",
    )
    plt.title(f"Categorical Cross-Entropy over time")

    ylim = ax.get_ylim()
    ax.set_ylim((0, ylim[1]))
    ax.set_ylabel("Categorical Cross-Entropy")
    ax.set_xlabel("Epoch")

    for key in ("val_loss", "loss"):
        ax.text(
            len(history.history[key]) - 1,
            history.history[key][-1],
            str(np.round(history.history[key][-1], 4)),
            horizontalalignment="right",
            verticalalignment="bottom",
        )
        ax.scatter(
            [len(history.history[key]) - 1],
            [history.history[key][-1]],
            s=10,
        )

    ax.legend(["Validation", "Training", "Baseline"], loc="best")
    plt.tight_layout()
    plt.savefig(f"{d}/metrics.pdf")
    if show_figs:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrices(
    y_valid,
    y_pred_valid,
    y_train,
    y_pred_train,
    results,
    trimmed_config,
    i2g,
    show_figs,
    d,
    cbar=False,
    perc="both",
):
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    _ = conf_mat(y_valid, y_pred_valid, i2g, ax=ax, perc=perc, cbar=cbar)
    valid_f1 = np.round(results["valid_f1"], 4)
    valid_scce = np.round(results["val_loss"], 4)
    ax.set_title(
        f"Validation Set Confusion Matrix\nSupport={len(y_valid)}, $F_1$={valid_f1}, Categorical Cross-Entropy={valid_scce}"
    )
    plt.tight_layout()
    plt.savefig(f"{d}/val_confusion_matrices.pdf")
    if show_figs:
        plt.show()
    else:
        plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    _ = conf_mat(y_train, y_pred_train, i2g, ax=ax, perc=perc, cbar=cbar)
    train_f1 = np.round(results["train_f1"], 4)
    train_scce = np.round(results["loss"], 4)
    ax.set_title(
        f"Training Set Confusion Matrix\nsupport={len(y_train)}, $F_1$={train_f1}, Categorical Cross-Entropy={train_scce}"
    )

    plt.tight_layout()
    plt.savefig(f"{d}/train_confusion_matrices.pdf")
    if show_figs:
        plt.show()
    else:
        plt.close()


def plot_metrics(model, d, show_figs):
    # Plot the Precisions, recalls, and F1s for all classes
    shape = (len(model.reports), len(model.reports[0].keys()) - 3)
    precisions = np.zeros(shape)
    recalls = np.zeros(shape)
    f1s = np.zeros(shape)
    for i, report in enumerate(model.reports):
        filtered = {k: v for k, v in report.items() if k.startswith("gesture")}
        precisions[i] = np.array([v["precision"] for k, v in filtered.items()])
        recalls[i] = np.array([v["recall"] for k, v in filtered.items()])
        f1s[i] = np.array([v["f1-score"] for k, v in filtered.items()])

    labels = list(
        {k: v for k, v in model.reports[0].items() if k.startswith("gesture")}.keys()
    )
    _fig, axs = plt.subplots(3, 1, figsize=(6, 8))

    val_metrics = [
        (precisions, "Precision"),
        (recalls, "Recall"),
        (f1s, "$F_1$ Score"),
    ]

    for ax, (vals, metric) in zip(axs, val_metrics):
        values = vals[:, np.nonzero(np.array(labels) != "gesture0255")[0]]
        mean = np.mean(values, axis=1)
        std = np.std(values, axis=1)
        ax.fill_between(
            x=range(len(mean)),
            y1=mean - std,
            y2=mean + std,
            color="tab:orange",
            alpha=0.1,
            #         lw=2
        )
        ax.plot(
            vals[:, np.nonzero(np.array(labels) != "gesture0255")[0]],
            alpha=0.1,
        )
        ax.plot(mean, c="tab:orange", lw=2)
        ax.plot(
            vals[:, np.nonzero(np.array(labels) == "gesture0255")[0]],
            label="g255",
            c="tab:blue",
            lw=2,
        )
        ax.set_ylim((0, 1))
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"{metric.title()}")
        ax.set_title(f"{metric} for all gesture classes")

    axs[2].legend(
        [
            mpl.lines.Line2D([0], [0], color="tab:blue", lw=4),
            mpl.lines.Line2D([0], [0], color="tab:orange", lw=4),
        ],
        ["gesture0255", "mean$\pm$std.dev. for all other gestures"],
        bbox_to_anchor=(0.6, -0.25),
    )
    plt.tight_layout()
    plt.savefig(f"{d}/precision_recall_f1.pdf")
    if show_figs:
        plt.show()
    else:
        plt.close()


def eval_and_save(
    model,
    X_train,
    y_train,
    X_valid,
    y_valid,
    config,
    i2g,
    history,
    show_figs=False,
    cbar=True,
    make_plots=True,
    perc="both",
):
    print("Saving model")
    d = f'./models/{str(datetime.datetime.now()).replace(" ", "T")}'
    model.save(d)
    with open(f"{d}/config.yaml", "w") as file:
        config["class_weight"] = {int(k): v for k, v in config["class_weight"].items()}
        config["gestures"] = [int(i) for i in config["gestures"]]
        config["i2g"] = {int(i): str(i2g(i)) for i in config["gestures"]}
        config["g2i"] = {v: k for k, v in config["i2g"].items()}
        yaml.dump(config, file)

    keys = ["epochs", "label_expansion", "n_hidden_units", "window_size"]
    trimmed_config = {k: v for k, v in config.items() if k in keys}
    results = {k: v[-1] for k, v in history.history.items()}

    # Collect together the precision/recall/F1 reports and combine them into results
    filtered = {k: v for k, v in model.reports[-1].items() if k.startswith("gesture")}
    merged = [[{f"{k}.{ki}": vi} for ki, vi in v.items()] for k, v in filtered.items()]
    for item in sum(merged, []):
        results.update(item)

    if make_plots:
        print("Making plots")
        plot_losses(history, show_figs, d, results, trimmed_config, y_true=y_valid)
        plot_metrics(model, d, show_figs)
        print("Making predictions and confusion matrix")
        y_pred_valid = np.argmax(model.predict(X_valid, verbose=0), axis=1)
        y_pred_train = np.argmax(model.predict(X_train, verbose=0), axis=1)
        results.update(
            {
                "valid_f1": float(f1_score(y_valid, y_pred_valid, average="weighted")),
                "train_f1": float(f1_score(y_train, y_pred_train, average="weighted")),
            }
        )
        plot_confusion_matrices(
            y_valid,
            y_pred_valid,
            y_train,
            y_pred_train,
            results,
            trimmed_config,
            i2g,
            show_figs,
            d,
            cbar,
            perc=perc,
        )

    if os.path.exists("./models/results.jsonlines"):
        old = pd.read_json("./models/results.jsonlines", lines=True)
    else:
        old = pd.DataFrame()
    new = pd.json_normalize(results | config)
    pd.concat((old, new), ignore_index=True).to_json(
        "./models/results.jsonlines",
        orient="records",
        lines=True,
    )
    with open(f"{d}/results.yaml", "w") as file:
        yaml.dump(results, file)


def build_dataset(df, config):
    print(f"Making batches with window size of {config['window_size']}")
    X, y = make_batches(
        df.drop(["datetime", "gesture"], axis=1).to_numpy(),
        df["gesture"].to_numpy(),
        df["datetime"].to_numpy(),
        window_size=config["window_size"],
        window_skip=config["window_skip"],
    )

    # Offset the label by some amount
    if config.get("label_offset", 0) > 0:
        print(f"Offsetting the labels by {config['label_offset']}")
        padding = np.array(["gesture0255"] * config["label_offset"])
        y = np.concatenate((padding, y[: -config["label_offset"]]))
    elif config.get("center_label", False):
        offset = config["window_size"] // 2
        padding = np.array(["gesture0255"] * offset)
        y = np.concatenate((padding, y[:-offset]))

    config["label_before"] = config.get(
        "label_expansion", config.get("label_before", 0)
    )
    config["label_after"] = config.get("label_expansion", config.get("label_after", 0))
    # Extend the labels by a certain amount
    non255_idxs = np.where(y != "gesture0255")[0]
    for i in range(-config.get("label_before", 0), config.get("label_after", 0)):
        if non255_idxs.max() + i >= y.shape[0] or i == 0:
            continue
        y[non255_idxs + i] = y[non255_idxs]

    if config["label_before"] or config["label_after"]:
        print(
            f"Expanding labels to be in deltas of {config['label_before']} before to {config['label_after']} after"
        )

    # print(np.where(y != 'gesture0255')[0][:16])
    if config.get("allowlist", []):
        print(f"Only including gestures in {config['allowlist']}")
        X = X[np.isin(y, config["allowlist"])]
        y = y[np.isin(y, config["allowlist"])]

    if config.get("omit_0255", False):
        print(f"Omitting the 255 gesture")
        X = X[y != "gesture0255"]
        y = y[y != "gesture0255"]

    if config.get("g255_vs_rest", False):
        print(f"Grouping gestures to be gesture0255 vs rest")
        y = np.where(y == "gesture0255", "gesture0255", "gesture0256")
    config["label_size_over_window_size"] = (
        config["label_before"] + config["label_after"]
    ) / config["window_size"]
    # Get functions to convert between gestures and indices
    g2i, i2g = gestures_and_indices(y)

    labels = sorted(np.unique(y))
    g2i_dict = {g: i for i, g in enumerate(labels)}
    i2g_dict = {i: g for i, g in enumerate(labels)}
    with open("saved_models/idx_to_gesture.pickle", "wb") as f:
        pickle.dump(i2g_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("saved_models/gesture_to_idx.pickle", "wb") as f:
        pickle.dump(g2i_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    y = g2i(y)
    # Get functions to convert between indices and one hot encodings
    i2ohe, ohe2i = one_hot_and_back(y)

    total = len(y)
    n_unique = len(np.unique(y))
    config["gestures"] = list(np.unique(y))

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=config["test_frac"], random_state=42
    )

    class_weight = {
        int(class_): (1 / freq)
        for class_, freq in zip(*np.unique(y_train, return_counts=True))
    }
    class_weight = {
        int(k): float(v / sum(class_weight.values())) for k, v in class_weight.items()
    }
    default_class_weights = {int(g): float(1.0 / n_unique) for g in np.unique(y_train)}
    config["class_weight"] = (
        class_weight if config["use_class_weights"] else default_class_weights
    )

    return config, g2i, i2g, i2ohe, ohe2i, X, y, X_train, X_valid, y_train, y_valid


def calc_red(y_true, y_pred, i2g):
    """Given the ordered true and actual predictions, calculate the mean
    distance from predicted rising edge to true rising edge"""
    mean_dists = np.zeros((y_pred.shape[1] - 1,))
    num_preds = np.zeros((y_pred.shape[1] - 1,))
    num_trues = np.zeros((y_pred.shape[1] - 1,))
    for i in range(mean_dists.shape[0]):
        gesture = i2g(i)
        pred_is_gesture = np.argmax(y_pred, axis=-1) == i
        correct_pred = np.argmax(y_pred, axis=-1) == y_true
        correct_gesture = i == y_true

        rising_pred = np.diff(pred_is_gesture.astype(int))
        count_pred = (rising_pred == 1).sum()

        rising_true = np.diff(correct_gesture.astype(int))
        count_true = (rising_true == 1).sum()

        nz_true = np.nonzero(rising_true)[0]
        nz_pred = np.nonzero(rising_pred)[0]
        num_preds[i] = len(nz_pred)
        num_trues[i] = len(nz_true)

        t = 0
        p = 0
        d = 0
        while t < nz_true.shape[0] and p < nz_pred.shape[0]:
            curr_true = nz_true[t]
            next_true = nz_true[t + 1] if t != nz_true.shape[0] - 1 else float("inf")
            curr_pred = nz_pred[p]
            next_pred = nz_pred[p + 1] if p != nz_pred.shape[0] - 1 else float("inf")
            dists = np.zeros((2, 2))
            dists[0][0] = np.abs(curr_true - curr_pred)
            dists[0][1] = np.abs(curr_true - next_pred)
            dists[1][0] = np.abs(next_true - curr_pred)
            dists[1][1] = np.abs(next_true - next_pred)
            d += dists[0][0] if dists[0][0] < 80 else 0
            mean_dists[i] += dists[0][0] if dists[0][0] < 80 else 0

            if dists[1][1] <= dists[0][1] and dists[1][1] <= dists[1][0]:
                t += 1
                p += 1
            elif dists[1][0] <= dists[0][1]:
                t += 1
            elif dists[0][1] <= dists[1][0]:
                p += 1
        mean_dists[i] = mean_dists[i] / len(nz_true)

    return mean_dists, num_preds, num_trues


def predict_tf(obs, config, model, i2g) -> List[Tuple[int, float]]:
    probabilities = model(np.array([obs])).numpy()[0]
    obs_pred = np.argmax(probabilities)

    # Sort the predictions
    predictions = sorted(enumerate(probabilities), key=lambda ip: -ip[1])
    # Convert the indexes to gestures and return the predictions
    return [(i2g[idx], proba) for idx, proba in predictions]


def load_model(filepath):
    """Load and return the model located at `filepath`."""
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model
