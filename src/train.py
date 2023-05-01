"""Model training and hyper-parameter optimisation.
This script takes care of training various models. It also orchestrates
hyper-parameter optimisation over all those models. The basic layout of the
hyper parameters can be thought of as a heirarchy of hyper-parameters:

1. Which data preprocessing/feature engineering strategy to use?
2. Which model architecture to use?
3. Which model hyperparameters to use?

Note that the choice of feature engineering strategy will likely constrain what
model architectures are even valid (since some forms of feature engineering
might disallow certain model architectures).

Similarly, the choice of model architecture will likely constrain which
hyperparameters are valid.
"""

import keras
from pprint import pprint
import datetime
import logging as l
import sys

import common
import models
import numpy as np
import optuna
import pandas as pd
import read
import sklearn
import sklearn.model_selection

MAX_OBSERVATIONS = None


def main():
    l.info("Reading data")
    df: pd.DataFrame = read.read_data(offsets="offsets.csv")
    if MAX_OBSERVATIONS is not None:
        df = df.head(MAX_OBSERVATIONS)
    trn = np.load("./gesture_data/trn.npz")
    X = trn["X_trn"]
    y = trn["y_trn"]

    X_trn, X_val, y_trn, y_val = sklearn.model_selection.train_test_split(
        X, y, stratify=y
    )

    # TODO Implement pruning: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.SuccessiveHalvingPruner.html#optuna.pruners.SuccessiveHalvingPruner
    now = datetime.datetime.now().isoformat(sep="T")[:-7]

    optuna.logging.get_logger("optuna").addHandler(l.StreamHandler(sys.stdout))
    study = optuna.create_study(
        study_name=f"optimizers-{now}",
        direction="minimize",
        storage="sqlite:///db.sqlite3",
        load_if_exists=False
        # pruner=optuna.pruners.HyperbandPruner(),
    )
    study.optimize(
        lambda trial: objective_wrapper(trial, X_trn, y_trn, X_val, y_val),
        n_trials=1000,
        gc_after_trial=True,
    )


def objective_wrapper(trial, X_trn, y_trn, X_val, y_val):
    architecture = trial.suggest_categorical(
        "architecture",
        [
            "ffnn",
            # "hmm",
            # "cusum",
        ],
    )
    if architecture == "ffnn":
        return objective_nn(trial, X_trn, y_trn, X_val, y_val)
    elif architecture == "hmm":
        return objective_hmm(trial, X_trn, y_trn, X_val, y_val)
    elif architecture == "cusum":
        return objective_cusum(trial, X_trn, y_trn, X_val, y_val)
    else:
        raise NotImplementedError


def objective_hmm(trial, X_trn, y_trn, X_val, y_val):
    config = {
        "n_timesteps": X_trn.shape[1],
        "hmm": {
            "n_iter": 1,
            "limit": 100,
        },
    }
    model = models.HMMClassifier(config=config)
    start = datetime.datetime.now()
    model.fit(X_trn, y_trn, validation_data=(X_val, y_val), verbose=True)
    finsh = datetime.datetime.now()

    duration = finsh - start
    duration_ms = duration.seconds * 1000 + duration.microseconds / 1000
    trial.set_user_attr("duration_ms", duration_ms)

    print("Calculating training loss")
    print(y_trn[:10].shape)
    print(model.predict_score(X_trn[:10]))
    trn_loss = keras.losses.sparse_categorical_crossentropy(
        y_trn[:100], model.predict(X_trn[:100]), from_logits=False
    )
    trial.set_user_attr("trn_loss", trn_loss)

    print("Calculating validation loss")
    val_loss = keras.losses.sparse_categorical_crossentropy(
        y_val[:100], model.predict(X_val[:100]), from_logits=False
    )
    trial.set_user_attr("val_loss", val_loss)

    return val_loss


def objective_cusum(trial, X_trn, y_trn, X_val, y_val):
    config = {
        "n_timesteps": X_trn.shape[1],
    }
    model = models.CuSUMClassifier(config=config)
    l.info("Fitting model")
    start = datetime.datetime.now()
    model.fit(X_trn, y_trn, validation_data=(X_val, y_val))
    finsh = datetime.datetime.now()

    duration = finsh - start
    duration_ms = duration.seconds * 1000 + duration.microseconds / 1000
    trial.set_user_attr("duration_ms", duration_ms)

    final_loss = 0  # TODO
    return final_loss


def objective_nn(trial, X_trn, y_trn, X_val, y_val):
    # Keras has memory leak issues. `clear_session` reportedly fixes this
    # https://github.com/optuna/optuna/issues/4587#issuecomment-1511564031
    keras.backend.clear_session()
    num_layers = trial.suggest_int("num_layers", 1, 3)
    nodes_per_layer = [
        trial.suggest_int(f"nodes_per_layer.{layer_idx+1}", 4, 512, log=True)
        for layer_idx in range(num_layers)
    ]
    config = {
        "n_timesteps": X_trn.shape[1],
        "nn": {
            "epochs": 20,
            "batch_size": trial.suggest_int("batch_size", 64, 256, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True),
            "optimizer": "adam",
        },
        "ffnn": {"nodes_per_layer": nodes_per_layer},
    }
    pprint(config)
    model = models.FFNNClassifier(config=config)
    l.info("Fitting model")
    start = datetime.datetime.now()
    model.fit(X_trn, y_trn, validation_data=(X_val, y_val))
    finsh = datetime.datetime.now()

    # Save information about the model
    now = datetime.datetime.now().isoformat(sep="T")[:-7]
    model_dir = f"saved_models/ffnn/{now}"
    model.write(model_dir, dump_model=False)
    trial.set_user_attr("model_dir", model_dir)

    duration = finsh - start
    duration_ms = duration.seconds * 1000 + duration.microseconds / 1000
    trial.set_user_attr("duration_ms", duration_ms)

    trial.set_user_attr("val_loss", model.history.history["val_loss"][-1])
    trial.set_user_attr("trn_loss", model.history.history["loss"][-1])

    final_loss = model.history.history["val_loss"][-1]
    return final_loss


if __name__ == "__main__":
    common.init_logs()
    main()
