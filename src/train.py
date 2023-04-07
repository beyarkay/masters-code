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


import common
import logging as l
import models
import numpy as np
import optuna
import pandas as pd
import read
import sklearn
import sys
import tqdm


MAX_OBSERVATIONS = None


def main():
    l.info("Reading data")
    df: pd.DataFrame = read.read_data(offsets="offsets.csv")
    if MAX_OBSERVATIONS is not None:
        df = df.head(MAX_OBSERVATIONS)
    l.info("Making windows")
    X, y_str = read.make_windows(
        df,
        30,
        pbar=tqdm.tqdm(total=len(df), desc="Making windows"),
    )
    g2i, i2g = common.make_gestures_and_indices(y_str)
    y = g2i(y_str)
    X_trn, X_val, y_trn, y_val = sklearn.model_selection.train_test_split(X, y)

    # TODO Implement pruning: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.SuccessiveHalvingPruner.html#optuna.pruners.SuccessiveHalvingPruner
    optuna.logging.get_logger("optuna").addHandler(l.StreamHandler(sys.stdout))
    study = optuna.create_study(
        direction="minimize",
        storage="sqlite:///db.sqlite3",
        # pruner=optuna.pruners.HyperbandPruner(),
    )
    study.optimize(
        lambda trial: objective(trial, X_trn, y_trn, X_val, y_val),
        n_trials=30,
        gc_after_trial=True,
    )


def objective(trial, X_trn, y_trn, X_val, y_val):
    config = {
        "n_timesteps": 30,
        "nn": {
            "epochs": 5,
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True),
            "optimizer": "adam",
        },
        "ffnn": {
            "nodes_per_layer": [
                trial.suggest_categorical(
                    "nodes_per_layer.1", [10 * i for i in range(1, 20)]
                ),
                trial.suggest_categorical(
                    "nodes_per_layer.2", [10 * i for i in range(0, 20)]
                ),
            ],
        },
    }
    model = models.FFNNClassifier(config=config)
    l.info("Fitting model")
    model.fit(X_trn, y_trn, validation_data=(X_val, y_val))
    evaluation = model.evaluate(X_val, y_val)

    # Convert the `np.int64`s into `int`s so that optuna doesn't complain
    evaluation = {
        (k.item() if type(k) == np.int64 else k): v for k, v in evaluation.items()
    }

    # Store some data to the trial, to be used during model evaluation
    trial.set_user_attr("evaluation", evaluation)
    trial.set_user_attr("history", model.history.history)

    final_loss = model.history.history["val_loss"][-1]
    return final_loss


if __name__ == "__main__":
    common.init_logs()
    main()
