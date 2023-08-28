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

from numpy.lib.npyio import NpzFile

import tensorflow as tf
import pandas as pd
import common
import models
import numpy as np
import optuna
import sklearn
import sklearn.model_selection
import sys


def main():
    l.info("Reading data")
    trn: NpzFile = np.load("./gesture_data/trn_20.npz")
    X: np.ndarray = trn["X_trn"]
    y: np.ndarray = trn["y_trn"]
    dt: np.ndarray = trn["dt_trn"]
    trn.close()

    (
        X_trn,
        X_val,
        y_trn,
        y_val,
        dt_trn,
        dt_val,
    ) = sklearn.model_selection.train_test_split(X, y, dt, stratify=y)

    now = datetime.datetime.now().isoformat(sep="T")[:-10]
    study_name = f"optimizers-{now}" if len(sys.argv) != 2 else sys.argv[1]

    optuna.logging.get_logger("optuna").addHandler(l.StreamHandler(sys.stdout))
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage="sqlite:///db.sqlite3",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=5, interval_steps=5
        ),
    )
    study.optimize(
        lambda trial: objective_nn(
            trial, X_trn, y_trn, dt_trn, X_val, y_val, dt_val),
        n_trials=1000,
        gc_after_trial=True,
    )


def objective_wrapper(trial, X_trn, y_trn, dt_trn, X_val, y_val, dt_val):
    architecture = trial.suggest_categorical(
        "architecture",
        [
            "ffnn",
            # "hmm",
            # "cusum",
        ],
    )
    if architecture == "ffnn":
        return objective_nn(trial, X_trn, y_trn, dt_trn, X_val, y_val, dt_val)
    elif architecture == "hmm":
        return objective_hmm(trial, X_trn, y_trn, dt_trn, X_val, y_val, dt_val)
    elif architecture == "cusum":
        return objective_cusum(trial, X_trn, y_trn, dt_trn, X_val, y_val, dt_val)
    else:
        raise NotImplementedError


def objective_hmm(trial, X_trn, y_trn, dt_trn, X_val, y_val, dt_val):
    config = {
        "hmm": {
            "n_iter": 1,
            "limit": 100,
        },
    }
    model = models.HMMClassifier(config=config)
    start = datetime.datetime.now()
    # FIXME model.fit should also take in dt
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


def objective_cusum(trial, X_trn, y_trn, dt_trn, X_val, y_val, dt_val):
    config = {}
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


def objective_nn(trial, X_trn, y_trn, dt_trn, X_val, y_val, dt_val):
    # Keras has memory leak issues. `clear_session` reportedly fixes this
    # https://github.com/optuna/optuna/issues/4587#issuecomment-1511564031
    keras.backend.clear_session()
    num_layers = trial.suggest_int("num_layers", 1, 3)
    nodes_per_layer = [
        trial.suggest_int(f"nodes_per_layer.{layer_idx+1}", 4, 512, log=True)
        for layer_idx in range(num_layers)
    ]
    config: models.ConfigDict = {
        "model_type": "FFNN",
        "preprocessing": {
            'seed': 42,
            'n_timesteps': 20,
            'delay': 0,
            'max_obs_per_class': None,
            'gesture_allowlist': list(range(51)),
            'num_gesture_classes': None,
            'rep_num': 0
        },
        "nn": {
            "epochs": 20,
            "batch_size": trial.suggest_int("batch_size", 64, 256, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True),
            "optimizer": "adam",
        },
        "ffnn": {
            "nodes_per_layer": nodes_per_layer,
            "l2_coefficient": trial.suggest_float("l2_coefficient", 1e-6, 1e-1, log=True),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.6),
        },
        "cusum": None,
        "lstm": None,
        "hmm": None,
    }
    pprint(config)
    clf = models.FFNNClassifier(config=config)
    print("Fitting model")
    start = datetime.datetime.now()

    clf.fit(
        X_trn,
        y_trn,
        dt_trn,
        validation_data=(X_val, y_val, dt_val),
        verbose=True,
        callbacks=[
            models.DisplayConfMat(
                validation_data=(X_val, y_val, dt_val),
                conf_mat=False,
                fig_path=f'fig_{trial.number}.png',
            ),
            OptunaPruningCallback(
                validation_data=(X_val, y_val, dt_val),
                trial=trial,
            ),
        ]
    )
    finsh = datetime.datetime.now()

    # Save information about the model
    # now = datetime.datetime.now().isoformat(sep="T")[:-7]
    # model_dir = f"saved_models/ffnn/{now}"
    # model.write(model_dir, dump_model=False)
    # trial.set_user_attr("model_dir", model_dir)

    duration = finsh - start
    duration_ms = duration.seconds * 1000 + duration.microseconds / 1000
    trial.set_user_attr("duration_ms", duration_ms)

    trial.set_user_attr("val_loss", clf.history.history["val_loss"][-1])
    trial.set_user_attr("trn_loss", clf.history.history["loss"][-1])

    y_pred = clf.predict(X_val)
    report = sklearn.metrics.classification_report(
        y_pred.astype(int),
        y_val.astype(int),
        output_dict=True,
        zero_division=0,
    )
    print(sklearn.metrics.classification_report(
        y_pred.astype(int),
        y_val.astype(int),
        zero_division=0,
    ))

    clf_report = pd.json_normalize(report)
    trial.set_user_attr("val.macro avg.f1-score",
                        clf_report['macro avg.f1-score'].values[0])
    trial.set_user_attr("val.macro avg.precision",
                        clf_report['macro avg.precision'].values[0])
    trial.set_user_attr("val.macro avg.recall",
                        clf_report['macro avg.recall'].values[0])

    return clf_report['macro avg.f1-score'].values[0]


class OptunaPruningCallback(keras.callbacks.Callback):
    def __init__(self, validation_data, trial):
        self.validation_data = validation_data
        self.X_val = validation_data[0]
        self.y_val = validation_data[1]
        self.dt_val = validation_data[2]
        self.trial = trial
        self.history = {'loss': [], 'val_loss': []}

    def on_epoch_end(self, _epoch, logs=None):
        assert hasattr(self, 'history')
        assert hasattr(self, 'X_val')
        assert hasattr(self, 'y_val')
        assert hasattr(self, 'validation_data')
        assert hasattr(self, 'model') and self.model is not None
        assert logs is not None
        self.history['loss'].append(logs.get('loss', None))
        self.history['val_loss'].append(logs.get('val_loss', None))

        y_pred = np.argmax(tf.nn.softmax(
            (self.model(self.X_val))).numpy(), axis=1)

        report = sklearn.metrics.classification_report(
            y_pred.astype(int),
            self.y_val.astype(int),
            output_dict=True,
            zero_division=0,
        )
        clf_report = pd.json_normalize(report)

        intermediate_value = clf_report['macro avg.f1-score'].values[0]
        self.trial.report(intermediate_value, len(self.history['loss']) - 1)
        if self.trial.should_prune():
            print(f"Pruning: macro f1={intermediate_value}")
            raise optuna.TrialPruned()
        else:
            print(f"Not pruning: macro f1={intermediate_value}")


if __name__ == "__main__":
    main()
