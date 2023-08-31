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
import os

from numpy.lib.npyio import NpzFile

import tensorflow as tf
import pandas as pd
import models
import numpy as np
import optuna
import sklearn
import sklearn.model_selection


def main():
    l.info("Reading data")
    trn: NpzFile = np.load("./gesture_data/trn_20_10.npz")
    X: np.ndarray = trn["X_trn"]
    y: np.ndarray = trn["y_trn"]
    dt: np.ndarray = trn["dt_trn"]
    trn.close()

    now = datetime.datetime.now().isoformat(sep="T")[:-10]
    study_name = f"optimizers-{now}" if len(sys.argv) != 2 else sys.argv[1]
    if not os.path.exists(f'saved_models/{study_name}'):
        os.makedirs(f'saved_models/{study_name}')

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
        lambda trial: objective_wrapper(trial, X, y, dt, study_name),
        n_trials=1000,
        gc_after_trial=True,
    )


def objective_wrapper(trial, X, y, dt, study_name):
    architecture = trial.suggest_categorical("architecture", [
        "ffnn",
        # "hmm",
        # "cusum",
    ])
    preprocessing: models.PreprocessingConfig = {
        'seed': 42 + trial.number,
        'n_timesteps': 20,
        'max_obs_per_class': 200 if architecture == "hmm" else None,
        'gesture_allowlist': list(range(51)),
        'num_gesture_classes': 51,
        'rep_num': 0
    }

    (X_trn, X_val, y_trn, y_val, dt_trn, dt_val) = sklearn.model_selection.train_test_split(
        X, y, dt, stratify=y, random_state=preprocessing["seed"]
    )

    if architecture == "ffnn":
        return objective_nn(trial, X_trn, y_trn, dt_trn, X_val, y_val, dt_val,
                            study_name, preprocessing)
    elif architecture == "hmm":
        return objective_hmm(trial, X_trn, y_trn, dt_trn, X_val, y_val, dt_val,
                             study_name, preprocessing)
    elif architecture == "cusum":
        return objective_cusum(trial, X_trn, y_trn, dt_trn, X_val, y_val,
                               dt_val, study_name, preprocessing)
    else:
        raise NotImplementedError


def objective_hmm(trial, X_trn, y_trn, dt_trn, X_val, y_val, dt_val,
                  study_name, preprocessing):
    config: models.ConfigDict = {
        "model_type": "HMM",
        "preprocessing": preprocessing,
        "hmm": {
            "n_iter": 20,
        },
        "cusum": None,
        "lstm": None,
        "ffnn": None,
        "nn": None,
    }
    clf = models.HMMClassifier(config=config)
    start = datetime.datetime.now()
    clf.fit(
        X_trn,
        y_trn,
        dt_trn,
        validation_data=(X_val, y_val, dt_val),
        verbose=True,
    )
    finsh = datetime.datetime.now()
    score = calc_metrics(trial, start, finsh, clf)
    return score


def objective_cusum(trial, X_trn, y_trn, dt_trn, X_val, y_val, dt_val,
                    study_name, preprocessing):
    config: models.ConfigDict = {
        "model_type": "CuSUM",
        "preprocessing": preprocessing,
        "cusum": {
            "thresh": trial.suggest_categorical("thresh", [5, 10, 20, 40, 60, 80, 100]),
        },
        "hmm": None,
        "lstm": None,
        "ffnn": None,
        "nn": None,
    }
    clf = models.CuSUMClassifier(config=config)
    l.info("Fitting model")
    start = datetime.datetime.now()
    clf.fit(
        X_trn,
        y_trn,
        dt_trn,
        validation_data=(X_val, y_val, dt_val),
        verbose=True,
    )

    finsh = datetime.datetime.now()
    score = calc_metrics(trial, start, finsh, clf)
    return score


def objective_nn(trial, X_trn, y_trn, dt_trn, X_val, y_val, dt_val, study_name, preprocessing):
    # TODO train-validation split inside the objective
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
        "preprocessing": preprocessing,
        "nn": {
            "epochs": 40,
            "batch_size": trial.suggest_int("batch_size", 64, 256, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True),
            "optimizer": "adam",
        },
        "ffnn": {
            "nodes_per_layer": nodes_per_layer,
            "l2_coefficient": trial.suggest_float("l2_coefficient", 1e-7, 1e-4, log=True),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
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
            # NOTE: This uses the wrong validation data if num_gesture_classes < 51
            # models.DisplayConfMat(
            #     validation_data=(X_val, y_val, dt_val),
            #     conf_mat=False,
            #     fig_path=f'saved_models/{study_name}/trial_{trial.number}.png',
            # ),
            OptunaPruningCallback(
                clf=clf,
                trial=trial,
            ),
        ]
    )
    finsh = datetime.datetime.now()
    score = calc_metrics(trial, start, finsh, clf)
    return score


def calc_metrics(trial, start, finsh, clf):
    model_type = clf.config['model_type']
    X_val = clf.validation_data[0]
    y_val = clf.validation_data[1]
    print("y_val unique: ", np.unique(y_val))
    duration = finsh - start
    duration_ms = duration.seconds * 1000 + duration.microseconds / 1000
    trial.set_user_attr("duration_ms", duration_ms)

    if model_type == "FFNN":
        trial.set_user_attr("val_loss", clf.history.history["val_loss"][-1])
        trial.set_user_attr("trn_loss", clf.history.history["loss"][-1])

    jsonl_path = f"saved_models/results_{model_type.lower()}_optuna.jsonl"
    model_dir = clf.write_as_jsonl(jsonl_path)
    trial.set_user_attr("model_dir", model_dir)

    y_pred = clf.predict(X_val)
    clf_report = pd.json_normalize(sklearn.metrics.classification_report(
        y_val.astype(int),
        y_pred.astype(int),
        output_dict=True,
        zero_division=0,
    ))
    print(sklearn.metrics.classification_report(
        y_val.astype(int),
        y_pred.astype(int),
        output_dict=False,
        zero_division=0,
    ))

    trial.set_user_attr("val.macro avg.f1-score",
                        clf_report['macro avg.f1-score'].values[0])
    trial.set_user_attr("val.macro avg.precision",
                        clf_report['macro avg.precision'].values[0])
    trial.set_user_attr("val.macro avg.recall",
                        clf_report['macro avg.recall'].values[0])
    return clf_report['macro avg.f1-score'].values[0]


class OptunaPruningCallback(keras.callbacks.Callback):
    def __init__(self, clf: models.TFClassifier, trial):
        if hasattr(clf, 'X_'):
            self.validation_data = clf.validation_data
            self.X_val = self.validation_data[0]
            self.y_val = self.validation_data[1]
            self.dt_val = self.validation_data[2]
        self.clf = clf
        self.trial = trial
        self.history = {'loss': [], 'val_loss': []}

    def on_epoch_end(self, _epoch, logs=None):
        # Exit if we haven't got checked X, y, dt values yet
        if not hasattr(self.clf, 'X_'):
            print("WARN: Doesn't have X_")
            return
        # Ensure we've got checked validation data
        if not hasattr(self, 'validation_data'):
            self.validation_data = self.clf.validation_data
        self.X_val = self.validation_data[0]
        self.y_val = self.validation_data[1]
        self.dt_val = self.validation_data[2]
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
            self.y_val.astype(int),
            y_pred.astype(int),
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
