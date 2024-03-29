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

import typing
import keras
from pprint import pprint
import datetime
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
    trn: NpzFile = np.load("./gesture_data/trn_20_10.npz")
    X: np.ndarray = trn["X_trn"]
    y: np.ndarray = trn["y_trn"]
    dt: np.ndarray = trn["dt_trn"]
    trn.close()

    if len(sys.argv) == 2 and sys.argv[1].startswith("optimizers-"):
        model_type = sys.argv[1].split("-")[1]
        num_gesture_classes = int(sys.argv[1].split("-")[2])
        study_name = sys.argv[1]
    else:
        model_type = input("What model type?: ")
        num_gesture_classes = int(input("How many gesture classes?: "))
        study_name = f"optimizers-{model_type}-{num_gesture_classes:0>2}"
    if not os.path.exists(f'saved_models/{study_name}'):
        os.makedirs(f'saved_models/{study_name}')

    repetitions = 5

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage="sqlite:///db.sqlite3",
        load_if_exists=True,
        sampler=optuna.samplers.RandomSampler(),
    )
    # print("WARN: Using TPE sampler")
    study.optimize(
        lambda trial: objective_wrapper(
            trial,
            X,
            y,
            dt,
            study_name,
            model_type,
            num_gesture_classes,
            repetitions=repetitions
        ),
        n_trials=200,
        gc_after_trial=True,
    )


def objective_wrapper(
    trial,
    X,
    y,
    dt,
    study_name,
    model_type,
    num_gesture_classes,
    repetitions=1
) -> float:

    if model_type == "FFNN":
        objective_func = objective_nn
    elif model_type == "HMM":
        objective_func = objective_hmm
    elif model_type == "CuSUM":
        objective_func = objective_cusum
    elif model_type == "HFFNN":
        objective_func = objective_hffnn
    elif model_type == "SVM":
        objective_func = objective_svm
    else:
        raise NotImplementedError(f"Model type {model_type} is not known")

    scores = []

    for rep_num in range(repetitions):
        print(f"[{rep_num}/{repetitions}] Repetition starting for model {model_type}")
        preprocessing: models.PreprocessingConfig = {
            'seed': 42 + rep_num + repetitions * trial.number,
            'n_timesteps': 20,
            'max_obs_per_class': 1000 if model_type == "HMM" else None,
            'gesture_allowlist': list(range(num_gesture_classes)),
            'num_gesture_classes': num_gesture_classes,
            'rep_num': rep_num,
        }

        (X_trn, X_val, y_trn, y_val, dt_trn, dt_val) = sklearn.model_selection.train_test_split(
            X, y, dt, stratify=y, random_state=preprocessing["seed"]
        )

        score = objective_func(
            trial,
            X_trn, y_trn, dt_trn,
            X_val, y_val, dt_val,
            study_name,
            preprocessing
        )
        trial.report(score, step=rep_num)
        scores.append(score)

    trial.set_user_attr("scores.all", scores)
    trial.set_user_attr("scores.min", np.min(scores))
    trial.set_user_attr("scores.max", np.max(scores))
    trial.set_user_attr("scores.mean", np.mean(scores))
    trial.set_user_attr("scores.std_dev", np.std(scores))
    return np.mean(scores).astype(float)


def objective_hmm(trial, X_trn, y_trn, dt_trn, X_val, y_val, dt_val,
                  study_name, preprocessing):
    config: models.ConfigDict = {
        "model_type": "HMM",
        "preprocessing": preprocessing,
        "hmm": {
            "n_iter": 20,
            "covariance_type": trial.suggest_categorical(
                "hmm.covariance_type",
                ["spherical", "diag", "full", "tied"],
            )
        },
        "cusum": None,
        "lstm": None,
        "ffnn": None,
        "nn": None,
        "svm": None,
        "hffnn": None,
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
        "svm": None,
        "hffnn": None,
    }
    clf = models.CuSUMClassifier(config=config)
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
    # Keras has memory leak issues. `clear_session` reportedly fixes this
    # https://github.com/optuna/optuna/issues/4587#issuecomment-1511564031
    keras.backend.clear_session()
    # TODO: reset this back to randomly selected nlayers
    num_layers = 2  # trial.suggest_int("num_layers", 2, 2)
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
        "svm": None,
        "hffnn": None,
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
            # OptunaPruningCallback(
            #     clf=clf,
            #     trial=trial,
            # ),
        ]
    )
    finsh = datetime.datetime.now()
    score = calc_metrics(trial, start, finsh, clf)
    return score


def objective_hffnn(trial, X_trn, y_trn, dt_trn, X_val, y_val, dt_val, study_name, preprocessing):
    # Keras has memory leak issues. `clear_session` reportedly fixes this
    # https://github.com/optuna/optuna/issues/4587#issuecomment-1511564031
    keras.backend.clear_session()

    majority_config = None
    minority_config = None

    for clf_type in ('majority', 'minority'):
        num_layers = trial.suggest_int(f"{clf_type}.num_layers", 1, 3)
        nodes_per_layer = [
            trial.suggest_int(
                f"{clf_type}.nodes_per_layer.{layer_idx+1}", 4, 512, log=True)
            for layer_idx in range(num_layers)
        ]
        gesture_allowlist = (
            [0, 1] if clf_type == "majority" else list(range(50))
        )
        config: models.ConfigDict = {
            "model_type": "FFNN",
            "preprocessing": preprocessing | {
                'gesture_allowlist': gesture_allowlist,
                'num_gesture_classes': len(gesture_allowlist),

            },
            "nn": {
                "epochs": trial.suggest_int(f"{clf_type}.epochs", 5, 40),
                "batch_size": trial.suggest_int(f"{clf_type}.batch_size", 64, 256, log=True),
                "learning_rate": trial.suggest_float(f"{clf_type}.learning_rate", 1e-6, 1e-1, log=True),
                "optimizer": "adam",
            },
            "ffnn": {
                "nodes_per_layer": nodes_per_layer,
                "l2_coefficient": trial.suggest_float(f"{clf_type}.l2_coefficient", 1e-7, 1e-4, log=True),
                "dropout_rate": trial.suggest_float(f"{clf_type}.dropout_rate", 0.0, 0.5),
            },
            "cusum": None,
            "lstm": None,
            "hmm": None,
            "svm": None,
            "hffnn": None,
        }
        if clf_type == "majority":
            majority_config = config
            print("Majority config: ", majority_config)
        elif clf_type == "minority":
            minority_config = config
            print("Minority config: ", minority_config)
        else:
            raise NotImplementedError(f"{clf_type=} not implemented")

    assert majority_config is not None
    assert minority_config is not None

    meta_clf = models.MetaClassifier(
        majority_config=majority_config,
        minority_config=minority_config,
        preprocessing=preprocessing,
    )

    print("Fitting Meta model")
    start = datetime.datetime.now()

    meta_clf.fit(
        X_trn,
        y_trn,
        dt_trn,
        validation_data=(X_val, y_val, dt_val),
        verbose=True,
    )
    finsh = datetime.datetime.now()
    score = calc_metrics(trial, start, finsh, meta_clf)
    return score


def objective_svm(trial, X_trn, y_trn, dt_trn, X_val, y_val, dt_val, study_name, preprocessing):
    clf = models.SVMClassifier(config={
        "model_type": "SVM",
        "preprocessing": preprocessing,
        "svm": {
            "c": trial.suggest_float('svm.c', 1e-6, 1, log=True),
            "class_weight": trial.suggest_categorical('svm.class_weight', ["balanced", None]),
            "max_iter": 200,
        },
        "nn": None,
        "ffnn": None,
        "cusum": None,
        "lstm": None,
        "hmm": None,
        "hffnn": None,
    })
    print("Fitting")
    start = datetime.datetime.now()
    clf.fit(
        X_trn,
        y_trn,
        dt_trn,
        validation_data=(X_val, y_val, dt_val),
        verbose=False,
    )
    finsh = datetime.datetime.now()
    print(f"SVM fitting duration: {(datetime.datetime.now()-start)}")

    score = calc_metrics(trial, start, finsh, clf)
    return score


def calc_metrics(trial, start, finsh, clf):
    model_type = clf.config['model_type']
    print(f"[{datetime.datetime.now()}] Calculating metrics for {model_type}")
    X_val = clf.validation_data[0]
    y_val = clf.validation_data[1]
    print("y_val unique: ", np.unique(y_val))
    duration = finsh - start
    duration_ms = duration.seconds * 1000 + duration.microseconds / 1000
    trial.set_user_attr("duration_ms", duration_ms)

    if model_type == "FFNN":
        trial.set_user_attr("val_loss", clf.history.history["val_loss"][-1])
        trial.set_user_attr("trn_loss", clf.history.history["loss"][-1])

    print(f"[{datetime.datetime.now()}] Writing as jsonl")
    jsonl_path = f"saved_models/results_{model_type.lower()}_optuna.jsonl"
    model_dir = clf.write_as_jsonl(jsonl_path)
    trial.set_user_attr("model_dir", model_dir)

    y_pred_path = f'{model_dir}/y_val_true_y_val_pred.npz'
    print(f"[{datetime.datetime.now()}] Reading y_pred from {y_pred_path}")
    try:
        y_pred = np.load(y_pred_path)['y_pred']
    except OSError as e:
        print(
            f"[{datetime.datetime.now()}] Failed to read {y_pred_path}: {e}, clf.predict'ing instead")
        y_pred = clf.predict(X_val)

    print(f"[{datetime.datetime.now()}] Calculating metrics")
    clf_report = pd.json_normalize(sklearn.metrics.classification_report(
        y_val.astype(int),
        y_pred.astype(int),
        output_dict=True,
        zero_division=0,
    ))
    print(f"[{datetime.datetime.now()}] Printing Metrics")
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
