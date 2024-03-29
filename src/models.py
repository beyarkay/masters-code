"""Defines the models which are used for prediction/classification."""
import datetime
from wrapt_timeout_decorator import timeout

import pandas as pd
from sklearn.metrics import classification_report
import os
import pickle
import dill
from typing import Optional, Literal, TypedDict
import time
from sklearn import svm

import common
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import tensorflow as tf
import tqdm
import vis
import yaml
from hmmlearn import hmm
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow import keras


def calc_class_weights(y):
    # NOTE: The np.log(weight) is *very* important.
    class_weight = {
        int(class_): np.log(1.0 / count * 1_000_000)
        for class_, count in zip(*np.unique(y, return_counts=True))
    }
    return class_weight


class HMMConfig(TypedDict):
    # The number of iterations for which *each* HMM will be trained
    n_iter: int
    covariance_type: Optional[Literal["spherical", "diag", "full", "tied"]]


class CusumConfig(TypedDict):
    thresh: int


class NNConfig(TypedDict):
    # The number of epochs for which each NN will be trained
    epochs: int
    # The learning rate
    learning_rate: float
    # The optimiser to use
    optimizer: str
    # The batch size to use during training
    batch_size: int


class LSTMConfig(TypedDict):
    units: int


class SVMConfig(TypedDict):
    c: float
    class_weight: Optional[Literal["balanced"]]
    max_iter: int


class FFNNConfig(TypedDict):
    nodes_per_layer: list[int]
    l2_coefficient: float
    dropout_rate: float


class PreprocessingConfig(TypedDict):
    # TODO Remove timesteps as an option?
    n_timesteps: int
    max_obs_per_class: Optional[int]
    gesture_allowlist: list[int]
    seed: int
    num_gesture_classes: Optional[int]
    rep_num: Optional[int]


class MetaClassifierConfig(TypedDict):
    preprocessing: PreprocessingConfig
    nn: Optional[NNConfig]
    ffnn: Optional[FFNNConfig]
    model_type: Literal["FFNN"]


class HFFNNConfig(TypedDict):
    majority: MetaClassifierConfig
    minority: MetaClassifierConfig


class ConfigDict(TypedDict):
    preprocessing: PreprocessingConfig
    cusum: Optional[CusumConfig]
    nn: Optional[NNConfig]
    ffnn: Optional[FFNNConfig]
    lstm: Optional[LSTMConfig]
    hmm: Optional[HMMConfig]
    svm: Optional[SVMConfig]
    hffnn: Optional[HFFNNConfig]
    model_type: Optional[Literal["FFNN", "HMM", "CuSUM", "HFFNN", "SVM"]]


class TemplateClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self, config_path: Optional[str] = None, config: Optional[ConfigDict] = None
    ):
        """From the sklearn docs:

        As `model_selection.GridSearchCV` uses `set_params` to apply parameter
        setting to estimators, it is essential that calling `set_params` has the
        same effect as setting parameters using the `__init__` method. The
        easiest and recommended way to accomplish this is to *not do any
        parameter validation* in `__init__`.

        All logic behind estimator parameters, like translating string
        arguments into functions, should be done in fit.

        """
        self.config_path = config_path
        self.config = config
        self.fit_start_time: Optional[float] = None
        self.fit_finsh_time: Optional[float] = None
        self.predict_start_time: Optional[float] = None
        self.predict_finsh_time: Optional[float] = None

    def _check_model_params(self, X, y, dt, validation_data):
        """Validate model parameters before fitting.

        This will read in the config (if applicable), check X,y are valid,
        store the classes, and perform some general pre-fit chores."""

        X_val, y_val, dt_val = validation_data
        print(
            f"Checking model params: {X.shape=} {y.shape=} {X_val.shape=} {y_val.shape=}"
        )

        print(
            f"Numbers of classes:\ny: {pd.Series(y).value_counts().unique()},\ny_val: {pd.Series(y_val).value_counts().unique()}"  # noqa: E501
        )

        # Assert that exactly one of (config_path, config) is not None
        if bool(self.config_path is None) == bool(self.config is None):
            raise ValueError(
                "Exactly one of (config_path, config) must be not None, but "
                f"config_path is {self.config_path} and config is {self.config}"
            )
        if self.config_path is not None:
            with open(self.config_path, "r") as f:
                self.config: Optional[ConfigDict] = yaml.safe_load(f)

        assert self.config is not None

        # Remove any gestures which aren't on the allowlist
        allowlist = self.config["preprocessing"]["gesture_allowlist"]
        allowed_trn_gestures = np.isin(y, allowlist)
        X = X[allowed_trn_gestures]
        y = y[allowed_trn_gestures]
        dt = dt[allowed_trn_gestures]
        allowed_val_gestures = np.isin(y_val, allowlist)
        X_val = X_val[allowed_val_gestures]
        y_val = y_val[allowed_val_gestures]
        dt_val = dt_val[allowed_val_gestures]
        g255 = 50 if len(allowlist) == 51 else None
        print(f"{g255=}, {allowlist=}")
        self.g2i, self.i2g = common.make_gestures_and_indices(
            y,
            to_i=lambda g: g,
            to_g=lambda i: i,
            g255=g255,
        )
        print(
            f"Shapes after allowlist: {X.shape=} {y.shape=} {X_val.shape=} {y_val.shape=}\n"  # noqa: E501
            f"to_i is the identity, to_g is the identity, g255={g255}\n"  # noqa: E501
            f"g2i: y {list(zip(self.g2i(np.unique(y)), np.unique(y)))}"  # noqa: E501
        )

        # Ensure that there are no more than `max_obs_per_class` observations
        # per class
        max_obs_per_class = self.config["preprocessing"]["max_obs_per_class"]
        if max_obs_per_class is not None:
            indexes_trn = []
            print("Not limiting number of validation observations")
            # indexes_val = []
            for cls in np.unique(y):
                num_trn_observations = (y == cls).sum()
                indexes_trn.extend(
                    np.random.choice(
                        np.nonzero(y == cls)[0],
                        min(num_trn_observations, max_obs_per_class),
                        replace=False,
                    )
                )
                # num_val_observations = (y_val == cls).sum()
                # indexes_val.extend(
                #     np.random.choice(
                #         np.nonzero(y_val == cls)[0],
                #         min(num_val_observations, max_obs_per_class),
                #         replace=False,
                #     )
                # )
            X = X[indexes_trn]
            y = y[indexes_trn]
            dt = dt[indexes_trn]
            # X_val = X_val[indexes_val]
            # y_val = y_val[indexes_val]
            # dt_val = dt_val[indexes_val]
        print(
            f"Shapes after {max_obs_per_class=}: {X.shape=} {y.shape=} {X_val.shape=} {y_val.shape=}"  # noqa: E501
        )

        # Ensure that there are no more than `n_timesteps` timesteps for each
        # observation
        # TODO is the timestep modifier working?
        n_timesteps = self.config["preprocessing"]["n_timesteps"]
        if n_timesteps > X.shape[1]:
            print(
                f"WARN: {n_timesteps=} > {X.shape[1]=}, which means that"
                " n_timesteps isn't doing anything useful"
            )
        X = X[:, -n_timesteps:, :]
        X_val = X_val[:, -n_timesteps:, :]
        print(
            f"Shapes after {n_timesteps=}: {X.shape=} {y.shape=} {X_val.shape=} {y_val.shape=}"  # noqa: E501
        )

        # Check that X and y have correct shape
        X, y = common.check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = sklearn.utils.multiclass.unique_labels(y)
        self.X_ = X
        self.y_ = y
        self.dt_ = dt
        self.validation_data = (X_val, y_val, dt_val)

    @timeout(3600)
    def fit(self, X, y):
        raise NotImplementedError

    @timeout(3600)
    def predict(self, X):
        raise NotImplementedError

    def write_as_jsonl(self, path):

        # Create a directory for the supplementary data
        assert self.config is not None
        assert self.config['model_type'] is not None
        now = datetime.datetime.now().isoformat(sep="T")
        model_type = self.config['model_type'].lower()
        model_dir = f"saved_models/{model_type}_{now}"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Calculate the columns that should be included in the CSV
        # Create some placeholder variables which will be used to construct the
        # columns list
        prefixes = ["trn", "val"]
        classes = [str(i) for i in list(range(0, 51))] + \
            ["weighted avg", "macro avg"]
        metrics = ["precision", "recall", "f1-score", "support"]
        # Each column is of the form {prefix}.{cls}.{metric}
        columns = [
            f"{prefix}.{cls}.{metric}"
            for prefix in prefixes
            for cls in classes
            for metric in metrics
        ]
        # Except for the accuracy column, which is just {prefix}.accuracy
        columns += [
            f"{prefix}.accuracy"
            for prefix in prefixes
        ]

        # Calculate the classification report for training and validation
        datasets = [(self.X_, self.y_, self.dt_), self.validation_data]
        row_of_data = pd.DataFrame()
        for prefix, (X, y, dt) in zip(prefixes, datasets):
            # Make predictions, attempting to mitigate the effect of a timeout
            while True:
                try:
                    print(
                        f"[{datetime.datetime.now()}] Predicting {prefix} with X.shape = {X.shape}")
                    y_pred = self.predict(X)
                    break
                except TimeoutError as e:
                    print(f"Timed out while predicting: {e}")
                    if len(X) == 0:
                        raise e
                    X = X[:len(X)//2]
                    y = y[:len(y)//2]
                    dt = dt[:len(dt)//2]
                    print(f"New shape: {X.shape}")
                    continue

            print(f"[{datetime.datetime.now()}] Saving y_pred and y_true")
            np.savez(
                f"{model_dir}/y_{prefix}_true_y_{prefix}_pred.npz",
                y_pred=y_pred.astype(int),
                y_true=y.astype(int)
            )

            print(f"[{datetime.datetime.now()}] Getting classification report")
            # Get a classification_report formatted as a pandas DF
            report = classification_report(  # type: ignore
                y.astype(int),
                y_pred.astype(int),
                output_dict=True,
                zero_division=0,  # type: ignore
            )
            clf_report = pd.json_normalize(report)  # type: ignore

            # Rename the columns to start with the correct prefix
            clf_report.columns = [f'{prefix}.{c}' for c in clf_report.columns]
            actual_cols = clf_report.columns
            # Ensure that the calculated columns are a super set of the actual
            # columns
            assert len(set(actual_cols).union(set(columns))) == len(columns)

            # Collect metrics about the time taken to predict
            pred_time = None
            if self.predict_finsh_time is not None and self.predict_start_time is not None:
                pred_time = self.predict_finsh_time - self.predict_start_time

            perf_metrics = pd.DataFrame({
                f'{prefix}.pred_time': [pred_time],
                f'{prefix}.num_observations': [X.shape[0]],
            })

            # Append the new data as new columns in the DF
            row_of_data = pd.concat(
                (row_of_data, clf_report, perf_metrics),
                axis='columns'
            )

        # Get a DF of the model's config
        cfg = pd.json_normalize(self.config)  # type: ignore

        # If the model is a NN, store it's loss
        trn_loss: float = (
            np.nan
            if self.config['model_type'] != 'FFNN' else
            self.model.history.history['loss'][-1]
        )
        # FIXME: Training loss and validation loss aren't comparable because
        # Keras uses the class weights for the training loss but *not* for the
        # validation loss. Trying to use class weights for validation loss
        # directly in Keras uses too much memory to be practical, for some
        # reason.
        # https://datascience.stackexchange.com/q/53012/139026
        # They're also not directly comparable because of dropout
        val_loss: float = (
            np.nan
            if self.config['model_type'] != 'FFNN' else
            self.model.history.history['val_loss'][-1]
        )

        # Keep track of how long it took to fit the model
        fit_time = None
        if self.fit_finsh_time is not None and self.fit_start_time is not None:
            fit_time = self.fit_finsh_time - self.fit_start_time
        extra_data = pd.DataFrame({
            'fit_time': [fit_time],
            'saved_at': [now],
            'model_dir': [model_dir],
            'val.loss': [val_loss],
            'trn.loss': [trn_loss],
        })

        to_save = pd.concat(
            (cfg, row_of_data, extra_data),
            axis='columns',
            sort=True
        )

        print(f"[{datetime.datetime.now()}] Saving data to jsonlines {path}")
        to_save.to_json(
            path,
            lines=True,
            orient='records',
            mode='a',
        )

        print(
            f"[{datetime.datetime.now()}] Dumping model config to {model_dir}/config.yaml")
        with open(f"{model_dir}/config.yaml", "w") as f:
            yaml.safe_dump(self.config, f)
        return model_dir

    def write(
        self,
        model_dir,
        dump_model=True,
        dump_conf_mat_plots=True,
        dump_loss_plots=True,
        dump_distribution_plots=True,
    ):
        assert hasattr(self, 'config') and self.config is not None

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        with open(f"{model_dir}/config.yaml", "w") as f:
            yaml.safe_dump(self.config, f)

        history = (
            {}
            if not hasattr(getattr(self, "model", None), "history")
            else self.model.history.history
        )

        # Calculate the time taken to fit the model
        fit_time = None
        if self.fit_finsh_time is not None and self.fit_start_time is not None:
            fit_time = self.fit_finsh_time - self.fit_start_time

        # Save the training stats
        y_trn_pred = self.predict(self.X_)

        # Calculate the time taken to predict with the model
        pred_time_trn = None
        if self.predict_finsh_time is not None and self.predict_start_time is not None:
            pred_time_trn = self.predict_finsh_time - self.predict_start_time
        else:
            print(
                f"WARN: {self.predict_finsh_time=}, {self.predict_start_time=}")

        clf_report_trn = pd.json_normalize(classification_report(
            self.y_,
            y_trn_pred,
            output_dict=True,
            zero_division=0,
        ))
        clf_report_trn.columns = [f'trn.{c}' for c in clf_report_trn.columns]

        cfg = pd.json_normalize(self.config)
        to_save = pd.concat(
            (clf_report_trn, cfg),
            axis=1
        )

        conf_mat_trn = self.confusion_matrix(self.y_, y_pred=y_trn_pred)
        results_trn = {
            "time_to_predict": pred_time_trn,
            "num_observations": self.X_.shape[0],
            "prediction_time_per_obs":
                None
                if pred_time_trn is None
                else pred_time_trn / self.X_.shape[0],
            "confidence_matrix": conf_mat_trn.tolist(),
            "history": {k: v for k, v in history.items() if "val" not in k},
        }
        np.savez(f"{model_dir}/y_pred_y_trn.npz",
                 y_pred=y_trn_pred, y_trn=self.y_)
        np.savez(f"{model_dir}/conf_mat_trn.npz", conf_mat_trn)

        # Save the validation stats
        X_val, y_val, dt_val = self.validation_data
        y_val_pred = self.predict(X_val)
        pred_time_val = None
        if self.predict_finsh_time is not None and self.predict_start_time is not None:
            pred_time_val = self.predict_finsh_time - self.predict_start_time
        else:
            print(
                f"WARN: {self.predict_finsh_time=}, {self.predict_start_time=}")

        prefixes = ["trn", "val"]
        datasets = [(self.y_, self.X_, self.dt_), self.validation_data]
        df = pd.DataFrame()
        for prefix, (X, y, dt) in zip(prefixes, datasets):
            print(f"Predicting for {prefix}")
            # Make predictions
            y_pred = self.predict(X)
            # Get a classification_report formatted as a pandas DF
            clf_report = pd.json_normalize(classification_report(
                y,
                y_pred,
                output_dict=True,
                zero_division=0,
            ))
            # Rename the columns to start with the correct prefix
            clf_report.columns = [f'{prefix}.{c}' for c in clf_report.columns]
            raise

            df = pd.concat(
                (df, clf_report),
                axis=1
            )

        conf_mat_val = self.confusion_matrix(y_val, y_pred=y_val_pred)
        results_val = {
            "time_to_predict": pred_time_val,
            "num_observations": self.X_.shape[0],
            "prediction_time_per_obs":
                None
                if pred_time_val is None
                else pred_time_val / self.X_.shape[0],
            "history": {
                (k.replace("val_", "")): v for k, v in history.items() if "val" in k
            },
            "confidence_matrix": conf_mat_val.tolist(),
        }
        np.savez(f"{model_dir}/y_pred_y_val.npz",
                 y_pred=y_val_pred, y_val=y_val)
        np.savez(f"{model_dir}/conf_mat_val.npz", conf_mat_val)

        with open(f"{model_dir}/results.yaml", "w") as f:
            yaml.safe_dump(
                {
                    "trn": results_trn,
                    "val": results_val,
                    "fit_time": fit_time,
                },
                f,
                default_flow_style=True,
            )

        # Save the model
        if dump_model:
            with open(f"{model_dir}/model.pkl", "wb") as f:
                pickle.dump(self, f)

        # Save plots of the confusion matrices
        if dump_conf_mat_plots:
            X_val, y_val, dt_val = self.validation_data
            vis.plot_conf_mats(
                self,
                (self.X_, X_val),
                (self.y_, y_val),
                ("Training", "Validation"),
            )
            plt.savefig(f"{model_dir}/conf_mats.png")

        # Save plots of the loss over time
        if (
            dump_loss_plots
            and hasattr(self, "history")
            and hasattr(getattr(self, "history"), "history")
        ):
            h = self.history.history
            fig, axs = plt.subplots(
                1, len(h.items()), figsize=(4 * len(h.items()), 3))
            for ax, (key, values) in zip(axs, h.items()):
                ax.plot(self.model.history.epoch, values, label=key)
                ax.set_title(key)
                ax.set(title=key, ylim=(0, np.max(values)))
            plt.savefig(f"{model_dir}/loss_plots.png")

        if dump_distribution_plots:
            fig, axs = vis.plot_distributions(y_val, self.predict(X_val))
            plt.savefig(f"{model_dir}/prediction_distributions.png")

    def evaluate(self, X_val, y_val):
        return sklearn.metrics.classification_report(
            y_val,
            self.predict(X_val),
            target_names=self.classes_,
            output_dict=True,
        )

    def confusion_matrix(self, y_true, y_pred=None, X_to_pred=None) -> np.ndarray:
        """Calculate the confusion matrix of some predictions.

        Either pass in the alread-predicted values via y_pred, or pass in some
        X data which will be predicted and then used to calculate the confusion
        matrix."""
        # Assert that exactly one of (y_pred, X_to_pred) is not None
        if bool(y_pred is None) == bool(X_to_pred is None):
            raise ValueError(
                f"Exactly one of(y_pred, X_to_pred) must be not None, but y_pred is {y_pred} and X_to_pred is {X_to_pred}"
            )
        if X_to_pred is not None:
            y_pred = self.predict(X_to_pred)

        return tf.math.confusion_matrix(y_true, y_pred).numpy()

    def set_random_seed(self, seed: int):
        tf.random.set_seed(seed)
        np.random.seed(seed)


class MeanClassifier(TemplateClassifier):
    @timeout(3600)
    def fit(self, X, y, dt):
        assert hasattr(self, 'config') and self.config is not None
        self.fit_start_time = time.time()
        self.set_random_seed(self.config["preprocessing"]["seed"])
        self._check_model_params(X, y, dt)
        n_timesteps = self.config["preprocessing"]["n_timesteps"]
        self.means = np.zeros((51, n_timesteps, 30))
        for g in range(51):
            data = X[y == g]
            self.means[g] = data.mean(axis=0)
        self.is_fitted_ = True
        self.fit_finsh_time = time.time()
        return self

    @timeout(3600)
    def predict(self, X):
        self.predict_start_time = time.time()
        assert self.is_fitted_
        result = np.empty((X.shape[0]))
        for i, xi in enumerate(X):
            result[i] = np.argmin(np.linalg.norm(self.means - xi, axis=(1, 2)))
        self.predict_finsh_time = time.time()
        return result


class HMMClassifier(TemplateClassifier):
    def __init__(
        self, config_path: Optional[str] = None, config: Optional[ConfigDict] = None
    ):
        super().__init__()
        self.config_path: Optional[str] = config_path
        self.config: Optional[ConfigDict] = config
        self.config["model_type"] = self.config.get("model_type", "HMM")

    def fit(self, X, y, dt, validation_data=None, verbose=False, **kwargs) -> None:
        assert hasattr(self, 'config') and self.config is not None
        self.fit_start_time = time.time()
        self.set_random_seed(self.config["preprocessing"]["seed"])
        self._check_model_params(X, y, dt, validation_data)

        self.models_ = {}
        iterator = (
            tqdm.tqdm(np.unique(self.y_))
            if kwargs.get("verbose", False)
            else np.unique(self.y_)
        )
        assert self.config['hmm'] is not None
        for yi in iterator:
            # if verbose:
            print(
                f"HMM: Training {self.i2g(yi)} on {len(self.X_[self.y_ == yi])} observations"  # noqa: E501
            )
            if (self.y_ == yi).sum() < (self.X_.shape[1] + 2):
                print(
                    f"WARN: gesture {self.i2g(yi)} has only {(self.y_ == yi).sum()} observations "  # noqa: E501
                    f" but {self.X_.shape[1] + 2} states. This might make things unstable."  # noqa: E501
                )
            self.models_[yi] = hmm.GaussianHMM(
                n_components=self.X_.shape[1] + 2,
                covariance_type=self.config["hmm"]["covariance_type"],
                n_iter=self.config["hmm"]["n_iter"],
                verbose=verbose,
                random_state=self.config["preprocessing"]["seed"],
            ).fit(np.concatenate(self.X_[self.y_ == yi]))

        # Now check that every model's transition matrix has rows summing to 1
        # (so that every HMM state has an entry (or exit?) point)
        for key, m in self.models_.items():
            rows = m.transmat_.sum(axis=1)
            if not np.isclose(rows, 1).all():
                print(
                    f"WARN: HMM for gesture {key} has rows not all summing to 1: {rows}"
                )

        self.is_fitted_ = True
        self.fit_finsh_time = time.time()

    def predict(self, X, verbose=True):
        self.predict_start_time = time.time()
        predictions = np.empty(X.shape[0])
        if verbose:
            pbar = tqdm.tqdm(total=X.shape[0])

        for i, xi in enumerate(X):
            all_failed = True
            best_key = None
            best_score = float("-inf")
            for key, m in self.models_.items():
                try:
                    score = m.score(xi)
                    all_failed = False
                except ValueError as e:
                    print(
                        f"Value error for HMM {self.i2g(key)}, observation {i}: {e}")
                    score = float("-inf")
                if score > best_score:
                    best_score = score
                    best_key = key
            if all_failed:
                print("All HMMs failed to classify, aborting.")
                break
            predictions[i] = best_key
            if verbose:
                pbar.update(1)
        self.predict_finsh_time = time.time()
        return predictions

    def predict_score(self, X, verbose=False):
        self.predict_score_start_time = time.time()
        scores = np.empty((X.shape[0], len(self.models_)))

        if verbose:
            pbar = tqdm.tqdm(total=X.shape[0])

        for i, xi in enumerate(X):
            scores[i] = [model.score(xi) for model in self.models_.values()]
            if verbose:
                pbar.update(1)
        self.predict_score_finsh_time = time.time()
        return scores


class SVMClassifier(TemplateClassifier):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.config = config

    @timeout(3600)
    def fit(self, X, y, dt, validation_data, verbose=False, **kwargs) -> None:
        assert self.config is not None
        assert self.config["svm"] is not None
        self.fit_start_time = time.time()
        self.set_random_seed(self.config["preprocessing"]["seed"])
        self._check_model_params(X, y, dt, validation_data)

        self.model = svm.LinearSVC(
            dual="auto",
            C=self.config["svm"]["c"],
            class_weight=self.config["svm"]["class_weight"],
            random_state=self.config["preprocessing"]["seed"],
            max_iter=self.config['svm']['max_iter'],
            verbose=verbose,
        )
        print("Fitting SVM")
        self.model.fit(
            self.X_.reshape(self.X_.shape[0], -1),
            self.y_
        )

        self.is_fitted_ = True
        self.fit_finsh_time = time.time()
        print(
            f"SVM fit in {self.fit_finsh_time - self.fit_start_time} seconds")

    @timeout(3600)
    def predict(self, X):
        assert self.config is not None
        assert self.config["svm"] is not None
        self.predict_start_time = time.time()

        y_pred = self.model.predict(X.reshape(X.shape[0], -1))

        self.predict_finsh_time = time.time()
        return y_pred


class CuSUMClassifier(TemplateClassifier):
    """A classifier that uses the CuSUM algorithm.

    CuSUM can detect changes from a reference distribution. This can be used to
    detect when a single sensor's readings change from 'normal'. A hard-coded
    mapping can then convert a list of deviant sensor readings into a gesture
    prediction.

    See [wikipedia](https://en.wikipedia.org/wiki/CUSUM) for details on the
    algorithm."""

    def __init__(self, config_path=None, config=None):
        super().__init__()
        self.config_path = config_path
        self.config = config
        if self.config is not None:
            self.config["model_type"] = self.config.get("model_type", "CuSUM")

    def _cusum(self, x, target=None, std_dev=None, allowed_std_devs=4):
        """Calculate the Cumulative Sum of some data.

        If no target is provided, the mean of the first 5 values of `x` is
        used as the value from which `x` should not deviate.
        """
        # Use the mean of the first five observations as the target if none is supplied
        target = x[:5].mean() if target is None else target
        # Use the std dev of the first five observations if none is supplied
        std_dev = x[:5].std() if std_dev is None else std_dev
        allowed_deviance = std_dev * allowed_std_devs

        # Get the upper and lower limits
        upper_limit = target + allowed_deviance
        lower_limit = target - allowed_deviance

        # Calculate the cusum for the upper limit
        cusum_pos = np.zeros(len(x))
        cusum_pos[0] = max(0, x[0] - upper_limit)
        # Calculate the cusum for the lower limit
        cusum_neg = np.zeros(len(x))
        cusum_neg[0] = min(0, x[0] - lower_limit)
        for n in range(1, len(x)):
            cusum_pos[n] = max(0, cusum_pos[n - 1] + x[n] - upper_limit)
            cusum_neg[n] = min(0, cusum_neg[n - 1] + x[n] - lower_limit)

        # Create arrays of booleans describing if the value was too high/too low
        too_high = np.where(cusum_pos == 0, 0, 1)
        too_low = np.where(cusum_neg == 0, 0, 1)

        return {
            "x": x,
            "cusum_pos": cusum_pos,
            "cusum_neg": cusum_neg,
            "too_high": too_high,
            "too_low": too_low,
            "target": target,
            "std_dev": std_dev,
            "allowed_std_devs": allowed_std_devs,
            "upper_limit": upper_limit,
            "lower_limit": lower_limit,
        }

    @timeout(3600)
    def fit(self, X, y, dt, validation_data, **kwargs) -> None:
        # Sanity checks
        assert self.config is not None
        assert self.config["cusum"] is not None
        self.fit_start_time = time.time()
        self.set_random_seed(self.config["preprocessing"]["seed"])
        self._check_model_params(X, y, dt, validation_data)

        # Extract some constants for ease-of-use
        threshold = self.config["cusum"]["thresh"]
        self.const: common.ConstantsDict = common.read_constants()

        n_gesture_classes = len(np.unique(self.y_))

        # TODO what does this doe
        records = np.zeros((n_gesture_classes, self.const["n_sensors"]))

        # Use a tqdm progress bar if `verbose`
        gesture_iter = (
            range(n_gesture_classes)
            if kwargs.get("verbose", False)
            else tqdm.trange(n_gesture_classes)
        )
        for gesture_idx in gesture_iter:
            if isinstance(gesture_iter, tqdm.tqdm):
                gesture_iter.set_description(
                    f"CuSUM gesture: {self.i2g(gesture_idx)}")
            # Extract just the observations for this class
            data = self.X_[self.y_ == gesture_idx]
            # TODO what does this doe
            gesture_ood = np.zeros((data.shape[0], self.const["n_sensors"]))

            # Loop over all observations matching that gesture
            obs_iter = (
                tqdm.trange(data.shape[0])
                if kwargs.get("verbose", False)
                else range(data.shape[0])
            )
            for observation_idx in obs_iter:
                subset = data[observation_idx, :, :]

                # Loop over each sensor
                for sensor_idx in range(self.const["n_sensors"]):
                    # Calculate the CuSUM statistics
                    csm = self._cusum(
                        subset[:, sensor_idx],
                        target=subset[:10, sensor_idx].mean(),
                    )
                    too_high = (np.abs(csm["cusum_neg"]) > threshold).any()
                    too_low = (np.abs(csm["cusum_pos"]) > threshold).any()
                    # record whether or not the cusum statistic passes the threshold
                    gesture_ood[observation_idx, sensor_idx] = int(
                        too_high or too_low)

            # We don't care about all the details, just the sum of all the
            # times the statistic was over the threshold for a given gesture
            records[gesture_idx] = gesture_ood.sum(axis=0)
        # Some gestures have more observations than others. Normalise the data
        # so we can treat all gestures equally.
        self.normalised = (records.T / records.T.sum(axis=0)).T

        # Bookkeeping
        self.is_fitted_ = True
        self.fit_finsh_time = time.time()

    @timeout(3600)
    def predict(self, X):
        assert self.config is not None
        assert self.config["cusum"] is not None
        self.predict_start_time = time.time()
        preds = np.empty(X.shape[0])
        threshold = self.config["cusum"]["thresh"]
        for i, xi in enumerate(X):
            # Zero out the binary OOD detection for each sensor
            sensor_ood = np.zeros(self.const["n_sensors"])
            for sensor_idx in range(self.const["n_sensors"]):

                csm = self._cusum(
                    # Perform CuSUM on just the i-th sensor's time series
                    xi[:, sensor_idx],
                    # The reference value from which the time series should not
                    # deviate is the first 10 observations in the time series
                    target=xi[:10, sensor_idx].mean(),
                )
                # Check if any of the time series observations were too high
                too_high = (np.abs(csm["cusum_neg"]) > threshold).any()
                # Check if any of the time series observations were too low
                too_low = (np.abs(csm["cusum_pos"]) > threshold).any()
                # A sensor is OOD if _any_ of the time series observations is
                # out of distribution
                sensor_ood[sensor_idx] = int(too_high or too_low)
            # Calculate how far away the observed OODs are from the expected
            # OODs for every gesture
            diff_from_expected = np.linalg.norm(
                self.normalised - sensor_ood,
                axis=1
            )
            # The prediction is the gesture index for which the difference
            # between the observed and expected sensor values are the smallest.
            preds[i] = np.argmin(diff_from_expected)
        # Keep track of the start and end time for bookkeeping
        self.predict_finsh_time = time.time()
        return preds


class TFClassifier(TemplateClassifier):
    """Just an abstract class for TensorFlow-style models"""

    @timeout(3600)
    def predict(self, X):
        """Give label predictions for each observation in X"""
        self.predict_start_time = time.time()
        preds = self.i2g(np.argmax(self.predict_proba(X), axis=1))
        self.predict_finsh_time = time.time()
        return preds

    @timeout(3600)
    def predict_proba(self, X):
        """Give label probabilities for each observation in X"""
        self.predict_proba_start_time = time.time()
        pred_probas = tf.nn.softmax((self.model(X))).numpy()
        self.predict_proba_finsh_time = time.time()
        return pred_probas

    def _resolve_optimizer(self):
        """Returns the Keras Optimizer object given a config dict defining
        learning rate, optimiser type, and other optimiser related
        shenanigans"""
        assert self.config is not None
        nn_config = self.config.get("nn", {})
        assert nn_config is not None
        optimizer_string = nn_config.get("optimizer", "adam")
        learning_rate = nn_config.get("learning_rate", 2.5e-5)
        if optimizer_string == "adam":
            # if self.uses_validation_weights:
            #     print("Using legacy adam")
            #     return tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
            # else:
            return keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_string == "rmsprop":
            return keras.optimizers.RMSprop(learning_rate=2.5e-5)
        else:
            return keras.optimizers.Adam(learning_rate=2.5e-5)

    def dump(self, directory: str):
        assert self.config is not None
        # Check that the model is a FFNN
        assert self.config['model_type'] == "FFNN", "TemplateClassifier.dump is not implemented for non-FFNN models yet"

        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            print(f"Dump[{directory}]: Creating directory")
            os.makedirs(directory)

        # Dump the actual model
        print(f"Dump[{directory}]: Saving model")
        self.model.save(f'{directory}/model.keras')  # type: ignore
        # Dump the training data
        print(f"Dump[{directory}]: Saving training data")
        np.savez(
            f"{directory}/trn.npz",
            X_trn=self.X_,
            y_trn=self.y_,
            dt_trn=self.dt_
        )
        # Dump the validation data
        print(f"Dump[{directory}]: Saving validation data")
        np.savez(
            f"{directory}/val.npz",
            X_val=self.validation_data[0],
            y_val=self.validation_data[1],
            dt_val=self.validation_data[2],
        )
        # Dump the config
        print(f"Dump[{directory}]: Saving config")
        with open(f"{directory}/config.yaml", 'w') as f:
            yaml.safe_dump(self.config, f)
        # Dump the i2g and g2i vectorized functions using dill
        print(f"Dump[{directory}]: Saving i2g and g2i")
        with open(f"{directory}/i2g.dill", 'wb') as f:
            dill.dump(self.i2g, f)
        with open(f"{directory}/g2i.dill", 'wb') as f:
            dill.dump(self.g2i, f)


def load_tf(directory: str):
    clf = TFClassifier(None, None)
    # Load the model
    print(f"Load[{directory}]: loading the model")
    clf.model = keras.models.load_model(f'{directory}/model.keras')
    # Load the training data
    print(f"Load[{directory}]: loading training data")
    clf.X_, clf.y_, clf.dt_ = np.load(f"{directory}/trn.npz").values()
    # load the validation data
    print(f"Load[{directory}]: loading validation data")
    X_val, y_val, dt_val = np.load(f"{directory}/val.npz").values()
    clf.validation_data = (X_val, y_val, dt_val)
    # Load the config
    print(f"Load[{directory}]: Loading config")
    with open(f"{directory}/config.yaml", 'r') as f:
        clf.config = yaml.safe_load(f)
    # Load the i2g and g2i vectorized functions using dill
    print(f"Load[{directory}]: Loading i2g and g2i")
    with open(f"{directory}/i2g.dill", 'rb') as f:
        clf.i2g = dill.load(f)
    with open(f"{directory}/g2i.dill", 'rb') as f:
        clf.g2i = dill.load(f)
    return clf


class DisplayConfMat(keras.callbacks.Callback):
    def __init__(self, validation_data, fig_path=None, conf_mat=False):
        self.validation_data = validation_data
        self.X_val = validation_data[0]
        self.y_val = validation_data[1]
        self.history = {'loss': [], 'val_loss': []}
        self.conf_mat = conf_mat
        self.fig_path: Optional[str] = fig_path

    def on_epoch_end(self, _epoch, logs=None):
        assert hasattr(self, 'history')
        assert hasattr(self, 'X_val')
        assert hasattr(self, 'y_val')
        assert hasattr(self, 'conf_mat')
        assert hasattr(self, 'validation_data')
        assert hasattr(self, 'model') and self.model is not None
        assert logs is not None
        self.history['loss'].append(logs.get('loss', None))
        self.history['val_loss'].append(logs.get('val_loss', None))
        y_val_pred = np.argmax(tf.nn.softmax(
            self.model(self.X_val)).numpy(), axis=-1)
        cm_val = tf.math.confusion_matrix(
            self.y_val.flatten(),
            y_val_pred.flatten()
        ).numpy()
        # print(cm_val)
        # print("max predicted: ", y_val_pred.max())

        fig, axs = plt.subplots(1, 3, figsize=(20, 5))

        vis.conf_mat(cm_val, ax=axs[2])
        axs[2].set_title('Validation Confusion Matrix')

        axs[0].plot(self.history['loss'])
        max_loss: float = max(self.history['loss'])
        axs[0].set_ylim((0, max_loss * 1.1))
        axs[0].set_title(f'Loss ({self.history["loss"][-1]:.6f})')

        axs[1].plot(self.history['val_loss'])
        max_val_loss: float = max(self.history['val_loss'])
        axs[1].set_ylim((0, max_val_loss * 1.1))
        axs[1].set_title(f'Val. loss ({self.history["val_loss"][-1]:.6f})')
        if self.fig_path is not None:
            plt.savefig(self.fig_path, bbox_inches='tight')
        else:
            plt.show()
        plt.close()


class FFNNClassifier(TFClassifier):
    """ Some example docstring"""

    def __init__(
        self, config_path: Optional[str] = None, config: Optional[ConfigDict] = None
    ):
        super().__init__()
        keras.backend.clear_session()
        self.config_path: Optional[str] = config_path
        self.config: Optional[ConfigDict] = config
        if self.config is not None:
            self.config["model_type"] = self.config.get("model_type", "FFNN")
        self.normalizer = None

    @timeout(3600)
    def fit(self, X, y, dt, validation_data=None, **kwargs) -> None:
        self.fit_start_time = time.time()
        assert hasattr(self, 'config') and self.config is not None
        assert self.config['nn'] is not None
        assert self.config['ffnn'] is not None

        # -----------------
        # Set up everything
        # -----------------
        self.set_random_seed(self.config["preprocessing"]["seed"])
        keras.backend.clear_session()
        print("Checking fit of X, y")
        self._check_model_params(X, y, dt, validation_data)

        # ----------------------
        # Actually fit the model
        # ----------------------
        # Fit the normalizer if not already fitted.
        self.normalizer = keras.layers.Normalization(axis=-2)
        if self.config["nn"]["epochs"] is not None:
            print("Fitting normalizer")
            self.normalizer.adapt(self.X_)
        else:
            print("WARN: Not fitting normalizer because epochs is None")

        # Construct the fully connected layers from the model config
        dense_layers = [
            [
                keras.layers.Dense(
                    units=npl,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l2(
                        self.config['ffnn']['l2_coefficient']),
                ),
                keras.layers.Dropout(self.config["ffnn"]["dropout_rate"])
            ]
            for npl in self.config["ffnn"]["nodes_per_layer"]
        ]

        # Construct the model as a sequence of layers
        self.model = tf.keras.Sequential(
            [
                keras.layers.Input(shape=self.X_.shape[1:]),
                self.normalizer,
                keras.layers.Flatten(),
                # Flatten the list of (dense, dropout) tuples
                # https://stackoverflow.com/a/952952/14555505
                *[item for sublist in dense_layers for item in sublist],
                # NOTE: Last layer isn't softmax because it's impossible to get
                # a stable loss calculation using softmax output
                keras.layers.Dense(len(np.unique(self.y_))),
            ]
        )

        print("Compiling model")
        optimizer = self._resolve_optimizer()
        # Compile the model using SCCE loss
        self.model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )

        print("Fitting model")
        # Add the validation data to the kwargs iff they're not None. The last
        # element of validation data is the dt array, which gets omitted.
        kwargs.update(
            {}
            if self.validation_data is None
            else {"validation_data": self.validation_data[:-1]}
        )
        # Fit the model to the data, with a number of epochs dictated by config
        if self.config["nn"]["epochs"] is not None:
            self.history = self.model.fit(
                self.X_,
                self.g2i(self.y_),
                batch_size=self.config["nn"]["batch_size"],
                epochs=self.config["nn"]["epochs"],
                class_weight=calc_class_weights(self.g2i(self.y_)),
                **kwargs,
            )
        else:
            print("WARN: Not fitting FFNN because epochs is None")

        # Sklearn expects is_fitted_ to be True after fitting
        self.is_fitted_ = True
        self.fit_finsh_time = time.time()


class MetaClassifier(TemplateClassifier):
    def __init__(
        self,
        majority_config: ConfigDict,
        minority_config: ConfigDict,
        preprocessing: PreprocessingConfig,
    ):
        super().__init__()
        self.majority_config = majority_config
        self.minority_config = minority_config
        # This type error is okay. Technically, we'd want arbitrarily recursive
        # config dicts, since a MetaClassifier could be made up of other
        # MetaClassifiers. However, Python doesn't allow recursive
        # datastructures, so we have to fudge the type system a bit.
        hffnn_config: HFFNNConfig = {
            'majority': self.majority_config,
            'minority': self.minority_config,
        }
        self.config: ConfigDict = {
            "model_type": "HFFNN",
            "preprocessing": preprocessing,
            "cusum": None,
            "hmm": None,
            "nn": None,
            "ffnn": None,
            "lstm": None,
            "svm": None,
            "hffnn": hffnn_config,
        }

    @timeout(7200)
    def fit(self, X, y, dt, validation_data=None, **kwargs) -> None:
        self.fit_start_time = time.time()
        self.validation_data = validation_data
        self.X_ = X
        self.y_ = y
        self.dt_ = dt

        # Figure out the majority config
        self.majority_clf = None
        if self.majority_config['model_type'] == "FFNN":
            self.majority_clf = FFNNClassifier(config=self.majority_config)
        elif self.majority_config['model_type'] == "HMM":
            self.majority_clf = HMMClassifier(config=self.majority_config)
        elif self.majority_config['model_type'] == "CuSUM":
            self.majority_clf = CuSUMClassifier(config=self.majority_config)
        else:
            raise NotImplementedError(
                f"Model type of {self.majority_config['model_type']} is not supported")
        assert self.majority_clf is not None

        # Figure out the minority config
        self.minority_clf = None
        if self.minority_config['model_type'] == "FFNN":
            self.minority_clf = FFNNClassifier(config=self.minority_config)
        elif self.minority_config['model_type'] == "HMM":
            self.minority_clf = HMMClassifier(config=self.minority_config)
        elif self.minority_config['model_type'] == "CuSUM":
            self.minority_clf = CuSUMClassifier(config=self.minority_config)
        else:
            raise NotImplementedError(
                f"Model type of {self.minority_config['model_type']} is not supported")
        assert self.minority_clf is not None

        print("Fitting majority classifier")
        self.majority_clf.fit(
            X,
            np.where(y == 50, 1, 0),
            dt,
            validation_data=None if validation_data is None else (
                validation_data[0],
                np.where(validation_data[1] == 50, 1, 0),
                validation_data[2]
            ),
            **kwargs
        )
        print("Fitting minority classifier")
        mask_trn = y != 50
        mask_val = None if validation_data is None else (
            validation_data[1] != 50)
        self.minority_clf.fit(
            X[mask_trn],
            y[mask_trn],
            dt[mask_trn],
            validation_data=None if validation_data is None else (
                validation_data[0][mask_val],
                validation_data[1][mask_val],
                validation_data[2][mask_val]
            ),
            **kwargs
        )

        # Sklearn expects is_fitted_ to be True after fitting
        self.is_fitted_ = True
        self.fit_finsh_time = time.time()

    @timeout(3600)
    def predict(self, X):
        assert self.majority_clf is not None
        assert self.minority_clf is not None
        self.predict_start_time = time.time()
        print(f"Predicting {len(X)} observations with minority classifier")
        majority_pred = self.majority_clf.predict(X)
        print(f"Predicting {len(X)} observations with minority classifier")
        minority_pred = self.minority_clf.predict(X)

        y_pred = np.where(
            majority_pred == 0,
            minority_pred,
            50,
        )

        self.predict_finsh_time = time.time()
        return y_pred


class RNNClassifier(TFClassifier):
    def __init__(self, config_path=None, config=None):
        super().__init__()
        self.config_path: Optional[str] = config_path
        self.config: Optional[ConfigDict] = config
        self.config["model_type"] = self.config.get("model_type", "RNN")
        self.normalizer = None

    @timeout(3600)
    def fit(self, X, y, dt, **kwargs):
        raise NotImplementedError("RNN hasn't been updated to use self.X_")
        self.fit_start_time = time.time()
        self.set_random_seed(self.config["preprocessing"]["seed"])
        raise NotImplementedError
        self._check_model_params(X, y, dt)
        # Fit the normalizer if not already fitted.
        if self.normalizer is None:
            print("Fitting normalizer")
            self.normalizer = keras.layers.Normalization(axis=-2)
            self.normalizer.adapt(X)

        # Construct the fully connected layers from the model config
        dense_layers = [
            keras.layers.Dense(
                units=num_nodes,
                activation="relu",
            )
            for num_nodes in self.config["ffnn"]["nodes_per_layer"]
        ]

        # Construct the model as a sequence of layers
        self.model = tf.keras.Sequential(
            [
                keras.layers.Input(shape=X.shape[1:]),
                self.normalizer,
                keras.layers.Flatten(),
                *dense_layers,
                # NOTE: Last layer isn't softmax because it's impossible to get
                # a stable loss calculation using softmax output
                keras.layers.Dense(len(np.unique(y))),
            ]
        )

        # Compile the model using the ADAM optimiser and SCCE loss
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=2.5e-5),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )

        # Fit the model to the data, with a number of epochs dictated by config
        self.history = self.model.fit(
            X, y, batch_size=128, epochs=self.config["nn"]["epochs"], **kwargs
        )

        # Sklearn expects is_fitted_ to be True after fitting
        self.is_fitted_ = True
        self.fit_finsh_time = time.time()


class LSTMClassifier(TFClassifier):
    def __init__(self, config_path=None, config=None):
        super().__init__()
        self.config_path: Optional[str] = config_path
        self.config: Optional[ConfigDict] = config
        self.config["model_type"] = self.config.get("model_type", "LSTM")
        self.normalizer = None

    @timeout(3600)
    def fit(self, X, y, dt, validation_data=None, **kwargs):
        raise NotImplementedError("LSTM hasn't been updated to use self.X_")
        self.fit_start_time = time.time()
        self.set_random_seed(self.config["preprocessing"]["seed"])
        # tensorflow's LSTM *requires* floats for matmul operations
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        self._check_model_params(X, y, dt)
        # Fit the normalizer if not already fitted.
        if self.normalizer is None:
            print("Fitting normalizer")
            self.normalizer = keras.layers.Normalization(axis=-2)
            self.normalizer.adapt(X)

        self.model = keras.models.Sequential(
            [
                # Shape [batch, time, features] => [batch, time, lstm_units]
                keras.layers.LSTM(
                    self.config["lstm"]["units"], return_sequences=False),
                # Shape => [batch, time, features]
                keras.layers.Dense(len(self.classes_)),
            ]
        )

        self.model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=self._resolve_optimizer(),
        )
        kwargs.update(
            {} if validation_data is None else {"validation_data": validation_data}
        )

        print(self.config)
        # Fit the model to the data, with a number of epochs dictated by config
        if self.config["nn"]["epochs"] is not None:
            self.history = self.model.fit(
                X,
                y,
                batch_size=self.config["nn"]["batch_size"],
                epochs=self.config["nn"]["epochs"],
                class_weight=calc_class_weights(y),
                **kwargs,
            )
        else:
            print("WARN: Not fitting LSTM because epochs is None")

        # Sklearn expects is_fitted_ to be True after fitting
        self.is_fitted_ = True
        self.fit_finsh_time = time.time()
