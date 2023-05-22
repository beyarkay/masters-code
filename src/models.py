"""Defines the models which are used for prediction/classification."""

import logging as l
import os
import pickle
import typing
from typing import Optional
import time

import common
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.utils.validation as sk_validation
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


class HMMConfig(typing.TypedDict):
    # The number of iterations for which *each* HMM will be trained
    n_iter: int


class CusumConfig(typing.TypedDict):
    thresh: int


class NNConfig(typing.TypedDict):
    # The number of epochs for which each NN will be trained
    epochs: int
    # The learning rate
    lr: float
    # The optimiser to use
    optimizer: str
    # The batch size to use during training
    batch_size: int


class LSTMConfig(typing.TypedDict):
    units: int


class FFNNConfig(typing.TypedDict):
    nodes_per_layer: list[int]


class PreprocessingConfig(typing.TypedDict):
    n_timesteps: int
    delay: int
    max_obs_per_class: int
    gesture_allowlist: list[int]
    seed: int


class ConfigDict(typing.TypedDict):
    preprocessing: PreprocessingConfig
    n_timesteps: int
    cusum: Optional[CusumConfig]
    nn: Optional[NNConfig]
    ffnn: Optional[FFNNConfig]
    lstm: Optional[LSTMConfig]
    hmm: Optional[HMMConfig]


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

    def _check_model_params(self, X, y, dt, validation_data):
        """Validate model parameters before fitting.

        This will read in the config (if applicable), check X,y are valid,
        store the classes, and perform some general pre-fit chores."""
        X_val, y_val, dt_val = validation_data
        print(
            f"Checking model params: {X.shape=} {y.shape=} {X_val.shape=} {y_val.shape=}"
        )

        # Assert that exactly one of (config_path, config) is not None
        if bool(self.config_path is None) == bool(self.config is None):
            raise ValueError(
                "Exactly one of (config_path, config) must be not None, but "
                f"config_path is {self.config_path} and config is {self.config}"
            )
        if self.config_path is not None:
            with open(self.config_path, "r") as f:
                self.config: ConfigDict = yaml.safe_load(f)

        delay = self.config["preprocessing"]["delay"]
        # First sort all the arrays with the datetime as the key
        argsort = np.argsort(dt)
        X = X[argsort]
        y = y[argsort]
        dt = dt[argsort]
        # Then chop off either the front or the end of the arrays by amount `delay`
        # https://stackoverflow.com/a/76252774/14555505
        # ------Original--------
        # X: [0 1 2 3 4 5 6]
        # y: [0 1 2 3 4 5 6]
        # ------Delayed by 2------
        # X[+2:None] =>  [    2 3 4 5 6]
        # y[None:-2] =>      [0 1 2 3 4    ]
        # ------Delayed by -2------
        # X[None:-2] =>     [0 1 2 3 4    ]
        # y[+2:None] => [    2 3 4 5 6]
        start_index = delay if delay > 0 else None
        finsh_index = delay if delay < 0 else None
        X = X[start_index:finsh_index]
        X_val = X_val[start_index:finsh_index]
        dt = dt[start_index:finsh_index]
        dt_val = dt_val[start_index:finsh_index]
        # We need to shift the labels in the opposite direction so that they
        # line up correctly. So trimming the last element of X should also mean
        # we trim the first element of y, and vice versa. This is equivalent to
        # negating the delay
        start_index = -delay if delay < 0 else None
        finsh_index = -delay if delay > 0 else None
        y = y[start_index:finsh_index]
        y_val = y_val[start_index:finsh_index]
        print(
            f"Shapes after {delay=}: {X.shape=} {y.shape=} {X_val.shape=} {y_val.shape=}"  # noqa: E501
        )

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
        self.g2i, self.i2g = common.make_gestures_and_indices(
            y,
            not_g255=lambda g: g != 50,
            to_i=lambda g: g,
            to_g=lambda i: i,
            g255=50,
        )
        print(
            f"Shapes after allowlist: {X.shape=} {y.shape=} {X_val.shape=} {y_val.shape=}"  # noqa: E501
        )

        # Ensure that there are no more than `max_obs_per_class` observations
        # per class
        max_obs_per_class = self.config["preprocessing"]["max_obs_per_class"]
        if max_obs_per_class is not None:
            indexes_trn = []
            indexes_val = []
            for cls in np.unique(y):
                num_trn_observations = (y == cls).sum()
                indexes_trn.extend(
                    np.random.choice(
                        np.nonzero(y == cls)[0],
                        min(num_trn_observations, max_obs_per_class),
                        replace=False,
                    )
                )
                num_val_observations = (y_val == cls).sum()
                indexes_val.extend(
                    np.random.choice(
                        np.nonzero(y_val == cls)[0],
                        min(num_val_observations, max_obs_per_class),
                        replace=False,
                    )
                )
            X = X[indexes_trn]
            y = y[indexes_trn]
            dt = dt[indexes_trn]
            X_val = X_val[indexes_val]
            y_val = y_val[indexes_val]
            dt_val = dt_val[indexes_val]
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

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def write(
        self,
        model_dir,
        dump_model=True,
        dump_conf_mat_plots=True,
        dump_conf_mats=True,
        dump_config=True,
        dump_loss_plots=True,
        dump_predictions=True,
        dump_distribution_plots=True,
    ):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        with open(f"{model_dir}/config.yaml", "w") as f:
            yaml.safe_dump(self.config, f)

        history = (
            {}
            if not hasattr(getattr(self, "model", None), "history")
            else self.model.history.history
        )

        fit_time = self.fit_finsh_time - self.fit_start_time
        # Save the training stats
        y_trn_pred = self.predict(self.X_)
        pred_time_trn = self.predict_finsh_time - self.predict_start_time
        conf_mat_trn = self.confusion_matrix(self.y_, y_pred=y_trn_pred)
        results_trn = {
            "time_to_predict": pred_time_trn,
            "num_observations": self.X_.shape[0],
            "prediction_time_per_obs": pred_time_trn / self.X_.shape[0],
            "confidence_matrix": conf_mat_trn.tolist(),
            "history": {k: v for k, v in history.items() if "val" not in k},
        }
        np.savez(f"{model_dir}/y_pred_y_trn.npz", y_pred=y_trn_pred, y_trn=self.y_)
        np.savez(f"{model_dir}/conf_mat_trn.npz", conf_mat_trn)

        # Save the validation stats
        X_val, y_val, dt_val = self.validation_data
        y_val_pred = self.predict(X_val)
        pred_time_val = self.predict_finsh_time - self.predict_start_time
        conf_mat_val = self.confusion_matrix(y_val, y_pred=y_val_pred)
        results_val = {
            "time_to_predict": pred_time_val,
            "num_observations": self.X_.shape[0],
            "prediction_time_per_obs": pred_time_val / self.X_.shape[0],
            "history": {
                (k.replace("val_", "")): v for k, v in history.items() if "val" in k
            },
            "confidence_matrix": conf_mat_val.tolist(),
        }
        np.savez(f"{model_dir}/y_pred_y_val.npz", y_pred=y_val_pred, y_val=y_val)
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
            fig, axs = plt.subplots(1, len(h.items()), figsize=(4 * len(h.items()), 3))
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

    def confusion_matrix(self, y_true, y_pred=None, X_to_pred=None):
        """Calculate the confusion matrix of some predictions.

        Either pass in the alread-predicted values via y_pred, or pass in some
        X data which will be predicted and then used to calculate the confusion
        matrix."""
        # Assert that exactly one of (y_pred, X_to_pred) is not None
        if bool(y_pred is None) == bool(X_to_pred is None):
            raise ValueError(
                f"Exactly one of (y_pred, X_to_pred) must be not None, but \
                y_pred is {y_pred} and X_to_pred is {X_to_pred}"
            )
        if X_to_pred is not None:
            y_pred = self.predict(X_to_pred)

        return tf.math.confusion_matrix(y_true, y_pred).numpy()

    def set_random_seed(self, seed: int):
        tf.random.set_seed(seed)
        np.random.seed(seed)


class MeanClassifier(TemplateClassifier):
    def fit(self, X, y, dt):
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
        self.config_path: Optional[str] = config_path
        self.config: Optional[ConfigDict] = config
        self.config["model_type"] = self.config.get("model_type", "HMM")

    def fit(self, X, y, dt, validation_data=None, verbose=False, **kwargs) -> None:
        self.fit_start_time = time.time()
        self.set_random_seed(self.config["preprocessing"]["seed"])
        self._check_model_params(X, y, dt, validation_data)

        self.models_ = {}
        iterator = (
            tqdm.tqdm(np.unique(self.y_))
            if kwargs.get("verbose", False)
            else np.unique(self.y_)
        )
        for yi in iterator:
            if verbose:
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
                covariance_type="diag",
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
        return self

    def predict(self, X, verbose=False):
        self.predict_start_time = time.time()
        predictions = np.empty(X.shape[0])
        if verbose:
            pbar = tqdm.tqdm(total=X.shape[0])

        for i, xi in enumerate(X):
            best_key = None
            best_score = float("-inf")
            for key, m in self.models_.items():
                try:
                    score = m.score(xi)
                except ValueError as e:
                    print(f"Value error for HMM {self.i2g(key)}, observation {i}: {e}")
                    score = float("-inf")
                if score > best_score:
                    best_score = score
                    best_key = key
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


class CuSUMClassifier(TemplateClassifier):
    """A classifier that uses the CuSUM algorithm.

    CuSUM can detect changes from a reference distribution. This can be used to
    detect when a single sensor's readings change from 'normal'. A hard-coded
    mapping can then convert a list of deviant sensor readings into a gesture
    prediction.

    See [wikipedia](https://en.wikipedia.org/wiki/CUSUM) for details on the
    algorithm."""

    def __init__(self, config_path=None, config=None):
        self.config_path = config_path
        self.config = config
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
        for i in range(1, len(x)):
            cusum_pos[i] = max(0, cusum_pos[i - 1] + x[i] - upper_limit)
            cusum_neg[i] = min(0, cusum_neg[i - 1] + x[i] - lower_limit)

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

    def fit(self, X, y, dt, validation_data, **kwargs) -> None:
        self.fit_start_time = time.time()
        self.set_random_seed(self.config["preprocessing"]["seed"])
        self._check_model_params(X, y, dt, validation_data)
        threshold = self.config["cusum"]["thresh"]
        self.const: common.ConstantsDict = common.read_constants()

        num_gestures = len(np.unique(self.y_))

        records = np.zeros((num_gestures, self.const["n_sensors"]))

        pbar = (
            range(num_gestures)
            if kwargs.get("verbose", False)
            else tqdm.trange(num_gestures)
        )
        # Loop over all gestures
        for gesture_idx in pbar:
            pbar.set_description(f"CuSUM gesture: {self.i2g(gesture_idx)}")
            data = self.X_[self.y_ == gesture_idx]
            record = np.zeros((data.shape[0], self.const["n_sensors"]))

            # Loop over all observations matching that gesture
            iterator = (
                tqdm.trange(data.shape[0])
                if kwargs.get("verbose", False)
                else range(data.shape[0])
            )
            for observation_idx in iterator:
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
                    record[observation_idx, sensor_idx] = int(too_high or too_low)

            # We don't care about all the details, just the sum of all the
            # times the statistic was over the threshold for a given gesture
            records[gesture_idx] = record.sum(axis=0)
        # Some gestures have more observations than others. Normalise the data
        # so we can treat all gestures equally.
        self.normalised = (records.T / records.T.sum(axis=0)).T
        self.is_fitted_ = True
        self.fit_finsh_time = time.time()

    def predict(self, X):
        self.predict_start_time = time.time()
        preds = np.empty(X.shape[0])
        threshold = self.config["cusum"]["thresh"]
        for i, xi in enumerate(X):
            values = np.zeros(self.const["n_sensors"])
            for sensor_idx in range(self.const["n_sensors"]):
                csm = self._cusum(
                    xi[:, sensor_idx],
                    target=xi[:10, sensor_idx].mean(),
                )
                too_high = (np.abs(csm["cusum_neg"]) > threshold).any()
                too_low = (np.abs(csm["cusum_pos"]) > threshold).any()
                values[sensor_idx] = int(too_high or too_low)
            preds[i] = np.argmin(np.linalg.norm(self.normalised - values, axis=1))
        self.predict_finsh_time = time.time()
        return preds


class TFClassifier(TemplateClassifier):
    """Just an abstract class for TensorFlow-style models"""

    def predict(self, X):
        """Give label predictions for each observation in X"""
        self.predict_start_time = time.time()
        preds = self.i2g(np.argmax(self.predict_proba(X), axis=1))
        self.predict_finsh_time = time.time()
        return preds

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
        optimizer_string = self.config.get("nn", {}).get("optimizer", "adam")
        learning_rate = self.config.get("nn", {}).get("learning_rate", 2.5e-5)
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


class FFNNClassifier(TFClassifier):
    def __init__(
        self, config_path: Optional[str] = None, config: Optional[ConfigDict] = None
    ):
        keras.backend.clear_session()
        self.config_path: Optional[str] = config_path
        self.config: Optional[ConfigDict] = config
        self.config["model_type"] = self.config.get("model_type", "FFNN")
        self.normalizer = None

    def fit(self, X, y, dt, validation_data=None, **kwargs) -> None:
        self.fit_start_time = time.time()

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
            keras.layers.Dense(
                units=npl,
                activation="relu",
            )
            for npl in self.config["ffnn"]["nodes_per_layer"]
        ]

        # Construct the model as a sequence of layers
        self.model = tf.keras.Sequential(
            [
                keras.layers.Input(shape=self.X_.shape[1:]),
                self.normalizer,
                keras.layers.Flatten(),
                *dense_layers,
                # NOTE: Last layer isn't softmax because it's impossible to get
                # a stable loss calculation using softmax output
                keras.layers.Dense(len(np.unique(y))),
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


class RNNClassifier(TFClassifier):
    def __init__(self, config_path=None, config=None):
        self.config_path: Optional[str] = config_path
        self.config: Optional[ConfigDict] = config
        self.config["model_type"] = self.config.get("model_type", "RNN")
        self.normalizer = None

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
        self.config_path: Optional[str] = config_path
        self.config: Optional[ConfigDict] = config
        self.config["model_type"] = self.config.get("model_type", "LSTM")
        self.normalizer = None

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
                keras.layers.LSTM(self.config["lstm"]["units"], return_sequences=False),
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
