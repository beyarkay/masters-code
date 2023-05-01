"""Defines the models which are used for prediction/classification.

These models all use the [sklearn
API](https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator))
for estimators to ensure that they can be used with sklearn's GridSearchCV and
RandomSearchCV. This will make checking multimple hyperparameters a lot easier.

TODO:
- Make a RandomSearchCV pipeline for every model, that will check a very wide
  range of hyperparameters
- Also allow that pipeline to check a wide range of preprocessing steps, such
  as:
    - dimensionality reduction via PCA
    - feature selection
    - different `n_timesteps`
    - others?
- Make sure to weight the various classes automatically
"""

import logging as l
import os
import pickle
import typing
from typing import Optional

import common
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    # I don't know why, but it makes everything work
    class_weight = {
        int(class_): np.log(1.0 / count * 1_000_000)
        for class_, count in zip(*np.unique(y, return_counts=True))
    }
    return class_weight


class CusumConfig(typing.TypedDict):
    thresh: int


class NNConfig(typing.TypedDict):
    epochs: int
    lr: float
    optimizer: str
    batch_size: int


class FFNNConfig(typing.TypedDict):
    nodes_per_layer: list[int]


class ConfigDict(typing.TypedDict):
    n_timesteps: int
    cusum: Optional[CusumConfig]
    ffnn: Optional[FFNNConfig]
    nn: Optional[NNConfig]


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

    def _check_fit(self, X, y):
        """Validate model parameters before fitting.

        This will read in the config (if applicable), check X,y are valid,
        store the classes, and perform some general pre-fit chores."""
        # Assert that exactly one of (config_path, config) is not None
        if bool(self.config_path is None) == bool(self.config is None):
            raise ValueError(
                "Exactly one of (config_path, config) must be not None, but "
                f"config_path is {self.config_path} and config is {self.config}"
            )
        if self.config_path is not None:
            with open(self.config_path, "r") as f:
                self.config: ConfigDict = yaml.safe_load(f)

        # Check that X and y have correct shape
        X, y = common.check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = sklearn.utils.multiclass.unique_labels(y)

        self.X_ = X
        self.y_ = y

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def write(
        self,
        model_dir,
        dump_model=True,
        dump_conf_mat_plots=True,
        dump_config=True,
        dump_loss_plots=True,
        dump_predictions=True,
        dump_distribution_plots=True,
    ):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Save the model
        if dump_model:
            with open(f"{model_dir}/model.pkl", "wb") as f:
                pickle.dump(self, f)

        # Save plots of the confusion matrices
        if dump_conf_mat_plots:
            X_val, y_val = self.validation_data
            vis.plot_conf_mats(
                self,
                (self.X_, X_val),
                (self.y_, y_val),
                ("Training", "Validation"),
            )
            plt.savefig(f"{model_dir}/conf_mats.png")

        # Save plots of the loss over time
        if dump_loss_plots:
            h = self.history.history
            fig, axs = plt.subplots(1, len(h.items()), figsize=(4 * len(h.items()), 3))
            for ax, (key, values) in zip(axs, h.items()):
                ax.plot(self.model.history.epoch, values, label=key)
                ax.set_title(key)
                ax.set(title=key, ylim=(0, np.max(values)))
            plt.savefig(f"{model_dir}/loss_plots.png")

        # Save the predictions
        if dump_predictions:
            X_val, y_val = self.validation_data
            y_val_pred = self.predict(X_val)
            y_trn_pred = self.predict(self.X_)
            np.savez(f"{model_dir}/y_pred_y_val.npz", y_pred=y_val_pred, y_val=y_val)
            np.savez(f"{model_dir}/y_pred_y_trn.npz", y_pred=y_trn_pred, y_trn=self.y_)

        # Save the config
        if dump_config:
            with open(f"{model_dir}/config.yaml", "w") as f:
                yaml.safe_dump(self.config, f)

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


class MeanClassifier(TemplateClassifier):
    def fit(self, X, y):
        self._check_fit(X, y)
        self.means = np.zeros((51, 30))
        for g in range(51):
            data = X[y == g]
            self.means[g] = data.mean(axis=0)
        # Return the classifier
        self.is_fitted_ = True
        return self

    def predict(self, X):
        # Check is fit had been called
        sk_validation.check_is_fitted(self, ["X_", "y_"])

        # Input validation
        X = sk_validation.check_array(X)
        result = np.empty((X.shape[0]))
        for i, xi in enumerate(X):
            result[i] = np.argmin(np.linalg.norm(self.means - xi, axis=(1, 2)))
        return result


class OneNearestNeighbourClassifier(TemplateClassifier):
    """Classifies data based on the single nearest neighbour"""

    def fit(self, X, y):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        self._check_fit(X, y)
        # Return the classifier
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """A reference implementation of a prediction for a classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        sk_validation.check_is_fitted(self, ["X_", "y_"])

        # Input validation
        X = sk_validation.check_array(X)

        closest = np.argmin(sklearn.metrics.euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]


class HMMClassifier(TemplateClassifier):
    def fit(self, X, y, validation_data, verbose=False, limit=None, **kwargs) -> None:
        self._check_fit(X, y)
        assert (
            X.shape[1] == self.config["n_timesteps"]
        ), f"{X.shape[1]} != {self.config['n_timesteps']}"

        # FIXME this assumes all X are continuous, which they're not
        X = np.concatenate(X)
        y = np.repeat(y, self.config["n_timesteps"])

        self.X_ = X
        self.y_ = y

        self.models_ = {}
        should_limit = limit is not None
        if verbose:
            if should_limit:
                total = sum(min(limit, len(X[y == yi])) for yi in np.unique(y))
            else:
                total = X.shape[0]
            pbar = tqdm.tqdm(total=total)
        for yi in np.unique(y):
            if not should_limit:
                limit = np.sum(y == yi)
            if verbose:
                pbar.desc = f"{yi}: ({limit}) {len(X[y == yi][:limit])}"
            self.models_[yi] = hmm.GaussianHMM(
                n_components=self.config["n_timesteps"] + 2,
                covariance_type="diag",
                n_iter=10,
                verbose=False,
            ).fit(X[y == yi][:limit])
            if verbose:
                pbar.update(len(X[y == yi][:limit]))

        self.is_fitted_ = True
        return self

    def predict(self, X, verbose=False):
        predictions = np.empty(X.shape[0])
        if verbose:
            pbar = tqdm.tqdm(total=X.shape[0])

        for i, xi in enumerate(X):
            best_key = None
            best_score = float("-inf")
            for key, m in self.models_.items():
                score = m.score(xi)
                if score > best_score:
                    best_score = score
                    best_key = key
            predictions[i] = best_key
            if verbose:
                pbar.update(1)
        return predictions

    def predict_score(self, X, verbose=False):
        scores = np.empty((X.shape[0], len(self.models_)))

        if verbose:
            pbar = tqdm.tqdm(total=X.shape[0])

        for i, xi in enumerate(X):
            scores[i] = [model.score(xi) for model in self.models_.values()]
            if verbose:
                pbar.update(1)
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

    def _deviant_sensors_to_gesture(self, sensors) -> str | None:
        gestures_to_sensors: dict[str, set[str]] = {}
        # Construct a dictionary that maps between sensors and the gestures
        # they're often associated with.
        for i in range(51):
            l_or_r = "l" if i % 10 < 5 else "r"
            number = {
                0: "5",
                1: "4",
                2: "3",
                3: "2",
                4: "1",
                5: "1",
                6: "2",
                7: "3",
                8: "4",
                9: "5",
            }[i % 10]
            xy_or_yz = "xy" if i % 10 not in (4, 5) else "yz"
            gestures_to_sensors[f"gesture{i:0>4}"] = {
                f"{l_or_r}{number}{xy_or_yz[0]}",
                f"{l_or_r}{number}{xy_or_yz[1]}",
            }

        for g, s in gestures_to_sensors.keys():
            if s.issubset(sensors):
                return g
        else:
            return None

    def _cusum(x, target=None, std_dev=None, allowed_std_devs=5):
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
        cusum_pos = pd.Series(np.zeros(len(x)))
        cusum_pos[0] = max(0, x[0] - upper_limit)
        for i in range(1, len(x)):
            cusum_pos[i] = max(0, cusum_pos[i - 1] + x[i] - upper_limit)

        # Calculate the cusum for the lower limit
        cusum_neg = pd.Series(np.zeros(len(x)))
        cusum_neg[0] = min(0, x[0] - lower_limit)
        for i in range(1, len(x)):
            cusum_neg[i] = min(0, cusum_neg[i - 1] + x[i] - lower_limit)

        # Create arrays of booleans describing if the value was too high/too low
        too_high = cusum_pos.apply(lambda cp: 0 if cp == 0 else 1)
        too_low = cusum_neg.apply(lambda cn: 0 if cn == 0 else 1)

        return pd.DataFrame(
            {
                "x": x,
                "target": target,
                "std_dev": std_dev,
                "allowed_std_devs": allowed_std_devs,
                "upper_limit": upper_limit,
                "lower_limit": lower_limit,
                "cusum_pos": cusum_pos,
                "cusum_neg": cusum_neg,
                "too_high": too_high,
                "too_low": too_low,
            }
        )

    def fit(self, X, y, validation_data, **kwargs) -> None:
        self._check_fit(X, y)
        raise NotImplementedError
        self.is_fitted_ = True

    def predict(self, X):
        # For each xi in X
        # For each time series ts in xi
        # perform CuSUM with upper and lower values learnt from fitting
        # Alert if CuSUM passes threshold
        raise NotImplementedError


class TFClassifier(TemplateClassifier):
    """Just an abstract class for TensorFlow-style models"""

    def predict(self, X):
        """Give label predictions for each observation in X"""
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        """Give label probabilities for each observation in X"""
        logits = self.model(X)
        l.info(logits)
        return tf.nn.softmax(logits).numpy()

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
    """

    Examples
    --------
    >>> X_trn, X_val, y_trn, y_val = train_test_split(X, y)
    >>> m = models.FFNNClassifier(config={
    ...     "n_timesteps": 30,
    ...     "ffnn": {
    ...         "epochs": 3,
    ...         "nodes_per_layer": [100],
    ...     }
    ... })
    >>> m.fit(X_trn, y_trn, validation_data=(X_val, y_val))
    >>> m.predict(X_trn[:1])[0]
    32
    """

    def __init__(
        self, config_path: Optional[str] = None, config: Optional[ConfigDict] = None
    ):
        self.config_path: Optional[str] = config_path
        self.config: Optional[ConfigDict] = config
        self.normalizer = None

    def fit(self, X, y, validation_data, **kwargs) -> None:
        l.info("Checking fit of X, y")
        self._check_fit(X, y)
        self.validation_data = validation_data
        # Fit the normalizer if not already fitted.
        if self.normalizer is None:
            l.info("Fitting normalizer")
            self.normalizer = keras.layers.Normalization(axis=-2)
            self.normalizer.adapt(X)

        # Construct the fully connected layers from the model config
        dense_layers = [
            keras.layers.Dense(
                units=npl,
                activation="relu",
            )
            for npl in self.config["ffnn"]["nodes_per_layer"]
        ]

        # TODO this function is unused
        def init_biases(shape, dtype=None):
            inv_freqs = np.array(
                [1 / count for _class, count in zip(*np.unique(y, return_counts=True))]
            )
            return np.log(inv_freqs)

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
        # If validation and class weights have been supplied, then add the
        # sample weights to the validation data
        # validation_data = kwargs.get("validation_data", [])
        # print(f"val data: {validation_data}")
        # class_weight = kwargs.get("class_weight", False)
        # print(f"class_weight: {class_weight}")
        # if len(validation_data) == 2 and class_weight:
        #     print("Mutating validation data")
        #     self.uses_validation_weights = True
        #     X_val, y_val = kwargs["validation_data"]
        #     self.weights_val = np.diag(list(class_weight.values()))[y_val].sum(axis=1)
        #     print(f"weights_validation = {self.weights_val.shape}")
        #     kwargs["validation_data"] = (X_val, y_val, self.weights_val)

        l.info("Compiling model")
        optimizer = self._resolve_optimizer()
        # Compile the model using SCCE loss
        self.model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )

        l.info("Fitting model")
        # Fit the model to the data, with a number of epochs dictated by config
        self.history = self.model.fit(
            X,
            y,
            batch_size=self.config["nn"]["batch_size"],
            epochs=self.config["nn"]["epochs"],
            class_weight=calc_class_weights(y),
            validation_data=validation_data,
            **kwargs,
        )

        # Sklearn expects is_fitted_ to be True after fitting
        self.is_fitted_ = True


class RNNClassifier(TFClassifier):
    def __init__(self, config_path=None, config=None):
        self.config_path: Optional[str] = config_path
        self.config: Optional[ConfigDict] = config
        self.normalizer = None

    def fit(self, X, y, **kwargs):
        raise NotImplementedError
        self._check_fit(X, y)
        # Fit the normalizer if not already fitted.
        if self.normalizer is None:
            print("Fitting normalizer")
            self.normalizer = keras.layers.Normalization(axis=-2)
            self.normalizer.adapt(X)

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


class LSTMClassifier(TFClassifier):
    def __init__(self, config_path=None, config=None):
        self.config_path: Optional[str] = config_path
        self.config: Optional[ConfigDict] = config
        self.normalizer = None

    def fit(self, X, y, **kwargs):
        # TF.LSTM *requires* floats for matmul operations
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        self._check_fit(X, y)
        # Fit the normalizer if not already fitted.
        if self.normalizer is None:
            print("Fitting normalizer")
            self.normalizer = keras.layers.Normalization(axis=-2)
            self.normalizer.adapt(X)

        self.model = keras.models.Sequential(
            [
                # Shape [batch, time, features] => [batch, time, lstm_units]
                keras.layers.LSTM(35, return_sequences=False),
                # Shape => [batch, time, features]
                keras.layers.Dense(len(self.classes_)),
            ]
        )
        # Compile the model using the ADAM optimiser and SCCE loss
        self.model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.Adam(learning_rate=2.5e-5),
        )

        # Fit the model to the data, with a number of epochs dictated by config
        self.history = self.model.fit(
            X, y, batch_size=128, epochs=self.config["nn"]["epochs"], **kwargs
        )

        # Sklearn expects is_fitted_ to be True after fitting
        self.is_fitted_ = True
