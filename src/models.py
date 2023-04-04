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
"""

from hmmlearn import hmm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import unique_labels
from typing import Optional
import common
import numpy as np
import pickle
import sklearn.utils.validation as sk_validation
import tqdm
import typing
import yaml
import tensorflow as tf
from tensorflow import keras
from keras import layers
import logging as l


class CusumConfig(typing.TypedDict):
    thresh: int


class NNConfig(typing.TypedDict):
    epochs: int


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
        # Assert that exactly one of (config_path, config) is not None using
        # boolean xor `^`
        if bool(self.config_path is None) ^ bool(self.config is None):
            raise ValueError("Exactly one of `config` and `config_path` must be None")
        if self.config_path is not None:
            with open(self.config_path, "r") as f:
                self.config: ConfigDict = yaml.safe_load(f)

        # Check that X and y have correct shape
        X, y = common.check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def write(self, model_path):
        with open(f"{model_path}.pkl", "wb") as f:
            pickle.dump(self, f)

    def confusion_matrix(self, y_true, y_pred=None, X_to_pred=None):
        """Calculate the confusion matrix of some predictions.

        Either pass in the alread-predicted values via y_pred, or pass in some
        X data which will be predicted and then used to calculate the confusion
        matrix."""
        # Assert that exactly one of (y_pred, X_to_pred) is not None using
        # boolean xor `^`.
        if bool(y_pred is None) ^ bool(X_to_pred is None):
            raise ValueError("Exactly one of (y_pred, X_to_pred) must be not None")
        if X_to_pred is not None:
            y_pred = self.predict(X_to_pred)

        return tf.math.confusion_matrix(y_true, y_pred)


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

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]


class HMMClassifier(TemplateClassifier):
    def fit(self, X, y, verbose=False, limit=None):
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

    def _cusum(self, data: np.ndarray, lower: float, upper: float):
        """Calculate the cumulative sum of the data, using the upper/lower bounds."""
        mu = data.mean()
        sigma = data.std()
        z = (data - mu) / sigma
        s_upper = np.zeros(z.shape)
        s_lower = np.zeros(z.shape)
        for i in range(1, len(z)):
            s_upper[i] = max(0, s_upper[i - 1] + z[i] - upper)
            s_lower[i] = min(0, s_lower[i - 1] + z[i] + lower)
        return s_lower, s_upper

    def fit(self, X, y):
        self._check_fit(X, y)
        # Return the classifier
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

    def __init__(self, config_path=None, config=None):
        self.config_path: Optional[str] = config_path
        self.config: Optional[ConfigDict] = config
        self.normalizer = None

    def fit(self, X, y, **kwargs):
        l.info("Checking fit of X, y")
        self._check_fit(X, y)
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

        l.info("Compiling model")
        # Compile the model using the ADAM optimiser and SCCE loss
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=2.5e-5),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )

        l.info("Fitting model")
        # Fit the model to the data, with a number of epochs dictated by config
        self.history = self.model.fit(
            X, y, batch_size=128, epochs=self.config["nn"]["epochs"], **kwargs
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
