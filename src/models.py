"""Defines the models which are used for prediction/classification.

These models all use the [sklearn
API](https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator))
for estimators to ensure that they can be used with sklearn's GridSearchCV and
RandomSearchCV. This will make checking multimple hyperparameters a lot easier.

"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import sklearn.utils.validation as sk_validation
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
import typing
import yaml


class ConfigDict(typing.TypedDict):
    n_timesteps: int


class TemplateClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, config_path):
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

    def fit(self, X, y):
        with open(self.config_path, "r") as f:
            self.config: ConfigDict = yaml.safe_load(f)
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


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
        with open(self.config_path, "r") as f:
            self.config: ConfigDict = yaml.safe_load(f)
            self.config["n_timesteps"] = 1
        # Check that X and y have correct shape
        X, y = sk_validation.check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
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
    def __init__(self, config_path):
        raise NotImplementedError

    def fit(self, X, y):
        with open(self.config_path, "r") as f:
            self.config: ConfigDict = yaml.safe_load(f)
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class NNClassifier(TemplateClassifier):
    def __init__(self, config_path):
        raise NotImplementedError

    def fit(self, X, y):
        with open(self.config_path, "r") as f:
            self.config: ConfigDict = yaml.safe_load(f)
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class CuSUMClassifier(TemplateClassifier):
    def __init__(self, config_path):
        raise NotImplementedError

    def fit(self, X, y):
        with open(self.config_path, "r") as f:
            self.config: ConfigDict = yaml.safe_load(f)
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
