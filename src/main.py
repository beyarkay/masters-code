import logging as l

import common
import models
import numpy as np
import pred
import read
import sklearn
import vis


def _():
    """
    1. train some models
    2. save y_pred, y_true, and the indexes of y_true to disk
    3. save the model to disk
    4. save some summary statistics to disk, so long as they don't take too
    long to calculate and don't slow down training
        - Maybe save confusion matrix as a np array?
        - save loss
    """
    pass


def main():
    common.init_logs()
    model: models.TemplateClassifier = get_model()
    handlers = [
        read.ReadLineHandler(mock="gesture_data/train/2022-10-19T19:22:46.781569.csv"),
        read.ParseLineHandler(),
        pred.PredictGestureHandler(model),
        vis.StdOutHandler(),
    ]
    print("Executing handlers")
    read.execute_handlers(handlers)


def get_model() -> models.TemplateClassifier:
    pass


def make_ffnn() -> models.TemplateClassifier:
    l.info("Making model")
    model = models.FFNNClassifier(
        config={
            "n_timesteps": 30,
            "nn": {"epochs": 2},
            "ffnn": {
                "nodes_per_layer": [100],
            },
        }
    )
    l.info("Reading data")
    trn = np.load("./gesture_data/trn.npz")
    X = trn["X_trn"]
    y = trn["y_trn"]

    X_trn, X_val, y_trn, y_val = sklearn.model_selection.train_test_split(
        X, y, stratify=y
    )

    l.info("Fitting model")
    model.fit(X_trn, y_trn)
    return model


if __name__ == "__main__":
    main()
