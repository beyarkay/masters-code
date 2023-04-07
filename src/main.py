import models
import vis
import pred
import read
import os
import logging as l
import datetime
import pandas as pd
import tqdm
import common
import sklearn
import sys


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
    l.info("Making model")
    model = models.FFNNClassifier(
        config={
            "n_timesteps": 30,
            "ffnn": {
                "epochs": 2,
                "nodes_per_layer": [100],
            },
        }
    )
    l.info("Reading data")
    df: pd.DataFrame = read.read_data().iloc[:50_000]
    l.info("Making windows")
    X, y_str = read.make_windows(
        df,
        model.config["n_timesteps"],
        pbar=tqdm.tqdm(total=len(df), desc="Making windows"),
    )
    g2i, i2g = common.make_gestures_and_indices(y_str)
    y = g2i(y_str)

    l.info("Fitting model")
    model.fit(X, y)
    return model


if __name__ == "__main__":
    main()
