import models
import vis
import pred
import read
import os
import logging as l
import datetime
import pandas as pd
import numpy as np


def main():
    init_logs()
    model = models.OneNearestNeighbourClassifier("src/config.yaml")
    df: pd.DataFrame = read.read_data()
    y: np.ndarray = df["gesture"].values
    X: np.ndarray = df.drop(columns=["gesture", "datetime"]).values
    model.fit(X, y)
    handlers = [
        read.ReadLineHandler(mock="gesture_data/train/2022-10-19T19:22:46.781569.csv"),
        read.ParseLineHandler(),
        pred.PredictGestureHandler(model),
        vis.StdOutHandler(),
    ]
    read.execute_handlers(handlers)


def init_logs():
    """Initiate logging to a file with the current timestamp as the filename.

    This function configures the logging module to write log messages to a file
    with a name that includes the current timestamp. The log file is created in
    a "logs" directory in the current working directory. The logging level is
    set to INFO, which means that messages with a severity level of INFO or
    higher will be logged.

    Returns:
        None
    """
    if not os.path.exists("logs"):
        os.mkdir("logs")

    l.basicConfig(
        filename=f"logs/{str(datetime.datetime.now()).replace(' ', 'T')}.log",
        format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=l.INFO,
    )


if __name__ == "__main__":
    main()
