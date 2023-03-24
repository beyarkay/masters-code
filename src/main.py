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


def main():
    init_logs()
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
