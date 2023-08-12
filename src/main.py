import argparse
import typing
from typing import cast
import os
import pandas as pd
import colorama as C
import tqdm
import itertools
import numpy as np
import tensorflow as tf

import datetime
import common
import models
import pred
import read
import save
import vis
from sklearn.model_selection import train_test_split

C.init()
const = common.read_constants()

# This defines the default config for preprocessing the data
preprocessing_config: models.PreprocessingConfig = {
    "seed": 42,
    "n_timesteps": 40,
    "num_gesture_classes": None,
    "rep_num": None,
    "delay": 0,
    "max_obs_per_class": None,
    "gesture_allowlist": [
        # fmt: off
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
        50,
        # fmt: on
    ],
}


def main(args):
    common.init_logs()

    model: models.TemplateClassifier = make_ffnn(preprocessing_config)
    # TODO read model from file
    model_path = "saved_models/ffnn/2023-05-13T11:53:30/config.yaml"
    reading = read.find_port()
    if reading is not None:
        port_name, baud_rate = reading
    handlers = [
        # read.ReadLineHandler(port_name=port_name, baud_rate=baud_rate),
        read.ReadLineHandler(
            mock="gesture_data/train/2022-11-06T18:49:26.928149.csv"),
        read.ParseLineHandler(),
        pred.PredictGestureHandler(model),
        pred.SpellCheckHandler(),
        vis.StdOutHandler(),
        # save.SaveHandler("tmp.csv"),
    ]
    print("Executing handlers")
    read.execute_handlers(handlers)


def run_ffnn_hpar_opt(args):
    print("Executing 'FFNN Hyperparameter Optimisation'")
    trn = np.load("./gesture_data/trn_40.npz")
    X = trn["X_trn"]
    y = trn["y_trn"]
    dt = trn["dt_trn"]

    REP_DOMAIN = range(30)
    LEARNING_RATE_DOMAIN = np.power(10, -np.linspace(2.0, 6.0, 5))
    NODES_IN_LAYER_1_DOMAIN = np.linspace(20, 200, 3)
    NODES_IN_LAYER_2_DOMAIN = np.linspace(20, 200, 3)
    iterables = [
        REP_DOMAIN,
        NODES_IN_LAYER_1_DOMAIN,
        NODES_IN_LAYER_2_DOMAIN,
        LEARNING_RATE_DOMAIN,
    ]

    n_timesteps = 40
    epochs = 20
    max_obs_per_class = None
    delay = 0
    num_gesture_classes = 51

    items = itertools.product(*iterables)
    num_tests = int(np.prod([len(iterable) for iterable in iterables]))
    pbar = tqdm.tqdm(items, total=num_tests)
    for item in pbar:
        (
            rep_num,
            nodes_in_layer_1,
            nodes_in_layer_2,
            learning_rate,
        ) = item

        now = datetime.datetime.now()
        pbar.set_description(
            f"""{C.Style.BRIGHT}{now}  \
            rep:{rep_num: >2}  \
            nodes: [{nodes_in_layer_1}, {nodes_in_layer_2}] \
            lr: {learning_rate:#7.2g}"""
        )
        print(f"{C.Style.DIM}")
        preprocessing_config["n_timesteps"] = n_timesteps
        preprocessing_config["max_obs_per_class"] = max_obs_per_class
        preprocessing_config["delay"] = delay
        preprocessing_config["gesture_allowlist"] = list(
            range(num_gesture_classes))
        preprocessing_config["num_gesture_classes"] = num_gesture_classes
        preprocessing_config["rep_num"] = rep_num
        preprocessing_config["seed"] = 42 + rep_num
        print(f"Making classifiers with preprocessing: {preprocessing_config}")
        (X_trn, X_val, y_trn, y_val, dt_trn, dt_val,) = train_test_split(
            X, y, dt, stratify=y, random_state=preprocessing_config["seed"]
        )
        hpars_path = "saved_models/hpars_ffnn_opt.csv"
        print(rep_num, nodes_in_layer_1, nodes_in_layer_2, learning_rate)
        cont, hpars = should_continue(
            hpars_path,
            rep_num=rep_num,
            nodes_in_layer_1=nodes_in_layer_1,
            nodes_in_layer_2=nodes_in_layer_2,
            learning_rate=learning_rate,
        )
        if cont:
            continue

        clf = models.FFNNClassifier(config={
            "model_type": "FFNN",
            "preprocessing": preprocessing_config,
            "nn": {
                "epochs": epochs,
                "batch_size": 256,
                "learning_rate": learning_rate,
                "optimizer": "adam",
            },
            "ffnn": {
                "nodes_per_layer": [nodes_in_layer_1, nodes_in_layer_2],
            },
            "cusum": None,
            "lstm": None,
            "hmm": None,
            "n_timesteps": -1,
        })

        tf.keras.backend.clear_session()
        try:
            clf.fit(
                X_trn,
                y_trn,
                dt_trn,
                validation_data=(X_val, y_val, dt_val),
                verbose=True,
            )
            print(f"{clf.X_.shape=}, {clf.validation_data[0].shape=}")
        except TimeoutError as e:
            print(f"Timed out while fitting: {e}")
        now = datetime.datetime.now().isoformat(sep="T")[:-7]
        print("Saving model")
        clf.write_as_jsonl("saved_models/results_ffnn_opt.jsonl")
        # NOTE: This save MUST come last, so that we don't accidentally
        # record us having trained a model when we have not.
        hpars.to_csv(hpars_path, index=False)


def run_experiment_01(args):
    print("Executing experiment 01")
    trn = np.load("./gesture_data/trn_40.npz")
    X = trn["X_trn"]
    y = trn["y_trn"]
    dt = trn["dt_trn"]

    REP_DOMAIN = range(30)
    NUM_GESTURE_CLASSES_DOMAIN = (5, 20, 35, 50, 51)
    iterables = [
        REP_DOMAIN,
        NUM_GESTURE_CLASSES_DOMAIN,
    ]
    items = itertools.product(*iterables)
    num_tests = int(np.prod([len(iterable) for iterable in iterables]))
    pbar = tqdm.tqdm(items, total=num_tests)
    for item in pbar:
        (
            rep_num,
            num_gesture_classes,
        ) = item
        n_timesteps = 40
        max_obs_per_class = 200
        delay = 0

        now = datetime.datetime.now()
        pbar.set_description(
            f"{C.Style.BRIGHT}{now} rep:{rep_num: >2} nClasses:{num_gesture_classes: >2}"  # noqa: E501
        )
        print(f"{C.Style.DIM}")
        preprocessing_config["n_timesteps"] = n_timesteps
        preprocessing_config["max_obs_per_class"] = max_obs_per_class
        preprocessing_config["delay"] = delay
        preprocessing_config["gesture_allowlist"] = list(
            range(num_gesture_classes))
        preprocessing_config["num_gesture_classes"] = num_gesture_classes
        preprocessing_config["rep_num"] = rep_num
        preprocessing_config["seed"] = 42 + rep_num
        print(f"Making classifiers with preprocessing: {preprocessing_config}")
        model_types = {"HMM": make_hmm, "CuSUM": make_cusum, "FFNN": make_ffnn}
        (X_trn, X_val, y_trn, y_val, dt_trn, dt_val,) = train_test_split(
            X, y, dt, stratify=y, random_state=preprocessing_config["seed"]
        )
        for model_type, make_model_fn in model_types.items():
            hpars_path = "saved_models/hpars.csv"
            cont, hpars = should_continue(
                hpars_path,
                rep_num=rep_num,
                n_timesteps=n_timesteps,
                num_gesture_classes=num_gesture_classes,
                max_obs_per_class=max_obs_per_class,
                delay=delay,
                model_type=model_type,
            )
            if cont:
                continue
            clf = make_model_fn(preprocessing_config)
            print(
                f"{C.Style.BRIGHT}{C.Fore.BLUE}Training model {clf.config['model_type']}{C.Fore.RESET}{C.Style.DIM}"
            )
            tf.keras.backend.clear_session()
            try:
                clf.fit(
                    X_trn,
                    y_trn,
                    dt_trn,
                    validation_data=(X_val, y_val, dt_val),
                    verbose=False,
                )
                print(f"{clf.X_.shape=}, {clf.validation_data[0].shape=}")
            except TimeoutError as e:
                print(f"Timed out while fitting: {e}")
            now = datetime.datetime.now().isoformat(sep="T")[:-7]
            print("Saving model")
            clf.write_as_jsonl("saved_models/results.jsonl")
            # NOTE: This save MUST come last, so that we don't accidentally
            # record us having trained a model when we have not.
            hpars.to_csv(hpars_path, index=False)


def should_continue(
    hpars_path, **kwargs
) -> tuple[bool, pd.DataFrame]:
    hpars = (
        pd.read_csv(hpars_path)
        if os.path.exists(hpars_path)
        else pd.DataFrame(columns=list(kwargs.keys()))
    )
    new_hpar_line = {k: [v] for k, v in kwargs.items()}
    print("new_hpar_line", new_hpar_line)
    print("kwargs", kwargs)
    hpars = pd.concat([pd.DataFrame(new_hpar_line), hpars], ignore_index=True)
    duplicated = hpars.duplicated()
    if duplicated.any():
        print(
            f"{C.Style.NORMAL}{C.Fore.YELLOW}Already trained this model:\n{hpars[duplicated]}{C.Style.DIM}{C.Fore.RESET}"  # noqa: E501
        )
        hpars = cast(pd.DataFrame, hpars.drop_duplicates())
        return (True, hpars)
    return (False, hpars)


def make_cusum(preprocessing_config: models.PreprocessingConfig):
    cusum_clf = models.CuSUMClassifier(
        config={"preprocessing": preprocessing_config, "cusum": {"thresh": 100}}
    )
    return cusum_clf


def make_hmm(preprocessing_config: models.PreprocessingConfig):
    hmm_clf = models.HMMClassifier(
        config={
            "preprocessing": preprocessing_config,
            "hmm": {"n_iter": 20},
        }
    )
    return hmm_clf


def make_ffnn(preprocessing_config: models.PreprocessingConfig):
    config: models.ConfigDict = {
        "model_type": "FFNN",
        "preprocessing": preprocessing_config,
        "nn": {
            "epochs": 20,
            "batch_size": 205,
            "learning_rate": 0.0005229989312667862,
            "optimizer": "adam",
        },
        "ffnn": {
            "nodes_per_layer": [36, 84, 271],
        },
        "cusum": None,
        "lstm": None,
        "hmm": None,
        "n_timesteps": -1,
    }

    ffnn_clf = models.FFNNClassifier(config=config)
    return ffnn_clf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The main entrypoint for Ergo")
    # Optional positional argument
    parser.add_argument(
        "--experiment",
        type=int,
        help="The experiment number to run",
    )

    args = parser.parse_args()
    if args.experiment == 1:
        run_experiment_01(args)
    if args.experiment == 2:
        run_ffnn_hpar_opt(args)
    else:
        main(args)
