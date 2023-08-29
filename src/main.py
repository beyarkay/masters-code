from sklearn.model_selection import train_test_split
import vis
import save
import read
import pred
import models
import common
import datetime
import tensorflow as tf
import numpy as np
import itertools
import tqdm
import colorama as C
import pandas as pd
import os
from typing import cast
import typing
import argparse
import sys


# init terminal colours
C.init()
const = common.read_constants()

# This defines the default config for preprocessing the data
preprocessing_config: models.PreprocessingConfig = {
    "seed": 42,
    "n_timesteps": 20,
    "num_gesture_classes": None,
    "rep_num": None,
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


def load_dataset(path="./gesture_data/trn_20.npz"):
    trn = np.load(path)
    X = trn["X_trn"]
    y = trn["y_trn"]
    dt = trn["dt_trn"]
    return (X, y, dt)


def run_ffnn_hpar_opt(args):
    hpars_path = "saved_models/hpars_ffnn_opt.csv"
    jsonl_path = "saved_models/results_ffnn_opt_bigger.jsonl"
    print("Executing 'FFNN Hyperparameter Optimisation'")
    X, y, dt = load_dataset()

    hyperparameters = {
        "rep_num": range(5),
        "num_gesture_classes": (5, 50, 51),
        "nodes_in_layer_1": (20, 60, 100),
        "nodes_in_layer_2": (20, 60, 100),
        "learning_rate": (1e-2, 1e-3, 1e-4),
        "dropout_rate": (0.0, 0.3, 0.6),
        "l2_coefficient": (0, 1e-5, 1e-2),
    }
    iterables = list(hyperparameters.values())

    n_timesteps = 20
    epochs = 10
    max_obs_per_class = None

    items = itertools.product(*iterables)
    num_tests = int(np.prod([len(iterable) for iterable in iterables]))
    pbar = tqdm.tqdm(items, total=num_tests)
    for hpars_tuple in pbar:
        hpars = {k: v for k, v in zip(hyperparameters.keys(), hpars_tuple)}

        cont, hpars_df = should_continue(hpars_path, **hpars)
        if cont:
            continue

        now = datetime.datetime.now().isoformat(sep="T")[:-7]
        pbar.set_description(
            f"""{C.Style.BRIGHT}{now}  \
rep:{hpars['rep_num']: >2}  \
#classes:{hpars['num_gesture_classes']: >2}  \
nodes: [{hpars['nodes_in_layer_1']}, {hpars['nodes_in_layer_2']}] \
lr: {hpars['learning_rate']:#7.2g} \
dropout: {hpars['dropout_rate']:.2f} \
l2: {hpars['l2_coefficient']:#7.2g} \
"""
        )
        print(f"{C.Style.DIM}")
        preprocessing_config["n_timesteps"] = n_timesteps
        preprocessing_config["max_obs_per_class"] = max_obs_per_class
        preprocessing_config["gesture_allowlist"] = list(
            range(hpars['num_gesture_classes']))
        preprocessing_config["num_gesture_classes"] = hpars['num_gesture_classes']
        preprocessing_config["rep_num"] = hpars['rep_num']
        preprocessing_config["seed"] = 42 + hpars['rep_num']
        print(f"Making classifiers with preprocessing: {preprocessing_config}")
        (X_trn, X_val, y_trn, y_val, dt_trn, dt_val,) = train_test_split(
            X, y, dt, stratify=y, random_state=preprocessing_config["seed"]
        )
        nn_config: models.NNConfig = {
            "epochs": epochs,
            "batch_size": 256,
            "learning_rate": hpars['learning_rate'],
            "optimizer": "adam",
        }
        ffnn_config: models.FFNNConfig = {
            "nodes_per_layer": [hpars['nodes_in_layer_1'], hpars['nodes_in_layer_2']],
            "l2_coefficient": hpars['l2_coefficient'],
            "dropout_rate": hpars['dropout_rate'],
        }
        config: models.ConfigDict = {
            "model_type": "FFNN",
            "preprocessing": preprocessing_config,
            "nn": nn_config,
            "ffnn": ffnn_config,
            "cusum": None,
            "lstm": None,
            "hmm": None,
        }
        clf = models.FFNNClassifier(config=config)

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
        clf.write_as_jsonl(jsonl_path)
        # NOTE: This save MUST come last, so that we don't accidentally
        # record us having trained a model when we have not.
        hpars_df.to_csv(hpars_path, index=False)


def run_experiment_hmm(args):
    hpars_path = "saved_models/hpars_hmm_opt.csv"
    jsonl_path = "saved_models/results_hmm_opt_bigger.jsonl"
    print("Executing 'HMM Hyperparameter Optimisation'")
    X, y, dt = load_dataset()

    hyperparameters = {
        "rep_num": range(30),
        "num_gesture_classes": (5, 50, 51),
    }
    iterables = list(hyperparameters.values())

    n_timesteps = 20
    max_obs_per_class = 200

    items = itertools.product(*iterables)
    num_tests = int(np.prod([len(iterable) for iterable in iterables]))
    pbar = tqdm.tqdm(items, total=num_tests)
    for hpars_tuple in pbar:
        hpars = {k: v for k, v in zip(hyperparameters.keys(), hpars_tuple)}

        cont, hpars_df = should_continue(hpars_path, **hpars)
        if cont:
            continue

        now = datetime.datetime.now().isoformat(sep="T")[:-7]
        pbar.set_description(
            f"""{C.Style.BRIGHT}{now}  \
rep:{hpars['rep_num']: >2}  \
#classes:{hpars['num_gesture_classes']: >2}  \
"""
        )
        print(f"{C.Style.DIM}")
        preprocessing_config["n_timesteps"] = n_timesteps
        preprocessing_config["max_obs_per_class"] = max_obs_per_class
        preprocessing_config["gesture_allowlist"] = list(
            range(hpars['num_gesture_classes']))
        preprocessing_config["num_gesture_classes"] = hpars['num_gesture_classes']
        preprocessing_config["rep_num"] = hpars['rep_num']
        preprocessing_config["seed"] = 42 + hpars['rep_num']
        print(f"Making classifier with preprocessing: {preprocessing_config}")
        (X_trn, X_val, y_trn, y_val, dt_trn, dt_val,) = train_test_split(
            X, y, dt, stratify=y, random_state=preprocessing_config["seed"]
        )
        config: models.ConfigDict = {
            "model_type": "HMM",
            "preprocessing": preprocessing_config,
            "hmm": {
                "n_iter": 20,
            },
            "nn": None,
            "ffnn": None,
            "cusum": None,
            "lstm": None,
        }
        clf = models.HMMClassifier(config=config)
        try:
            clf.fit(
                X_trn,
                y_trn,
                dt_trn,
                validation_data=(X_val, y_val, dt_val),
                verbose=False,
            )
        except TimeoutError as e:
            print(f"Timed out while fitting: {e}")
        print("Saving model")
        clf.write_as_jsonl(jsonl_path)
        # NOTE: This save MUST come last, so that we don't accidentally
        # record us having trained a model when we have not.
        hpars_df.to_csv(hpars_path, index=False)


def run_experiment_cusum(args):
    hpars_path = "saved_models/hpars_cusum_opt.csv"
    jsonl_path = "saved_models/results_cusum_opt_bigger.jsonl"
    print("Executing 'cusum Hyperparameter Optimisation'")
    X, y, dt = load_dataset()

    hyperparameters = {
        "rep_num": range(5),
        "num_gesture_classes": (5, 50, 51),
        "thresh": (5, 10, 20, 40, 60, 80, 100),
    }
    iterables = list(hyperparameters.values())

    n_timesteps = 20
    max_obs_per_class = 200

    items = itertools.product(*iterables)
    num_tests = int(np.prod([len(iterable) for iterable in iterables]))
    pbar = tqdm.tqdm(items, total=num_tests)
    for hpars_tuple in pbar:
        hpars = {k: v for k, v in zip(hyperparameters.keys(), hpars_tuple)}

        cont, hpars_df = should_continue(hpars_path, **hpars)
        if cont:
            continue

        now = datetime.datetime.now().isoformat(sep="T")[:-7]
        pbar.set_description(
            f"""{C.Style.BRIGHT}{now}  \
rep:{hpars['rep_num']: >2}  \
#classes:{hpars['num_gesture_classes']: >2}  \
"""
        )
        print(f"{C.Style.DIM}")
        preprocessing_config["n_timesteps"] = n_timesteps
        preprocessing_config["max_obs_per_class"] = max_obs_per_class
        preprocessing_config["gesture_allowlist"] = list(
            range(hpars['num_gesture_classes']))
        preprocessing_config["num_gesture_classes"] = hpars['num_gesture_classes']
        preprocessing_config["rep_num"] = hpars['rep_num']
        preprocessing_config["seed"] = 42 + hpars['rep_num']
        print(f"Making classifier with preprocessing: {preprocessing_config}")
        (X_trn, X_val, y_trn, y_val, dt_trn, dt_val,) = train_test_split(
            X, y, dt, stratify=y, random_state=preprocessing_config["seed"]
        )
        config: models.ConfigDict = {
            "model_type": "CuSUM",
            "preprocessing": preprocessing_config,
            "cusum": {
                "thresh": hpars['thresh'],
            },
            "nn": None,
            "ffnn": None,
            "hmm": None,
            "lstm": None,
        }
        clf = models.CuSUMClassifier(config=config)
        try:
            clf.fit(
                X_trn,
                y_trn,
                dt_trn,
                validation_data=(X_val, y_val, dt_val),
                verbose=False,
            )
        except TimeoutError as e:
            print(f"Timed out while fitting: {e}")
        print("Saving model")
        clf.write_as_jsonl(jsonl_path)
        # NOTE: This save MUST come last, so that we don't accidentally
        # record us having trained a model when we have not.
        hpars_df.to_csv(hpars_path, index=False)


def should_continue(
    hpars_path, **kwargs
) -> tuple[bool, pd.DataFrame]:
    hpars = (
        pd.read_csv(hpars_path)
        if os.path.exists(hpars_path)
        else pd.DataFrame(columns=list(kwargs.keys()))
    )
    new_hpar_line = {k: [v] for k, v in kwargs.items()}
    hpars = pd.concat([pd.DataFrame(new_hpar_line), hpars], ignore_index=True)
    duplicated = hpars.duplicated()
    if duplicated.any():
        print(
            f"{C.Style.NORMAL}{C.Fore.YELLOW}Already trained this model:\n{hpars[duplicated]}{C.Style.DIM}{C.Fore.RESET}"  # noqa: E501
        )
        hpars = cast(pd.DataFrame, hpars.drop_duplicates())
        return (True, hpars)
    return (False, hpars)


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
            "l2_coefficient": 0.0,
            "dropout_rate": 0.0,
        },
        "cusum": None,
        "lstm": None,
        "hmm": None,
        # "n_timesteps": -1,
    }

    ffnn_clf = models.FFNNClassifier(config=config)
    return ffnn_clf


def predict_from_serial(args):
    now = datetime.datetime.now().isoformat(sep="T")[:-7]
    print(f"Predicting from serial with model '{args.predict_with}'")
    clf = models.load_tf(args.predict_with)
    reading = read.find_port()
    if reading is None:
        print("Could not infer serial port number")
        sys.exit(1)
    port_name, baud_rate = reading

    handlers = [
        read.ReadLineHandler(port_name=port_name, baud_rate=baud_rate),
        read.ParseLineHandler(),
        pred.PredictGestureHandler(clf),
        # pred.SpellCheckHandler(),
        vis.StdOutHandler(),
        save.SaveHandler(f"tmp_{now}.csv"),
    ]
    print("Executing handlers")
    read.execute_handlers(handlers)

    sys.exit(0)


def save_from_serial(args):
    print("saving from serial")
    reading = read.find_port()
    if reading is None:
        print("Could not infer serial port number")
        sys.exit(1)
    port_name, baud_rate = reading
    labels = [[s.strip() for s in label.split(",")] for label in args.labels]
    labels = [item for sublist in labels for item in sublist]
    now = datetime.datetime.now().isoformat(sep="T")[:-7]
    handlers = [
        read.ReadLineHandler(port_name=port_name, baud_rate=baud_rate),
        read.ParseLineHandler(),
        vis.InsertLabelHandler(labels=labels),
        vis.StdOutHandler(),
        save.SaveHandler(f"gesture_data/tmp_{now}.csv"),
    ]
    print("Executing handlers")
    read.execute_handlers(handlers)

    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The main entrypoint for Ergo")
    # Optional positional argument
    parser.add_argument(
        "-p",
        "--predict-with",
        action='store',
        help="The directory containing the model",
    )
    parser.add_argument(
        "-s",
        "--save",
        action='store_true',
        help="Whether or not to save the data from the serial port",
    )

    parser.add_argument(
        "-l",
        "--labels",
        action='append',
        help="The labels of gestures to cycle through, like g0001, or g0045",
    )

    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        help="The experiment to run",
    )

    args = parser.parse_args()
    if args.predict_with:
        predict_from_serial(args)
    elif args.save:
        save_from_serial(args)
    elif args.experiment == 'cusum':
        run_experiment_cusum(args)
    elif args.experiment == 'ffnn':
        run_ffnn_hpar_opt(args)
    elif args.experiment == 'hmm':
        run_experiment_hmm(args)
    else:
        main(args)
