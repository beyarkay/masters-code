#!/usr/bin/env python
from ml_utils import *
from wandb.keras import WandbCallback
import sys
import wandb
import yaml


def main():
    # wandb.login()
    # with open("sweep.yaml", "r") as file:
    #     sweep_config = yaml.safe_load(file)
    # sweep_id = wandb.sweep(sweep_config, project="ergo")
    wandb.agent(f"beyarkay/ergo/{sys.argv[1]}", train, count=25)


def train(wb_config=None):
    with wandb.init(config=wb_config):
        wb_config = wandb.config
        df = parse_csvs()
        with open("config.yaml", "r") as file:
            brk_config = yaml.safe_load(file)

        intersection = list(set(wb_config.keys()) & set(brk_config.keys()))

        brk_config.update({k: wb_config[k] for k in intersection})

        (
            brk_config,
            g2i,
            i2g,
            i2ohe,
            ohe2i,
            X,
            y,
            X_train,
            X_valid,
            y_train,
            y_valid,
        ) = build_dataset(df, brk_config)

        print("Starting to fit")
        history, model = compile_and_fit(
            X_train,
            y_train,
            X_valid,
            y_valid,
            brk_config,
            i2g,
            verbose=0,
        )
        y_pred = model.predict(X, verbose=0)
        mean_dist = calc_mean_dist(y, y_pred, i2g).mean()

        print("Fit complete")
        wandb.log(
            {"rising_edge_dist": mean_dist, "val_loss": history.history["val_loss"][-1]}
        )


if __name__ == "__main__":
    main()
