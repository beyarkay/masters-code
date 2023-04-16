#!/usr/bin/env python
import ml_utils
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
        df = ml_utils.parse_csvs()
        with open("config.yaml", "r") as file:
            brk_config = yaml.safe_load(file)

        intersection = list(set(wb_config.keys()) & set(brk_config.keys()))
        print(intersection)

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
        ) = ml_utils.build_dataset(df, brk_config)

        print("Starting to fit")
        history, model = ml_utils.compile_and_fit(
            X_train,
            y_train,
            X_valid,
            y_valid,
            brk_config,
            i2g,
            verbose=1,
        )
        print("Evaluating model...")
        # Now load the (contiguous) test dataset and calculate metrics on
        # that
        df_test = ml_utils.parse_csvs("../gesture_data/test/")
        (_, _, _, _, _, X_test, y_test, _, _, _, _) = ml_utils.build_dataset(
            df_test, brk_config
        )
        y_test_pred = model.predict(X_test, verbose=0)
        mean_dtw = ml_utils.dtw_evaluation(y_test, y_test_pred, i2g)
        print("Evaluation complete, logging to W&B")
        wandb.log(
            {
                "dtw.mean": mean_dtw.mean(),
                "val_loss": history.history["val_loss"][-1],
            }
            | {f"dtw.{i2g(i)}": d for i, d in enumerate(mean_dtw)}
        )


if __name__ == "__main__":
    main()
