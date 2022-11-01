import unittest
import numpy as np
import yaml
import pandas as pd
import ergo as E
import ml_utils as utils


class TestErgo(unittest.TestCase):
    def test_serial_port(self):
        self.assertRaisesRegex(SystemExit, "1", E.get_serial_port)

    def test_gesture_info(self):
        val = E.gesture_info()
        keys = [f"gesture{i:0>4}" for i in (list(range(0, 50)) + [255])]
        self.assertEqual(list(val.keys()), keys)

    def test_format_prediction(self):
        val = E.format_prediction("gesture0042", 0.9, g2k={"gesture0042": "g"})
        self.assertEqual(val, "\x1b[30;48;2;247;131;16mg90%\x1b[0m")

        val = E.format_prediction("gesture0042", 0.9, g2k={"gesture0042": "\t"})
        self.assertEqual(val, "\x1b[30;48;2;247;131;16m\\t90%\x1b[0m")

        val = E.format_prediction("gesture0042", 0.9, g2k={"gesture0042": "\r"})
        self.assertEqual(val, "\x1b[30;48;2;247;131;16m\r90%\x1b[0m")

        val = E.format_prediction("gesture0042", 0.9, g2k={"gesture0042": "\n"})
        self.assertEqual(
            val,
            "\x1b[30;48;2;247;131;16m\\n90%\x1b[0m",
        )

        val = E.format_prediction("gesture0042", 0.9)
        self.assertEqual(val, "\x1b[30;48;2;247;131;16mg42 90%\x1b[0m")

        val = E.format_prediction("gesture0042", 0.1)
        self.assertEqual(val, "\x1b[37;48;2;15;9;45mg42 10%\x1b[0m")

        val = E.format_prediction("gesture0255", 0.9)
        self.assertEqual(val, "\x1b[30;48;2;247;131;16mg255 90%\x1b[0m")

    def test_get_colored_string(self):
        val = E.get_colored_string(20, 100)
        self.assertEqual(val, "\x1b[30;48;2;70;130;248m 20\x1b[0m")

        val = E.get_colored_string(2, 10)
        self.assertEqual(val, "\x1b[30;48;2;54;168;249m  2\x1b[0m")

    def test_color_bg(self):
        val = E.color_bg("10")
        self.assertEqual(val, "\x1b[48;2;86;39;114m10\x1b[0m")

        val = E.color_bg("abdc10")
        self.assertEqual(val, "\x1b[48;2;86;39;114mabdc10\x1b[0m")

        val = E.color_bg("gesture0001")
        self.assertEqual(val, "\x1b[48;2;66;123;94mgesture0001\x1b[0m")

        self.assertRaises(ValueError, lambda: E.color_bg("abcd"))

    def test_get_color(self):
        self.assertEqual(E.get_color("asdf123"), "rgb(63, 52, 114)")
        self.assertEqual(E.get_color("42"), "rgb(48, 153, 74)")
        self.assertEqual(E.get_color("381"), "rgb(54, 86, 121)")
        self.assertRaises(ValueError, lambda: E.get_color("blueandyellow"))

    def test_get_gesture_counts(self):
        should_be = {
            "g255": 1160,
            "g001": 13,
            "g002": 12,
            "g004": 12,
            "g000": 10,
            "g003": 10,
        }
        self.assertEqual(E.get_gesture_counts(root="mocks/"), should_be)

    def test_main(self):
        self.assertRaisesRegex(
            SystemExit,
            "1",
            lambda: E.main({"predict": False, "save": False, "as_keyboard": ""}),
        )
        self.assertRaisesRegex(
            SystemExit,
            "1",
            lambda: E.main({"predict": False, "save": True, "as_keyboard": "blah"}),
        )
        self.assertRaisesRegex(
            SystemExit,
            "1",
            lambda: E.main({"predict": True, "save": False, "as_keyboard": ""}),
        )

        self.assertRaisesRegex(
            SystemExit,
            "1",
            lambda: E.main({"predict": False, "save": False, "as_keyboard": ""}),
        )


class TestUtils(unittest.TestCase):
    def test_makes_batches(self):
        df = utils.parse_csvs(root="mocks/")
        X = df.drop(["datetime", "gesture"], axis=1).to_numpy()
        y = df["gesture"].to_numpy()
        t = pd.to_datetime(df["datetime"]).to_numpy()
        batched_X, batched_y = utils.make_batches(X, y, t)
        should_be_X = np.array(
            [[[562.0, 534.0, 433.0, 533.0, 591.0, 442.0, 548.0, 584.0, 472.0, 425.0]]]
        )
        should_be_y = np.array(
            [
                "gesture0255",
                "gesture0255",
                "gesture0255",
                "gesture0255",
                "gesture0255",
                "gesture0255",
                "gesture0255",
                "gesture0255",
                "gesture0255",
                "gesture0255",
            ]
        )

        self.assertTrue((batched_X[0, 0, :10] == should_be_X).all())
        self.assertTrue((batched_y[:10] == should_be_y).all())

    def test_gestures_and_indices(self):
        g2i, i2g = utils.gestures_and_indices(np.array(["a", "b", "c"]))
        self.assertEqual(g2i(i2g(1)), 1)
        self.assertEqual(i2g(g2i("a")), "a")

    def test_conf_mat(self):
        g2i, i2g = utils.gestures_and_indices(np.array(["0", "1"]))
        cm = utils.conf_mat([1, 1, 0, 0], [1, 0, 1, 0], i2g)
        self.assertTrue((np.array([[1, 1], [1, 1]]) == cm).all())

        cm = utils.conf_mat([1, 1, 0, 0], [1, 0, 1, 0], i2g, perc="cols")
        self.assertTrue((np.array([[50.0, 50.0], [50.0, 50.0]]) == cm).all())

        cm = utils.conf_mat([1, 1, 0, 0], [1, 0, 1, 0], i2g, perc="rows")
        self.assertTrue((np.array([[50.0, 50.0], [50.0, 50.0]]) == cm).all())

        cm = utils.conf_mat([1, 1, 0, 0], [1, 0, 1, 0], i2g, perc="both")
        self.assertTrue((np.array([[25.0, 25.0], [25.0, 25.0]]) == cm).all())

    def test_per_class_cb(self):
        g2i, i2g = utils.gestures_and_indices(np.array([0, 1, 2, 3, 4]))
        val_data = ([1, 1, 1, 1], [2, 3, 4, 1])
        cb = utils.PerClassCallback(val_data, i2g)
        self.assertEqual(cb.validation_data, val_data)

    def test_build_dataset(self):
        df = utils.parse_csvs(root="mocks/")
        with open("mocks/config.yaml", "r") as file:
            config = yaml.safe_load(file)
        config["epochs"] = 1
        (
            config,
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
        ) = utils.build_dataset(df, config)
        should_be_X_train = np.array(
            [563.0, 526.0, 437.0, 536.0, 580.0, 446.0, 568.0, 583.0, 466.0, 430.0]
        )
        should_be_y_train = np.array([4, 1, 3, 5, 5, 5, 5, 5, 4, 3])
        self.assertTrue((should_be_X_train == X_train[0, 0, :10]).all())
        self.assertTrue((should_be_y_train == y_train[:10]).all())

    def test_compile_and_fit_adam(self):
        df = utils.parse_csvs(root="mocks/")
        with open("mocks/config.yaml", "r") as file:
            config = yaml.safe_load(file)
        (
            config,
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
        ) = utils.build_dataset(df, config)
        config["epochs"] = 1
        config["use_wandb"] = False
        history, model = utils.compile_and_fit(
            X_train, y_train, X_valid, y_valid, config, i2g
        )
        should_be_history = {
            "loss": [3.150095224380493],
            "val_loss": [1.367781162261963],
        }
        self.assertEqual(history.history["loss"], should_be_history["loss"])
        self.assertEqual(history.history["val_loss"], should_be_history["val_loss"])
        self.assertTrue(True)

    def test_calc_red(self):
        mean_dists, num_preds, num_trues = utils.calc_red(
            np.array([1, 0, 0, 0, 1]),
            np.array(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ),
            lambda i: {0: "0", 1: "1"}[i],
        )
        print(repr(mean_dists))
        print(repr(num_preds))
        print(repr(num_trues))
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
