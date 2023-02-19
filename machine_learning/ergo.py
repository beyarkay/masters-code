print("Importing libraries")
from time import sleep, time
from collections import Counter
from colors import color
from matplotlib import cm
from serial.tools.list_ports import comports
from typing import Callable, List, Any
import argparse
import colorsys
from ml_utils import *
import datetime
import keyboard
import numpy as np
import os
import serial
import sys
import yaml

# Get better print options
np.set_printoptions(threshold=sys.maxsize, linewidth=250)
# This magic ANSI string clears a line that's been partially written
CLR = "\x1b[2K\r"
should_create_new_file = True
GESTURE_WINDOW_MS = 30
GESTURES = [f"gesture{i:0>4}" for i in range(50)]
COUNTDOWN_MS = 1000


def main(args):
    print(args)
    # Get the correct callback dependant on the cmdline argvalues
    callbacks = []
    if args["predict"]:
        callbacks.append(predict_cb)
    if args["save"]:
        callbacks.append(save_cb)
    if args["as_keyboard"] == "os" or (
        args["as_keyboard"] is not None and len(args["as_keyboard"]) > 0
    ):
        callbacks.append(keyboard_cb)

    if args["save"]:
        # Add a listener that will remove the last 120 lines (3 seconds) from the
        # training data. Just in case some poor gesture data gets recorded by
        # mistake
        def burn_most_recent_observations():
            global should_create_new_file
            should_create_new_file = True
            root = "../gesture_data/train/"
            paths = [f for f in sorted(os.listdir(root)) if f.endswith(".csv")]
            path = root + paths[-1]
            num_seconds = 1.5
            newlines_per_second = 40
            num_newlines = newlines_per_second * num_seconds
            cursor_pos = 0
            print(f"{CLR}{num_newlines=}, {num_seconds=}")
            # Delete the last `num_newlines` lines efficiently
            # https://stackoverflow.com/a/10289740/14555505
            with open(path, "r+", encoding="utf-8") as file:
                # Move the pointer to the end of the file
                file.seek(0, os.SEEK_END)
                # Seek to the final character. We pass over the \0 byte at the end
                # of the file in the first iteration of the while loop.
                cursor_pos = file.tell()
                # Read backwards from pos until we've found `num_newlines` newlines
                while cursor_pos > 0 and num_newlines > 0:
                    cursor_pos -= 1
                    file.seek(cursor_pos, os.SEEK_SET)
                    if file.read(1) == "\n":
                        num_newlines -= 1

                # Check that we didn't get to the beginning of the file, then
                # delete all characters from `cursor_pos` to the end
                cursor_pos = max(cursor_pos, 0)
                file.seek(cursor_pos, os.SEEK_SET)
                file.truncate()
            # If we've removed every line of a file, just delete it
            if cursor_pos == 0:
                os.remove(path)
            print(
                CLR,
                color(
                    f"Burnt the last {int(newlines_per_second * num_seconds)} lines ({int(num_seconds)} seconds) of data from {path}",
                    bg="firebrick",
                ),
            )
            sleep(2)

        keyboard.add_hotkey("space", burn_most_recent_observations)
        gestures = GESTURES
        print(f"{CLR}Possible gestures: {gestures}")

    while True:
        # Get the correct serial port, exiting if none exists
        port = get_serial_port()
        # Define the baudrate (number of bits per second sent over the serial port)
        baudrate = 19_200
        print(f"Reading from {port} with baudrate {baudrate}")
        # Start up an infinite loop that calls the callback every time one set of
        # new data is available over the serial port.
        with serial.Serial(port=port, baudrate=baudrate, timeout=1) as serial_port:
            # `loop_over_serial_stream` will only return if the connection to
            # the Arduino is broken, so that we can loop back and try reconnect
            # to the new serial port
            loop_over_serial_stream(serial_port, callbacks, args)
        # The broken connection will take a little while to restart, so sleep
        # for a bit to give it time to do so
        sleep(2)


def get_serial_port() -> str:
    """Look at the open serial ports and return one if it's correctly
    formatted, otherwise exit with status code 1.

    Only considers serial ports starting with `/dev/cu.usbmodem` and will offer
    the user a choice if more than one port is available."""
    # Read in all the available ports starting with `/dev/cu.usbmodem`
    ports = [p.device for p in comports() if p.device.startswith("/dev/cu.usbmodem")]
    print(f"Looking through ports {ports}")
    if len(ports) >= 1:
        # check that the available ports are actually communicating
        filtered_ports = []
        for i, port in enumerate(ports):
            try:
                with serial.Serial(
                    port=port, baudrate=19_200, timeout=1
                ) as serial_port:
                    if serial_port.isOpen():
                        # flush the port so the line we read will definitely
                        # start from the beginning of the line
                        line = serial_port.readline().decode("utf-8").strip()
                        if line and "#" not in line:
                            print(
                                f"Port {port} is returning non empty non-comment lines"
                            )
                            filtered_ports.append(port)
            except Exception as e:
                continue
        if len(filtered_ports) != 1:
            for i, port in filtered_ports:
                print(f"[{i}]: {port}")
            idx = int(input(f"Please choose a port index [0..{len(ports)-1}]: "))
            port = ports[idx]
        elif len(filtered_ports) == 0:
            print("No ports beginning with `/dev/cu.usbmodem` found")
            sys.exit(1)
        else:
            port = filtered_ports[0]
    elif len(ports) == 0:
        # If there are no ports available, exit with status code 1
        print("No ports beginning with `/dev/cu.usbmodem` found")
        sys.exit(1)
    # Finally, return the port
    return port


def loop_over_serial_stream(
    serial_handle: serial.serialposix.Serial,
    callbacks: List[Callable[[np.ndarray, dict[str, Any]], dict[str, Any]]],
    args: dict,
) -> None:
    model_path = args["model"]
    print(model_path)
    from tensorflow import keras

    model = keras.models.load_model(model_path)
    # Get the config
    with open(model_path + "/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    idx_to_gesture = config["i2g"]
    gesture_to_idx = config["g2i"]
    # Define some constants
    n_timesteps = config["window_size"]
    n_sensors = 30
    # The first loop can sometimes contain incomplete data, so define a flag so
    # that we can skip it if required
    first_loop = True

    with open("../gesture_data/gesture_info.yaml", "r") as f:
        g2k = yaml.safe_load(f.read()).get("gestures", {})
    g2k = {k: v.get("key", "unknown") for k, v in g2k.items()}

    # Create a dictionary of data to pass to the callback
    cb_data: dict[str, Any] = {
        "n_timesteps": n_timesteps,
        "n_sensors": n_sensors,
        "curr_offset": 0,
        "prev_offset": 0,
        "curr_idx": 0,
        "prev_idx": 0,
        "gesture_idx": None,
        "obs": np.full((n_timesteps, n_sensors), np.nan),
        "time_ms": int(time() * 1000),
        "prev_time_ms": int(time() * 1000),
        "idx_to_gesture": idx_to_gesture,
        "gesture_to_idx": gesture_to_idx,
        "model": model,
        "config": config,
        "prediction": "no pred",
        "g2k": g2k,
        "gesture_info": gesture_info(),
        "args": args,
    }
    # Read in one line just to make sure the next read starts on a newline
    if serial_handle.isOpen():
        _ = serial_handle.readline()

    # Now that all the setup is complete, loop forever (or until the serial
    # port is unplugged), read in the data, pre-process the data, and call
    # the callback.
    while serial_handle.isOpen():
        try:
            # record the current and previous times so we can calculate the
            # duration between measurements
            cb_data["prev_time_ms"] = cb_data["time_ms"]
            cb_data["time_ms"] = int(time() * 1000)
            before_split = serial_handle.readline().decode("utf-8")
            # Comments starting with `#` act as heartbeats
            if before_split.startswith("#"):
                log(f"Found comment: {before_split}")
                continue
            # Parse the values
            raw_values: List[str] = before_split.strip().split(",")[:-1]
            # Ensure there are exactly 32 values
            if len(raw_values) == 0:
                print(
                    f"{CLR}No values found from serial connection, try restarting the device\n{raw_values=}, {before_split=}"
                )
                sleep(1)
                cb_data["time_ms"] = time_ms()
                cb_data["prev_time_ms"] = time_ms()
                cb_data["obs"] = np.full((n_timesteps, n_sensors), np.nan)
                continue
            if len(raw_values) != n_sensors + 2:
                continue
            # Update the dictionary with some useful values
            cb_data["prev_gesture_idx"] = cb_data["gesture_idx"]
            if not raw_values[0]:
                continue
            # Shuffle along the previous and current offsets
            cb_data["prev_offset"] = cb_data["curr_offset"]
            cb_data["curr_offset"] = int(raw_values[1])
            # Clamp the `curr_idx` so that it doesn't cause an array index out
            # of bounds error
            cb_data["curr_idx"] = min(
                n_timesteps - 1, round(cb_data["curr_offset"] / 25)
            )
            # If this is the first time looping through, then wait until the
            # beginning of a gesture comes around
            if first_loop:
                if cb_data["curr_idx"] > 0:
                    continue
                else:
                    first_loop = False
            # The aligned_offset is the number of milliseconds from the start
            # of the gesture, but rounded to the nearest 25 milliseconds
            cb_data["aligned_offset"] = cb_data["curr_idx"] * 25
        except serial.serialutil.SerialException as e:
            print(f"Ergo has been disconnected: {e}")
            # Return so the connection can be restarted
            return

        # Convert the values to integers and clamp between `lower_bound` and
        # `upper_bound`
        upper_bound = 900
        lower_bound = 300
        try:
            new_measurements = np.array(
                [min(upper_bound, max(lower_bound, int(val))) for val in raw_values[2:]]
            )
        except ValueError as e:
            print(f"value error: {e}, {raw_values=}")
            continue

        cb_data = calc_obs(new_measurements, cb_data)
        # Call the callback with the new measurements and the callback data
        for callback in callbacks:
            cb_data = callback(new_measurements, cb_data)

        # Format the new measurements nicely so they can be drawn to the
        # terminal
        cb_data = write_debug_line(new_measurements, cb_data)
        serial_handle.flush()
    else:
        print("Serial port closed")


def save_cb(new_measurements: np.ndarray, d: dict[str, Any]) -> dict[str, Any]:
    """Append data line-by-line to a csv file"""
    # only write to file if we've got a non-None gesture index
    if d["gesture_idx"] is None:
        return d
    root = f"../gesture_data/train/"
    now_str = datetime.datetime.now().isoformat()
    global should_create_new_file
    if should_create_new_file:
        path = root + now_str + ".csv"
        print(f"{CLR}Sleeping due to burnt observations, new path is {path}")
        sleep(2)
        should_create_new_file = False
        d["time_ms"] = time_ms()
        d["prev_time_ms"] = time_ms()
    else:
        paths = [f for f in sorted(os.listdir(root)) if f.endswith(".csv")]
        path = root + paths[-1]
    with open(path, "a") as f:
        # Only label the measurements as being an actual gesture (as opposed
        # to the null 0255 gesture) in the final 50ms of the countdown.
        if COUNTDOWN_MS - (time_ms() - d["last_gesture"]) <= GESTURE_WINDOW_MS:
            gesture = f'gesture{d["gesture_idx"]:04}'
        else:
            gesture = "gesture0255"
        items = [now_str, gesture] + [f"{m:.0f}" for m in new_measurements.tolist()]
        f.write(",".join(items) + "\n")
    return d


def predict_cb(new_measurements: np.ndarray, d: dict[str, Any]) -> dict[str, Any]:
    """Predict the current gesture, printing the probabilities to stdout."""

    d = print_predictions(d)
    return d


def keyboard_cb(new_measurements: np.ndarray, d: dict[str, Any]) -> dict[str, Any]:
    """Take the predicted gesture and convert it to a character, writing to file."""
    # If we have any NaNs, don't try predict anything
    if np.isnan(d["obs"]).any():
        return d

    log(d["idx_to_gesture"])
    gesture_preds = predict_tf(d["obs"], d["config"], d["model"], d["idx_to_gesture"])

    predictions = np.array(list(zip(*gesture_preds))[1])
    gestures = np.array(list(zip(*gesture_preds))[0])
    best_idx = np.argmax(predictions)
    best_pred = predictions[best_idx]
    best_gest = gestures[best_idx]

    d["prediction"] = format_prediction(best_gest, best_pred)

    if "last_predicted" not in d.keys():
        d["last_predicted"] = {g: time_ms() - 1000 for g in gestures}
    if "last_keystroke" not in d.keys():
        d["last_keystroke"] = ""

    # Find the most recent non-g255 gesture prediction
    most_recent_key, most_recent_keypress = max(
        [(k, v) for k, v in d["last_predicted"].items() if k != "gesture0255"],
        key=lambda x: x[1],
    )
    log(
        f"{best_idx=}, {best_gest=}, {best_pred=}, {most_recent_key=}, {most_recent_keypress=}, {time_ms()=}"
    )
    # Only press a key if the most recent non-g255 prediction was more than
    # 500ms in the past and the current key isn't gesture0255
    if time_ms() - most_recent_keypress > 500 and best_pred > 0.5:
        # Log the time at which this keystroke was made
        d["last_predicted"][best_gest] = time_ms()
        if best_gest != "gesture0255":
            # Convert the gesture to a keystroke, defaulting to <gesture_label> if
            # no keystroke is specified
            keystroke = str(d["g2k"].get(best_gest, f"<{best_gest}>"))
            log(f"{keystroke=}")

            # Either write to the provided file, or send the keystrokes to the OS
            # directly
            if d["args"]["as_keyboard"] == "os" and keystroke:
                d, fn, args = process_keystrokes(d, keystroke)
                if fn is not None and len(args) > 0:
                    fn(*args)
            elif d["args"]["as_keyboard"] is not None:
                with open(d["args"]["as_keyboard"], "a") as f:
                    f.write(keystroke)
    return d


def calc_obs(new_measurements: np.ndarray, d: dict[str, Any]) -> dict[str, Any]:
    """Calculate how much time has passed between this measurement and the
    previous measurement, rounded to the nearest 25ms"""
    diff = round((d["time_ms"] - d["prev_time_ms"]) / 25) * 25
    if diff < 25:
        # If less than 25ms has passed, just replace the most recent
        # measurement with the new measurement
        d["obs"][0, :] = new_measurements
    elif diff == 25:
        # If 25ms has passed, shuffle all measurements along and then insert
        # the new measurement
        d["obs"][1:, :] = d["obs"][:-1, :]
        d["obs"][0, :] = new_measurements
    elif diff > 25:
        # If more than 25ms has passed, shuffle all measurements along by the
        # number of 25ms increments
        shift_by = diff // 25
        d["obs"][shift_by:, :] = d["obs"][:-shift_by, :]
        # Then add the new measurements to the first slot
        d["obs"][0, :] = new_measurements
        # And impute the missing values as the mean of the most recent
        # measurements and the new measurements
        avg = np.mean((d["obs"][0, :], d["obs"][shift_by, :]), axis=0)
        for idx in range(1, shift_by):
            d["obs"][idx, :] = avg
    return d


def gesture_info(path="../gesture_data/gesture_info.yaml"):
    with open(path, "r") as f:
        gesture_info = yaml.safe_load(f.read())["gestures"]
    return gesture_info


def write_debug_line(
    new_measurements, cb_data: dict[str, Any], end="\r"
) -> dict[str, Any]:
    """Write the new measurements with some helpful information to the terminal."""
    gestures = GESTURES
    curr_idx = cb_data["curr_idx"]
    cb_data["last_gesture"] = cb_data.get("last_gesture", time_ms())
    cb_data["lineup"] = cb_data.get("lineup", gestures)

    colors = ["left:"]
    dims = ["x", "y", "z"]
    max_value = 900
    for i, val in enumerate(new_measurements):
        # Append an ANSI-coloured string to the colors array
        colors.append(
            get_colored_string(int(val), max_value, fstring=(dims[i % 3] + "{value:3}"))
        )
        if i == 14:
            # The 14th value is the middle, so add a space to separate
            colors[-1] += "   right:"
        elif i % 3 == 2 and i > 0:
            # Every (i%3==2) value deliminates a triplet of (x,y,z) values, so
            # add a space to separate
            colors[-1] += " "

    # Join all the colour strings together
    colors = "".join(colors)

    if time_ms() - cb_data["last_gesture"] >= COUNTDOWN_MS:
        cb_data["last_gesture"] = time_ms()
        popped = cb_data["lineup"].pop(0)
        cb_data["lineup"].append(popped)
        if cb_data["args"]["save"]:
            relevant = {
                k: v
                for k, v in get_gesture_counts().items()
                if k.replace("g", "gesture0") in cb_data["lineup"]
            }
            print(CLR, "Count of gestures: ", relevant)

    gesture = cb_data["lineup"][0]
    description = (
        cb_data["gesture_info"]
        .get(gesture)["description"]
        .lower()
        .replace("left", "l")
        .replace("right", "r")
    )
    if cb_data["args"]["save"]:
        curr_gesture_str = (
            f"{gesture.replace('gesture0', 'g')}: {description: <10}".replace(
                "thumb", "1"
            )
            .replace("index", "2")
            .replace("middle", "3")
            .replace("ring", "4")
            .replace("little", "5")
            .replace("Right", "->")
            .replace("Left", "<-")
        )
        progress = len(curr_gesture_str) - round(
            ((time_ms() - cb_data["last_gesture"]) / COUNTDOWN_MS)
            * len(curr_gesture_str)
        )
        regular = color(
            curr_gesture_str[:progress], fg="white", bg=get_color(gesture), style="bold"
        )
        inverse = color(
            curr_gesture_str[progress:], fg=get_color(gesture), bg="white", style="bold"
        )
    else:
        inverse = ""
        regular = ""
    cb_data["gesture_idx"] = int(gesture.replace("gesture", ""))
    prediction = cb_data["prediction"]
    countdown = cb_data["lineup"][0]
    print(f"{regular}{inverse} {colors}{prediction}", end=end)
    return cb_data


def format_prediction(
    gesture: str,
    proba: float,
    shortness=0,
    cmap="inferno",
    threshold=0.99,
    g2k=None,
):
    """Given a gesture and it's probability, return an ANSI coloured string
    colour mapped to it's probability."""
    if g2k is not None and gesture != "gesture0255":
        gesture = g2k.get(gesture, gesture)
        if gesture == "\n":
            gesture = r"\n"
        elif gesture == "\t":
            gesture = r"\t"
    else:
        gesture = (
            gesture.replace("gesture0", "g").replace("g0", "g").replace("g0", "g") + " "
        )
        gesture = gesture if shortness <= 1 else gesture.replace("g", "")
    fstring = f"{gesture}" + (f"{{value:>2}}%" if shortness < 1 else "")
    colored = get_colored_string(
        min(99, int(proba * 100)),
        100,
        fstring=fstring,
        cmap=cmap,
        highlight=proba >= threshold,
    )
    return colored


def get_colored_string(
    value: int, n_values: int, fstring="{value:3}", cmap="turbo", highlight=False
) -> str:
    """Given a value and the total number of values, return a colour mapped and formatted string of that value"""
    colours = cm.get_cmap(cmap, n_values)
    rgb = [
        int(val * 255)
        for val in colours(round(value * (1.0 if highlight else 0.8)))[:-1]
    ]
    mag = np.sqrt(sum([x * x for x in rgb]))
    coloured = color(
        fstring.format(value=value),
        "black" if mag > 180 else "white",
        f"rgb({rgb[0]},{rgb[1]},{rgb[2]})",
    )
    return coloured


def print_predictions(d):
    """Attempt to predict the gesture based on the observation in `d`."""
    # If we have any NaNs, don't try predict anything
    if np.isnan(d["obs"]).any():
        return d

    gesture_preds = predict_tf(d["obs"], d["config"], d["model"], d["idx_to_gesture"])
    predictions = np.array(list(zip(*gesture_preds))[1])
    gestures = np.array(list(zip(*gesture_preds))[0])
    best_idx = np.argmax(predictions)
    best_pred = predictions[best_idx]
    best_gest = gestures[best_idx]

    d["prediction"] = format_prediction(best_gest, best_pred)
    print(f"{CLR}", end="")
    # Only count a prediction if it's over 98% probable
    for i, (gesture, proba) in enumerate(sorted(gesture_preds, key=lambda gp: gp[0])):
        print(
            format_prediction(
                gesture,
                proba,
                shortness=len(gesture_preds) // 25,
                cmap="inferno",
                g2k=d["g2k"],
            ),
            end="   " if i % 10 == 9 else (" " if i % 5 == 4 else ""),
        )
    if d["prediction"]:
        print("Prediction: ", d["prediction"])
    else:
        print()
    return d


def time_ms() -> float:
    """Get the time in milliseconds."""
    return time() * 1_000


def color_bg(string) -> str:
    """Just a wrapper to color a string and use that string as a seed to
    dictate what color to use."""
    return color(string, bg=get_color(string))


def get_color(string) -> str:
    """Use the given string as a seed to get a deterministic background color.
    See https://www.mattgroeber.com/utilities/random-color-generator"""
    np.random.seed(int("".join(c for c in string if c.isdigit())))
    # Hue in [0, 360]
    h_range = (0.0, 1.0)
    # Saturation in [30%, 52%]
    s_range = (0.30, 0.60)
    # Light in [27%, 39%]
    l_range = (0.30, 0.40)
    r, g, b = colorsys.hls_to_rgb(
        np.random.random() * (h_range[1] - h_range[0]) + h_range[0],
        np.random.random() * (l_range[1] - l_range[0]) + l_range[0],
        np.random.random() * (s_range[1] - s_range[0]) + s_range[0],
    )
    return f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"


def get_gesture_counts(root="../gesture_data/train/"):
    paths = [p for p in os.listdir(root) if p.endswith(".csv")]
    counter = Counter()
    for path in paths:
        with open(root + path, "r") as f:
            # Extract just the gesture from the lines
            gestures = ["g" + l[35:38] for l in f.readlines()]
            counter.update(gestures)
    return counter


def log(s):
    with open("tmp.txt", "a") as f:
        f.write(f"{datetime.datetime.now().isoformat()} {s}\n")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            prog="python3 ergo.py",
            description="Record, classify, and predict *Ergo* sensor data in real time.",
        )
        parser.add_argument(
            "-m",
            "--model",
            help="Path to the TensorFlow model to use for predictions",
            action="store",
        )
        parser.add_argument(
            "-p",
            "--predict",
            help="Predict gestures based on sensor data.",
            action="store_true",
        )
        parser.add_argument(
            "-s",
            "--save",
            help="Save the incoming data line-by-line to a csv file",
            action="store_true",
        )
        parser.add_argument(
            "-k",
            "--as-keyboard",
            help="Predict gestures, convert them to keystrokes, and write the keystrokes to the file AS_KEYBOARD",
            action="store",
            nargs="?",
            const="os",
        )

        args = parser.parse_args()
        main(vars(args))
    except KeyboardInterrupt:
        print(f"{CLR}Finishing")
        sys.exit(0)
