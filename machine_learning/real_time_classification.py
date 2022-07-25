print("Importing libraries")
from colors import color
from matplotlib import cm
from serial.tools.list_ports import comports
from typing import Callable, List, Any
import datetime
import numpy as np
from time import time
import os
import serial
import sys
import pickle
import yaml
from yaml import Loader, Dumper
script_dir = os.path.dirname( __file__ )
utils_dir = os.path.join(script_dir, '..', 'machine_learning')
sys.path.append(utils_dir)
import common_utils as utils
np.set_printoptions(threshold=sys.maxsize, linewidth=250)
CLR = "\x1b[2K\r"


def main():
    if len(sys.argv) not in [2,3]:
        print("Usage: \n\tpython3 save_serial_to_disc.py save\n\tpython3 save_serial_to_disc.py predict\n\tpython3 save_serial_to_disc.py predict <filename>")
    if len(sys.argv) == 2 and sys.argv[1] not in ['predict', 'save', 'p', 's']:
        print("Usage: \n\tpython3 save_serial_to_disc.py save\n\tpython3 save_serial_to_disc.py predict")
        sys.exit(1)
    callback = save_to_disc_cb if sys.argv[1].startswith('s') else predict_cb
    port = get_serial_port()
    if port is None:
        print("Port not found")
        sys.exit(1)
    baudrate = 19_200
    print(f"Reading from {port} with baudrate {baudrate}")
    with serial.Serial(port=port, baudrate=baudrate, timeout=1) as ser:
        loop_over_serial_stream(ser, callback)

def gestures_to_keystrokes():
    with open('../machine_learning/gestures_to_keystrokes.yaml', 'r') as f:
        g2k = yaml.load(f.read(), Loader=Loader)
    return g2k

def get_serial_port() -> str | None:
    ports = [p.device for p in comports() if p.device.startswith("/dev/cu.usbmodem")]
    if len(ports) > 1:
        for i, port in enumerate(ports):
            print(f"[{i}]: {port}")
        idx = int(input(f"Please choose a port index [0..{len(ports)-1}]: "))
        port = ports[idx]
    elif len(ports) == 0:
        print("No ports beginning with `/dev/cu.usbmodem` found")
        return
    else:
        port = ports[0]
    return port

def loop_over_serial_stream(
    serial_handle: serial.serialposix.Serial,
    callback: Callable[[np.ndarray, dict[str, Any]], dict[str, Any]]
) -> None:
    """Read in one set of measurements from `serial_handle` and pass to
    `callable`. Some pre-processing and error checking is done to ensure things
    happen nicely."""
    n_timesteps = 20
    n_sensors = 30
    first_loop = True

    with open('../machine_learning/saved_models/idx_to_gesture.pickle', 'rb') as f:
        idx_to_gesture = pickle.load(f)
    # cb_data contains all data that get passed to the CallBack
    scaler = utils.load_model(
        '../machine_learning/saved_models/StandardScaler().pickle'
    )
    model_paths = sorted(['../machine_learning/saved_models/' + p for p in os.listdir('../machine_learning/saved_models/') if "Classifier" in p])
    clf = utils.load_model(model_paths[0])
    cb_data: dict[str, Any] = {
        "n_timesteps": n_timesteps,
        "n_sensors": n_sensors,
        "curr_offset": 0,
        "prev_offset": 0,
        "curr_idx": 0,
        "prev_idx": 0,
        "n_measurements": 0,
        "gesture_idx": 0,
        "obs": np.full((n_timesteps, n_sensors), np.nan),
        "time_ms": int(time() * 1000),
        "prev_time_ms": int(time() * 1000),
        "idx_to_gesture": idx_to_gesture,
        "scaler": scaler,
        "clf": clf,
        "prediction": "no prediction",
        "g2k": gestures_to_keystrokes(),
    }

    while serial_handle.isOpen():
        try:
            # record the current and previous times so we can calculate the
            # duration between measurements
            cb_data["prev_time_ms"] = cb_data["time_ms"]
            cb_data["time_ms"] = int(time() * 1000)
            before_split = serial_handle.readline().decode('utf-8')
            raw_values: List[str] = before_split.strip().split(",")[:-1]
            # Ensure there are exactly 32 values
            if len(raw_values) == 0:
                print(f"{CLR}No values found from serial connection, try unplugging the device ({raw_values=}), {before_split=}")
                sys.exit(1)
            if len(raw_values) != n_sensors+2:
                print(f"{CLR}Raw values are length {len(raw_values)}, not {n_sensors+2}: {raw_values}")
                continue
            # TODO for some reason the observation is mostly NaNs when `gesture_idx==0`
            cb_data["prev_gesture_idx"] = cb_data["gesture_idx"]
            cb_data["gesture_idx"] = int(raw_values[0])
            cb_data["prev_offset"] = cb_data["curr_offset"]
            cb_data["curr_offset"] = int(raw_values[1])
            cb_data["curr_idx"] = round(cb_data["curr_offset"] / 25)
            # If this is the first time looping through, then wait until the
            # beginning of a gesture comes around
            if first_loop:
                if cb_data["curr_idx"] > 0:
                    continue
                else:
                    first_loop = False

            if cb_data["curr_idx"] == n_timesteps:
                cb_data["curr_idx"] = n_timesteps - 1 
            aligned_offset = cb_data["curr_idx"] * 25
        except serial.serialutil.SerialException as e:
            print(f"Ergo has been disconnected: {e}")
            sys.exit(1)

        upper_bound = 800
        lower_bound = 300
        # Convert the values to integers and clamp between `lower_bound` and
        # `upper_bound`
        try:
            new_measurements = np.array([
                min(upper_bound, max(lower_bound, int(val)))
                for val in raw_values[2:]
            ])
        except ValueError as e:
            print(f"value error: {e}, {raw_values=}")
            continue

        # Call the callback
        cb_data = callback(new_measurements, cb_data)

        now_str = datetime.datetime.now().isoformat()[11:-3]
        curr_idx = cb_data["curr_idx"]
        colors = []
        dims = ['x', 'y', 'z']
        for i, val in enumerate(raw_values[2:]):
            colors.append(get_colored_string(
                int(val) - lower_bound,
                upper_bound - lower_bound,
                fstring=(dims[i%3]+'{value:3}')
            ))
            if i == 14:
                colors[-1] += '   '
            elif i % 3 == 2 and i > 0:
                colors[-1] += ' '
            else:
                colors[-1] += ''
        colors = ''.join(colors)
        print(f'{now_str} {aligned_offset: >3} [{curr_idx: >2}]: {raw_values[0]: >3} {raw_values[1]: >3} {colors}{cb_data["prediction"]}', end="\r")

        serial_handle.flush()
    else:
        print("Serial port closed")

def save_to_disc_cb(new_measurements: np.ndarray, d: dict[str, Any]) -> dict[str, Any]:
    """Given a series of new measurements, populate an observation and save to
    disc as require."""
    clf = d["clf"]
    scaler = d["scaler"]

    # If the current time offset is < the previous time offset, then the
    # gesture has ended and we should 1) save the current observation to
    # disc and 2) set the observation to all NaNs in preparation for the
    # next gesture
    if d["curr_offset"] < d["prev_offset"] and d["n_measurements"] >= d["n_timesteps"]:
        # If there was no measurement for 975ms, just use the one for 000ms
        if np.isnan(d["obs"][-1]).any():
            d["obs"][-1, :] = new_measurements
        try:
            predictions = utils.predict_nicely(d["obs"], clf, scaler, d["idx_to_gesture"])
            d["prediction"] = format_prediction(*predictions[0])
        except Exception as e:
            d["prediction"] = "Classifier exception"

        directory = f'../gesture_data/train/gesture{d["gesture_idx"]:04}'
        now_str = datetime.datetime.now().isoformat()
        utils.write_obs_to_disc(d["obs"], f'{directory}/{now_str}.csv')
        num_obs = len(os.listdir(directory))
        print(f"\x1b[2K\r{d['n_measurements']} measurements taken, wrote observation as `{directory}/{now_str}.csv` ({num_obs} total)")
        for idx in range(d["curr_idx"]):
            # If the curr_idx > 0 then we've skipped over the first
            # measurement. Therefore impute the first measurements as the mean
            # of the last measurement of the previous observation and the new
            # measurement of this observation.
            d["obs"][idx, :] = np.mean((d["obs"][-1, :], new_measurements), axis=0)
        # Reset the observation to be mostly NaNs
        d["obs"][d["curr_idx"]:] = np.full((d["n_timesteps"]-d["curr_idx"], d["n_sensors"]), np.nan)
        d["n_measurements"] = 0

    # If we've skipped over an index, impute with the mean between the
    # last recorded observation and the most recent observation
    d["obs"][d["curr_idx"]] = new_measurements
    for idx in range(d["prev_idx"] + 1, d["curr_idx"]):
        # print(f"imputing index {idx} using the mean of idx {d['prev_idx']} and of idx {d['curr_idx']}")
        d["obs"][idx, :] = np.mean((d["obs"][d["prev_idx"], :], d["obs"][d["curr_idx"], :]), axis=0)
    d["prev_idx"] = d["curr_idx"]
    # Place this observation at it's place in curr_idx
    d["n_measurements"] += 1

    return d

def format_prediction(gesture, proba, shortness=0, cmap='inferno', threshold=0.99):
    gesture = gesture.replace('gesture0', 'g').replace('g0', 'g').replace('g0', 'g')
    gesture = gesture if shortness <= 1 else gesture.replace('g', '')
    fstring = f'{gesture}' + (f':{{value:>02}}%' if shortness < 1 else '')
    colored = get_colored_string(int(proba*100), 100, fstring=fstring, cmap=cmap, highlight=proba >= threshold)
    return colored

def predict_cb(new_measurements: np.ndarray, d: dict[str, Any]) -> dict[str, Any]:
    """Predict the current gesture, printing the probabilities to stdout."""
    clf = d["clf"]
    scaler = d["scaler"]
    # Calculate how much time has passed between this measurement and the
    # previous measurement, rounded to the nearest 25ms
    diff = round((d["time_ms"] - d["prev_time_ms"]) / 25) * 25
    if diff == 0:
        # If no time has passed, just replace the most recent measurement with
        # the new measurement
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

    # If we have any NaNs, don't try predict anything
    if np.isnan(d["obs"]).any():
        print(f"{CLR}Not predicting, (obs contains {(np.isnan(d['obs'])).sum()} NaNs)")
        return d

    predictions = utils.predict_nicely(d["obs"], clf, scaler, d["idx_to_gesture"])
    print(f"{CLR}", end='')
    d["prediction"] = format_prediction(*predictions[0])
    # Only count a prediction if it's over 98% probable
    best_proba = 0.98
    best_gesture = None
    for gesture, proba in sorted(predictions, key=lambda gp: gp[0]):
        if proba > best_proba:
            best_proba = proba
            best_gesture = gesture
        print(format_prediction(gesture, proba, shortness=len(predictions) // 25, cmap='inferno'), end=' ')
    print()
    if len(sys.argv) == 3 and best_gesture not in [None, 'gesture0255']:
        with open(sys.argv[2], 'a') as f:
            f.write(d["g2k"].get(best_gesture, best_gesture))
    return d

def get_colored_string(value, n_values, fstring='{value:3}', cmap='turbo', highlight=False) -> str:
    colours = cm.get_cmap(cmap, n_values)
    rgb = [int(val * 255) for val in colours(round(value * (1.0 if highlight else 1.0)))[:-1]]
    mag = np.sqrt(sum([x * x for x in rgb]))
    coloured = color(
        fstring.format(value=value),
        'black' if mag > 180 else 'white',
        f'rgb({rgb[0]},{rgb[1]},{rgb[2]})',
    )
    return coloured

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"{CLR}Finishing")
        sys.exit(0)
