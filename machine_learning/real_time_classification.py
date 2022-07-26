print("Importing libraries")
import threading
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
import common_utils as utils
from sklearn.neural_network import MLPClassifier
# Get better print options
np.set_printoptions(threshold=sys.maxsize, linewidth=250)
# This magic ANSI string clears a line that's been partially written
CLR = "\x1b[2K\r"


def main():
    # Check that the usage is correct
    if len(sys.argv) not in [2,3] or sys.argv[1] not in ['predict', 'save', 'drive', 'p', 's', 'd']:
        print("Usage: \n\tpython3 save_serial_to_disc.py save\n\tpython3 save_serial_to_disc.py predict\n\tpython3 save_serial_to_disc.py drive [<FILENAME>]")
        sys.exit(1)
    # Get the correct callback dependant on the cmdline argvalues
    callback = None
    if sys.argv[1].startswith('s'):
        callback = save_cb
    elif sys.argv[1].startswith('p'):
        callback = predict_cb
    elif sys.argv[1].startswith('d'):
        callback = driver_cb
    # Get the correct serial port, exiting if none exists
    port = get_serial_port()
    # Define the baudrate (number of bits per second sent over the serial port)
    baudrate = 19_200
    print(f"Reading from {port} with baudrate {baudrate}")
    # Start up an infinite loop that calls the callback every time one set of
    # new data is available over the serial port.
    with serial.Serial(port=port, baudrate=baudrate, timeout=1) as serial_port:
        loop_over_serial_stream(serial_port, callback)

def get_serial_port() -> str:
    """Look at the open serial ports and return one if it's correctly
    formatted, otherwise exit with status code 1.

    Only considers serial ports starting with `/dev/cu.usbmodem` and will offer
    the user a choice if more than one port is available."""
    # Read in all the available ports starting with `/dev/cu.usbmodem`
    ports = [p.device for p in comports() if p.device.startswith("/dev/cu.usbmodem")]
    if len(ports) > 1:
        # If there's more than one, offer the user a choice
        for i, port in enumerate(ports):
            print(f"[{i}]: {port}")
        idx = int(input(f"Please choose a port index [0..{len(ports)-1}]: "))
        port = ports[idx]
    elif len(ports) == 0:
        # If there are no ports available, exit with status code 1
        print("No ports beginning with `/dev/cu.usbmodem` found")
        sys.exit(1)
    else:
        # If there's only one port, then assign it to the variable `port`
        port = ports[0]
    # Finally, return the port
    return port

def gestures_to_keystrokes(path='gestures_to_keystrokes.yaml'):
    """Simple utility to read in the gestures_to_keystrokes yaml file and
    return it as a dictionary. 

    The dictionary structured like Dict{gesture->keystroke} and is not
    guaranteed to have every gesture defined. Undefined gestures should result
    in no keys being pressed."""
    with open(path, 'r') as f:
        g2k = yaml.load(f.read(), Loader=Loader)
    return g2k

def loop_over_serial_stream(
    serial_handle: serial.serialposix.Serial,
    callback: Callable[[np.ndarray, dict[str, Any]], dict[str, Any]]
) -> None:
    """Read in one set of measurements from `serial_handle` and pass to
    `callable`. 

    Some pre-processing and error checking is done to ensure things
    transpire nicely. The model used for predictions is the
    alphabetically-first Classifier available in `saved_models/`. If no model
    is there, then no predictions will be given but everything will still
    function appropriately."""
    # Define some constants
    n_timesteps = 20
    n_sensors = 30
    # The first loop can sometimes contain incomplete data, so define a flag so
    # that we can skip it if required
    first_loop = True

    # Read in the index to gesture mappings used by the machine learning model
    with open('saved_models/idx_to_gesture.pickle', 'rb') as f:
        idx_to_gesture = pickle.load(f)
    with open('saved_models/gesture_to_idx.pickle', 'rb') as f:
        gesture_to_idx = pickle.load(f)
    # Read in the scaler used by the machine learning model to scale the input
    # data
    scaler = utils.load_model(
        'saved_models/StandardScaler().pickle'
    )
    # Get a list of all model paths that are Classifiers
    model_paths = sorted(['saved_models/' + p for p in os.listdir('saved_models/') if "Classifier" in p], reverse=True)
    # Read in the first model alphabetically
    clf = utils.load_model(model_paths[0])
    # Create a dictionary of data to pass to the callback
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
        "gesture_to_idx": gesture_to_idx,
        "scaler": scaler,
        "clf": clf,
        "prediction": "no pred",
        "g2k": gestures_to_keystrokes(),
        "new_X": [],
        "new_y": [],
        "thread": None,
    }

    # Now that all the setup is complete, loop forever (or until the serial
    # port is unplugged), read in the data, pre-process the data, and call
    # the callback.
    while serial_handle.isOpen():
        try:
            # record the current and previous times so we can calculate the
            # duration between measurements
            cb_data["prev_time_ms"] = cb_data["time_ms"]
            cb_data["time_ms"] = int(time() * 1000)
            before_split = serial_handle.readline().decode('utf-8')
            # Parse the values
            raw_values: List[str] = before_split.strip().split(",")[:-1]
            # Ensure there are exactly 32 values
            if len(raw_values) == 0:
                print(f"{CLR}No values found from serial connection, try unplugging the device ({raw_values=}), {before_split=}")
                sys.exit(1)
            if len(raw_values) != n_sensors+2:
                # print(f"{CLR}Raw values are length {len(raw_values)}, not {n_sensors+2}: {raw_values}")
                continue
            # Update the dictionary with some useful values
            cb_data["prev_gesture_idx"] = cb_data["gesture_idx"]
            cb_data["gesture_idx"] = int(raw_values[0])
            # Shuffle along the previous and current offsets
            cb_data["prev_offset"] = cb_data["curr_offset"]
            cb_data["curr_offset"] = int(raw_values[1])
            # Clamp the `curr_idx` so that it doesn't cause an array index out
            # of bounds error
            cb_data["curr_idx"] = min(n_timesteps - 1, round(cb_data["curr_offset"] / 25))
            # If this is the first time looping through, then wait until the
            # beginning of a gesture comes around
            if first_loop:
                if cb_data["curr_idx"] > 0:
                    continue
                else:
                    first_loop = False
            # The aligned_offset is the number of milliseconds from the start
            # of the gesture, but rounded to the nearest 25 milliseconds
            cb_data['aligned_offset'] = cb_data["curr_idx"] * 25
        except serial.serialutil.SerialException as e:
            print(f"Ergo has been disconnected: {e}")
            sys.exit(1)

        # Convert the values to integers and clamp between `lower_bound` and
        # `upper_bound`
        upper_bound = 800
        lower_bound = 300
        try:
            new_measurements = np.array([
                min(upper_bound, max(lower_bound, int(val))) for val in raw_values[2:]
            ])
        except ValueError as e:
            print(f"value error: {e}, {raw_values=}")
            continue

        # Call the callback with the new measurements and the callback data
        cb_data = callback(new_measurements, cb_data)

        # Format the new measurements nicely so they can be drawn to the
        # terminal
        write_measurements_to_terminal(new_measurements, cb_data)
        serial_handle.flush()
    else:
        print("Serial port closed")

def write_measurements_to_terminal(new_measurements, cb_data: dict[str, Any], end='\r'):
    """Write the new measurements with some helpful information to the terminal."""
    curr_idx = cb_data["curr_idx"]
    colors = ['left:']
    dims = ['x', 'y', 'z']
    max_value = 900
    for i, val in enumerate(new_measurements):
        # Append an ANSI-coloured string to the colors array
        colors.append(get_colored_string(int(val), max_value, fstring=(dims[i%3]+'{value:3}')))
        if i == 14:
            # The 14th value is the middle, so add a space to separate
            colors[-1] += '   right:'
        elif i % 3 == 2 and i > 0:
            # Every (i%3==2) value deliminates a triplet of (x,y,z) values, so
            # add a space to separate
            colors[-1] += ' '
    # Join all the colour strings together
    colors = ''.join(colors)
    # Finally print out the full string, ended with a `\r` so that we can write
    # over it afterwards
    aligned_offset = cb_data["aligned_offset"]
    prediction = cb_data["prediction"]
    gesture_idx = cb_data['gesture_idx']
    millis_offset = cb_data['curr_offset']
    print(f'{millis_offset: >3}ms ({aligned_offset: >3}ms) ms//25={curr_idx: >2} gesture:{gesture_idx: <3} {colors}{prediction}', end=end)

def format_prediction(gesture: str, proba: float, shortness=0, cmap='inferno', threshold=0.99):
    """Given a gesture and it's probability, return an ANSI coloured string
    colour mapped to it's probability."""
    gesture = gesture.replace('gesture0', 'g').replace('g0', 'g').replace('g0', 'g')
    gesture = gesture if shortness <= 1 else gesture.replace('g', '')
    fstring = f'{gesture}' + (f':{{value:>02}}%' if shortness < 1 else '')
    colored = get_colored_string(int(proba*100), 100, fstring=fstring, cmap=cmap, highlight=proba >= threshold)
    return colored

def get_colored_string(value: int, n_values: int, fstring='{value:3}', cmap='turbo', highlight=False) -> str:
    """Given a value and the total number of values, return a colour mapped and formatted string of that value"""
    colours = cm.get_cmap(cmap, n_values)
    rgb = [int(val * 255) for val in colours(round(value * (1.0 if highlight else 0.8)))[:-1]]
    mag = np.sqrt(sum([x * x for x in rgb]))
    coloured = color(
        fstring.format(value=value),
        'black' if mag > 180 else 'white',
        f'rgb({rgb[0]},{rgb[1]},{rgb[2]})',
    )
    return coloured

def save_cb(new_measurements: np.ndarray, d: dict[str, Any]) -> dict[str, Any]:
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

        gesture_idx = f'gesture{d["gesture_idx"]:04}'
        directory = f'../gesture_data/train/{gesture_idx}'
        now_str = datetime.datetime.now().isoformat()
        utils.write_obs_to_disc(d["obs"], f'{directory}/{now_str}.csv')
        d["new_X"].append(utils.to_flat(d["obs"]))
        d["new_y"].append(d["gesture_to_idx"][gesture_idx])
        num_obs = len(os.listdir(directory))
        print(f"{CLR}{d['n_measurements']} measurements taken, wrote observation as `{directory}/{now_str}.csv` ({num_obs} total)")
        for gesture, proba in sorted(predictions, key=lambda gp: gp[0]):
            print(format_prediction(gesture, proba, shortness=len(predictions) // 25, cmap='inferno'), end=' ')
        print()
        # Retrain the model when a certain number of new observations have emerged
        if num_obs % 100 == 0:
            if d["thread"] is not None and d["thread"].is_alive():
                print("Waiting for thread to join")
                d["thread"].join()
            model_paths = sorted(['saved_models/' + p for p in os.listdir('saved_models/') if "Classifier" in p], reverse=True)
            print(f"Starting thread to train new model, current model is: {model_paths[0]}")
            d["clf"] = utils.load_model(model_paths[0])
            d["thread"] = threading.Thread(target=train_model, args=(d,), kwargs={})
            d["thread"].start()
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

def train_model(d):
    start = time()
    X, y, paths = utils.read_to_numpy(
            include=list(d['gesture_to_idx'].keys()),
            min_obs=0,
            verbose=-1,
    )
    n_classes = np.unique(y).shape[0]
    X_train, X_test, y_train, y_test, paths_train, paths_test = utils.train_test_split_scale(X, y, paths)
    start = time()
    d["clf"] = d["clf"].fit(X_train, y_train)
    score = d["clf"].score(X_test, y_test)
    path = utils.save_model(d["clf"])
    d["new_X"] = []
    d["new_y"] = []
    print(f'{CLR}Trained classifier with {len(y)} observations in {time() - start:.3f}s, {score=:.4f}, {path=}\n')

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
    return d

def driver_cb(new_measurements: np.ndarray, d: dict[str, Any]) -> dict[str, Any]:
    """Take the predicted gesture and convert it to a character, writing to
    stdout."""
    if 'bucket' not in d:
        d['bucket'] = 0
        d['curr_gesture'] = None
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

    predictions = utils.predict_nicely(d["obs"], d["clf"], d["scaler"], d["idx_to_gesture"])
    print(f"{CLR}", end='')
    d["prediction"] = format_prediction(*predictions[0])
    # Only count a prediction if it's over 98% probable
    best_proba = 0.0
    best_gesture = None
    for gesture, proba in sorted(predictions, key=lambda gp: gp[0]):
        if proba > best_proba:
            best_proba = proba
            best_gesture = gesture

    MIN_PROBABILITY = 0.98
    REQ_BUCKET_QUANTITY = 2

    # print(f'\t\t{best_gesture} with {best_proba} and bucket={d["bucket"]}, current={d["curr_gesture"]}')
    if best_proba > MIN_PROBABILITY:
        if best_gesture != d['curr_gesture']:
            d['bucket'] = 0
            d['curr_gesture'] = best_gesture
        d['bucket'] += 1
        if d['bucket'] == REQ_BUCKET_QUANTITY and best_gesture != 'gesture0255':
            now_str = datetime.datetime.now().isoformat()
            print(now_str, d["g2k"].get(best_gesture, best_gesture), f" <{best_gesture}>")
            if len(sys.argv) == 3:
                with open(sys.argv[2], 'a') as f:
                    f.write(d["g2k"].get(best_gesture, best_gesture))
    return d

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"{CLR}Finishing")
        sys.exit(0)
