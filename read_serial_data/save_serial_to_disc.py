print("Importing libraries")
from time import time
from typing import Callable, List, Any
import datetime
from colors import color
import os
import numpy as np
from matplotlib import cm
import serial
from serial.tools.list_ports import comports
import sys
np.set_printoptions(threshold=sys.maxsize, linewidth=250)


def main():
    port = get_serial_port()
    if port is None:
        print("Port not found")
        sys.exit(1)
    baudrate = 19_200
    print(f"Reading from {port} with baudrate {baudrate}")
    with serial.Serial(port=port, baudrate=baudrate, timeout=1) as ser:
        loop_over_serial_stream(ser, save_to_disc)


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

def write_obs_to_disc(obs: np.ndarray, filename: str) -> None:
    np.savetxt(filename, obs, delimiter=",", fmt='%4.0f')

def loop_over_serial_stream(
    serial_handle: serial.serialposix.Serial,
    callback: Callable[[np.ndarray, dict[str, Any]], dict[str, Any]]
) -> None:
    """Read in one set of measurements from `serial_handle` and pass to
    `callable`. Some pre-processing and error checking is done to ensure things
    happen nicely."""
    n_timesteps = 40
    n_sensors = 30
    first_loop = True
    callback_data: dict[str, Any] = {
        "n_timesteps": 40,
        "n_sensors": 30,
        "curr_offset": 0,
        "prev_offset": 0,
        "curr_idx": 0,
        "prev_idx": 0,
        "n_measurements": 0,
        "gesture_idx": 0,
        "obs": np.full((n_timesteps, n_sensors), np.nan),
    }

    while serial_handle.isOpen():
        try:
            raw_values: List[str] = serial_handle.readline().decode('utf-8').strip().split(",")[:-1]
            # Ensure there are exactly 32 values
            if len(raw_values) == 0:
                print(f"No values found from serial connection, try unplugging the device ({raw_values=})")
                sys.exit(1)
            if len(raw_values) != n_sensors+2:
                print(f"raw values are length {len(raw_values)}, not {n_sensors+2}: {raw_values}")
                continue
            callback_data["prev_gesture_idx"] = callback_data["gesture_idx"]
            callback_data["gesture_idx"] = int(raw_values[0])
            callback_data["prev_offset"] = callback_data["curr_offset"]
            callback_data["curr_offset"] = int(raw_values[1])
            callback_data["curr_idx"] = round(callback_data["curr_offset"] / 25)
            # If this is the first time looping through, then wait until the
            # beginning of a gesture comes around
            if first_loop:
                if callback_data["curr_idx"] > 0:
                    continue
                else:
                    first_loop = False

            if callback_data["curr_idx"] == 40:
                callback_data["curr_idx"] = 39
            aligned_offset = callback_data["curr_idx"] * 25
        except serial.serialutil.SerialException as e:
            print(f"Ergo has been disconnected: {e}")
            sys.exit(1)
        except ValueError:
            continue

        upper_bound = 800
        lower_bound = 300
        # Convert the values to integers and clamp between `lower_bound` and
        # `upper_bound`
        new_measurements = np.array([
            min(upper_bound, max(lower_bound, int(val))) 
            for val in raw_values[2:]
        ])

        # Call the callback
        callback_data = callback(new_measurements, callback_data)

        now_str = datetime.datetime.now().isoformat()[11:-3]
        curr_idx = callback_data["curr_idx"]
        colors = []
        dims = ['x', 'y', 'z']
        for i, val in enumerate(raw_values[2:]):
            colors.append(dims[i%3] + get_colored_string(int(val) - lower_bound, upper_bound - lower_bound)) 
            if i % 15 == 14 and i > 0:
                colors[-1] += '     '
            elif i % 3 == 2 and i > 0:
                colors[-1] += ' '
            else:
                colors[-1] += ''
        colors = ''.join(colors)
        print(f'{now_str} {aligned_offset: >3} [{curr_idx: >2}]: {raw_values[0]: >3} {raw_values[1]: >3} {colors}', end="\r")

        serial_handle.flush()
    else:
        print("Serial port closed")

def save_to_disc(new_measurements: np.ndarray, d: dict[str, Any]) -> dict[str, Any]:
    """Given a series of new measurements, populate an observation and save to
    disc as require."""

    # If the current time offset is < the previous time offset, then the
    # gesture has ended and we should 1) save the current observation to
    # disc and 2) set the observation to all NaNs in preparation for the
    # next gesture
    if d["curr_offset"] < d["prev_offset"] and d["n_measurements"] >= 40:
        # If there was no measurement for 975ms, just use the one for 000ms
        if np.isnan(d["obs"][-1]).any():
            d["obs"][-1, :] = new_measurements

        directory = f'../gesture_data/train/gesture{d["gesture_idx"]:04}'
        now_str = datetime.datetime.now().isoformat()
        write_obs_to_disc(d["obs"], f'{directory}/{now_str}.csv')
        num_obs = len(os.listdir(directory))
        print(f"\x1b[2K\r{d['n_measurements']} measurements taken, wrote observation as `{directory}/{now_str}.csv` ({num_obs} total)")
        # Reset the observation to be all NaNs
        d["obs"] = np.full((d["n_timesteps"], d["n_sensors"]), np.nan)
        d["n_measurements"] = 0

    # If we've skipped over an index, impute with the mean between the
    # last recorded observation and the most recent observation
    d["obs"][d["curr_idx"], :] = new_measurements
    for idx in range(d["prev_idx"] + 1, d["curr_idx"]):
        # print(f"imputing index {idx} using the mean of idx {d['prev_idx']} and of idx {d['curr_idx']}")
        d["obs"][idx, :] = np.mean((d["obs"][d["prev_idx"], :], d["obs"][d["curr_idx"], :]), axis=0)
    d["prev_idx"] = d["curr_idx"]
    # Place this observation at it's place in curr_idx
    d["n_measurements"] += 1
    return d

def get_colored_string(value, n_values) -> str:
    colours = cm.get_cmap('turbo', n_values)
    rgb = [int(val * 255) for val in colours(value)[:-1]]
    coloured = color(f'{str(value):3}', 'black', f'rgb({rgb[0]},{rgb[1]},{rgb[2]})')
    return coloured

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\x1b[2K\rFinishing")
        sys.exit(0)
