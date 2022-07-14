print("Importing models")
from time import time
import datetime
from common_utils import *
import serial
import subprocess
import sys
np.set_printoptions(threshold=sys.maxsize, linewidth=250)

def main():
    # Set the options for the serial port
    baudrate = 19_200
    ports = subprocess.run(
            ["python", "-m", "serial.tools.list_ports"],
            stdout=subprocess.PIPE,
            text=True
    ).stdout.strip().split("\n")
    ports = [p for p in ports if p.startswith("/dev/cu.usbmodem")]
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

    print(f"Reading from {port} with baudrate {baudrate}")
    with serial.Serial(port=port, baudrate=baudrate, timeout=1) as ser:
        predict_from_serial_stream(ser)

def predict_from_serial_stream(ser):
    n_sensors = 30
    n_timesteps = 20
    # Read in the model
    model_paths = ['saved_models/' + p for p in os.listdir('saved_models') if 'Classifier' in p]
    assert model_paths, "There must be at least one classifier in `saved_models/`"
    clf = load_model(model_paths[0])
    print(f"Loaded model at {model_paths[0]}")
    # Read in the scaler
    scaler_path = f"saved_models/StandardScaler().pickle"
    scaler = load_model(scaler_path)
    # Read in the index-to-gesture mapping
    with open('saved_models/idx_to_gesture.pickle', 'rb') as f:
        idx_to_gesture = pickle.load(f)

    obs = np.zeros((n_timesteps, n_sensors))
    print(ser)
    timestep = 0
    last_write = int(time() * 1000)
    raw_values = None

    while True:
        # If it's been at least 25ms and the serial port is open
        if int(time() * 1000) - last_write >= 25 and ser.isOpen():
            last_write = int(time() * 1000)
            try:
                raw_values = ser.readline().decode('utf-8').strip().split(",")
            except serial.serialutil.SerialException:
                # If Ergo has been disconnected, end the program
                return
            # Occasionally a line won't be completely populated. In this case,
            # just carry over the previous readings to the new dataset
            if len(raw_values) != n_sensors+3:
                print(f'raw values arent of length {n_sensors+3}: {raw_values}')
                continue
            values = [int(val) for val in raw_values[2:-1]]
            new_values = np.array(values)

            obs[1:, :] = obs[:-1, :]
            obs[0, :] = new_values

            timestep += 1
            # Only attempt a prediction if there's valid data populating the
            # observation
            if timestep % 5 == 0 and timestep > n_timesteps:
                timestep = 0

                # Write the observation to disc
                now = datetime.datetime.now().isoformat()
                filename = f'../gesture_data/self-classified/unknown/{now}.txt'
                with open(filename, 'w') as f:
                    lines = []
                    for i, arr in enumerate(obs.tolist()):
                        joined_arr = ','.join([str(int(ai)) for ai in arr])
                        lines.append(f'{i*25},{joined_arr}\n')
                    f.writelines(lines)

                # Then fetch the observation from disc and predict
                predictions = predict_nicely(
                    read_to_df(filename),
                    clf,
                    scaler,
                    idx_to_gesture
                )
                tot = 0
                for gesture, pred_proba in predictions:
                    tot += pred_proba
                    pretty = f'{pred_proba*100:.2f}%'
                    print(f'{gesture:>11}: {pretty:<10}', end='')
                    if tot >= 0.95:
                        break
                print()
                if predictions[0][1] > 0.9:
                    # If the model is confident in the prediction, then save
                    # the observation for future analysis.
                    now = datetime.datetime.now().isoformat()
                    gesture = predictions[0][0]
                    filename = f'../gesture_data/self-classified/{gesture}/{now}.txt'
                    # print(f"Saving as {filename}:\n{obs[:5]}")
                    # with open(filename, 'w') as f:
                    #     lines = []
                    #     for i, arr in enumerate(obs.tolist()):
                    #         joined_arr = ','.join([str(int(ai)) for ai in arr])
                    #         lines.append(f'{i*25},{joined_arr}\n')
                    #     f.writelines(lines)
            ser.flush()


if __name__ == "__main__":
    main()
