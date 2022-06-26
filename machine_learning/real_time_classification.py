print("Importing models")
from time import sleep, time
import datetime
from common_utils import *
import serial
import os
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
    # Read in the model
    model_path = f"saved_models/MLPClassifier(activation='tanh',alpha=5.532519953153552e-05,hidden_layer_sizes=(400,200),max_iter=1000,solver='lbfgs').pickle"
    model = load_model(model_path)
    # Read in the scaler
    scaler_path = f"saved_models/StandardScaler().pickle"
    scaler = load_model(scaler_path)
    # Read in the index-to-gesture mapping
    with open('saved_models/idx_to_gesture.pickle', 'rb') as f:
        idx_to_gesture = pickle.load(f)

    obs = np.zeros((40, 30))
    print(ser)
    filled_cols = 0
    last_write = int(time() * 1000)
    raw_values = None
    old_values = None
    while True:
        # If it's been at least 25ms and the serial port is open
        if int(time() * 1000) - last_write >= 25 and ser.isOpen():
            last_write = int(time() * 1000)
            try:
                if raw_values is not None:
                    old_values = raw_values[:]
                raw_values = ser.readline().decode('utf-8').strip().split(",")
            except serial.serialutil.SerialException:
                # If Ergo has been disconnected, end the program
                return
            # Occasionally a line won't be completely populated. In this case,
            # just carry over the previous readings to the new dataset
            if len(raw_values) != 33:
                if old_values is not None and len(old_values) == 33:
                    raw_values = old_values[:]
                    print('continuing old values onwards')
                else:
                    continue
            values = [int(val) for val in raw_values[2:-1]]
            new_values = np.array(values)
            # Shift all the values across by one row
            obs[1:, :] = obs[:-1, :]
            # Then populate the first row with the new values
            obs[0, :] = new_values

            if filled_cols < obs.shape[1]:
                filled_cols += 1
            else:
                flat_scaled = scaler.transform(obs.flatten()[..., np.newaxis].T)
                prediction = model.predict_proba(flat_scaled)
                predictions = []
                for i, prob in enumerate(prediction[0]):
                    predictions.append((i, prob))
                predictions.sort(key=lambda ip: -ip[1])

                if predictions[0][1] > 0.9:
                    # If the model is confident in the prediction, then save
                    # the observation for future analysis
                    now = datetime.datetime.now().isoformat()
                    gesture = idx_to_gesture[predictions[0][0]]
                    with open(f'../gesture_data/self-classified/{gesture}/{now}.txt', 'w') as f:
                        f.writelines([str(i*25) + ',' + ','.join([str(int(ai)) for ai in a]) + '\n' for i, a in enumerate(obs.tolist())])
                    # Also print the prediction
                    tot = 0
                    for i, pred in predictions:
                        tot += pred
                        pretty = f'{pred*100:.2f}%'
                        print(f'{gesture:>11}: {pretty:<10}', end='')
                        if tot >= 0.95:
                            break
                    print()
            ser.flush()

if __name__ == "__main__":
    main()
