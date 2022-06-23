print("Importing models")
from time import sleep
from common_utils import *
import serial
import os
import subprocess

def main() -> None:
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
    # model_paths = os.listdir("saved_models")
    # Read in the model
    model_path = f"saved_models/MLPClassifier(activation='tanh',alpha=5.532519953153552e-05,hidden_layer_sizes=(400,200),max_iter=1000,solver='lbfgs').pickle"
    model = load_model(model_path)
    # Read in the scaler
    scaler_path = f"saved_models/StandardScaler().pickle"
    scaler = load_model(scaler_path)
    # Read in the index-to-gesture mapping
    with open('saved_models/idx_to_gesture.pickle', 'rb') as f:
        idx_to_gesture = pickle.load(f)

    obs = np.zeros((30, 40))
    print(ser)
    filled_cols = 0
    while True:
        if ser.isOpen():
            try:

                values = ser.readline().decode('utf-8').strip().split(",")
            except serial.serialutil.SerialException:
                # If Ergo has been disconnected, end the program
                return
            if len(values) != 33:
                print(values)
                continue
            values = [int(val) for val in values[2:-1]]
            new_values = np.array(values)
            # Shift all the values across by one column
            obs[:, 1:] = obs[:, :-1]
            # Then populate the first column with the new values
            obs[:, 0] = values
            print(obs)

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
                    # Only bother printing info if the model is confident
                    tot = 0
                    for i, pred in predictions:
                        tot += pred
                        pretty = f'{pred*100:.2f}%'
                        print(f'{idx_to_gesture[i]:>11}: {pretty:<10}', end='')
                        if tot >= 0.95:
                            break
                    print()
            ser.flush()

if __name__ == "__main__":
    main()
