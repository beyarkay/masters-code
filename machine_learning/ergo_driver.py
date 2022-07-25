import common_utils as utils
import keyboard
import os
import pickle

IS_ACTIVE = True
def main():
    n_timesteps = 20
    n_sensors = 30
    with open('saved_models/idx_to_gesture.pickle', 'rb') as f:
        idx_to_gesture = pickle.load(f)
    scaler = utils.load_model('../machine_learning/saved_models/StandardScaler().pickle')
    model_paths = sorted(['../machine_learning/saved_models/' + p for p in os.listdir('../machine_learning/saved_models/') if "Classifier" in p])
    clf = utils.load_model(model_paths[0])
    print(f"Loaded model {clf}")
    # while True:
    #     if IS_ACTIVE:
    keyboard.wait('esc')
    print("Pressed <ESC>")
    keyboard.write('asdf')

    # break;

if __name__ == "__main__":
    main()
