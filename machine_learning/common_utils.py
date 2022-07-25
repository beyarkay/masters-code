import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import matplotlib as mpl
# By default use a larger figure size
mpl.rcParams['figure.figsize'] = [12, 12]
mpl.rcParams['figure.dpi'] = 200
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import os
import datetime
from time import time
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
import yaml
from yaml import Loader, Dumper
import re
import pickle

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

# Setup seaborn
sns.set()

# Define some variables for creating feature names like 'left-thumb-x' or 'right-index-z'
finger_names = [
    'little', 'ring', 'middle', 'index', 'thumb',
    'thumb', 'index',  'middle',  'ring',  'little',
]
dimensions = ['x', 'y', 'z']
# Actually create some feature names to give the data meaningful labels
fingers = []
fingers_short = []
for i, finger_name in enumerate(finger_names):
    for dimension in dimensions:
        hand = ['left', 'right'][i // 5]
        fingers.append(f'{hand}-{finger_name}-{dimension}')
        if dimension == dimensions[1]:
            fingers_short.append(f'{dimension}\n{finger_name}\n{hand}')
        else:
            fingers_short.append(f'{dimension}')

n_timesteps = 20
n_sensors = 30

def write_obs_to_disc(obs: np.ndarray, filename: str) -> None:
    np.savetxt(filename, obs, delimiter=",", fmt='%4.0f')

def to_flat(arr):
    """Given a 2D array, flatten it to `(n_timesteps*n_sensors,)`"""
    if type(arr) is pd.core.frame.DataFrame:
        arr = arr.to_numpy()
    assert arr.shape == (n_timesteps, n_sensors), f"Shape isn't ({n_timesteps}, {n_sensors})"
    return arr.flatten()

def from_flat(flat):
    """Given a flattened array, reshape it to `(n_timesteps, n_sensors)`"""
    assert flat.shape == (n_timesteps*n_sensors,), f"Shape isn't ({n_timesteps*n_sensors},)"
    return flat.reshape((n_timesteps, n_sensors))

def scale_single(arr, scaler):
    """Scale a single observation by the given scaler.
    Returns the result in the same shape as the input."""
    input_shape = arr.shape
    if type(arr) is pd.core.frame.DataFrame:
        arr = to_flat(arr)
    assert arr.shape != (n_sensors, n_timesteps), f"Shape is transposed"
    if arr.shape == (n_timesteps, n_sensors,):
        arr = to_flat(arr)
    assert arr.shape == (n_timesteps*n_sensors,), f"Shape isn't ({n_timesteps*n_sensors},)"

    return scaler.transform(np.array([arr]))[0].reshape(input_shape)


def get_gesture_info():
    """Get a dictionary of gestures to their text descriptions."""
    with open('../gesture_data/gesture_info.yaml', 'r') as f:
        gesture_info = yaml.load(f.read(), Loader=Loader)
    return gesture_info['gestures']

def get_dir_files(root_path='../gesture_data/train'):
    """Get a dictionary of directories to a list of raw gesture files."""
    # Get a listing of all directories and their files
    dir_files = {
        d: os.listdir(f'{root_path}/{d}')
        for d in os.listdir(root_path)
        if d != ".DS_Store"
    }
    # Filter out all directories which don't have any files in them
    dir_files = {
        d: files
        for d, files in sorted(dir_files.items())
        if len(files) > 0
    }
    return dir_files

def read_to_numpy(root_dir='../gesture_data/train', include=None, ignore=None, min_obs=180, verbose=0):
    """Given a root directory, read in all valid gesture observations
    in that directory and convert to a `X`, `y`, and `paths` arrays."""
    start = time()
    if ignore is not None and include is not None:
        raise Exception("You can only specify one of `ignore` and `include`")
    if ignore is None:
        ignore = []
    dir_files = {k: v for k, v in get_dir_files(root_dir).items() if k not in ignore}
    if include is not None:
        dir_files = {k: v for k, v in dir_files.items() if k in include}
    gesture_info = get_gesture_info()
    max_val = 0

    if verbose > 1:
        format_string = "\n- ".join([
            f'{k}: {gesture_info.get(k, {}).get("description", "<No description>"):<40} ({len(v)} files)'
            for k,v in dir_files.items()
        ])
        print(f'The following gestures have data recorded for them:\n- {format_string}')

    # Exclude all classes which do not have at least `min_obs` observations
    n_classes = len([d for d,fs in dir_files.items() if len(fs) > min_obs])
    n_obs = sum([len(fs) for d, fs in dir_files.items() if len(fs) > min_obs])
    if verbose > 0:
        print(f'{n_classes=}, {n_obs=}')

    # Create arrays to store the observations and labels
    X = np.zeros((n_obs, n_timesteps * n_sensors))
    y = np.zeros((n_obs,))
    # Also keep track of the paths from which each observation originated
    paths = []

    # Create hashmaps to easily convert from and from indexes (0, 1, 2, 3, ...)
    # and gestures ('gesture0001', 'gesture0002', ...)
    idx_to_gesture = {}
    gesture_to_idx = {}

    # The `obs_idx` increments with every observation
    obs_idx = 0
    # The `label_idx` increments with every label iff
    # there's > `min_obs` observations for that label
    label_idx = 0

    # Iterate over every gesture
    if verbose >= 0:
        print(f'{len(dir_files.keys())} gestures, {n_obs} total observations')
    for gesture_index, filenames in dir_files.items():
        if len(filenames) > min_obs:
            # Populate the idx <-> gesture mapping
            idx_to_gesture[label_idx] = gesture_index
            gesture_to_idx[gesture_index] = label_idx
            description = gesture_info.get(gesture_index, {}).get("description", "<No description>")
            if verbose > 0:
                print(f'  {gesture_index}: {description:<40} ({len(filenames)} observations)')

            # Iterate over every observation for the current gesture
            for file in filenames:
                path = f'{root_dir}/{gesture_index}/{file}'
                # Read in the raw sensor data.
                df = read_to_ndarray(path)
                # Make sure the data is the correct shape
                if df.shape != (n_timesteps, n_sensors):
                    print(df)
                obs = to_flat(df)
                max_val = max(max_val, df.max().max())
                if np.any(np.isnan(obs)):
                    print(f'rm {path}')
                paths.append(path)
                X[obs_idx] = obs
                y[obs_idx] = label_idx
                obs_idx += 1
            label_idx += 1

    with open('saved_models/idx_to_gesture.pickle', 'wb') as f:
        pickle.dump(idx_to_gesture, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('saved_models/gesture_to_idx.pickle', 'wb') as f:
        pickle.dump(gesture_to_idx, f, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose >= 0:
        print(f'done in {round(time() - start, 2)}s')
    return X, y, np.array(paths)



def train_test_split_scale(X, y, paths):
    """Given arrays X, y, and paths, test-train split the data with a 25%
    test size, and return the splits."""
    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
        X, y, paths, test_size=0.25, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    save_model(scaler)
    return X_train, X_test, y_train, y_test, paths_train, paths_test


def plot_raw_gesture(
    arr,
    title=None,
    ax=None,
    show_cbar=True,
    show_xticks=True,
    show_yticks=True,
    show_values=False,
    delim_lw=2.5,
):
    """ Given an array of data, create a heatmap of the sensor data.
    Returns the fig and ax"""
    # If no ax is specified, create one.
    if ax is None:
        # Create a new plot
        fig, ax = plt.subplots(1, figsize=(15, 7))
        if title is not None:
            # with title referencing to the origin of the data
            fig.suptitle(title)
    else:
        # If an ax is specified, then use it and
        # get a reference to the current figure
        fig = plt.gcf()
        if title is not None:
            ax.title.set_text(title)

    assert type(arr) is np.ndarray, f"Type is {type(arr)}, not np.array"
    assert arr.shape == (n_timesteps, n_sensors), f"Shape isn't ({n_timesteps}, {n_sensors})"

    time = np.array(list(range(0, (n_timesteps-1)*25+1, 25)))

    # Actually draw the heatmap, with square blocks.
    img = ax.imshow(arr, cmap='viridis', aspect='equal')
    if show_cbar:
        fig.colorbar(img, ax=ax)
    if show_yticks:
        # Set the y-axis ticks to be the elapsed miliseconds since the gesture started
        ax.set_yticks([i for i in range(len(arr))])
        ax.set_yticklabels([int(t) for t in time])
        ax.set_ylabel("Milliseconds")
    else:
        ax.set_yticks([])

    if show_xticks:
        # Set the x-axis ticks to the names of the different fingers like 'right-index-y'
        ax.set_xticks(range(len(fingers)))
        ax.set_xticklabels(fingers_short)
        # Rotate the x-ticks so they're visible
        ax.tick_params(axis='x', rotation=0)
    else:
        ax.set_xticks([])

    # remove the grid
    ax.grid(visible=None)

    # Draw some horizontal separators between the fingers
    for i in range(1, 10):
        # These constants had to be hand-tuned
        x_offset = -0.5
        ax.plot(
            [i*3 + x_offset, i*3 + x_offset],
            [-0.5, n_timesteps-0.5],
            c='white',
            lw=delim_lw if i != 5 else delim_lw*3
        )
    if show_values:
        for (j,i), label in np.ndenumerate(arr):
            if abs(label) > 10:
                label = '{:g}'.format(float('{:.3g}'.format(round(label, 3))))
            else:
                label = '{:g}'.format(float('{:.2g}'.format(round(label, 2))))
            ax.text(i, j, label, size=7, ha='center', va='center', color='white')

    return fig, ax


def read_to_ndarray(filename):
    return np.loadtxt(filename, delimiter=',')

def read_to_df(filename, normalise=False):
    """ Given a filename, read in the file to a Pandas DataFrame.
    The columns are 'milliseconds' along with one column for each finger+dimension
    ('left-index-z', 'right-middle-y', etc). Each row is one instant of sensor measurements, with the
    time of those measurements given by the `milliseconds` column.
    """
    print("`read_to_df` is deprecated. Use read_to_ndarray instead")
    should_return_nans = False
    # Read in the raw data values
    df = pd.read_csv(filename, header=None, names=fingers)
    # Set the index to be a timedelta index
    df.index = pd.TimedeltaIndex(df.index, unit='ms', name='offset_ms')

    # Check to see that we've got enough measurements to roughly fill the df
    if len(df.index) < 28:
        # And if not, return a df of NaNs
        should_return_nans = True
    # If the start and end items don't explicitly exist => add them
    start = pd.Timedelta('0 days 00:00:00.000')
    end = pd.Timedelta('0 days 00:00:00.975')
    if start not in df.index:
        df.loc[start] = pd.Series(dtype='float64')
    if end not in df.index:
        df.loc[end] = pd.Series(dtype='float64')

    # Remove any outliers, they're likely invalid readings
    lower_bound = 300
    upper_bound = 800
    df[df < lower_bound] = np.nan
    df[df > upper_bound] = np.nan

    # Resample the data so we've got values exactly every 25ms
    df = df.resample("25ms").mean().ffill()

    if np.any(np.isnan(df.to_numpy())):
        should_return_nans = True

    # If we've got any samples that are after 0.975 or before 0.000, drop them
    df = df[df.index <= end]
    df = df[start <= df.index]

    # Normalise each column to have 0 mean and 1 std dev.
    # Means are computed per sensor, but std dev is computed over all measurements
    if normalise:
        df = (df - df.mean()) / df.std()
    if should_return_nans:
        df.loc[:] = np.nan
    return df

def predict_nicely(obs, clf, scaler, idx_to_gesture):
    """Given a single observation, a classifier, and a scaler, return a list of
    predictions in the format [(gesture, probability), ...].

    For example:
    ```
    >>> df = read_to_df(f'../gesture_data/self-classified/{gesture}/{file}')
    >>> predictions = predict_nicely(df, clf, scaler)
    >>> for gesture, proba in predictions:
    ...     if proba < 0.0001:
    ...         break
    >>>     print(f'{gesture}: {proba*100:.2f}%')
    gesture0008: 96.36%
    gesture0005: 3.64%
    ```
    """
    if type(obs) is pd.core.frame.DataFrame:
        obs = obs.to_numpy()
    assert obs.shape == (n_timesteps, n_sensors), "shape must be (n_timesteps, n_sensors)"
    # Scalar transform the observation and predict the gesture from it
    wrap_scale_flat = np.array([to_flat(scale_single(obs, scaler))])
    predict_proba = clf.predict_proba(wrap_scale_flat)
    # Sort the predictions
    predictions = sorted(enumerate(predict_proba[0]), key=lambda ip: -ip[1])
    # Convert the indexes to gestures and return the predictions
    return [(idx_to_gesture[idx], proba) for idx, proba in predictions]


def save_model(model):
    """Given a model, save it to the directory `./saved_models/` as a pickle.

    Returns the filepath to which it was saved."""
    now_str = datetime.datetime.now().isoformat()
    filepath = re.sub(r'\s+', '', f'saved_models/{now_str}_{model}.pickle')
    with open(filepath, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    return filepath

def load_model(filepath):
    """Load and return the model located at `filepath`."""
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model
