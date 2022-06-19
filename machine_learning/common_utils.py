import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import matplotlib as mpl
# By default use a larger figure size
mpl.rcParams['figure.figsize'] = [12, 12]
mpl.rcParams['figure.dpi'] = 72
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import os
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
import yaml
from yaml import Loader, Dumper
# Setup seaborn
sns.set()

# Define some variables for creating feature names like 'left-thumb-x' or 'right-index-z'
finger_names = [
    'left-little', 'left-ring', 'left-middle', 'left-index', 'left-thumb',
    'right-thumb', 'right-index',  'right-middle',  'right-ring',  'right-little',
]
dimensions = ['x', 'y', 'z']
# Actually create some feature names to give the data meaningful labels
fingers = []
for finger_name in finger_names:
    for dimension in dimensions:
        fingers.append(f'{finger_name}-{dimension}')

def get_gesture_info():
    """Get a dictionary of gestures to their text descriptions."""
    with open('../gesture_data/gesture_info.yaml', 'r') as f:
        gesture_info = yaml.load(f.read(), Loader=Loader)
    return gesture_info['gestures']

def get_dir_files():
    """Get a dictionary of directories to a list of raw gesture files."""
    # Get a listing of all directories and their files
    dir_files = {
        d: os.listdir(f'../gesture_data/train/{d}')
        for d in os.listdir(f'../gesture_data/train')
        if d != ".DS_Store"
    }
    # Filter out all directories which don't have any files in them
    dir_files = {
        d: files
        for d, files in sorted(dir_files.items())
        if len(files) > 0
    }
    return dir_files


def plot_raw_gesture(arr, title):
    """ Given an array of data and a title, create a heatmap of the sensor data."""
    # Create a new plot
    fig, ax = plt.subplots(1, figsize=(20, 10))
    # with title referencing to the origin of the data
    fig.suptitle(title)
    # If we've got a dataframe
    if type(arr) is pd.core.frame.DataFrame:
        # Plot the data
        img = ax.imshow(arr.T, cmap='viridis', aspect='auto')
        fig.colorbar(img, ax=ax)
        # Set the xticks to be time since the start of the gesture in ms
        ax.set_xticks([i for i in range(len(arr.index))])
        ax.set_xticklabels([t.microseconds // 1000 for t in arr.index])
        ax.set_xlabel("Milliseconds")

    elif type(arr) is np.ndarray:
        if arr.shape == (1200,):
            arr = arr.reshape((30, 40)).T
            time = list(range(0, 975+1, 25))
        elif arr.shape == (40, 31):
            # extract the x-axis labels
            time = arr[:,0]
            # And the rest of the data
            arr = arr[:,1:]
        # Actually draw the heatmap, with non-square blocks
        img = ax.imshow(arr.T, cmap='viridis', aspect='auto')
        fig.colorbar(img, ax=ax)
        # Set the x-axis ticks to be the elapsed miliseconds since the gesture started
        ax.set_xticks([i for i in range(len(arr))])
        ax.set_xticklabels([int(t) for t in time])

    # Set the y-axis ticks to the names of the different fingers like 'right-index-y'
    ax.set_yticks([i for i in range(len(fingers))])
    ax.set_yticklabels(fingers)
    # remove the grid
    ax.grid(visible=None)
    return fig, ax


def read_to_df(filename, normalise=False):
    """ Given a filename, read in the file to a Pandas DataFrame.
    The columns are 'milliseconds' along with one column for each finger+dimension
    ('left-index-z', 'right-middle-y', etc). Each row is one instant of sensor measurements, with the
    time of those measurements given by the `milliseconds` column.
    """
    # Read in the raw data values
    df = pd.read_csv(filename, header=None, names=fingers)
    # Set the index to be a timedelta index
    df.index = pd.TimedeltaIndex(df.index, unit='ms', name='offset_ms')
    # If the start and end items don't explicitly exist => add them
    start = pd.Timedelta('0 days 00:00:00.000')
    end = pd.Timedelta('0 days 00:00:00.975')
    if start not in df.index:
        df.loc[start] = pd.Series(dtype='float64')
    if end not in df.index:
        df.loc[end] = pd.Series(dtype='float64')
    # Resample the data so we've got values exactly every 25ms
    df = df.resample("25ms").mean().ffill()

    # If we've got any samples that are after 0.975 or before 0.000, drop them
    df = df[df.index <= end]
    df = df[start <= df.index]


    # Clamp any outliers
    lower, upper = df.stack().quantile([0.01, 0.95])
    df[df < lower] = lower
    df[df > upper] = upper

    # Normalise each column to have 0 mean and 1 std dev.
    # Means are computed per sensor, but std dev is computed over all measurements
    if normalise:
        df = (df - df.mean()) / df.std()

    return df
