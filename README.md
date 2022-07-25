---
title: Ergo
subtitle: A novel keyboard-like device that doesn't cause chronic wrist pain
author: 
- Boyd Kane (26723077)
---
# Ergo

#### A novel keyboard-like device that doesn't cause chronic wrist pain

## Description of Goals

This student-proposed project aims to design, build, and evaluate a sensor
suite capable of recognising hand gestures and converting them into
keystrokes. This alternative to a regular computer keyboard aims to be superior
in terms of its ergonomics and range of situations in which it can be used.


## Subsystems involved

There are several systems which all connect together in order for this project
to work. In no particular order:

- **The Arduino hardware platform** which formats sensor data into a form that can
  be parsed by the machine learning system.
- **The machine learning system** which accepts sensor data and makes two passes on
  it. The first pass pre-processes the data from raw sensor data into a form
  that is hopefully more suited for machine learning, for example:
    - omitting low-variance dimensions, 
    - scaling the data,
    - re-sampling the data,
    - dimensionality reduction,
    - removing outliers and influential points

  Once this first pass is done, a number of candidate machine-learning methods
  are applied to the pre-processed data and those models are evaluated against
  each other.
- **The Bluetooth connection system** establishes a connection with external
  devices or alternatively broadcasts itself as a Bluetooth keyboard if no
  known devices are in range.
- **The prediction system** has an embedded model that can use real-time sensor
  data to predict which gesture is likely being performed at the current
  moment. These gestures are then mapped onto keystrokes or combinations of
  keystrokes.

## List of candidate gestures
Please see `gestures.md`.

## Installation

To install and run all sub-systems of the project you will need:

- The [Arduino software](https://github.com/arduino/Arduino/#installation) in
  order to upload code to the Arduino. With [Homebrew](https://brew.sh/) installed:
```sh
brew install --cask arduino
```
- The `Arduino Mbed OS Nano Boards` package from the Boards Manager in
    order to use the Arduino Nano 33 BLE board. Upon plugging in the Arduino
    Nano 33 BLE board, a prompt in the Arduino software will guide you in
    installing this package.
- The `Mux.h` library, which can be installed from the Arduino IDE by going to
  Tools > Manage Libraries... and then searching for `Analog-Digital
  Multiplexers` by Stefano Chizzolini version 3.0.0. The github repository for
  this library is available [here](https://github.com/stechio/arduino-ad-mux-lib).
  Not installing this library will cause a `Mux.h not found` exception when
  trying to compile or upload the `collect_training_data.ino` sketch.
- `Python3` with the libraries listed in `requirements.txt` installed:

```sh
pip install -r requirements.txt
```

- Link to include: https://www.rokoko.com/products/smartgloves

## Troubleshooting

- (2022-05-10) The multiplexor has 16 inputs, but only 15 are used. This means
  that the 16th input must be left empty. Somehow the 1st input was left empty
  instead of the 16th, causing nonsense input. Shuffling all the inputs down by
  1 fixed the issue.
