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
What follows is first a definition of the terminology used to describe the
various gestures (taken from medical literature) and then a list of gestures
that the models will be trained to recognise.

### Terminology used for cataloguing the gestures
The language used to describe gestures is taken from medical literature. For
the readers that don't have a background in medicine, a plain English
explanation is provided here.

- Movements of the forearm
    - **Pronation**: When the forearm is rotated so that the palm faces
      downwards.
    - **Supination** When the forearm is rotated so that the palm faces
      upwards.
- Movements of the wrist or fingers
    - **Extension**: When the wrist or finger is rotated to bring the back of the
      hand closer to the outer forearm.
    - **Hyperextension**: When the finger is rotated above the plane formed by
      the palm
    - **Flexion**: When the wrist or finger is rotated to bring the palm closer to
      the inner forearm.
    - **Radial deviation**: When the wrist or finger is rotated to bring the thumb
      towards the forearm.
    - **Ulnar deviation**: When the wrist or finger is rotated to bring the pinky
      finger towards the forearm.
- Movements of the fingers
    - **Abduction**: When the fingers are splayed away from each other.
    - **Adduction**: When the fingers are brought together in a flat plane.
- **Palmar**: The area of the hand closer to the palm than to the back of the
  hand.

### Candidate gestures

There are several defined dimensions along which a gesture can change. Each
dimension is listed below, along with its possible values. Note that not every
combination is physically possible, this is merely an upper bound.

- Handedness (2): right or left
- Fingers (31): any of $_5C_{\{1, \dots, 5\}} = 5 + 10 + 10 + 5 + 1 =
  31$ for each of the combinations of 1, 2, 3, 4, or 5 fingers on each hand.
- Motion (3): flexion, extension, hyperextension
- Contact (3): No contact, thumb touches finger pad, thumb touches finger nail

Which totals to 558 different gestures, excluding any gestures made between two
hands

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

