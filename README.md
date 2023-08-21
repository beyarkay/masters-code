# _Ergo_

It is not possible to run the full program as this requires that the hardware
associated with _Ergo_ be plugged in and accessible.

The script `machine_learning/ergo.py` is used to control the hardware and to
use it as a keyboard.

## Tasks

```
- for `rep_num` in 0..10:
  - for `num_gesture_classes` in (5, 50, 51):
    - match `model`:
      - CuSUM: threshold in (TODO, TODO)
      - HMM: no hpars
      - FFNN:
        - for `num_layers` in (1,2,3)
          - for `nodes_per_layer` in (20, 40, 60, 80, 100)
            - for `learning_rate` in (1e-2, 1e-3, 1e-4, 1e-5)
              - for `dropout_rate` in (0.0, 0.5, 1.0)
                - for `l2_coefficient` in (1e-2, 1e-3, 1e-4, 1e-5)

```

- [ ] Collect enough data that all classes have equal number of observations
  - [ ] This will require re-labelling the new data ):
- [ ] Collect "real life" training data
- [ ] Change code to reweight the validation loss
- [ ] Maybe reimplement the HMM code to be able to handle many observations?
- [ ] Check the implementation of CuSUM and how it works in detail.
- [ ] Run experiments with hyperparameter tuning on the regular data
  - [ ] Record trn_loss:val_loss ratio
  - Hyperparameters:
    - Overall: number of gesture classes
    - CuSUM: Threshold
    - HMM: None
    - FFNN: Number of layers, nodes per layer, learning rate, dropout, L1/L2
      normalization, (batch norm?)
- [ ] Run experiments with the "real life" data
- [ ] Run experiments with the "real life" data _and_ the spell checker

## Experiments to run

Research questions

- Do a test with both implicit and explicit segmentation
- Only ever use 40ms of historical data

---

- Testing HMMs
- Testing NNs
- Testing CuSUM
- Running live predictions
- Running predictions on stored experiments
- Integrating prediction model with some sort of spell correct

NOTE: The HMM just cannot handle 100k observations, it makes training
untenable so the max observations was set to 200...

## Installing dependencies

This can be done by:

```
cd machine_learning
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Unit Tests

Once the dependencies are installed, the tests can be run by:

```
cd machine_learning
source .venv/bin/activate
make test
```
