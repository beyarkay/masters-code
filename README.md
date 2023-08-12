# _Ergo_

It is not possible to run the full program as this requires that the hardware
associated with _Ergo_ be plugged in and accessible.

The script `machine_learning/ergo.py` is used to control the hardware and to
use it as a keyboard.

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
