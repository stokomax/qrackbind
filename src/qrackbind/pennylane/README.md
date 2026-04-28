<!--
This README documents the qrackbind PennyLane integration package.
-->

# qrackbind PennyLane module

`qrackbind.pennylane` provides the PennyLane device plugin backed by
`qrackbind.QrackSimulator`. After installing qrackbind with the PennyLane extra,
the device is available as:

```python
import pennylane as qml

dev = qml.device("qrackbind.simulator", wires=2)
```

## Installation

```bash
pip install "qrackbind[pennylane]"
```

For development in this repository, use the project environment:

```bash
uv sync --extra pennylane
```

## Basic usage

```python
import pennylane as qml

dev = qml.device("qrackbind.simulator", wires=2)

@qml.qnode(dev)
def bell_zz():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

print(bell_zz())  # approximately 1.0
```

Constructor keyword arguments not consumed by PennyLane are forwarded to
`QrackSimulator`, so backend options such as `isTensorNetwork`, `isOpenCL`, or
`isStabilizerHybrid` can be supplied through `qml.device(...)`.

## Measurements and differentiation

The device supports the core PennyLane measurement processes used by QNodes:

- `qml.expval(...)`
- `qml.var(...)`
- `qml.probs(...)`
- `qml.state()`
- `qml.sample(...)`
- `qml.counts(...)`

Parameterized circuits support PennyLane's parameter-shift workflow:

```python
dev = qml.device("qrackbind.simulator", wires=1)

@qml.qnode(dev, diff_method="parameter-shift")
def circuit(theta):
    qml.RY(theta, wires=0)
    return qml.expval(qml.PauliZ(0))

grad = qml.grad(circuit, argnums=0)(0.123)
```

## Operation support

Gate application is centralized in `_dispatch.py`, where PennyLane operation
names are mapped to `QrackSimulator` methods. Unsupported operations are passed
through PennyLane's decomposition transform before execution.

The native dispatch table covers common single-qubit, two-qubit, controlled,
multi-controlled, unitary, and state-preparation operations, including:

- `Hadamard`, `PauliX`, `PauliY`, `PauliZ`, `S`, `T`, `SX`
- `RX`, `RY`, `RZ`, `PhaseShift`, `Rot`, `U`, `U2`, `U3`
- `CNOT`, `CY`, `CZ`, `CH`, `CRX`, `CRY`, `CRZ`, `ControlledPhaseShift`
- `SWAP`, `ISWAP`, `Toffoli`, `MultiControlledX/Y/Z`
- `QubitUnitary`, `ControlledQubitUnitary`, `BasisState`, `StatePrep`

PennyLane `PhaseShift` is intentionally implemented through an explicit matrix
to match PennyLane's relative phase `diag(1, exp(iφ))`; the lower-level Qrack
`r1` binding behaves as a global-phase-like operation in this stack.

## Testing

Focused tests for this plugin live in `tests/test_pennylane_device.py`.
Compatibility-style tests live in `tests/pennylane/` and were adapted from the
Unitary Foundation PennyLane-Qrack test suite:

<https://github.com/unitaryfoundation/pennylane-qrack/tree/master/tests>

Run the PennyLane tests with:

```bash
uv run pytest tests/test_pennylane_device.py tests/pennylane -q
```
