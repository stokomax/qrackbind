<!--
This README documents the qrackbind PennyLane integration package.
-->

# qrackbind PennyLane module

`qrackbind.pennylane` provides three PennyLane device plugins:

| Entry point                    | Backend                 | Notes                                       |
|-------------------------------|-------------------------|---------------------------------------------|
| `qrackbind.simulator`         | `QrackSimulator`        | Full gate surface, dense simulation         |
| `qrackbind.stabilizer`        | `QrackStabilizer`       | Clifford-only, polynomial memory (tableau)  |
| `qrackbind.stabilizer_hybrid` | `QrackStabilizerHybrid` | Clifford until first non-Clifford gate, then dense |

After installing qrackbind with the PennyLane extra, devices are available as:

```python
import pennylane as qml

dev = qml.device("qrackbind.simulator", wires=2)
dev_stab = qml.device("qrackbind.stabilizer", wires=2)
dev_hybrid = qml.device("qrackbind.stabilizer_hybrid", wires=2)
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
the underlying simulator. For `qrackbind.simulator`, supported options include
`isTensorNetwork`, `isOpenCL`, and `isStabilizerHybrid`. For
`qrackbind.stabilizer_hybrid`, supported options are `isCpuGpuHybrid`,
`isOpenCL`, `isHostPointer`, and `isSparse`. The pure
`qrackbind.stabilizer` device takes no extra kwargs.

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

### Sample measurement shapes

The shape of `qml.sample()` results follows the standard PennyLane convention:

| Call form | Return shape | Values |
|-----------|--------------|--------|
| `qml.sample(qml.PauliZ(0))` | `(shots,)` | float eigenvalues: `+1.0` or `-1.0` |
| `qml.sample(wires=[0, 1])` | `(shots, num_wires)` | integer bits: `0` or `1` |

```python
dev = qml.device("qrackbind.simulator", wires=2, shots=1000)

@qml.qnode(dev)
def circuit_obs():
    qml.Hadamard(wires=0)
    return qml.sample(qml.PauliZ(0))   # observable → eigenvalues

result = circuit_obs()
print(result.shape)   # (1000,)   — float eigenvalues: +1.0 or -1.0

@qml.qnode(dev)
def circuit_wires():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.sample(wires=[0, 1])    # no observable → bit strings

result = circuit_wires()
print(result.shape)   # (1000, 2) — integer bits: 0 or 1
```

This behaviour applies to all three device backends (`qrackbind.simulator`,
`qrackbind.stabilizer`, and `qrackbind.stabilizer_hybrid`).

## Operation support

Gate application is centralized in `_dispatch.py`, which exports two dispatch
tables:

- **`GATE_DISPATCH`**: full gate surface used by `qrackbind.simulator` and
  `qrackbind.stabilizer_hybrid`.
- **`CLIFFORD_GATE_DISPATCH`**: Clifford-only subset used by
  `qrackbind.stabilizer`.

### `qrackbind.simulator` / `qrackbind.stabilizer_hybrid` gate set

Covers common single-qubit, two-qubit, controlled, multi-controlled, unitary,
and state-preparation operations:

- `Hadamard`, `PauliX`, `PauliY`, `PauliZ`, `S`, `T`, `SX`
- `RX`, `RY`, `RZ`, `PhaseShift`, `Rot`, `U`, `U2`, `U3`
- `CNOT`, `CY`, `CZ`, `CH`, `CRX`, `CRY`, `CRZ`, `ControlledPhaseShift`
- `SWAP`, `ISWAP`, `Toffoli`, `MultiControlledX/Y/Z`
- `QubitUnitary`, `ControlledQubitUnitary`, `BasisState`, `StatePrep`

### `qrackbind.stabilizer` gate set (Clifford only)

Only operations that map Pauli operators to Pauli operators:

- `Hadamard`, `PauliX`, `PauliY`, `PauliZ`, `S`, `SX`
- `CNOT`, `CY`, `CZ`, `CH`, `SWAP`, `ISWAP`
- `BasisState` (via `set_permutation`)

Non-Clifford gates (`T`, `RX`, `RY`, `RZ`, `PhaseShift`, `QubitUnitary`, …)
are not in the Clifford dispatch table. PennyLane will attempt to decompose
them; if no Clifford decomposition exists it raises a decomposition error.

**Measurement restrictions for `qrackbind.stabilizer`**: `qml.state()` and
`qml.probs()` are not supported (no dense state vector). Only `qml.expval()`,
`qml.var()` (Pauli observables), `qml.sample()`, and `qml.counts()` are
available.

PennyLane `PhaseShift` is intentionally implemented through an explicit matrix
to match PennyLane's relative phase `diag(1, exp(iφ))`; the lower-level Qrack
`r1` binding behaves as a global-phase-like operation in this stack.

## Stabilizer device example

```python
import pennylane as qml

# Pure Clifford: polynomial memory, Pauli measurements only
stab = qml.device("qrackbind.stabilizer", wires=2)

@qml.qnode(stab)
def bell_expval():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

print(bell_expval())  # ≈ 1.0

# Shot-based sampling with a Pauli observable → 1-D eigenvalue array
stab_shots = qml.device("qrackbind.stabilizer", wires=2, shots=1000)

@qml.qnode(stab_shots)
def bell_sample():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.sample(qml.PauliZ(0))

result = bell_sample()
print(result.shape)   # (1000,)  — +1.0 or -1.0

# Stabilizer-hybrid: starts as Clifford, falls back on first non-Clifford gate
hyb = qml.device("qrackbind.stabilizer_hybrid", wires=2)

@qml.qnode(hyb)
def t_circuit(theta):
    qml.Hadamard(wires=0)
    qml.RY(theta, wires=1)   # non-Clifford → triggers dense fallback
    qml.CNOT(wires=[0, 1])
    return qml.state()

print(t_circuit(0.5))
```

## Testing

Focused tests for the simulator and stabilizer devices live in `tests/pennylane/`:

| File | Coverage |
|------|----------|
| `test_apply.py` | State preparation, single- and two-qubit gates, parametrized gates, unitaries, Toffoli |
| `test_device.py` | Device registration, circuit execution, expval, probs, state, variance, gradients, VQE |
| `test_integration.py` | Load device, repeated executions, expectations, parameter-shift gradients, Bell state |
| `test_stabilizer_device.py` | Stabilizer expval/variance/sampling/gates and hybrid device — full measurement surface |
| `test_units.py` | Device attributes, analytic probabilities, shots, reset behaviour |

Run all PennyLane tests with:

```bash
uv run pytest tests/pennylane -q
# 123 passed
```
