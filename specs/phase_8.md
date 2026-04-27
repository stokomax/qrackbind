---
tags:
  - qrack
  - nanobind
  - python
  - pennylane
  - plugin
  - implementation
  - qrackbind
  - phase8
---
## qrackbind Phase 8 — PennyLane Device Plugin

Builds directly on [[qrackbind Phase 7]]. Phases 1–7 produce a complete, fully typed, published nanobind extension. Phase 8 integrates `qrackbind` with the quantum computing framework ecosystem. The primary deliverable is a PennyLane device plugin. The Qiskit integration is handled separately by updating the existing `qiskit-qrack-provider` package rather than building a parallel plugin from scratch.

**Prerequisite:** All Phase 7 checklist items passing. `uv run pyright` passing. Wheel published to (at minimum) TestPyPI.

---

## Nanobind Learning Goals

Phase 8 has no new nanobind binding work. The learning goals are framework-architecture focused:

| Topic | Where it appears |
|---|---|
| PennyLane `Device` API (v0.43+) — `execute()` and `preprocess_transforms()` | §2 — device class |
| TOML capabilities file format | §3 — gate declaration |
| Gate name mapping between PennyLane, Qiskit, and qrackbind | §4 — dispatch table |
| PennyLane `QuantumScript` IR — iterating operations | §5 — `execute()` implementation |
| Parameter-shift gradient rule — how PennyLane invokes it | §6 — gradient support |
| `qml.device` registration via entry points | §7 — packaging |

---

## Architecture

Phase 8 adds a pure-Python layer on top of the existing nanobind extension. No new C++ code is required.

```
┌────────────────────────────────────────────┐
│  User code                                 │
│  qml.device("qrackbind.simulator", ...)    │
├────────────────────────────────────────────┤
│  src/qrackbind/pennylane/                  │  ← Phase 8 (pure Python)
│    device.py   — QrackDevice               │
│    _dispatch.py — gate name mapping        │
│    qrack.toml  — TOML capabilities file    │
├────────────────────────────────────────────┤
│  src/qrackbind/_core (nanobind extension)  │  ← Phases 1–6
│    QrackSimulator, QrackCircuit, Pauli     │
└────────────────────────────────────────────┘
```

## File Structure

| File | Role |
|---|---|
| `src/qrackbind/pennylane/__init__.py` | Package init, exports `QrackDevice` |
| `src/qrackbind/pennylane/device.py` | `QrackDevice` class |
| `src/qrackbind/pennylane/_dispatch.py` | Gate name → qrackbind method dispatch table |
| `src/qrackbind/pennylane/qrack.toml` | TOML capabilities file declaring supported gates |
| `pyproject.toml` | Entry point registration under `pennylane.plugins` |
| `tests/test_phase8.py` | PennyLane integration tests |

---

## 1. Before Writing Code — PennyLane Gate Name Mapping

This is the prerequisite identified in the [[qrackbind Compatibility Review — April 2026]] (Issue 5). PennyLane uses its own operation names that differ from pyqrack, Qiskit, and Qrack's internal names. The mapping must be complete before the TOML file or `execute()` can be written correctly.

### Verified gate name mapping

| PennyLane operation | qrackbind method | Notes |
|---|---|---|
| `Hadamard` | `sim.h(q)` | |
| `PauliX` | `sim.x(q)` | |
| `PauliY` | `sim.y(q)` | |
| `PauliZ` | `sim.z(q)` | |
| `S` | `sim.s(q)` | |
| `T` | `sim.t(q)` | |
| `SX` | `sim.sx(q)` | PennyLane name for √X |
| `Adjoint(S)` | `sim.sdg(q)` | PennyLane wraps in `Adjoint` |
| `Adjoint(T)` | `sim.tdg(q)` | same |
| `Adjoint(SX)` | `sim.sxdg(q)` | same |
| `RX` | `sim.rx(angle, q)` | angle in radians |
| `RY` | `sim.ry(angle, q)` | |
| `RZ` | `sim.rz(angle, q)` | |
| `PhaseShift` | `sim.r1(angle, q)` | PennyLane `PhaseShift(φ)` = qrackbind `r1(φ)` |
| `Rot` | `sim.u(θ, φ, λ, q)` | Euler angles (θ, φ, λ) |
| `CNOT` | `sim.cnot(c, t)` | |
| `CY` | `sim.cy(c, t)` | |
| `CZ` | `sim.cz(c, t)` | |
| `CH` | `sim.ch(c, t)` | |
| `CRX` | `sim.crx(angle, c, t)` | |
| `CRY` | `sim.cry(angle, c, t)` | |
| `CRZ` | `sim.crz(angle, c, t)` | |
| `SWAP` | `sim.swap(q0, q1)` | |
| `ISWAP` | `sim.iswap(q0, q1)` | |
| `Toffoli` | `sim.mcx([c0, c1], t)` | 2-control X |
| `MultiControlledX` | `sim.mcx(controls, t)` | n-control X |
| `QubitUnitary` | `sim.mtrx(matrix, q)` | arbitrary 2×2 |
| `ControlledQubitUnitary` | `sim.mcmtrx(controls, matrix, t)` | arbitrary controlled |
| `Snapshot` | no-op | PennyLane state snapshot — use `state_vector` |
| `BasisState` | `sim.set_permutation(int(state))` | basis state preparation |
| `StatePrep` | `sim.set_state_vector(arr)` | arbitrary state preparation |

### Operations PennyLane decomposes automatically

The following PennyLane operations are decomposed by the preprocessor into the basis set above and therefore do **not** need direct bindings:

`CCZ`, `CSWAP`, `CPhaseShift`, `MultiRZ`, `IsingXX`, `IsingYY`, `IsingZZ`, `IsingXY`, `OrbitalRotation`, `FermionicSWAP` — PennyLane's transpiler handles these if they are absent from the TOML capabilities file.

---

## 2. TOML Capabilities File

PennyLane's `preprocess_transforms()` reads a TOML file to determine which operations are natively supported. Operations not listed are decomposed before `execute()` is called, so `execute()` never receives them.

### `src/qrackbind/pennylane/qrack.toml`

```toml
schema_version = 2

[device]
name = "qrackbind.simulator"
short_name = "QrackSimulator"
version = "0.1.0"
author = "qrackbind contributors"
wires = []

[operators]
# Gates directly supported by qrackbind with no decomposition needed.
# PennyLane will decompose any gate not listed here.
gates = [
    "Hadamard",
    "PauliX",
    "PauliY",
    "PauliZ",
    "S",
    "T",
    "SX",
    "RX",
    "RY",
    "RZ",
    "PhaseShift",
    "Rot",
    "CNOT",
    "CY",
    "CZ",
    "CH",
    "CRX",
    "CRY",
    "CRZ",
    "SWAP",
    "ISWAP",
    "Toffoli",
    "MultiControlledX",
    "QubitUnitary",
    "ControlledQubitUnitary",
    "BasisState",
    "StatePrep",
]

[observables]
# Observables supported for expectation value computation.
observables = [
    "PauliX",
    "PauliY",
    "PauliZ",
    "Identity",
    "Hadamard",
    "Hermitian",
    "Prod",
    "Sum",
    "SProd",
]

[measurement_processes]
# Measurement types this device can return natively.
# Shot-based: Expval, Var, Probs, Sample, Counts
# Exact (analytic): Expval, Var, Probs, State
exactness = "analytic"
supported_mid_circuit_measurements = false
```

---

## 3. Gate Dispatch Table

```python
# src/qrackbind/pennylane/_dispatch.py

from __future__ import annotations
import numpy as np
from qrackbind import QrackSimulator, Pauli

# PennyLane operation name → dispatch function
# Each function receives (sim, wires, params) and applies the gate.

def _apply_pauli_basis(sim: QrackSimulator, op_name: str,
                        wires: list[int], params: list[float]) -> None:
    """Apply a gate that PennyLane may express in Adjoint form."""
    raise NotImplementedError(f"Direct dispatch for {op_name} not implemented")


GATE_DISPATCH: dict[str, callable] = {
    # ── Single-qubit Clifford ──────────────────────────────────────────────
    "Hadamard":   lambda sim, w, p: sim.h(w[0]),
    "PauliX":     lambda sim, w, p: sim.x(w[0]),
    "PauliY":     lambda sim, w, p: sim.y(w[0]),
    "PauliZ":     lambda sim, w, p: sim.z(w[0]),
    "S":          lambda sim, w, p: sim.s(w[0]),
    "T":          lambda sim, w, p: sim.t(w[0]),
    "SX":         lambda sim, w, p: sim.sx(w[0]),
    # Adjoint wrappers — PennyLane expands these before dispatch in most cases;
    # include as fallback for device-level Adjoint support
    "Adjoint(S)":  lambda sim, w, p: sim.sdg(w[0]),
    "Adjoint(T)":  lambda sim, w, p: sim.tdg(w[0]),
    "Adjoint(SX)": lambda sim, w, p: sim.sxdg(w[0]),

    # ── Single-qubit rotation ──────────────────────────────────────────────
    "RX":          lambda sim, w, p: sim.rx(p[0], w[0]),
    "RY":          lambda sim, w, p: sim.ry(p[0], w[0]),
    "RZ":          lambda sim, w, p: sim.rz(p[0], w[0]),
    "PhaseShift":  lambda sim, w, p: sim.r1(p[0], w[0]),
    "Rot":         lambda sim, w, p: sim.u(p[0], p[1], p[2], w[0]),

    # ── Two-qubit gates ────────────────────────────────────────────────────
    "CNOT":  lambda sim, w, p: sim.cnot(w[0], w[1]),
    "CY":    lambda sim, w, p: sim.cy(w[0], w[1]),
    "CZ":    lambda sim, w, p: sim.cz(w[0], w[1]),
    "CH":    lambda sim, w, p: sim.ch(w[0], w[1]),
    "CRX":   lambda sim, w, p: sim.crx(p[0], w[0], w[1]),
    "CRY":   lambda sim, w, p: sim.cry(p[0], w[0], w[1]),
    "CRZ":   lambda sim, w, p: sim.crz(p[0], w[0], w[1]),
    "SWAP":  lambda sim, w, p: sim.swap(w[0], w[1]),
    "ISWAP": lambda sim, w, p: sim.iswap(w[0], w[1]),

    # ── Multi-qubit gates ──────────────────────────────────────────────────
    "Toffoli":          lambda sim, w, p: sim.mcx([w[0], w[1]], w[2]),
    "MultiControlledX": lambda sim, w, p: sim.mcx(w[:-1], w[-1]),

    # ── Arbitrary unitary ──────────────────────────────────────────────────
    "QubitUnitary": lambda sim, w, p: sim.mtrx(
        list(p[0].flatten().view(np.complex64)), w[0]),
    "ControlledQubitUnitary": lambda sim, w, p: sim.mcmtrx(
        w[:-1], list(p[0].flatten().view(np.complex64)), w[-1]),

    # ── State preparation ──────────────────────────────────────────────────
    "BasisState": lambda sim, w, p: sim.set_permutation(
        int(sum(int(b) << (len(p[0]) - 1 - i) for i, b in enumerate(p[0])))),
    "StatePrep":  lambda sim, w, p: sim.set_state_vector(
        np.array(p[0], dtype=np.complex64)),
}


def dispatch_gate(sim: QrackSimulator, op) -> None:
    """Apply a PennyLane operation to the simulator.

    op is a pennylane.Operation instance with .name, .wires, and .parameters.
    Raises KeyError if the operation name is not in GATE_DISPATCH.
    """
    name = op.name
    wires = list(op.wires)
    params = list(op.parameters)

    handler = GATE_DISPATCH.get(name)
    if handler is None:
        raise NotImplementedError(
            f"QrackDevice: operation '{name}' is not supported. "
            f"It should have been decomposed by preprocess_transforms().")
    handler(sim, wires, params)
```

---

## 4. `QrackDevice` — Device Class

```python
# src/qrackbind/pennylane/device.py

from __future__ import annotations
import pathlib
from typing import Sequence

import numpy as np
import pennylane as qml
from pennylane.devices import Device, ExecutionConfig
from pennylane.tape import QuantumScript, QuantumScriptOrBatch
from pennylane.transforms import TransformProgram
from pennylane.typing import ResultBatch

from qrackbind import QrackSimulator, Pauli, QrackException
from ._dispatch import dispatch_gate

_TOML_PATH = pathlib.Path(__file__).parent / "qrack.toml"


class QrackDevice(Device):
    """PennyLane device backed by the Qrack quantum simulator via qrackbind.

    Usage::

        import pennylane as qml
        dev = qml.device("qrackbind.simulator", wires=4)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

    Constructor keyword arguments are forwarded to QrackSimulator. Common
    options: isTensorNetwork, isOpenCL, isStabilizerHybrid. See
    QrackSimulator.__init__ for the full list.
    """

    config_filepath = _TOML_PATH
    pennylane_requires = ">=0.43"

    def __init__(self, wires=None, shots=None, **simulator_kwargs):
        super().__init__(wires=wires, shots=shots)
        self._simulator_kwargs = simulator_kwargs

    def _make_simulator(self, num_qubits: int) -> QrackSimulator:
        """Create a fresh QrackSimulator for a circuit execution."""
        return QrackSimulator(qubitCount=num_qubits, **self._simulator_kwargs)

    # ── PennyLane Device API ───────────────────────────────────────────────

    def preprocess_transforms(
        self,
        execution_config: ExecutionConfig | None = None,
    ) -> tuple[TransformProgram, ExecutionConfig]:
        """Declare supported gate set and return decomposition transforms.

        PennyLane reads the TOML capabilities file and inserts decomposition
        transforms for any operation not listed in [operators].gates. By the
        time execute() is called, all operations in the circuit are guaranteed
        to be in GATE_DISPATCH.
        """
        config = execution_config or ExecutionConfig()
        program = TransformProgram()

        # Standard PennyLane preprocessing — decomposes to supported gate set,
        # validates wires, handles mid-circuit measurements if needed.
        program.add_transform(qml.transforms.decompose,
                              stopping_condition=self.stopping_condition,
                              max_expansion=10)

        if self.shots is not None:
            program.add_transform(qml.transforms.broadcast_expand)

        return program, config

    def stopping_condition(self, op: qml.operation.Operator) -> bool:
        """Return True if an operation is natively supported (no decomposition)."""
        from ._dispatch import GATE_DISPATCH
        return op.name in GATE_DISPATCH

    def execute(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig | None = None,
    ) -> ResultBatch:
        """Execute one or more circuits and return measurement results.

        PennyLane calls this method after preprocessing — all operations are
        guaranteed to be in GATE_DISPATCH at this point.
        """
        if isinstance(circuits, QuantumScript):
            circuits = [circuits]

        return tuple(self._execute_one(circuit) for circuit in circuits)

    def _execute_one(self, circuit: QuantumScript):
        """Execute a single QuantumScript and return its measurement result."""
        num_qubits = len(circuit.wires)
        wire_map = {wire: idx for idx, wire in enumerate(circuit.wires)}

        sim = self._make_simulator(num_qubits)

        # Apply all gate operations
        for op in circuit.operations:
            # Remap PennyLane wire labels to simulator qubit indices
            remapped_op = _remap_wires(op, wire_map)
            dispatch_gate(sim, remapped_op)

        # Evaluate measurements
        results = []
        for m in circuit.measurements:
            results.append(self._evaluate_measurement(sim, m, wire_map, circuit))

        return results[0] if len(results) == 1 else tuple(results)

    def _evaluate_measurement(self, sim, m, wire_map, circuit):
        """Evaluate a single measurement process."""
        import pennylane.measurements as meas

        if isinstance(m, meas.ExpectationMP):
            return self._expval(sim, m.obs, wire_map)

        elif isinstance(m, meas.VarianceMP):
            return self._variance(sim, m.obs, wire_map)

        elif isinstance(m, meas.ProbabilityMP):
            wires = [wire_map[w] for w in m.wires]
            return self._probabilities(sim, wires)

        elif isinstance(m, meas.StateMP):
            return sim.state_vector.astype(np.complex128)

        elif isinstance(m, meas.SampleMP):
            return self._sample(sim, m, wire_map, circuit.shots)

        elif isinstance(m, meas.CountsMP):
            samples = self._sample(sim, m, wire_map, circuit.shots)
            unique, counts = np.unique(samples, axis=0, return_counts=True)
            return dict(zip(
                [''.join(str(b) for b in row) for row in unique],
                counts.tolist()
            ))

        raise NotImplementedError(
            f"QrackDevice: measurement type {type(m).__name__} not supported.")

    def _expval(self, sim: QrackSimulator, obs, wire_map: dict) -> float:
        """Compute expectation value of an observable."""
        paulis, qubits = self._observable_to_paulis(obs, wire_map)
        if paulis:
            return float(sim.exp_val_pauli(paulis, qubits))

        # Hermitian observable — use matrix expectation
        return float(self._matrix_expval(sim, obs, wire_map))

    def _variance(self, sim: QrackSimulator, obs, wire_map: dict) -> float:
        """Compute variance of an observable."""
        paulis, qubits = self._observable_to_paulis(obs, wire_map)
        if paulis:
            return float(sim.variance_pauli(paulis, qubits))
        ev = self._expval(sim, obs, wire_map)
        ev2 = self._matrix_expval(sim, qml.prod(obs, obs), wire_map)
        return float(ev2 - ev ** 2)

    def _observable_to_paulis(self, obs, wire_map: dict):
        """Convert a PennyLane observable to (list[Pauli], list[int]) if possible."""
        pauli_map = {
            "PauliX": Pauli.PauliX,
            "PauliY": Pauli.PauliY,
            "PauliZ": Pauli.PauliZ,
            "Identity": Pauli.PauliI,
        }

        if obs.name in pauli_map:
            return [pauli_map[obs.name]], [wire_map[obs.wires[0]]]

        if obs.name == "Prod":
            # Tensor product of Pauli operators
            paulis, qubits = [], []
            for factor in obs.operands:
                if factor.name not in pauli_map:
                    return [], []   # not a pure Pauli product
                paulis.append(pauli_map[factor.name])
                qubits.append(wire_map[factor.wires[0]])
            return paulis, qubits

        return [], []   # not a Pauli observable — fall back to matrix method

    def _probabilities(self, sim: QrackSimulator, qubits: list[int]) -> np.ndarray:
        """Marginal probabilities for the given qubits."""
        full = sim.probabilities
        # Marginalise over qubits not in the list
        n = sim.num_qubits
        out_size = 1 << len(qubits)
        out = np.zeros(out_size, dtype=np.float64)
        for basis in range(1 << n):
            idx = sum(
                ((basis >> q) & 1) << i
                for i, q in enumerate(reversed(qubits))
            )
            out[idx] += float(full[basis])
        return out

    def _sample(self, sim, m, wire_map, shots) -> np.ndarray:
        """Generate shot samples via MultiShotMeasureMask."""
        wires = [wire_map[w] for w in (m.wires if m.wires else sim.num_qubits)]
        results = sim.measure_shots(wires, shots)
        samples = []
        for outcome, count in results.items():
            bits = [(outcome >> (len(wires) - 1 - i)) & 1
                    for i in range(len(wires))]
            samples.extend([bits] * count)
        return np.array(samples, dtype=int)

    def _matrix_expval(self, sim, obs, wire_map) -> float:
        """Expectation value via state vector contraction for Hermitian observables."""
        sv = sim.state_vector.astype(np.complex128)
        matrix = qml.matrix(obs, wire_order=list(wire_map.keys()))
        n = sim.num_qubits
        rho_psi = matrix @ sv
        return float(np.real(np.dot(sv.conj(), rho_psi)))

    # ── Gradient support ───────────────────────────────────────────────────

    def supports_derivatives(
        self,
        execution_config: ExecutionConfig | None = None,
        circuit: QuantumScript | None = None,
    ) -> bool:
        """Declare parameter-shift gradient support for analytic execution."""
        if execution_config is None:
            return True
        return (execution_config.gradient_method == "parameter-shift"
                and self.shots is None)


def _remap_wires(op, wire_map: dict):
    """Return a copy of an operation with wire labels mapped to integer indices."""
    new_wires = [wire_map[w] for w in op.wires]
    return op.map_wires(dict(zip(op.wires, new_wires)))
```

---

## 5. Parameter-Shift Gradient Support

PennyLane's parameter-shift rule runs the circuit at `param + π/2` and `param - π/2` and computes the finite difference. It invokes `execute()` twice per parameter automatically — `QrackDevice` does not need to implement the derivative calculation itself. `supports_derivatives` returning `True` for `gradient_method == "parameter-shift"` is sufficient.

Verify gradient support works:

```python
import pennylane as qml
import numpy as np

dev = qml.device("qrackbind.simulator", wires=2)

@qml.qnode(dev, diff_method="parameter-shift")
def circuit(x, y):
    qml.RX(x, wires=0)
    qml.RY(y, wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

x = np.array(0.5, requires_grad=True)
y = np.array(0.3, requires_grad=True)

result = circuit(x, y)
grad_x, grad_y = qml.grad(circuit)(x, y)

print(f"expval: {result:.4f}")
print(f"d/dx:   {grad_x:.4f}")
print(f"d/dy:   {grad_y:.4f}")
```

---

## 6. Entry Point Registration

Register the device so `qml.device("qrackbind.simulator", wires=n)` works:

```toml
# pyproject.toml
[project.entry-points."pennylane.plugins"]
"qrackbind.simulator" = "qrackbind.pennylane.device:QrackDevice"
```

Verify after install:

```bash
uv pip install -e .
python -c "import pennylane as qml; print(qml.device('qrackbind.simulator', wires=2))"
```

---

## 7. VQE Example

A complete variational quantum eigensolver for the hydrogen molecule Hamiltonian, demonstrating that the full gradient pipeline works end-to-end:

```python
# examples/vqe_h2.py
import pennylane as qml
import numpy as np
from pennylane import qchem

# Hydrogen molecule Hamiltonian
symbols = ["H", "H"]
coordinates = np.array([0.0, 0.0, -0.66140414, 0.0, 0.0, 0.66140414])
H, num_qubits = qchem.molecular_hamiltonian(symbols, coordinates)

dev = qml.device("qrackbind.simulator", wires=num_qubits)

def ansatz(params, wires):
    qml.BasisState(np.array([1, 1, 0, 0]), wires=wires)
    for i in range(len(params)):
        qml.RY(params[i], wires=wires[i % num_qubits])
    for i in range(num_qubits - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])

@qml.qnode(dev, diff_method="parameter-shift")
def cost_fn(params):
    ansatz(params, range(num_qubits))
    return qml.expval(H)

params = np.random.uniform(-np.pi, np.pi, num_qubits)
opt = qml.GradientDescentOptimizer(stepsize=0.4)

for step in range(50):
    params, energy = opt.step_and_cost(cost_fn, params)
    if step % 10 == 0:
        print(f"Step {step:3d}  Energy = {energy:.6f} Ha")

print(f"\nFinal energy: {energy:.6f} Ha")
print(f"Exact (FCI):  -1.136189 Ha")
```

---

## 8. Qiskit Integration — Coordinate with qiskit-qrack-provider

As documented in [[qrackbind Compatibility Review — April 2026]] (Issue 3) and [[Framework Plugin Architecture (PennyLane + Qiskit)]], a `qiskit-qrack-provider` PyPI package already exists and wraps pyqrack as a Qiskit backend. Building a parallel Qiskit plugin inside qrackbind would fragment the Qiskit ecosystem.

**The correct approach for Phase 8:**

Open a PR to `qiskit-qrack-provider` that updates the package to use `qrackbind` instead of `pyqrack`. The diff is small — the same scope as the Bloqade migration PR:

1. Update `requirements.txt`: `pyqrack` → `qrackbind`
2. Update imports: `from pyqrack import QrackSimulator` → `from qrackbind import QrackSimulator`
3. Update any `sim.m(q)` calls → `sim.measure(q)` (deprecated alias handles this but the PR should clean it up)
4. Verify the existing `qiskit-qrack-provider` test suite passes

This is tracked as a Phase 8 coordination task, not a Phase 8 code-delivery task.

---

## 9. Test Suite

```python
# tests/test_phase8.py
import math
import numpy as np
import pytest

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not PENNYLANE_AVAILABLE,
    reason="PennyLane not installed")


@pytest.fixture
def dev2():
    return qml.device("qrackbind.simulator", wires=2)

@pytest.fixture
def dev4():
    return qml.device("qrackbind.simulator", wires=4)


# ── Device registration ────────────────────────────────────────────────────────

class TestDeviceRegistration:
    def test_device_importable(self):
        from qrackbind.pennylane.device import QrackDevice
        assert QrackDevice is not None

    def test_qml_device_creates_qrack_device(self):
        from qrackbind.pennylane.device import QrackDevice
        dev = qml.device("qrackbind.simulator", wires=2)
        assert isinstance(dev, QrackDevice)

    def test_device_has_correct_num_wires(self, dev2):
        assert len(dev2.wires) == 2


# ── Basic circuit execution ────────────────────────────────────────────────────

class TestCircuitExecution:
    def test_hadamard_expval(self, dev2):
        @qml.qnode(dev2)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))
        assert circuit() == pytest.approx(0.0, abs=1e-4)

    def test_pauli_x_expval(self, dev2):
        @qml.qnode(dev2)
        def circuit():
            qml.PauliX(wires=0)
            return qml.expval(qml.PauliZ(0))
        assert circuit() == pytest.approx(-1.0, abs=1e-5)

    def test_ground_state_expval(self, dev2):
        @qml.qnode(dev2)
        def circuit():
            return qml.expval(qml.PauliZ(0))
        assert circuit() == pytest.approx(1.0, abs=1e-5)

    def test_bell_state_zz_expval(self, dev2):
        @qml.qnode(dev2)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        assert circuit() == pytest.approx(1.0, abs=1e-4)

    def test_rx_expval(self, dev2):
        @qml.qnode(dev2)
        def circuit(angle):
            qml.RX(angle, wires=0)
            return qml.expval(qml.PauliZ(0))
        # <Z> = cos(angle)
        angle = 0.7
        assert circuit(angle) == pytest.approx(math.cos(angle), abs=1e-4)

    def test_state_measurement(self, dev2):
        @qml.qnode(dev2)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.state()
        sv = circuit()
        assert sv.shape == (4,)
        assert abs(sv[0] - 1 / math.sqrt(2)) < 1e-4
        assert abs(sv[2] - 1 / math.sqrt(2)) < 1e-4

    def test_probs_measurement(self, dev2):
        @qml.qnode(dev2)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.probs(wires=[0])
        probs = circuit()
        assert probs == pytest.approx([0.5, 0.5], abs=1e-4)


# ── Variance ──────────────────────────────────────────────────────────────────

class TestVariance:
    def test_z_variance_eigenstate(self, dev2):
        @qml.qnode(dev2)
        def circuit():
            return qml.var(qml.PauliZ(0))
        # |0> is Z eigenstate — variance = 0
        assert circuit() == pytest.approx(0.0, abs=1e-5)

    def test_z_variance_superposition(self, dev2):
        @qml.qnode(dev2)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.var(qml.PauliZ(0))
        # |+> in Z basis — variance = 1
        assert circuit() == pytest.approx(1.0, abs=1e-4)


# ── Parametric circuits and gradients ────────────────────────────────────────

class TestGradients:
    def test_parameter_shift_gradient(self, dev2):
        @qml.qnode(dev2, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        x = np.array(0.5)
        grad = qml.grad(circuit)(x)
        # d/dx <Z> = d/dx cos(x) = -sin(x)
        assert grad == pytest.approx(-math.sin(0.5), abs=1e-4)

    def test_two_parameter_gradient(self, dev2):
        @qml.qnode(dev2, diff_method="parameter-shift")
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        x = np.array(0.4, requires_grad=True)
        y = np.array(0.6, requires_grad=True)
        grad_x, grad_y = qml.grad(circuit)(x, y)
        assert isinstance(grad_x, float)
        assert isinstance(grad_y, float)

    def test_gradient_matches_finite_difference(self, dev2):
        @qml.qnode(dev2, diff_method="parameter-shift")
        def circuit(x):
            qml.RZ(x, wires=0)
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        x0 = 0.8
        grad_ps = qml.grad(circuit)(np.array(x0))

        eps = 1e-4
        fd = (circuit(x0 + eps) - circuit(x0 - eps)) / (2 * eps)
        assert grad_ps == pytest.approx(fd, rel=1e-3)


# ── Gate coverage ──────────────────────────────────────────────────────────────

class TestGateCoverage:
    def test_ry_gate(self, dev2):
        @qml.qnode(dev2)
        def circuit(angle):
            qml.RY(angle, wires=0)
            return qml.expval(qml.PauliZ(0))
        assert circuit(math.pi) == pytest.approx(-1.0, abs=1e-4)

    def test_toffoli_gate(self, dev4):
        @qml.qnode(dev4)
        def circuit():
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            qml.Toffoli(wires=[0, 1, 2])
            return qml.expval(qml.PauliZ(2))
        # Controls |1,1> → target flipped to |1>
        assert circuit() == pytest.approx(-1.0, abs=1e-4)

    def test_rot_gate(self, dev2):
        @qml.qnode(dev2)
        def circuit():
            qml.Rot(math.pi, 0, 0, wires=0)   # = X gate
            return qml.expval(qml.PauliZ(0))
        assert circuit() == pytest.approx(-1.0, abs=1e-4)

    def test_phase_shift_gate(self, dev2):
        @qml.qnode(dev2)
        def circuit(phi):
            qml.Hadamard(wires=0)
            qml.PhaseShift(phi, wires=0)
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))
        # PhaseShift(π) = Z; H·Z·H = X; <0|X|0> = 0
        assert circuit(math.pi) == pytest.approx(0.0, abs=1e-4)


# ── VQE smoke test ────────────────────────────────────────────────────────────

class TestVQE:
    def test_vqe_converges(self):
        """VQE minimisation reduces energy over iterations."""
        dev = qml.device("qrackbind.simulator", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(params):
            qml.RY(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        params = np.array([0.1, 0.2], requires_grad=True)
        opt = qml.GradientDescentOptimizer(stepsize=0.3)

        energy_start = circuit(params)
        for _ in range(20):
            params, _ = opt.step_and_cost(circuit, params)
        energy_final = circuit(params)

        assert energy_final < energy_start
```

---

## 10. Phase 8 Completion Checklist

```
□ pennylane package is a listed dependency in pyproject.toml [project.optional-dependencies]
□ qml.device("qrackbind.simulator", wires=n) creates a QrackDevice
□ Entry point registered under pennylane.plugins in pyproject.toml
□ qrack.toml capabilities file present and valid TOML
□ preprocess_transforms() returns a TransformProgram
□ stopping_condition() returns True for all gates in GATE_DISPATCH
□ Hadamard + CNOT Bell state executes — ZZ expval ≈ 1.0
□ RX(angle) → expval(Z) ≈ cos(angle)
□ RX gradient matches -sin(angle) via parameter-shift
□ Two-parameter gradient returns finite float values
□ Finite-difference and parameter-shift gradients agree to 1e-3 relative
□ qml.var(PauliZ) ≈ 0 for eigenstate, ≈ 1 for superposition
□ qml.probs() marginal probabilities sum to 1
□ qml.state() returns correct complex64 state vector
□ Toffoli gate: controls |11⟩ flips target
□ Rot(π,0,0) acts as X gate
□ PhaseShift(π) = Z gate
□ VQE energy decreases over 20 gradient descent steps
□ qiskit-qrack-provider issue or PR opened
□ uv run pytest tests/test_phase8.py — all green
□ uv run pytest tests/ — all phases green
```

---

## 11. What Phase 8 Leaves Out (Deferred)

| Item | Reason deferred |
|---|---|
| Catalyst QJIT support (`get_c_interface()`) | Requires a separate C++ `QuantumDevice` implementation conforming to Catalyst's `QuantumDevice.hpp` — significant additional C++ work |
| `adjoint` differentiation | Requires implementing `compute_vjp()` — the forward pass must store intermediate states; adds memory overhead |
| Mid-circuit measurement support | `supported_mid_circuit_measurements = false` in TOML; requires conditional logic in `_execute_one` |
| Qiskit plugin (built from scratch) | Superseded by the `qiskit-qrack-provider` PR strategy — avoid ecosystem fragmentation |
| Noise model integration | Requires `QrackSimulator(noise=λ)` constructor path and PennyLane's `NoiseModel` API |
| Batched execution parallelism | Multiple circuits could be run in parallel using the simulator registry; deferred optimisation |

---

## Related

- [[qrackbind Phase 7]]
- [[qrackbind Phase 6]]
- [[qrackbind Phase 4]]
- [[qrackbind Phase 9]]
- [[qrackbind Project Phase Breakdown]]
- [[Framework Plugin Architecture (PennyLane + Qiskit)]]
- [[qrackbind Compatibility Review — April 2026]]
- [[pyqrack Compatibility Strategy]]
