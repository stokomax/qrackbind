---
tags:
  - qrack
  - nanobind
  - python
  - circuit
  - implementation
  - qrackbind
  - phase6
---
## qrackbind Phase 6 — QrackCircuit and Circuit-Level Operations

Builds directly on [[qrackbind Phase 5]]. Phases 1–5 expose a complete stateful `QrackSimulator` API. Phase 6 adds a second class — `QrackCircuit` — which represents a quantum circuit as a replayable, optimisable, serialisable object independent of any specific simulator instance. This unlocks circuit-level optimisation (Qrack's gate fusion and simplification), QASM output, and the framework dispatch architecture described in [[Framework Plugin Architecture (PennyLane + Qiskit)]].

Phase 6 also delivers the deferred items from Phase 4: `exp_val_unitary`, `variance_unitary`, and `ExpectationBitsFactorized`.

**Prerequisite:** All Phase 5 checklist items passing. `uv run pytest tests/test_phase5.py` green.

---

## Nanobind Learning Goals

| Topic | Where it appears |
|---|---|
| Binding a second class in the same module | §1 — `QrackCircuit` alongside `QrackSimulator` |
| Cross-class method references — accepting a bound type as a parameter | §4 — `run(sim: QrackSimulator)` |
| `nb::keep_alive<>` — preventing use-after-free across class boundaries | §4 — circuit holds a sim reference during `run()` |
| Ownership transfer and `nb::rv_policy` for class return | §5 — `inverse()` returns a new `QrackCircuit` |
| `nb::type<T>()` — querying a registered type at runtime | §4 — runtime type check on sim arg |
| Binding a C++ enum used only within one class | §2 — `GateType` enum |

---

## File Structure

| File | Changes |
|---|---|
| `bindings/circuit.cpp` | **New file** — `bind_circuit()` defines `QrackCircuit` binding |
| `bindings/binding_core.h` | Add `#include "qcircuit.hpp"` (Qrack circuit header) |
| `bindings/module.cpp` | Add `void bind_circuit(nb::module_& m)` and call it after `bind_simulator` |
| `CMakeLists.txt` | Add `bindings/circuit.cpp` to sources |
| `bindings/simulator.cpp` | Add deferred Phase 4 items: `exp_val_unitary`, `variance_unitary`, `ExpectationBitsFactorized` |
| `src/qrackbind/__init__.py` | Export `QrackCircuit` |

---

## 1. `QrackCircuit` — Design Overview

`QrackCircuit` wraps Qrack's `QCircuit` class, which records gate operations and provides:

- **Gate accumulation** — gates are appended to an internal list rather than applied immediately
- **Optimisation** — adjacent gates that cancel or commute are fused or eliminated
- **Execution** — the recorded circuit is applied to any `QrackSimulator` instance
- **Inversion** — a new circuit that applies all gates in reverse with conjugate-transposed matrices
- **QASM output** — serialisation to OpenQASM 2.0 for interoperability

### C++ backing class

Qrack's `QCircuit` is in `qcircuit.hpp`. The Python-facing `QrackCircuit` wraps it:

```cpp
// ── File: bindings/circuit.cpp ────────────────────────────────────────────────
#include "binding_core.h"

// Include Qrack's circuit header
// Confirm path against the installed package:
// find /usr/include/qrack -name "qcircuit.hpp"
#include "qcircuit.hpp"

struct QrackCircuit {
    std::shared_ptr<Qrack::QCircuit> circuit;
    bitLenInt numQubits;

    explicit QrackCircuit(bitLenInt n)
        : numQubits(n)
        , circuit(std::make_shared<Qrack::QCircuit>()) {}

    // Clone constructor — used by inverse()
    explicit QrackCircuit(std::shared_ptr<Qrack::QCircuit> c, bitLenInt n)
        : numQubits(n), circuit(std::move(c)) {}
};
```

---

## 2. Gate Type Enum

`QrackCircuit` needs a way for Python to specify which gate to append. A `GateType` enum maps Python-friendly names to the gate application logic:

```cpp
// ── File: bindings/circuit.cpp, above bind_circuit() ─────────────────────────

enum class GateType {
    H, X, Y, Z, S, T, IS, IT,
    SqrtX, ISqrtX,
    RX, RY, RZ, R1,
    CNOT, CY, CZ, CH,
    MCX, MCZ, MCY,
    SWAP, ISWAP,
    U,          // arbitrary single-qubit unitary (3 angles)
    Mtrx,       // arbitrary 2x2 matrix
    MCMtrx,     // multi-controlled arbitrary 2x2 matrix
};
```

Register as a nanobind enum in `bind_circuit()`:

```cpp
nb::enum_<GateType>(m, "GateType",
    "Gate type identifier for QrackCircuit.append_gate().\n\n"
    "Used to specify which gate to add to the circuit without\n"
    "immediately applying it to a simulator.")
    .value("H",      GateType::H,      "Hadamard gate")
    .value("X",      GateType::X,      "Pauli X")
    .value("Y",      GateType::Y,      "Pauli Y")
    .value("Z",      GateType::Z,      "Pauli Z")
    .value("S",      GateType::S,      "S gate (phase π/2)")
    .value("T",      GateType::T,      "T gate (phase π/4)")
    .value("IS",     GateType::IS,     "S† (inverse S)")
    .value("IT",     GateType::IT,     "T† (inverse T)")
    .value("SqrtX",  GateType::SqrtX,  "√X gate")
    .value("ISqrtX", GateType::ISqrtX, "√X† gate")
    .value("RX",     GateType::RX,     "X rotation (1 angle param)")
    .value("RY",     GateType::RY,     "Y rotation (1 angle param)")
    .value("RZ",     GateType::RZ,     "Z rotation (1 angle param)")
    .value("R1",     GateType::R1,     "Phase rotation (1 angle param)")
    .value("CNOT",   GateType::CNOT,   "Controlled NOT (2 qubits)")
    .value("CY",     GateType::CY,     "Controlled Y (2 qubits)")
    .value("CZ",     GateType::CZ,     "Controlled Z (2 qubits)")
    .value("CH",     GateType::CH,     "Controlled H (2 qubits)")
    .value("MCX",    GateType::MCX,    "Multi-controlled X")
    .value("MCY",    GateType::MCY,    "Multi-controlled Y")
    .value("MCZ",    GateType::MCZ,    "Multi-controlled Z")
    .value("SWAP",   GateType::SWAP,   "SWAP gate (2 qubits)")
    .value("ISWAP",  GateType::ISWAP,  "iSWAP gate (2 qubits)")
    .value("U",      GateType::U,      "Arbitrary unitary (θ, φ, λ params)")
    .value("Mtrx",   GateType::Mtrx,   "Arbitrary 2x2 unitary (4 complex params)")
    .value("MCMtrx", GateType::MCMtrx, "Multi-controlled arbitrary 2x2");
```

---

## 3. `append_gate` — Adding Gates to the Circuit

The `append_gate` method dispatches on `GateType` and calls the appropriate `QCircuit` method. The `qubits` list and optional `params` list carry all gate operands:

```cpp
// ── File: bindings/circuit.cpp, inside bind_circuit() ────────────────────────

.def("append_gate",
    [](QrackCircuit& c,
       GateType gate,
       std::vector<bitLenInt> qubits,
       std::vector<float> params)
    {
        // Validate qubit indices
        for (auto q : qubits) {
            if (q >= c.numQubits)
                throw QrackError(
                    "append_gate: qubit " + std::to_string(q) +
                    " out of range [0, " + std::to_string(c.numQubits - 1) + "]",
                    QrackErrorKind::QubitOutOfRange);
        }

        // Static gate matrices for 1-qubit Clifford gates
        using C = Qrack::complex;
        static const C H_MTRX[4] = {
            {(Qrack::real1)M_SQRT1_2, 0}, {(Qrack::real1)M_SQRT1_2, 0},
            {(Qrack::real1)M_SQRT1_2, 0}, {-(Qrack::real1)M_SQRT1_2, 0}};
        static const C X_MTRX[4] = {{0,0},{1,0},{1,0},{0,0}};
        static const C Y_MTRX[4] = {{0,0},{0,-1},{0,1},{0,0}};
        static const C Z_MTRX[4] = {{1,0},{0,0},{0,0},{-1,0}};

        const bitLenInt q0 = qubits.empty() ? 0 : qubits[0];
        const bitLenInt q1 = qubits.size() > 1 ? qubits[1] : 0;

        switch (gate) {
            // ── Clifford 1-qubit ───────────────────────────────────────────
            case GateType::H:
                c.circuit->AppendGate(
                    std::make_shared<Qrack::QCircuitGate>(q0, H_MTRX));
                break;
            case GateType::X:
                c.circuit->AppendGate(
                    std::make_shared<Qrack::QCircuitGate>(q0, X_MTRX));
                break;
            case GateType::Y:
                c.circuit->AppendGate(
                    std::make_shared<Qrack::QCircuitGate>(q0, Y_MTRX));
                break;
            case GateType::Z:
                c.circuit->AppendGate(
                    std::make_shared<Qrack::QCircuitGate>(q0, Z_MTRX));
                break;
            // ── Rotation ──────────────────────────────────────────────────
            case GateType::RZ: {
                if (params.empty())
                    throw QrackError("RZ requires 1 angle param",
                                     QrackErrorKind::InvalidArgument);
                const float half = params[0] / 2.0f;
                const C ph0 = std::exp(C(0, -half));
                const C ph1 = std::exp(C(0,  half));
                const C mtrx[4] = {ph0, {0,0}, {0,0}, ph1};
                c.circuit->AppendGate(
                    std::make_shared<Qrack::QCircuitGate>(q0, mtrx));
                break;
            }
            // ── 2-qubit gates ─────────────────────────────────────────────
            case GateType::CNOT: {
                if (qubits.size() < 2)
                    throw QrackError("CNOT requires 2 qubits",
                                     QrackErrorKind::InvalidArgument);
                const std::set<bitLenInt> controls{q0};
                c.circuit->AppendGate(
                    std::make_shared<Qrack::QCircuitGate>(q1, X_MTRX, controls,
                        Qrack::ONE_BCI));
                break;
            }
            // ── Arbitrary matrix ──────────────────────────────────────────
            case GateType::Mtrx: {
                if (params.size() < 8)
                    throw QrackError("Mtrx requires 8 floats (4 complex)",
                                     QrackErrorKind::InvalidArgument);
                const C mtrx[4] = {
                    C{params[0], params[1]}, C{params[2], params[3]},
                    C{params[4], params[5]}, C{params[6], params[7]}};
                c.circuit->AppendGate(
                    std::make_shared<Qrack::QCircuitGate>(q0, mtrx));
                break;
            }
            default:
                throw QrackError(
                    "append_gate: unsupported gate type for circuit recording",
                    QrackErrorKind::InvalidArgument);
        }
    },
    nb::arg("gate"), nb::arg("qubits"), nb::arg("params") = std::vector<float>{},
    nb::sig(
        "def append_gate(\n"
        "    self,\n"
        "    gate: GateType,\n"
        "    qubits: list[int],\n"
        "    params: list[float] = [],\n"
        ") -> None"
    ),
    "Append a gate to the circuit without executing it.\n\n"
    "Gates are accumulated and can be optimised before running.\n"
    "params carries angle values for rotation gates, or\n"
    "complex components (real, imag pairs) for matrix gates.")
```

---

## 4. `run` — Executing the Circuit on a Simulator

This is the key cross-class method. `nb::keep_alive<1, 2>()` prevents the simulator from being garbage-collected while `run()` is executing.

```cpp
.def("run",
    [](QrackCircuit& c, QrackSim& sim) {
        if (sim.numQubits < c.numQubits)
            throw QrackError(
                "run: circuit has " + std::to_string(c.numQubits) +
                " qubits but simulator has only " +
                std::to_string(sim.numQubits),
                QrackErrorKind::InvalidArgument);
        c.circuit->Run(sim.sim);
    },
    nb::arg("simulator"),
    nb::keep_alive<1, 2>(),   // keep sim alive while circuit.run() executes
    nb::sig(
        "def run(self, simulator: QrackSimulator) -> None"
    ),
    "Apply the circuit to the given simulator.\n\n"
    "The simulator's state is updated in place. The circuit itself\n"
    "is not consumed — it can be run on multiple simulators.\n"
    "The simulator must have at least as many qubits as the circuit.")
```

---

## 5. `inverse` — Circuit Adjoint

Returns a new `QrackCircuit` containing the adjoint (reverse order, conjugate-transposed matrices) of the original.

```cpp
.def("inverse",
    [](const QrackCircuit& c) -> QrackCircuit {
        return QrackCircuit(c.circuit->Inverse(), c.numQubits);
    },
    nb::sig("def inverse(self) -> QrackCircuit"),
    "Return a new circuit that is the adjoint (inverse) of this circuit.\n\n"
    "Applies all gates in reverse order with conjugate-transposed matrices.\n"
    "Useful for uncomputation and ansatz construction.\n\n"
    "Example:\n"
    "  circ = QrackCircuit(2)\n"
    "  circ.append_gate(GateType.H, [0])\n"
    "  circ_inv = circ.inverse()   # applies H† = H\n"
    "  circ.run(sim); circ_inv.run(sim)  # net effect: identity")
```

---

## 6. `optimize` — In-Place Gate Simplification

Calls Qrack's built-in circuit simplification, which fuses adjacent gates, removes identity sequences, and eliminates inverse pairs.

```cpp
.def("optimize",
    [](QrackCircuit& c) {
        c.circuit->Optimize();
    },
    "Simplify the circuit in place using Qrack's gate fusion.\n\n"
    "Removes identity sequences (X·X = I), fuses adjacent rotations\n"
    "(RZ(a)·RZ(b) = RZ(a+b)), and eliminates inverse pairs.\n"
    "Call before run() for large circuits to reduce gate count.")

.def_prop_ro("gate_count",
    [](const QrackCircuit& c) -> size_t {
        return c.circuit->GetGateCount();
    },
    "Number of gates currently in the circuit. "
    "Check before and after optimize() to measure reduction.")
```

---

## 7. `append` — Combining Circuits

```cpp
.def("append",
    [](QrackCircuit& c, const QrackCircuit& other) {
        if (other.numQubits > c.numQubits)
            throw QrackError(
                "append: circuit qubit counts are incompatible",
                QrackErrorKind::InvalidArgument);
        c.circuit->AppendCircuit(other.circuit);
    },
    nb::arg("other"),
    nb::sig("def append(self, other: QrackCircuit) -> None"),
    "Append all gates from another circuit to the end of this circuit.\n"
    "The other circuit's qubit count must be <= this circuit's qubit count.")
```

---

## 8. QASM Output

```cpp
.def("to_qasm",
    [](const QrackCircuit& c) -> std::string {
        std::ostringstream oss;
        c.circuit->ToQasm(oss);
        return oss.str();
    },
    nb::sig("def to_qasm(self) -> str"),
    "Serialise the circuit to OpenQASM 2.0 format.\n\n"
    "Returns a string containing the complete QASM program.\n"
    "Useful for interoperability with Qiskit, Cirq, and other frameworks.")

.def("__repr__",
    [](const QrackCircuit& c) {
        return "QrackCircuit(qubits=" + std::to_string(c.numQubits) +
               ", gates=" + std::to_string(c.circuit->GetGateCount()) + ")";
    })
```

---

## 9. Deferred Phase 4 Items — `simulator.cpp`

These go inside `bind_simulator()` alongside the existing Phase 4 Pauli methods.

### `exp_val_unitary` — Arbitrary 2×2 Observable

```cpp
// ── File: bindings/simulator.cpp, inside bind_simulator() ────────────────────

.def("exp_val_unitary",
    [](QrackSim& s,
       std::vector<bitLenInt> qubits,
       std::vector<std::complex<float>> basisOps,
       std::vector<float> eigenVals) -> float
    {
        if (basisOps.size() != qubits.size() * 4)
            throw QrackError(
                "exp_val_unitary: basisOps must have 4 * len(qubits) elements",
                QrackErrorKind::InvalidArgument);
        for (auto q : qubits) s.check_qubit(q, "exp_val_unitary");

        std::vector<std::shared_ptr<Qrack::complex>> ops;
        ops.reserve(qubits.size());
        for (size_t i = 0; i < qubits.size(); ++i) {
            auto m = std::make_shared<Qrack::complex>(
                reinterpret_cast<const Qrack::complex*>(&basisOps[i * 4])[0]);
            ops.push_back(m);
        }

        std::vector<Qrack::real1_f> ev(eigenVals.begin(), eigenVals.end());
        return static_cast<float>(
            s.sim->ExpectationUnitaryAll(qubits, ops, ev));
    },
    nb::arg("qubits"), nb::arg("basis_ops"), nb::arg("eigen_vals") = std::vector<float>{},
    nb::sig(
        "def exp_val_unitary(\n"
        "    self,\n"
        "    qubits: list[int],\n"
        "    basis_ops: list[complex],\n"
        "    eigen_vals: list[float] = [],\n"
        ") -> float"
    ),
    "Expectation value of a tensor product of arbitrary 2x2 unitary observables.\n"
    "basis_ops is a flat list of 4 * len(qubits) complex values — one 2x2 matrix\n"
    "per qubit, in row-major order.")

.def("variance_unitary",
    [](QrackSim& s,
       std::vector<bitLenInt> qubits,
       std::vector<std::complex<float>> basisOps,
       std::vector<float> eigenVals) -> float
    {
        if (basisOps.size() != qubits.size() * 4)
            throw QrackError(
                "variance_unitary: basisOps must have 4 * len(qubits) elements",
                QrackErrorKind::InvalidArgument);
        for (auto q : qubits) s.check_qubit(q, "variance_unitary");

        std::vector<std::shared_ptr<Qrack::complex>> ops;
        ops.reserve(qubits.size());
        for (size_t i = 0; i < qubits.size(); ++i) {
            auto m = std::make_shared<Qrack::complex>(
                reinterpret_cast<const Qrack::complex*>(&basisOps[i * 4])[0]);
            ops.push_back(m);
        }

        std::vector<Qrack::real1_f> ev(eigenVals.begin(), eigenVals.end());
        return static_cast<float>(
            s.sim->VarianceUnitaryAll(qubits, ops, ev));
    },
    nb::arg("qubits"), nb::arg("basis_ops"), nb::arg("eigen_vals") = std::vector<float>{},
    nb::sig(
        "def variance_unitary(\n"
        "    self,\n"
        "    qubits: list[int],\n"
        "    basis_ops: list[complex],\n"
        "    eigen_vals: list[float] = [],\n"
        ") -> float"
    ),
    "Variance of a tensor product of arbitrary 2x2 unitary observables.")
```

### `exp_val_bits_factorized`

```cpp
.def("exp_val_bits_factorized",
    [](QrackSim& s,
       std::vector<bitLenInt> qubits,
       std::vector<bitCapInt> perms) -> float
    {
        for (auto q : qubits) s.check_qubit(q, "exp_val_bits_factorized");
        return static_cast<float>(
            s.sim->ExpectationBitsFactorized(qubits, perms));
    },
    nb::arg("qubits"), nb::arg("perms"),
    nb::sig(
        "def exp_val_bits_factorized(\n"
        "    self, qubits: list[int], perms: list[int]\n"
        ") -> float"
    ),
    "Per-qubit weighted expectation value using bitCapInt permutation weights.\n"
    "Low-level API used by Shor's and arithmetic expectation paths.")
```

---

## 10. `module.cpp` — Updated Registration

```cpp
void bind_exceptions(nb::module_& m);
void bind_pauli(nb::module_& m);
void bind_simulator(nb::module_& m);
void bind_circuit(nb::module_& m);      // ← new

NB_MODULE(_qrackbind_core, m) {
    m.doc() = "qrackbind — nanobind bindings for the Qrack quantum simulator";
    m.attr("__version__") = "0.1.0";

    bind_exceptions(m);
    bind_pauli(m);
    bind_simulator(m);
    bind_circuit(m);    // ← after simulator so QrackSimulator is registered first
}
```

---

## 11. `__init__.py` — Exports

```python
from ._qrackbind_core import QrackSimulator, Pauli
from ._qrackbind_core import QrackException, QrackQubitError, QrackArgumentError
from ._qrackbind_core import QrackCircuit, GateType

__all__ = [
    "QrackSimulator", "Pauli",
    "QrackException", "QrackQubitError", "QrackArgumentError",
    "QrackCircuit", "GateType",
]
```

---

## 12. Test Suite

```python
# tests/test_phase6.py
import math
import pytest
from qrackbind import QrackSimulator, QrackCircuit, GateType, Pauli, QrackArgumentError


# ── Construction ───────────────────────────────────────────────────────────────

class TestConstruction:
    def test_basic_construction(self):
        circ = QrackCircuit(3)
        assert repr(circ) == "QrackCircuit(qubits=3, gates=0)"

    def test_empty_gate_count(self):
        circ = QrackCircuit(2)
        assert circ.gate_count == 0


# ── append_gate ────────────────────────────────────────────────────────────────

class TestAppendGate:
    def test_append_h(self):
        circ = QrackCircuit(2)
        circ.append_gate(GateType.H, [0])
        assert circ.gate_count == 1

    def test_append_multiple(self):
        circ = QrackCircuit(2)
        circ.append_gate(GateType.H, [0])
        circ.append_gate(GateType.CNOT, [0, 1])
        assert circ.gate_count == 2

    def test_append_rotation(self):
        circ = QrackCircuit(1)
        circ.append_gate(GateType.RZ, [0], [math.pi / 2])
        assert circ.gate_count == 1

    def test_qubit_out_of_range_raises(self):
        from qrackbind import QrackQubitError
        circ = QrackCircuit(2)
        with pytest.raises(QrackQubitError):
            circ.append_gate(GateType.H, [5])

    def test_missing_params_raises(self):
        circ = QrackCircuit(1)
        with pytest.raises(Exception):
            circ.append_gate(GateType.RZ, [0])   # no params


# ── run ────────────────────────────────────────────────────────────────────────

class TestRun:
    def test_h_circuit_creates_superposition(self):
        circ = QrackCircuit(1)
        circ.append_gate(GateType.H, [0])
        sim = QrackSimulator(qubitCount=1)
        circ.run(sim)
        assert sim.prob(0) == pytest.approx(0.5, abs=1e-4)

    def test_bell_state_circuit(self):
        circ = QrackCircuit(2)
        circ.append_gate(GateType.H, [0])
        circ.append_gate(GateType.CNOT, [0, 1])
        sim = QrackSimulator(qubitCount=2)
        circ.run(sim)
        # Bell state: prob(qubit 0) = prob(qubit 1) = 0.5
        assert sim.prob(0) == pytest.approx(0.5, abs=1e-4)
        assert sim.prob(1) == pytest.approx(0.5, abs=1e-4)

    def test_run_multiple_times(self):
        circ = QrackCircuit(1)
        circ.append_gate(GateType.X, [0])
        for _ in range(5):
            sim = QrackSimulator(qubitCount=1)
            circ.run(sim)
            assert sim.prob(0) == pytest.approx(1.0, abs=1e-5)

    def test_run_on_larger_simulator(self):
        circ = QrackCircuit(1)
        circ.append_gate(GateType.X, [0])
        sim = QrackSimulator(qubitCount=4)   # more qubits than circuit
        circ.run(sim)
        assert sim.prob(0) == pytest.approx(1.0, abs=1e-5)

    def test_run_on_too_small_simulator_raises(self):
        circ = QrackCircuit(4)
        sim = QrackSimulator(qubitCount=2)
        with pytest.raises(QrackArgumentError):
            circ.run(sim)


# ── inverse ────────────────────────────────────────────────────────────────────

class TestInverse:
    def test_h_inverse_is_h(self):
        # H is self-inverse: H·H = I
        circ = QrackCircuit(1)
        circ.append_gate(GateType.H, [0])
        circ_inv = circ.inverse()
        sim = QrackSimulator(qubitCount=1)
        circ.run(sim)
        circ_inv.run(sim)
        # Should be back to |0>
        assert sim.prob(0) == pytest.approx(0.0, abs=1e-4)

    def test_inverse_returns_new_circuit(self):
        circ = QrackCircuit(2)
        circ.append_gate(GateType.H, [0])
        circ_inv = circ.inverse()
        assert circ_inv is not circ

    def test_rz_inverse(self):
        circ = QrackCircuit(1)
        circ.append_gate(GateType.RZ, [0], [0.7])
        circ_inv = circ.inverse()
        sim = QrackSimulator(qubitCount=1)
        sim.h(0)   # put in superposition to make phase observable
        circ.run(sim)
        circ_inv.run(sim)
        sim.h(0)
        assert sim.prob(0) == pytest.approx(0.0, abs=1e-4)


# ── optimize ──────────────────────────────────────────────────────────────────

class TestOptimize:
    def test_double_x_optimizes_to_identity(self):
        circ = QrackCircuit(1)
        circ.append_gate(GateType.X, [0])
        circ.append_gate(GateType.X, [0])   # X·X = I
        gates_before = circ.gate_count
        circ.optimize()
        assert circ.gate_count < gates_before

    def test_optimize_preserves_semantics(self):
        circ = QrackCircuit(1)
        circ.append_gate(GateType.H, [0])
        circ.append_gate(GateType.X, [0])
        circ.append_gate(GateType.X, [0])
        circ.optimize()
        sim = QrackSimulator(qubitCount=1)
        circ.run(sim)
        assert sim.prob(0) == pytest.approx(0.5, abs=1e-4)


# ── append ────────────────────────────────────────────────────────────────────

class TestAppend:
    def test_append_combines_circuits(self):
        circ1 = QrackCircuit(1)
        circ1.append_gate(GateType.H, [0])
        circ2 = QrackCircuit(1)
        circ2.append_gate(GateType.X, [0])
        circ1.append(circ2)
        assert circ1.gate_count == 2

    def test_append_and_run(self):
        circ1 = QrackCircuit(1)
        circ1.append_gate(GateType.X, [0])
        circ2 = QrackCircuit(1)
        circ2.append_gate(GateType.X, [0])
        circ1.append(circ2)   # X·X = I
        sim = QrackSimulator(qubitCount=1)
        circ1.run(sim)
        assert sim.prob(0) == pytest.approx(0.0, abs=1e-4)


# ── QASM output ───────────────────────────────────────────────────────────────

class TestQasm:
    def test_to_qasm_returns_string(self):
        circ = QrackCircuit(2)
        circ.append_gate(GateType.H, [0])
        circ.append_gate(GateType.CNOT, [0, 1])
        qasm = circ.to_qasm()
        assert isinstance(qasm, str)
        assert len(qasm) > 0

    def test_to_qasm_contains_qreg(self):
        circ = QrackCircuit(3)
        qasm = circ.to_qasm()
        assert "qreg" in qasm or "OPENQASM" in qasm


# ── Deferred Phase 4: exp_val_unitary ────────────────────────────────────────

class TestExpValUnitary:
    def test_z_matrix_matches_pauli_z(self):
        import math
        # Z matrix: [[1,0],[0,-1]] → real/imag pairs: 1,0, 0,0, 0,0, -1,0
        z_matrix = [1+0j, 0+0j, 0+0j, -1+0j]
        sim = QrackSimulator(qubitCount=1)
        ev_unitary = sim.exp_val_unitary([0], z_matrix)
        ev_pauli = sim.exp_val(Pauli.PauliZ, 0)
        assert ev_unitary == pytest.approx(ev_pauli, abs=1e-4)

    def test_mismatched_ops_raises(self):
        sim = QrackSimulator(qubitCount=2)
        with pytest.raises(QrackArgumentError):
            sim.exp_val_unitary([0, 1], [1+0j]*4)  # 4 elements, need 8
```

---

## 13. Phase 6 Completion Checklist

```
□ QrackCircuit importable from qrackbind
□ GateType enum importable from qrackbind
□ QrackCircuit(n) constructs with gate_count == 0
□ append_gate(GateType.H, [0]) increases gate_count to 1
□ append_gate with out-of-range qubit raises QrackQubitError
□ Bell state circuit run on fresh sim gives prob(0)=prob(1)≈0.5
□ Circuit can be run on multiple simulators independently
□ run() on too-small simulator raises QrackArgumentError
□ inverse() returns a new circuit
□ H·H·(circuit + inverse) = identity on sim
□ optimize() reduces gate count for X·X sequence
□ optimize() preserves circuit semantics
□ append() combines two circuits correctly
□ to_qasm() returns a non-empty string
□ bind_circuit called after bind_simulator in module.cpp
□ exp_val_unitary Z matrix matches exp_val(PauliZ)
□ exp_val_unitary mismatched ops raises QrackArgumentError
□ uv run pytest tests/test_phase1.py … tests/test_phase6.py — all green
```

---

## 14. What Phase 6 Leaves Out (Deferred)

| Item | Reason deferred |
|---|---|
| `from_qasm(qasm_str)` — circuit construction from QASM | Requires QASM parser integration; deferred to Phase 7 |
| `from_qiskit(circuit)` — construction from Qiskit `QuantumCircuit` | Requires Qiskit as a dependency; deferred to integration phase |
| Noise model attachment | Requires `QInterfaceNoisy` binding; deferred |
| `Hamiltonian` struct binding and `time_evolve` | Hamiltonian is a complex nested type; Phase 8 |
| Multi-GPU circuit distribution | `QrackCircuit` on `QINTERFACE_QUNIT_MULTI`; Phase 8 |

---

## Related

- [[qrackbind Phase 5]]
- [[qrackbind Phase 4]]
- [[qrackbind Phase 3]]
- [[qrackbind Project Phase Breakdown]]
- [[Framework Plugin Architecture (PennyLane + Qiskit)]]
- [[qrackbind Compatibility Review — April 2026]]
- [[qrack project/Reference/qinterface.hpp.md]]
