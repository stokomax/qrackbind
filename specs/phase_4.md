---
tags:
  - qrack
  - nanobind
  - python
  - pauli
  - implementation
  - qrackbind
  - phase4
---
## qrackbind Phase 4 — Enums, Pauli Operators, and Type Safety

Builds directly on [[qrackbind Phase 3]]. The `Pauli` enum was scaffolded in `pauli.cpp` during the starter phase, but the methods that consume it — `measure_pauli`, `exp_val_pauli`, `variance_pauli` — were deferred. Phase 4 completes this surface, adds the variance methods, and ensures the `Pauli` type is wired correctly through all call sites so nanobind's type system enforces correct usage at the stub level.

This is a short phase (3–4 days) but a critical one: PennyLane's expectation value pipeline and Bloqade's Pauli-basis measurement both depend entirely on this API.

**Prerequisite:** All Phase 3 checklist items passing. `uv run pytest tests/test_phase3.py` green.

---

## Nanobind Learning Goals

| Topic | Where it appears |
|---|---|
| `nb::enum_<T>` and `nb::is_arithmetic()` | §1 — `Pauli` enum registration |
| Enum values in `.pyi` stubs — `IntEnum` vs `Enum` | §1 — stub output |
| `nb::arg().none()` — allowing `None` for optional enum args | §2 — `measure_pauli` default basis |
| Using registered enum types in method signatures | §3 — `exp_val_pauli` parameter type |
| `stl/vector.h` with enum element type | §3 — `list[Pauli]` → `std::vector<Qrack::Pauli>` |
| Type safety enforcement at the Python boundary | §4 — integer vs Pauli arg rejection |

---

## File Structure

| File | Changes |
|---|---|
| `bindings/pauli.cpp` | Complete the `bind_pauli()` function — already scaffolded, confirm member names, add docstrings |
| `bindings/simulator.cpp` | Add `measure_pauli`, `exp_val_pauli`, `variance_pauli`, `exp_val_all` inside `bind_simulator()` |
| `bindings/binding_core.h` | Add `#include <nanobind/stl/vector.h>` if not already present (needed for `list[Pauli]`) |
| `src/qrackbind/__init__.py` | Export `Pauli` and add Phase 4 to module docstring |

No new `.cpp` files are needed.

---

## 1. Completing `pauli.cpp`

The starter scaffolded `bind_pauli()` with placeholder member names. Verify against `pauli.hpp` before finalising:

```bash
grep -E "PauliI|PauliX|PauliY|PauliZ|enum" /usr/include/qrack/common/pauli.hpp
```

The confirmed member names from the installed header are `PauliI`, `PauliX`, `PauliY`, `PauliZ`.

### Complete binding

```cpp
// ── File: bindings/pauli.cpp ──────────────────────────────────────────────────
#include "binding_core.h"

void bind_pauli(nb::module_& m) {
    nb::enum_<Qrack::Pauli>(m, "Pauli",
        nb::is_arithmetic(),
        "Pauli operator basis for single-qubit observables.\n\n"
        "Used by measure_pauli(), exp_val_pauli(), and variance_pauli().\n"
        "is_arithmetic() makes this an IntEnum-compatible type — integer\n"
        "values (0, 1, 2, 3) are accepted wherever Pauli is expected,\n"
        "which preserves compatibility with frameworks that pass Pauli\n"
        "bases as raw integers (PennyLane, Bloqade QASM interpreter).")
        .value("PauliI", Qrack::Pauli::PauliI,
               "Identity operator — no rotation applied.")
        .value("PauliX", Qrack::Pauli::PauliX,
               "Pauli X basis — measures in the X (Hadamard) basis.")
        .value("PauliY", Qrack::Pauli::PauliY,
               "Pauli Y basis — measures in the Y basis (S†H rotation).")
        .value("PauliZ", Qrack::Pauli::PauliZ,
               "Pauli Z basis — computational basis, no rotation needed.");
}
```

### What `nb::is_arithmetic()` does

Without `nb::is_arithmetic()`, nanobind generates a standard Python `Enum` — integer values are **not** interchangeable with the enum members. With it, nanobind generates an `IntEnum`-compatible type, meaning:

```python
# Without nb::is_arithmetic() — raises TypeError
sim.exp_val_pauli([3], [0])   # 3 is not a Pauli

# With nb::is_arithmetic() — accepted
sim.exp_val_pauli([3], [0])   # 3 == Pauli.PauliZ, works
sim.exp_val_pauli([Pauli.PauliZ], [0])   # also works
```

This matters for PennyLane's `qml.PauliZ` and Bloqade's QASM2 interpreter, both of which pass integer Pauli codes in certain code paths.

### Stub output

With `nb::is_arithmetic()`, stubgen generates:

```python
class Pauli(enum.IntEnum):
    PauliI = 0
    PauliX = 1
    PauliY = 3
    PauliZ = 2
```

Note the non-sequential values — `PauliZ = 2`, `PauliY = 3`. This is Qrack's convention and must be preserved exactly. Do not assume they are 0, 1, 2, 3 in order.

---

## 2. `measure_pauli` — Measurement in a Pauli Basis

Measuring in a Pauli basis means rotating the qubit into the computational basis, measuring, then rotating back. `QInterface` has no single `MeasurePauli` method — the rotation is applied manually using the gate set.

The rotation scheme matches pyqrack exactly:

| Basis | Rotation before measure | Rotation after measure |
|---|---|---|
| `PauliI` | none | none |
| `PauliX` | `H` | `H` |
| `PauliY` | `IS`, `H` (= adjoint SH) | `H`, `S` |
| `PauliZ` | none | none |

```cpp
// ── File: bindings/simulator.cpp — add above bind_simulator() ─────────────────

// Rotate into Pauli basis before measurement, then rotate back.
// Returns true if result is +1 eigenvalue, false if -1 eigenvalue.
static bool measure_in_basis(QrackSim& s, Qrack::Pauli basis, bitLenInt q)
{
    switch (basis) {
        case Qrack::Pauli::PauliX:
            s.sim->H(q);
            break;
        case Qrack::Pauli::PauliY:
            s.sim->IS(q);
            s.sim->H(q);
            break;
        case Qrack::Pauli::PauliI:
        case Qrack::Pauli::PauliZ:
        default:
            break;
    }

    const bool result = s.sim->M(q);

    // Rotate back to original basis
    switch (basis) {
        case Qrack::Pauli::PauliX:
            s.sim->H(q);
            break;
        case Qrack::Pauli::PauliY:
            s.sim->H(q);
            s.sim->S(q);
            break;
        case Qrack::Pauli::PauliI:
        case Qrack::Pauli::PauliZ:
        default:
            break;
    }

    return result;
}
```

```cpp
// ── File: bindings/simulator.cpp, inside bind_simulator() ─────────────────────

.def("measure_pauli",
    [](QrackSim& s, Qrack::Pauli basis, bitLenInt q) -> bool {
        s.check_qubit(q, "measure_pauli");
        return measure_in_basis(s, basis, q);
    },
    nb::arg("basis"), nb::arg("qubit"),
    nb::sig(
        "def measure_pauli(self, basis: Pauli, qubit: int) -> bool"
    ),
    "Measure a qubit in the specified Pauli basis.\n\n"
    "Rotates the qubit into the computational basis, measures, and\n"
    "rotates back. Returns True for the +1 eigenvalue, False for -1.\n"
    "Collapses the state in the chosen basis.\n\n"
    "Example:\n"
    "  sim.h(0)\n"
    "  result = sim.measure_pauli(Pauli.PauliX, 0)  # always +1 for |+>")
```

---

## 3. `exp_val_pauli` — Pauli Tensor Product Expectation Value

Maps directly to `QInterface::ExpectationPauliAll(bits, paulis)`. This is the primary method PennyLane uses for observable evaluation.

```cpp
.def("exp_val_pauli",
    [](QrackSim& s,
       std::vector<Qrack::Pauli> paulis,
       std::vector<bitLenInt> qubits) -> float
    {
        if (paulis.size() != qubits.size())
            throw std::invalid_argument(
                "exp_val_pauli: paulis and qubits must have the same length");
        for (auto q : qubits)
            s.check_qubit(q, "exp_val_pauli");
        return static_cast<float>(
            s.sim->ExpectationPauliAll(qubits, paulis));
    },
    nb::arg("paulis"), nb::arg("qubits"),
    nb::sig(
        "def exp_val_pauli(\n"
        "    self,\n"
        "    paulis: list[Pauli],\n"
        "    qubits: list[int],\n"
        ") -> float"
    ),
    "Expectation value of a Pauli tensor product observable.\n\n"
    "Returns <ψ|P₀⊗P₁⊗…⊗Pₙ|ψ> where each Pᵢ is a Pauli operator\n"
    "acting on the corresponding qubit. Result is in [-1.0, +1.0].\n"
    "Does not collapse the state.\n\n"
    "paulis and qubits must have equal length.\n\n"
    "Example:\n"
    "  # Measure <ZZ> on a Bell state — should be +1\n"
    "  sim.h(0); sim.cnot(0, 1)\n"
    "  result = sim.exp_val_pauli([Pauli.PauliZ, Pauli.PauliZ], [0, 1])")

// Single-qubit convenience alias — matches pyqrack's exp_val() signature
.def("exp_val",
    [](QrackSim& s, Qrack::Pauli basis, bitLenInt q) -> float {
        s.check_qubit(q, "exp_val");
        return static_cast<float>(
            s.sim->ExpectationPauliAll({q}, {basis}));
    },
    nb::arg("basis"), nb::arg("qubit"),
    nb::sig(
        "def exp_val(self, basis: Pauli, qubit: int) -> float"
    ),
    "Single-qubit Pauli expectation value. Equivalent to\n"
    "exp_val_pauli([basis], [qubit]). Result is in [-1.0, +1.0].\n"
    "Does not collapse the state.\n\n"
    "Example:\n"
    "  sim.h(0)\n"
    "  print(sim.exp_val(Pauli.PauliX, 0))  # → 1.0")
```

---

## 4. `variance_pauli` — Pauli Observable Variance

Maps to `QInterface::VariancePauliAll`. Useful for computing quantum Fisher information and uncertainty relations.

```cpp
.def("variance_pauli",
    [](QrackSim& s,
       std::vector<Qrack::Pauli> paulis,
       std::vector<bitLenInt> qubits) -> float
    {
        if (paulis.size() != qubits.size())
            throw std::invalid_argument(
                "variance_pauli: paulis and qubits must have the same length");
        for (auto q : qubits)
            s.check_qubit(q, "variance_pauli");
        return static_cast<float>(
            s.sim->VariancePauliAll(qubits, paulis));
    },
    nb::arg("paulis"), nb::arg("qubits"),
    nb::sig(
        "def variance_pauli(\n"
        "    self,\n"
        "    paulis: list[Pauli],\n"
        "    qubits: list[int],\n"
        ") -> float"
    ),
    "Variance of a Pauli tensor product observable.\n\n"
    "Returns <ψ|P²|ψ> - <ψ|P|ψ>² = 1 - <ψ|P|ψ>² for a Pauli P\n"
    "(since P² = I for all Pauli operators). Result is in [0.0, 1.0].\n"
    "Does not collapse the state.\n\n"
    "For an eigenstate: variance = 0. For a maximally mixed state: variance = 1.")
```

---

## 5. `exp_val_all` — Expectation Value of All Qubits

Computes the tensor product expectation value across all qubits using the same Pauli basis repeated. Useful for parity calculations.

```cpp
.def("exp_val_all",
    [](QrackSim& s, Qrack::Pauli basis) -> float {
        std::vector<bitLenInt> qubits(s.numQubits);
        std::iota(qubits.begin(), qubits.end(), 0);
        std::vector<Qrack::Pauli> paulis(s.numQubits, basis);
        return static_cast<float>(
            s.sim->ExpectationPauliAll(qubits, paulis));
    },
    nb::arg("basis"),
    nb::sig("def exp_val_all(self, basis: Pauli) -> float"),
    "Expectation value of the same Pauli operator applied to every qubit.\n"
    "Equivalent to exp_val_pauli([basis]*num_qubits, list(range(num_qubits))).")
```

---

## 6. Expectation Value with Floating-Point Weights

Maps to `QInterface::ExpectationFloatsFactorized` — useful for weighted observables where each qubit carries a different classical weight. Required for PennyLane's Hamiltonian expectation values.

```cpp
.def("exp_val_floats",
    [](QrackSim& s,
       std::vector<bitLenInt> qubits,
       std::vector<float> weights) -> float
    {
        if (qubits.size() != weights.size())
            throw std::invalid_argument(
                "exp_val_floats: qubits and weights must have the same length");
        for (auto q : qubits)
            s.check_qubit(q, "exp_val_floats");
        std::vector<Qrack::real1_f> w(weights.begin(), weights.end());
        return static_cast<float>(
            s.sim->ExpectationFloatsFactorized(qubits, w));
    },
    nb::arg("qubits"), nb::arg("weights"),
    nb::sig(
        "def exp_val_floats(\n"
        "    self,\n"
        "    qubits: list[int],\n"
        "    weights: list[float],\n"
        ") -> float"
    ),
    "Expectation value of a weighted sum of single-qubit Z observables.\n"
    "Each qubit's contribution is weighted by the corresponding float.\n"
    "Used by PennyLane's Hamiltonian expectation value path.")
```

---

## 7. Updated `__init__.py`

```python
from ._qrackbind_core import QrackSimulator, Pauli

__all__ = ["QrackSimulator", "Pauli"]
```

Module docstring addition:

```python
"""
New in Phase 4:
  Pauli                              — enum: PauliI, PauliX, PauliY, PauliZ
  sim.measure_pauli(basis, qubit)    — measure in Pauli basis, collapses state
  sim.exp_val(basis, qubit)          — single-qubit Pauli expectation value
  sim.exp_val_pauli(paulis, qubits)  — tensor product Pauli expectation value
  sim.variance_pauli(paulis, qubits) — Pauli observable variance
  sim.exp_val_all(basis)             — all-qubit same-basis expectation value
  sim.exp_val_floats(qubits, weights)— weighted float expectation value
"""
```

---

## 8. Known Pauli Expectation Values for Tests

These are the analytical results used to verify the binding. All can be derived from the quantum state without a simulator:

| State | Observable | Expected value |
|---|---|---|
| `\|0⟩` | Z | +1.0 |
| `\|1⟩` | Z | −1.0 |
| `\|+⟩ = H\|0⟩` | X | +1.0 |
| `\|−⟩ = HX\|0⟩` | X | −1.0 |
| `\|+⟩` | Z | 0.0 |
| `\|0⟩` | X | 0.0 |
| Bell state `(|00⟩+|11⟩)/√2` | ZZ | +1.0 |
| Bell state `(|00⟩+|11⟩)/√2` | ZI | 0.0 |
| Any state | I | +1.0 |
| Eigenstate | any | variance = 0 |
| `\|+⟩` | Z | variance = 1.0 |

---

## 9. Test Suite

```python
# tests/test_phase4.py
import math
import pytest
from qrackbind import QrackSimulator, Pauli


# ── Enum properties ────────────────────────────────────────────────────────────

class TestPauliEnum:
    def test_members_exist(self):
        assert hasattr(Pauli, "PauliI")
        assert hasattr(Pauli, "PauliX")
        assert hasattr(Pauli, "PauliY")
        assert hasattr(Pauli, "PauliZ")

    def test_integer_values(self):
        # Qrack's convention — not sequential
        assert int(Pauli.PauliI) == 0
        assert int(Pauli.PauliX) == 1
        assert int(Pauli.PauliZ) == 2
        assert int(Pauli.PauliY) == 3

    def test_is_arithmetic_integer_accepted(self):
        # nb::is_arithmetic() means integer codes are accepted
        sim = QrackSimulator(qubitCount=1)
        result = sim.exp_val(Pauli.PauliZ, 0)
        result_int = sim.exp_val(2, 0)   # 2 == PauliZ
        assert result == pytest.approx(result_int, abs=1e-5)


# ── measure_pauli ──────────────────────────────────────────────────────────────

class TestMeasurePauli:
    def test_z_basis_ground_state_always_plus(self):
        # |0> is Z eigenstate with eigenvalue +1
        for _ in range(20):
            sim = QrackSimulator(qubitCount=1)
            assert sim.measure_pauli(Pauli.PauliZ, 0) == True

    def test_z_basis_excited_state_always_minus(self):
        # |1> is Z eigenstate with eigenvalue -1
        for _ in range(20):
            sim = QrackSimulator(qubitCount=1)
            sim.x(0)
            assert sim.measure_pauli(Pauli.PauliZ, 0) == False

    def test_x_basis_plus_state_always_plus(self):
        # |+> = H|0> is X eigenstate with eigenvalue +1
        for _ in range(20):
            sim = QrackSimulator(qubitCount=1)
            sim.h(0)
            assert sim.measure_pauli(Pauli.PauliX, 0) == True

    def test_x_basis_minus_state_always_minus(self):
        # |-> = HX|0> is X eigenstate with eigenvalue -1
        for _ in range(20):
            sim = QrackSimulator(qubitCount=1)
            sim.x(0)
            sim.h(0)
            assert sim.measure_pauli(Pauli.PauliX, 0) == False

    def test_pauli_i_does_not_collapse(self):
        # PauliI measurement should not change state probabilities
        sim = QrackSimulator(qubitCount=1)
        sim.h(0)
        prob_before = sim.prob(0)
        sim.measure_pauli(Pauli.PauliI, 0)
        prob_after = sim.prob(0)
        assert prob_before == pytest.approx(prob_after, abs=1e-5)

    def test_returns_bool(self):
        sim = QrackSimulator(qubitCount=1)
        result = sim.measure_pauli(Pauli.PauliZ, 0)
        assert isinstance(result, bool)


# ── exp_val ────────────────────────────────────────────────────────────────────

class TestExpVal:
    def test_z_ground_state(self):
        sim = QrackSimulator(qubitCount=1)
        assert sim.exp_val(Pauli.PauliZ, 0) == pytest.approx(1.0, abs=1e-5)

    def test_z_excited_state(self):
        sim = QrackSimulator(qubitCount=1)
        sim.x(0)
        assert sim.exp_val(Pauli.PauliZ, 0) == pytest.approx(-1.0, abs=1e-5)

    def test_x_plus_state(self):
        sim = QrackSimulator(qubitCount=1)
        sim.h(0)
        assert sim.exp_val(Pauli.PauliX, 0) == pytest.approx(1.0, abs=1e-5)

    def test_x_minus_state(self):
        sim = QrackSimulator(qubitCount=1)
        sim.x(0)
        sim.h(0)
        assert sim.exp_val(Pauli.PauliX, 0) == pytest.approx(-1.0, abs=1e-5)

    def test_z_superposition_is_zero(self):
        # <+|Z|+> = 0
        sim = QrackSimulator(qubitCount=1)
        sim.h(0)
        assert sim.exp_val(Pauli.PauliZ, 0) == pytest.approx(0.0, abs=1e-4)

    def test_x_ground_state_is_zero(self):
        # <0|X|0> = 0
        sim = QrackSimulator(qubitCount=1)
        assert sim.exp_val(Pauli.PauliX, 0) == pytest.approx(0.0, abs=1e-4)

    def test_identity_always_one(self):
        sim = QrackSimulator(qubitCount=1)
        sim.h(0)
        assert sim.exp_val(Pauli.PauliI, 0) == pytest.approx(1.0, abs=1e-5)

    def test_result_in_range(self):
        sim = QrackSimulator(qubitCount=1)
        sim.rx(0.7, 0)
        for basis in [Pauli.PauliX, Pauli.PauliY, Pauli.PauliZ]:
            val = sim.exp_val(basis, 0)
            assert -1.0 - 1e-4 <= val <= 1.0 + 1e-4

    def test_does_not_collapse_state(self):
        sim = QrackSimulator(qubitCount=1)
        sim.h(0)
        prob_before = sim.prob(0)
        sim.exp_val(Pauli.PauliZ, 0)
        prob_after = sim.prob(0)
        assert prob_before == pytest.approx(prob_after, abs=1e-5)


# ── exp_val_pauli ──────────────────────────────────────────────────────────────

class TestExpValPauli:
    def test_zz_bell_state(self):
        # <Bell|ZZ|Bell> = +1 for (|00>+|11>)/√2
        sim = QrackSimulator(qubitCount=2)
        sim.h(0)
        sim.cnot(0, 1)
        result = sim.exp_val_pauli([Pauli.PauliZ, Pauli.PauliZ], [0, 1])
        assert result == pytest.approx(1.0, abs=1e-4)

    def test_zi_bell_state(self):
        # <Bell|ZI|Bell> = 0 (individual qubit is maximally mixed)
        sim = QrackSimulator(qubitCount=2)
        sim.h(0)
        sim.cnot(0, 1)
        result = sim.exp_val_pauli([Pauli.PauliZ, Pauli.PauliI], [0, 1])
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_xx_bell_state(self):
        # <Bell|XX|Bell> = +1
        sim = QrackSimulator(qubitCount=2)
        sim.h(0)
        sim.cnot(0, 1)
        result = sim.exp_val_pauli([Pauli.PauliX, Pauli.PauliX], [0, 1])
        assert result == pytest.approx(1.0, abs=1e-4)

    def test_mismatched_lengths_raises(self):
        sim = QrackSimulator(qubitCount=2)
        with pytest.raises(Exception):
            sim.exp_val_pauli([Pauli.PauliZ], [0, 1])

    def test_single_qubit_matches_exp_val(self):
        sim = QrackSimulator(qubitCount=2)
        sim.rx(1.2, 0)
        direct = sim.exp_val(Pauli.PauliX, 0)
        via_list = sim.exp_val_pauli([Pauli.PauliX], [0])
        assert direct == pytest.approx(via_list, abs=1e-5)


# ── variance_pauli ─────────────────────────────────────────────────────────────

class TestVariancePauli:
    def test_eigenstate_variance_is_zero(self):
        # |0> is Z eigenstate — variance should be 0
        sim = QrackSimulator(qubitCount=1)
        v = sim.variance_pauli([Pauli.PauliZ], [0])
        assert v == pytest.approx(0.0, abs=1e-5)

    def test_superposition_variance_is_one(self):
        # <+|Z²|+> - <+|Z|+>² = 1 - 0 = 1
        sim = QrackSimulator(qubitCount=1)
        sim.h(0)
        v = sim.variance_pauli([Pauli.PauliZ], [0])
        assert v == pytest.approx(1.0, abs=1e-4)

    def test_variance_in_range(self):
        sim = QrackSimulator(qubitCount=1)
        sim.rx(0.5, 0)
        v = sim.variance_pauli([Pauli.PauliZ], [0])
        assert 0.0 - 1e-4 <= v <= 1.0 + 1e-4

    def test_variance_equals_one_minus_expval_squared(self):
        # For Pauli operators: Var(P) = 1 - <P>²
        sim = QrackSimulator(qubitCount=1)
        sim.rx(0.8, 0)
        sim.ry(0.3, 0)
        ev = sim.exp_val(Pauli.PauliZ, 0)
        var = sim.variance_pauli([Pauli.PauliZ], [0])
        assert var == pytest.approx(1.0 - ev**2, abs=1e-4)


# ── exp_val_floats ────────────────────────────────────────────────────────────

class TestExpValFloats:
    def test_ground_state_weight_one(self):
        # |0> with weight 1.0 → expectation = 1.0 * prob(|0>) = 1.0
        sim = QrackSimulator(qubitCount=1)
        result = sim.exp_val_floats([0], [1.0])
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_mismatched_lengths_raises(self):
        sim = QrackSimulator(qubitCount=2)
        with pytest.raises(Exception):
            sim.exp_val_floats([0], [1.0, 2.0])


# ── Integration: exp_val and state_vector consistency ─────────────────────────

class TestPhase4Integration:
    def test_exp_val_consistent_with_state_vector(self):
        import numpy as np
        sim = QrackSimulator(qubitCount=1)
        sim.ry(1.1, 0)
        # <ψ|Z|ψ> = |α₀|² - |α₁|²
        sv = sim.state_vector
        expected_z = abs(sv[0])**2 - abs(sv[1])**2
        measured_z = sim.exp_val(Pauli.PauliZ, 0)
        assert measured_z == pytest.approx(float(expected_z), abs=1e-4)

    def test_exp_val_does_not_disturb_state_vector(self):
        import numpy as np
        sim = QrackSimulator(qubitCount=2)
        sim.h(0); sim.h(1)
        sv_before = sim.state_vector.copy()
        sim.exp_val_pauli([Pauli.PauliZ, Pauli.PauliX], [0, 1])
        sv_after = sim.state_vector
        assert np.allclose(np.abs(sv_before), np.abs(sv_after), atol=1e-5)
```

---

## 10. Phase 4 Completion Checklist

```
□ Pauli.PauliI, PauliX, PauliY, PauliZ all accessible
□ int(Pauli.PauliZ) == 2, int(Pauli.PauliY) == 3 (Qrack convention)
□ Integer codes accepted where Pauli is expected (is_arithmetic)
□ .pyi stub shows class Pauli(enum.IntEnum)
□ sim.measure_pauli(Pauli.PauliZ, 0) == True for |0> (20 trials)
□ sim.measure_pauli(Pauli.PauliX, 0) == True for |+> (20 trials)
□ sim.exp_val(Pauli.PauliZ, 0) ≈ 1.0 for |0>
□ sim.exp_val(Pauli.PauliZ, 0) ≈ -1.0 for |1>
□ sim.exp_val(Pauli.PauliX, 0) ≈ 1.0 for |+>
□ sim.exp_val(Pauli.PauliZ, 0) ≈ 0.0 for |+>
□ sim.exp_val(Pauli.PauliI, 0) ≈ 1.0 for any state
□ exp_val does not collapse the state
□ exp_val_pauli([PauliZ, PauliZ], [0,1]) ≈ 1.0 for Bell state
□ exp_val_pauli([PauliZ, PauliI], [0,1]) ≈ 0.0 for Bell state
□ exp_val_pauli mismatched lengths raises exception
□ variance_pauli ≈ 0 for eigenstate
□ variance_pauli ≈ 1 for |+> in Z basis
□ variance = 1 - exp_val² holds within tolerance
□ exp_val_floats accepts qubits and weights of equal length
□ exp_val consistent with manual state_vector calculation
□ uv run pytest tests/test_phase1.py … tests/test_phase4.py — all green
```

---

## 11. What Phase 4 Leaves Out (Deferred)

| Item | Reason deferred |
|---|---|
| `exp_val_unitary` / `variance_unitary` | Takes complex 2x2 matrices — requires `stl/complex.h` vector caster for matrix inputs; deferred to Phase 6 |
| `ExpectationBitsFactorized` binding | Low-level bitCapInt permutation weighting; niche use, Phase 6 |
| Pauli string parsing (`"XYZ"` → `[Pauli.PauliX, ...]`) | Python-side utility; add to `_compat.py` in a later phase |
| Hamiltonian expectation via `TimeEvolve` | Requires `Hamiltonian` struct binding; Phase 6 |

---

## Related

- [[qrackbind Phase 3]]
- [[qrackbind Phase 2]]
- [[qrackbind Phase 1 Revised]]
- [[qrackbind Project Phase Breakdown]]
- [[qrackbind Compatibility Review — April 2026]]
- [[Framework Plugin Architecture (PennyLane + Qiskit)]]
- [[qrack project/Reference/qinterface.hpp.md]]
