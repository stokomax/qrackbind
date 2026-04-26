---
tags:
  - qrack
  - nanobind
  - python
  - numpy
  - implementation
  - qrackbind
  - phase3
---
## qrackbind Phase 3 — State Vector Access and NumPy Integration

Builds directly on [[qrackbind Phase 2]]. Phase 1 added basic `state_vector` and `probabilities` properties as a sketch; Phase 3 completes them properly — zero-copy via DLPack, correct ownership semantics, writable state injection, per-amplitude access, and the reduced density matrix. Together these unlock everything downstream that works with state vectors: Qiskit's statevector backend interface, PennyLane's expectation value calculation, and any workflow that inspects or post-processes quantum states as NumPy arrays.

**Prerequisite:** All Phase 2 checklist items passing. `uv run pytest tests/test_phase2.py` green.

---

## Nanobind Learning Goals

Phase 3 is the `nb::ndarray` phase. It covers the full depth of nanobind's array exchange mechanism, which is the most complex single topic in nanobind.

| Topic | Where it appears |
|---|---|
| `nb::ndarray<>` template parameters — shape, dtype, device | §1 — `state_vector` property |
| Zero-copy vs copy semantics and when each applies | §1 vs §3 |
| DLPack and ownership: `nb::rv_policy::reference` vs `nb::rv_policy::copy` | §1 — state vector return policy |
| `nb::ndarray` as an input — accepting NumPy arrays from Python | §3 — `set_state_vector` |
| Capsule-based lifetime management for zero-copy returns | §1 — keeping the sim alive while the array is live |
| `nb::ndarray` shape and dtype validation in the binding layer | §3 — input validation |
| `stl/complex.h` caster for scalar `complex` return | §4 — `get_amplitude` |

---

## File Structure

All bindings in this phase go into the existing `bindings/simulator.cpp` inside `bind_simulator()`. No new `.cpp` files are needed.

One header addition is required in `bindings/binding_core.h` if not already present:

```cpp
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>    // for get_amplitude → complex return
```

---

## 1. `state_vector` — Zero-Copy Property

### Background

`QInterface::GetQuantumState(complex* outputState)` writes the full 2^n complex amplitudes into a caller-supplied buffer. The buffer must be allocated before the call. Two strategies exist:

**Copy strategy** — allocate a new buffer, call `GetQuantumState`, wrap as ndarray with `nb::rv_policy::copy`. Simple but allocates 2^n complexes on every access.

**Zero-copy strategy** — allocate a persistent buffer inside `QrackSim`, call `GetQuantumState` to fill it, then return an ndarray that *references* that buffer. The buffer stays alive as long as the `QrackSim` is alive. Zero allocation per access, but the returned array reflects the state *at the moment of the call*, not live simulation state.

Phase 3 uses the **copy strategy** as the default because zero-copy of quantum state is semantically misleading — the state changes after every gate. The returned array is always a snapshot. Zero-copy is reserved for `probabilities` (§2), which is a derived real-valued quantity where snapshot semantics are unambiguous.

### C++ buffer type alias — add above `bind_simulator()`

```cpp
// ── File: bindings/simulator.cpp — add above bind_simulator() ─────────────────
using cf32 = std::complex<float>;

// Convenience: compute state vector size from qubit count
static size_t state_size(const QrackSim& s) {
    return size_t(1) << s.numQubits;
}
```

### Binding

```cpp
// ── File: bindings/simulator.cpp, inside bind_simulator() ─────────────────────

.def_prop_ro("state_vector",
    [](QrackSim& s) -> nb::ndarray<nb::numpy, cf32, nb::shape<nb::any>>
    {
        const size_t n = state_size(s);

        // Allocate a new buffer for this snapshot
        cf32* buf = new cf32[n];

        // GetQuantumState writes into the buffer
        // Qrack uses its own complex type — reinterpret as std::complex<float>
        s.sim->GetQuantumState(reinterpret_cast<Qrack::complex*>(buf));

        // Wrap in a capsule so Python's GC owns the buffer lifetime
        nb::capsule owner(buf, [](void* p) noexcept {
            delete[] reinterpret_cast<cf32*>(p);
        });

        // Return as a 1-D NumPy array of complex64
        // rv_policy::take_ownership: capsule owns buf, Python owns the ndarray
        return nb::ndarray<nb::numpy, cf32, nb::shape<nb::any>>(
            buf,
            {n},         // shape: [2^numQubits]
            owner        // lifetime tied to this capsule
        );
    },
    nb::sig(
        "def state_vector(self) -> numpy.ndarray[numpy.complex64]"
    ),
    "Full state vector snapshot as a 1-D complex64 NumPy array of length 2^n.\n"
    "Returns a copy — modifying the array does not affect the simulator.\n"
    "To inject a state, use set_state_vector().")
```

> **Why `nb::capsule`?** The capsule is a Python object that holds a destructor for the C++ buffer. nanobind attaches it to the ndarray as its `base` object. When the ndarray's reference count drops to zero, Python calls the capsule's destructor, which calls `delete[]`. Without the capsule, `buf` would leak because Python has no other way to know how to free it.

---

## 2. `probabilities` — Zero-Copy Real Array

Probabilities are real-valued (`float32`) and derived from the state vector. The buffer can safely use zero-copy reference semantics because the returned array carries a capsule that ties its lifetime to a freshly allocated buffer, same as `state_vector`. The distinction from §1 is that the element type is `float` not `complex`.

```cpp
.def_prop_ro("probabilities",
    [](QrackSim& s) -> nb::ndarray<nb::numpy, float, nb::shape<nb::any>>
    {
        const size_t n = state_size(s);
        float* buf = new float[n];

        // GetProbs writes real1 (float) probabilities into the buffer
        s.sim->GetProbs(buf);

        nb::capsule owner(buf, [](void* p) noexcept {
            delete[] reinterpret_cast<float*>(p);
        });

        return nb::ndarray<nb::numpy, float, nb::shape<nb::any>>(
            buf, {n}, owner
        );
    },
    nb::sig(
        "def probabilities(self) -> numpy.ndarray[numpy.float32]"
    ),
    "Probability of each basis state as a 1-D float32 NumPy array of length 2^n.\n"
    "Equivalent to abs(state_vector)**2. Does not collapse the state.\n"
    "Values sum to 1.0 within floating-point tolerance.")
```

---

## 3. `set_state_vector` — Injecting State from Python

Accepts a NumPy array from Python and writes it into the simulator via `QInterface::SetQuantumState`. This is the inverse of `state_vector`.

```cpp
.def("set_state_vector",
    [](QrackSim& s,
       nb::ndarray<nb::numpy, const cf32, nb::ndim<1>> arr)
    {
        const size_t expected = state_size(s);

        // Validate shape before touching the C++ layer
        if (arr.shape(0) != expected)
            throw std::invalid_argument(
                "set_state_vector: array length " +
                std::to_string(arr.shape(0)) +
                " does not match state space size " +
                std::to_string(expected) +
                " (2^" + std::to_string(s.numQubits) + ")");

        // SetQuantumState takes a raw pointer — the array owns its buffer
        s.sim->SetQuantumState(
            reinterpret_cast<const Qrack::complex*>(arr.data()));
    },
    nb::arg("state"),
    nb::sig(
        "def set_state_vector(self, state: numpy.ndarray[numpy.complex64]) -> None"
    ),
    "Set the simulator's quantum state from a 1-D complex64 NumPy array.\n"
    "Array must have length 2^num_qubits and should be normalised to unit norm.\n"
    "The array is copied into the simulator — subsequent mutations to the array\n"
    "have no effect on the simulation state.\n\n"
    "Raises ValueError if the array length does not match the state space size.")
```

### Why `const cf32` in the template

The `const` qualifier on the ndarray element type tells nanobind the binding will only read the array, not write it. nanobind uses this to allow passing both writable and read-only NumPy arrays (e.g. arrays created with `np.array(..., copy=False)` or from `memoryview`). Without `const`, nanobind rejects read-only arrays at runtime.

### Non-NumPy arrays

`nb::numpy` in the template restricts the binding to NumPy arrays. If you later want to accept PyTorch tensors or CuPy arrays via DLPack, replace `nb::numpy` with `nb::device::cpu` and add the appropriate device tag:

```cpp
// Future: accept any CPU array via DLPack
nb::ndarray<nb::device::cpu, const cf32, nb::ndim<1>> arr
```

---

## 4. `get_amplitude` and `set_amplitude` — Per-Basis-State Access

Direct access to individual amplitudes. Useful for state preparation and for testing specific entries without copying the full state vector.

```cpp
// ── File: bindings/simulator.cpp, inside bind_simulator() ─────────────────────

.def("get_amplitude",
    [](QrackSim& s, bitCapInt perm) -> cf32
    {
        // GetAmplitude returns Qrack::complex — reinterpret as cf32
        const Qrack::complex amp = s.sim->GetAmplitude(perm);
        return cf32(amp.real(), amp.imag());
    },
    nb::arg("index"),
    nb::sig(
        "def get_amplitude(self, index: int) -> complex"
    ),
    "Get the complex amplitude of a specific basis state by its integer index.\n"
    "index must be in [0, 2^num_qubits). Does not collapse the state.\n\n"
    "Example: amp = sim.get_amplitude(3)  # amplitude of |011> for 3 qubits")

.def("set_amplitude",
    [](QrackSim& s, bitCapInt perm, cf32 amp)
    {
        s.sim->SetAmplitude(perm, Qrack::complex(amp.real(), amp.imag()));
    },
    nb::arg("index"), nb::arg("amplitude"),
    nb::sig(
        "def set_amplitude(self, index: int, amplitude: complex) -> None"
    ),
    "Set the complex amplitude of a specific basis state.\n"
    "Warning: does not re-normalise the state. You are responsible for ensuring\n"
    "the state vector remains unit-norm after modification.\n\n"
    "Example: sim.set_amplitude(0, 1+0j)  # set |000> amplitude to 1")
```

> **Caution on `set_amplitude`:** Setting individual amplitudes bypasses Qrack's internal normalisation tracking. After calling `set_amplitude`, call `sim.update_running_norm()` if the simulator's norm tracking is enabled, or ensure the state is normalised before measurement.

---

## 5. `get_reduced_density_matrix` — Partial Trace

Returns the reduced density matrix for a subset of qubits, tracing out all others. Useful for computing subsystem entropy and for PennyLane's partial trace expectation values.

```cpp
.def("get_reduced_density_matrix",
    [](QrackSim& s, std::vector<bitLenInt> qubits)
        -> nb::ndarray<nb::numpy, cf32, nb::shape<nb::any, nb::any>>
    {
        // Validate qubits
        for (auto q : qubits)
            s.check_qubit(q, "get_reduced_density_matrix");

        const size_t dim = size_t(1) << qubits.size();  // 2^len(qubits)
        const size_t total = dim * dim;                  // dim x dim matrix

        cf32* buf = new cf32[total];

        s.sim->GetReducedDensityMatrix(
            qubits,
            reinterpret_cast<Qrack::complex*>(buf));

        nb::capsule owner(buf, [](void* p) noexcept {
            delete[] reinterpret_cast<cf32*>(p);
        });

        // Return as a 2-D NumPy array of shape (dim, dim)
        return nb::ndarray<nb::numpy, cf32, nb::shape<nb::any, nb::any>>(
            buf, {dim, dim}, owner
        );
    },
    nb::arg("qubits"),
    nb::sig(
        "def get_reduced_density_matrix("
        "    self, qubits: list[int]"
        ") -> numpy.ndarray[numpy.complex64]"
    ),
    "Return the reduced density matrix of the specified qubits as a 2-D\n"
    "complex64 NumPy array of shape (2^k, 2^k), where k = len(qubits).\n"
    "All other qubits are traced out. The result is a valid density matrix:\n"
    "Hermitian, positive semi-definite, and trace-1.")
```

---

## 6. `update_running_norm` — Post-Mutation Cleanup

Exposed to Python to allow users to re-normalise after direct state manipulation via `set_amplitude`.

```cpp
.def("update_running_norm",
    [](QrackSim& s) { s.sim->UpdateRunningNorm(); },
    "Recompute and apply the state vector normalisation factor.\n"
    "Call after set_amplitude() or set_state_vector() if the injected state\n"
    "may not be exactly unit-norm.")

.def("first_nonzero_phase",
    [](QrackSim& s) -> float {
        return static_cast<float>(s.sim->FirstNonzeroPhase());
    },
    "Return the phase angle of the lowest-index nonzero amplitude, in radians.\n"
    "Useful for global phase normalisation before state comparison.")
```

---

## 7. `prob_all` — Probability of a Specific Basis State

A convenience wrapper around `QInterface::ProbAll`. More efficient than fetching the full `probabilities` array when only one entry is needed.

```cpp
.def("prob_all",
    [](QrackSim& s, bitCapInt perm) -> float {
        return static_cast<float>(s.sim->ProbAll(perm));
    },
    nb::arg("index"),
    nb::sig("def prob_all(self, index: int) -> float"),
    "Probability of a specific basis state by integer index.\n"
    "More efficient than sim.probabilities[index] for sparse queries.\n"
    "Does not collapse the state.")

.def("prob_mask",
    [](QrackSim& s, bitCapInt mask, bitCapInt permutation) -> float {
        return static_cast<float>(s.sim->ProbMask(mask, permutation));
    },
    nb::arg("mask"), nb::arg("permutation"),
    nb::sig("def prob_mask(self, mask: int, permutation: int) -> float"),
    "Probability that the masked qubits match the given permutation.\n"
    "mask selects which qubits to check; permutation gives their expected values.\n"
    "Bits not in mask should be 0 in permutation.")
```

---

## 8. Updated `__init__.py` Exports

Add Phase 3 items to the module docstring:

```python
# src/qrackbind/__init__.py — module docstring addition
"""
New in Phase 3:
  sim.state_vector                   — full state vector as complex64 ndarray
  sim.probabilities                  — basis state probabilities as float32 ndarray
  sim.set_state_vector(arr)          — inject state from a NumPy array
  sim.get_amplitude(index)           — complex amplitude of a basis state
  sim.set_amplitude(index, amp)      — set amplitude of a basis state
  sim.get_reduced_density_matrix(qubits) — partial trace density matrix
  sim.prob_all(index)                — probability of a specific basis state
  sim.prob_mask(mask, permutation)   — masked permutation probability
  sim.update_running_norm()          — re-normalise after direct state mutation
  sim.first_nonzero_phase()          — global phase of lowest nonzero amplitude
"""
```

---

## 9. Numerical Considerations

### State vector dtype

Qrack compiles with either `float` (`real1 = float`) or `double` (`real1 = double`) precision depending on the build flags. The binding always exposes `complex64` (float-based) to Python regardless. If `real1 = double`, `GetQuantumState` fills a `Qrack::complex` buffer using `double` precision, and the binding truncates to `float` on copy. This is acceptable — Python NumPy's default is `complex128`, and users who need double precision should use `complex128` explicitly in their NumPy work, not rely on the simulator's internal precision.

### Normalisation after `set_state_vector`

`SetQuantumState` does not normalise the input. A common mistake is injecting an unnormalised state (e.g. from a statevector computed outside Qrack) and then measuring with incorrect probabilities. Best practice:

```python
import numpy as np
psi = np.array([1, 1, 0, 0], dtype=np.complex64)
psi /= np.linalg.norm(psi)   # normalise before injecting
sim.set_state_vector(psi)
sim.update_running_norm()
```

---

## 10. Test Suite

```python
# tests/test_phase3.py
import math
import numpy as np
import pytest
from qrackbind import QrackSimulator


# ── state_vector ───────────────────────────────────────────────────────────────

class TestStateVector:
    def test_returns_ndarray(self):
        sim = QrackSimulator(qubitCount=2)
        sv = sim.state_vector
        assert isinstance(sv, np.ndarray)
        assert sv.dtype == np.complex64

    def test_ground_state_shape(self):
        sim = QrackSimulator(qubitCount=3)
        sv = sim.state_vector
        assert sv.shape == (8,)   # 2^3

    def test_ground_state_first_amplitude_is_one(self):
        sim = QrackSimulator(qubitCount=2)
        sv = sim.state_vector
        assert abs(sv[0] - 1.0) < 1e-5
        assert np.allclose(sv[1:], 0.0, atol=1e-5)

    def test_normalised(self):
        sim = QrackSimulator(qubitCount=3)
        sim.h(0); sim.h(1); sim.h(2)
        sv = sim.state_vector
        norm = np.sum(np.abs(sv) ** 2)
        assert abs(norm - 1.0) < 1e-5

    def test_bell_state_amplitudes(self):
        sim = QrackSimulator(qubitCount=2)
        sim.h(0)
        sim.cnot(0, 1)
        sv = sim.state_vector
        # Bell state: (|00> + |11>) / sqrt(2)
        assert abs(sv[0] - 1 / math.sqrt(2)) < 1e-5
        assert abs(sv[1]) < 1e-5
        assert abs(sv[2]) < 1e-5
        assert abs(sv[3] - 1 / math.sqrt(2)) < 1e-5

    def test_snapshot_independence(self):
        sim = QrackSimulator(qubitCount=1)
        sv1 = sim.state_vector
        sim.x(0)
        sv2 = sim.state_vector
        # sv1 should still show |0>
        assert abs(sv1[0] - 1.0) < 1e-5
        # sv2 should show |1>
        assert abs(sv2[1] - 1.0) < 1e-5


# ── probabilities ──────────────────────────────────────────────────────────────

class TestProbabilities:
    def test_returns_float32_ndarray(self):
        sim = QrackSimulator(qubitCount=2)
        p = sim.probabilities
        assert isinstance(p, np.ndarray)
        assert p.dtype == np.float32

    def test_sums_to_one(self):
        sim = QrackSimulator(qubitCount=4)
        sim.h(0); sim.h(1)
        p = sim.probabilities
        assert abs(float(np.sum(p)) - 1.0) < 1e-5

    def test_ground_state(self):
        sim = QrackSimulator(qubitCount=2)
        p = sim.probabilities
        assert abs(p[0] - 1.0) < 1e-5
        assert np.allclose(p[1:], 0.0, atol=1e-5)

    def test_equal_superposition(self):
        sim = QrackSimulator(qubitCount=2)
        sim.h(0); sim.h(1)
        p = sim.probabilities
        assert np.allclose(p, 0.25, atol=1e-5)


# ── set_state_vector ──────────────────────────────────────────────────────────

class TestSetStateVector:
    def test_roundtrip(self):
        sim = QrackSimulator(qubitCount=2)
        # Prepare a Bell state via gates, capture, reset, reinject
        sim.h(0); sim.cnot(0, 1)
        sv_orig = sim.state_vector.copy()
        sim.reset_all()
        sim.set_state_vector(sv_orig)
        sv_after = sim.state_vector
        assert np.allclose(sv_orig, sv_after, atol=1e-5)

    def test_wrong_size_raises(self):
        sim = QrackSimulator(qubitCount=2)
        with pytest.raises(Exception):
            sim.set_state_vector(np.zeros(3, dtype=np.complex64))

    def test_x_state_injection(self):
        sim = QrackSimulator(qubitCount=1)
        psi = np.array([0, 1], dtype=np.complex64)   # |1>
        sim.set_state_vector(psi)
        assert sim.prob(0) == pytest.approx(1.0, abs=1e-5)


# ── get_amplitude / set_amplitude ─────────────────────────────────────────────

class TestAmplitude:
    def test_get_amplitude_ground_state(self):
        sim = QrackSimulator(qubitCount=2)
        amp = sim.get_amplitude(0)
        assert abs(amp - 1.0) < 1e-5

    def test_get_amplitude_after_x(self):
        sim = QrackSimulator(qubitCount=1)
        sim.x(0)
        assert abs(sim.get_amplitude(0)) < 1e-5
        assert abs(sim.get_amplitude(1) - 1.0) < 1e-5

    def test_set_amplitude_and_read_back(self):
        sim = QrackSimulator(qubitCount=2)
        # Manually create |10> = basis state 2
        sim.set_amplitude(0, 0+0j)
        sim.set_amplitude(2, 1+0j)
        sim.update_running_norm()
        assert abs(sim.get_amplitude(2) - 1.0) < 1e-4


# ── prob_all / prob_mask ───────────────────────────────────────────────────────

class TestProbAll:
    def test_prob_all_ground_state(self):
        sim = QrackSimulator(qubitCount=2)
        assert sim.prob_all(0) == pytest.approx(1.0, abs=1e-5)
        assert sim.prob_all(1) == pytest.approx(0.0, abs=1e-5)

    def test_prob_all_matches_probabilities(self):
        sim = QrackSimulator(qubitCount=3)
        sim.h(0); sim.h(2)
        p_arr = sim.probabilities
        for i in range(8):
            assert sim.prob_all(i) == pytest.approx(float(p_arr[i]), abs=1e-5)

    def test_prob_mask_selects_subset(self):
        sim = QrackSimulator(qubitCount=3)
        sim.x(1)   # state is |010>
        # mask = 0b010 = 2 (check qubit 1), permutation = 0b010 = 2 (qubit 1 is 1)
        assert sim.prob_mask(0b010, 0b010) == pytest.approx(1.0, abs=1e-5)
        assert sim.prob_mask(0b010, 0b000) == pytest.approx(0.0, abs=1e-5)


# ── get_reduced_density_matrix ────────────────────────────────────────────────

class TestReducedDensityMatrix:
    def test_shape(self):
        sim = QrackSimulator(qubitCount=4)
        rho = sim.get_reduced_density_matrix([0, 1])
        assert rho.shape == (4, 4)   # 2^2 x 2^2
        assert rho.dtype == np.complex64

    def test_trace_is_one(self):
        sim = QrackSimulator(qubitCount=3)
        sim.h(0); sim.cnot(0, 1)
        rho = sim.get_reduced_density_matrix([0, 1])
        trace = np.trace(rho)
        assert abs(trace - 1.0) < 1e-4

    def test_pure_state_rho_squared_equals_rho(self):
        # For a pure state, Tr(rho^2) = 1
        sim = QrackSimulator(qubitCount=2)
        sim.h(0)   # product state — each qubit is pure
        rho = sim.get_reduced_density_matrix([0])
        rho_sq = rho @ rho
        assert abs(np.trace(rho_sq) - 1.0) < 1e-4

    def test_maximally_mixed_qubit_from_bell_state(self):
        # Tracing out half of a Bell state gives a maximally mixed qubit
        sim = QrackSimulator(qubitCount=2)
        sim.h(0); sim.cnot(0, 1)
        rho = sim.get_reduced_density_matrix([0])
        # Maximally mixed: [[0.5, 0], [0, 0.5]]
        assert abs(rho[0, 0] - 0.5) < 1e-4
        assert abs(rho[1, 1] - 0.5) < 1e-4
        assert abs(rho[0, 1]) < 1e-4


# ── Integration — all Phase 3 outputs are consistent ─────────────────────────

class TestConsistency:
    def test_state_vector_and_probabilities_consistent(self):
        sim = QrackSimulator(qubitCount=3)
        sim.h(0); sim.h(1); sim.cnot(0, 2)
        sv = sim.state_vector
        p_from_sv = np.abs(sv) ** 2
        p_direct = sim.probabilities
        assert np.allclose(p_from_sv, p_direct, atol=1e-5)

    def test_prob_all_and_probabilities_consistent(self):
        sim = QrackSimulator(qubitCount=2)
        sim.h(0)
        p = sim.probabilities
        for i in range(4):
            assert sim.prob_all(i) == pytest.approx(float(p[i]), abs=1e-5)
```

---

## 11. Phase 3 Completion Checklist

```
□ sim.state_vector returns a 1-D complex64 NumPy array of shape (2^n,)
□ sim.state_vector snapshot is independent — subsequent gates don't mutate it
□ sim.state_vector norm is 1.0 within tolerance
□ sim.probabilities returns a 1-D float32 NumPy array of shape (2^n,)
□ sim.probabilities sums to 1.0 within tolerance
□ sim.set_state_vector(arr) roundtrips correctly through state_vector
□ sim.set_state_vector raises on wrong-size input
□ sim.set_state_vector accepts read-only NumPy arrays
□ sim.get_amplitude(i) returns correct complex for known states
□ sim.set_amplitude(i, amp) + update_running_norm() produces correct state
□ sim.get_reduced_density_matrix(qubits) shape is (2^k, 2^k)
□ Reduced density matrix trace is 1.0
□ Bell state trace-out gives maximally mixed qubit
□ sim.prob_all(i) matches sim.probabilities[i] for all i
□ sim.prob_mask(mask, perm) gives correct conditional probability
□ sim.update_running_norm() runs without error
□ All ndarray returns have correct dtype (complex64 or float32)
□ .pyi stubs have correct NumPy type annotations on all new methods
□ uv run pytest tests/test_phase1.py tests/test_phase2.py tests/test_phase3.py — all green
```

---

## 12. What Phase 3 Leaves Out (Deferred)

| Item | Reason deferred |
|---|---|
| DLPack device support (CuPy, PyTorch tensors) | Phase 3 targets NumPy/CPU only; GPU tensor exchange requires additional device-tag handling in `nb::ndarray` |
| `set_quantum_state` from a 2-D density matrix | Input density matrices require square-root decomposition before `SetQuantumState` can use them; non-trivial |
| `prob_bits_all(bits, probs_out)` | Low demand; deferred to a utility expansion phase |
| `lossy_save_state_vector` / `lossy_load_state_vector` | File I/O binding; deferred |
| Double precision (`complex128`) state vector | Would require a build-flag check and conditional ndarray type; low priority |

---

## Related

- [[qrackbind Phase 2]]
- [[qrackbind Phase 1 Revised]]
- [[qrackbind Project Phase Breakdown]]
- [[nanobind Type Casting and std-bad_cast Troubleshooting]]
- [[qrack project/Reference/qinterface.hpp.md]]
