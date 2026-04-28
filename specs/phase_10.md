---
tags:
  - qrack
  - nanobind
  - python
  - stabilizer
  - implementation
  - qrackbind
  - phase10
---
## qrackbind Phase 10 — QrackStabilizer and QrackStabilizerHybrid Standalone Classes

Builds on [[qrackbind Phase 6]] and [[qrackbind Phase 7]]. Phases 1–6 deliver `QrackSimulator` and `QrackCircuit` with the full gate surface, and the `isStabilizerHybrid=True` constructor flag already routes the standard simulator over Qrack's Clifford-hybrid engine. Phase 10 adds two **standalone** classes — `QrackStabilizer` and `QrackStabilizerHybrid` — that expose the underlying engines directly, without the `QUnit` / tensor-network wrapping that `QrackSimulator` always applies on top.

The result is a typed, restricted API for Clifford-only workloads, a direct framework-plugin target, and a clean benchmarking surface for measuring stabilizer-engine performance without the cost of upper layers.

**Prerequisite:** All Phase 6 checklist items passing; Phase 7 stub pipeline configured. Phase 10 ships in the same Phase 9 wheel cycle (it slots in parallel to Phase 8).

---

## Why a Separate Class?

**Which class should you reach for?** `QrackStabilizerHybrid` is the default for almost everyone — it accepts the full gate set, falls back gracefully to dense simulation when non-Clifford gates appear, and supports the near-Clifford T-injection path for Clifford+RZ circuits. `QrackStabilizer` is a specialty / strict-mode tool: pick it when you want a typed Clifford-only contract that fails at the IDE level rather than at runtime, when you need the absolute lowest-overhead engine for benchmarking, or when you're proving to yourself a circuit is Clifford-pure.

`QrackSimulator(isStabilizerHybrid=True)` already routes through `QINTERFACE_STABILIZER_HYBRID` — but always *underneath* `QINTERFACE_TENSOR_NETWORK` and `QINTERFACE_QUNIT`. There are three reasons to expose the engines as their own classes in addition to the flag:

| Reason | What it enables |
|---|---|
| Type-stub-documented contract | `QrackStabilizer.h(0)` shows up in `.pyi`; `QrackStabilizer.rx(...)` does not exist — Pyright / IDE autocomplete enforces the Clifford restriction at edit time |
| Direct framework targets | PennyLane device names `qrackbind.stabilizer` and `qrackbind.stabilizer_hybrid` map onto these classes without flag plumbing through `QrackSimulator` kwargs |
| Stack-overhead control | Skips `QINTERFACE_TENSOR_NETWORK` and `QINTERFACE_QUNIT` layers — useful for benchmarking the bare engine and for circuits where the upper layers don't help |

The `isStabilizerHybrid` flag stays on `QrackSimulator` for backward compatibility with pyqrack and Bloqade users.

---

## Nanobind Learning Goals

| Topic | Where it appears |
|---|---|
| Multiple class bindings sharing C++ helpers via templates | §3 — `add_clifford_gates<T>()`, `add_measurement<T>()`, etc. |
| Restricting the API surface of an engine to a subset of its capabilities | §4 — `QrackStabilizer` exposes only Clifford gates |
| Engine-stack-aware factory helpers | §2 — `make_stabilizer()` vs `make_stabilizer_hybrid()` |
| Run-time engine introspection from Python | §5 — `is_clifford` property on the hybrid class |

---

## File Structure

| File | Changes |
|---|---|
| `bindings/stabilizer.cpp` | **New file** — `bind_stabilizer()` defines both classes |
| `bindings/gate_helpers.h` | **New file** — templated `add_clifford_gates<T>`, `add_rotation_gates<T>`, `add_measurement<T>`, `add_pauli_methods<T>`, `add_state_access<T>` |
| `bindings/simulator.cpp` | Refactor to use the templated helpers (no behaviour change) |
| `bindings/module.cpp` | Add `void bind_stabilizer(nb::module_& m)` and call it after `bind_simulator()` and `bind_circuit()` |
| `CMakeLists.txt` | Add `bindings/stabilizer.cpp` to sources |
| `src/qrackbind/__init__.py` | Export `QrackStabilizer`, `QrackStabilizerHybrid` |
| `src/qrackbind/_core.pyi` | Regenerate via `just stubs` (Phase 7 pipeline) |

---

## 1. Engine Stack Mapping

The two classes select different `QInterfaceEngine` stacks for `CreateQuantumInterface`:

```
QrackStabilizer         →  [QINTERFACE_STABILIZER]
QrackStabilizerHybrid   →  [QINTERFACE_STABILIZER_HYBRID, QINTERFACE_HYBRID]   (or _OPENCL / _CPU per flags)
```

`QINTERFACE_STABILIZER_HYBRID` requires a dense engine *underneath* it for the fallback path when non-Clifford gates appear — that is the second element. `QINTERFACE_STABILIZER` is standalone and rejects non-Clifford gates outright.

For comparison, `QrackSimulator(isStabilizerHybrid=True)` produces:

```
[QINTERFACE_TENSOR_NETWORK, QINTERFACE_QUNIT, QINTERFACE_STABILIZER_HYBRID, QINTERFACE_QPAGER, QINTERFACE_HYBRID]
```

Phase 10 strips this down to the two-layer (or one-layer) form.

---

## 2. Factory Helpers

```cpp
// ── File: bindings/stabilizer.cpp ────────────────────────────────────────────
#include "binding_core.h"
#include "gate_helpers.h"
#include "qfactory.hpp"

namespace {

QInterfacePtr make_stabilizer(bitLenInt n) {
    return CreateQuantumInterface(
        std::vector<QInterfaceEngine>{QINTERFACE_STABILIZER},
        n, /*initState=*/0, /*rgp=*/nullptr,
        CMPLX_DEFAULT_ARG, /*doNorm=*/false, /*randomGP=*/true,
        /*useHostMem=*/false, /*deviceId=*/-1, /*useHWRNG=*/true,
        /*isSparse=*/false);
}

struct StabHybridConfig {
    bool isCpuGpuHybrid = true;
    bool isOpenCL       = true;
    bool isHostPointer  = false;
    bool isSparse       = false;
};

QInterfacePtr make_stabilizer_hybrid(bitLenInt n, const StabHybridConfig& c) {
    std::vector<QInterfaceEngine> stack{QINTERFACE_STABILIZER_HYBRID};
    if (c.isCpuGpuHybrid && c.isOpenCL)
        stack.push_back(QINTERFACE_HYBRID);
    else if (c.isOpenCL)
        stack.push_back(QINTERFACE_OPENCL);
    else
        stack.push_back(QINTERFACE_CPU);

    return CreateQuantumInterface(
        stack, n, 0, nullptr, CMPLX_DEFAULT_ARG,
        false, true, c.isHostPointer, -1, true, c.isSparse);
}

} // namespace
```

---

## 3. Templated Gate Helpers — Shared with `QrackSimulator`

The Clifford gate `.def()` calls from [[qrackbind Phase 1 Revised]] are factored into templated functions so `QrackSim`, `QrackStab`, and `QrackStabHybrid` reuse the same dispatch logic without copy-paste. Each helper composes a different subset of the gate surface.

```cpp
// ── File: bindings/gate_helpers.h ────────────────────────────────────────────
#pragma once
#include "binding_core.h"

// Strict Clifford 1-qubit gates: H, X, Y, Z, S, S†, √X, √X†.
// Safe to apply to QrackStabilizer.
template <typename WrapperT>
void add_clifford_gates(nb::class_<WrapperT>& cls) {
    #define GATE1(pyname, cppfn, doc) \
        .def(pyname, [](WrapperT& w, bitLenInt q) { \
            w.check_qubit(q, pyname); w.sim->cppfn(q); }, \
            nb::arg("qubit"), doc)

    cls
        GATE1("h",     H,      "Hadamard gate.")
        GATE1("x",     X,      "Pauli X.")
        GATE1("y",     Y,      "Pauli Y.")
        GATE1("z",     Z,      "Pauli Z.")
        GATE1("s",     S,      "S gate (phase π/2).")
        GATE1("sdg",   IS,     "S† (inverse S).")
        GATE1("sx",    SqrtX,  "√X gate.")
        GATE1("sxdg",  ISqrtX, "√X†.")
    ;
    #undef GATE1
}

// Two-qubit Clifford: CNOT, CY, CZ, SWAP, iSWAP, plus multi-control variants.
template <typename WrapperT>
void add_clifford_two_qubit(nb::class_<WrapperT>& cls) { /* ...CNOT, CY, CZ, SWAP, iSWAP, MCX, MCY, MCZ, MACX, MACY, MACZ... */ }

// T, T† — non-Clifford phase gates. Only included on simulator and stabilizer-hybrid.
template <typename WrapperT>
void add_t_gates(nb::class_<WrapperT>& cls) { /* ...t, tdg... */ }

// Rotations RX, RY, RZ, R1 — non-Clifford.
template <typename WrapperT>
void add_rotation_gates(nb::class_<WrapperT>& cls) { /* ... */ }

// U, U2 — non-Clifford arbitrary single-qubit unitaries.
template <typename WrapperT>
void add_u_gates(nb::class_<WrapperT>& cls) { /* ... */ }

// Mtrx, MCMtrx, MACMtrx, multiplex1_mtrx — arbitrary 2×2 matrices.
template <typename WrapperT>
void add_matrix_gates(nb::class_<WrapperT>& cls) { /* ... */ }

// measure, measure_all, force_measure, prob, prob_all.
template <typename WrapperT>
void add_measurement(nb::class_<WrapperT>& cls) { /* ... */ }

// measure_pauli, exp_val, exp_val_pauli, variance_pauli.
template <typename WrapperT>
void add_pauli_methods(nb::class_<WrapperT>& cls) { /* ... */ }

// state_vector, probabilities, get_amplitude.
// Cost-bearing for stabilizer engines — only included on QrackSimulator and QrackStabilizerHybrid.
template <typename WrapperT>
void add_state_access(nb::class_<WrapperT>& cls) { /* ... */ }
```

Phase 1's `bind_simulator()` is updated to call these helpers in sequence instead of inlining the `.def()` calls. The behaviour is unchanged; tests from Phase 1 still pass.

> **Refactor scope**: this is a mechanical extraction. Only the gate `.def()` calls move; the `QrackSim` struct and the constructor binding stay where they are.

> **GIL handling**: the templated helpers in this phase do not yet attach `nb::call_guard<nb::gil_scoped_release>()` to their `.def()` calls. Phase 10 stays single-threaded for simplicity; [[qrackbind Phase 11]] retroactively adds the call guard to all long-running operations across these helpers (gate methods, measurement, state access). Phase 10's tests are GIL-transparent and continue to pass after the Phase 11 retrofit.

---

## 4. `QrackStabilizer` — Pure Clifford Engine

```cpp
struct QrackStab {
    QInterfacePtr sim;
    bitLenInt     numQubits;

    explicit QrackStab(bitLenInt n) : numQubits(n), sim(make_stabilizer(n)) {
        if (!sim) throw QrackError("QrackStabilizer: factory returned null",
                                    QrackErrorKind::InvalidArgument);
    }

    void check_qubit(bitLenInt q, const char* method) const {
        if (q >= numQubits)
            throw QrackError(std::string(method) + ": qubit out of range",
                             QrackErrorKind::QubitOutOfRange);
    }

    std::string repr() const {
        return "QrackStabilizer(qubits=" + std::to_string(numQubits) + ")";
    }
};

void bind_stabilizer_class(nb::module_& m) {
    auto cls = nb::class_<QrackStab>(m, "QrackStabilizer",
        "Pure Clifford-only quantum simulator. Polynomial memory in qubit count.\n\n"
        "Supports H, X, Y, Z, S, S†, √X, √X† single-qubit gates; CNOT, CY, CZ,\n"
        "SWAP, iSWAP two-qubit gates; and their multiply-controlled forms.\n"
        "Non-Clifford gates (RX, RY, RZ, U, T, T†, arbitrary matrices) are NOT\n"
        "exposed — use QrackStabilizerHybrid or QrackSimulator for those.")
        .def(nb::init<bitLenInt>(), nb::arg("qubitCount") = 0,
             "Create a stabilizer simulator on n qubits, initialised to |0...0>.")
        .def("__repr__", &QrackStab::repr);

    add_clifford_gates(cls);
    add_clifford_two_qubit(cls);
    add_measurement(cls);
    add_pauli_methods(cls);

    cls
        .def_prop_ro("num_qubits", [](const QrackStab& s) { return s.numQubits; })
        .def("reset_all", [](QrackStab& s) { s.sim->SetPermutation(0); })
        .def("set_permutation",
             [](QrackStab& s, bitCapInt p) { s.sim->SetPermutation(p); },
             nb::arg("permutation"))
        .def("__enter__", [](QrackStab& s) -> QrackStab& { return s; })
        .def("__exit__",
             [](QrackStab& s, nb::object, nb::object, nb::object) { s.sim.reset(); });
}
```

**Deliberately NOT exposed:**

| Method | Reason |
|---|---|
| `rx`, `ry`, `rz`, `r1`, `u`, `u2` | Non-Clifford rotations — use `QrackStabilizerHybrid` |
| `t`, `tdg` | Non-Clifford phase gates |
| `mtrx`, `mcmtrx`, `macmtrx`, `multiplex1_mtrx` | Arbitrary matrices not generally Clifford |
| `state_vector`, `probabilities`, `get_amplitude` | Stabilizer engines do not store amplitudes; would force expensive dense materialisation |
| `set_state_vector` | Same reason |
| `qft`, `iqft`, arithmetic gates | Decompose to non-Clifford operations |

The user gets a static error from Pyright if they attempt `stab.rx(...)`, instead of a runtime exception buried in C++.

---

## 5. `QrackStabilizerHybrid` — Clifford with Automatic Fallback

```cpp
struct QrackStabHybrid {
    QInterfacePtr     sim;
    bitLenInt         numQubits;
    StabHybridConfig  config;

    QrackStabHybrid(bitLenInt n, const StabHybridConfig& cfg)
        : numQubits(n), config(cfg), sim(make_stabilizer_hybrid(n, cfg)) {
        if (!sim) throw QrackError("QrackStabilizerHybrid: factory returned null",
                                    QrackErrorKind::InvalidArgument);
    }

    void check_qubit(bitLenInt q, const char* method) const {
        if (q >= numQubits)
            throw QrackError(std::string(method) + ": qubit out of range",
                             QrackErrorKind::QubitOutOfRange);
    }

    std::string repr() const {
        return "QrackStabilizerHybrid(qubits=" + std::to_string(numQubits) +
               ", clifford=" + (sim->isClifford() ? "true" : "false") + ")";
    }
};

void bind_stabilizer_hybrid_class(nb::module_& m) {
    auto cls = nb::class_<QrackStabHybrid>(m, "QrackStabilizerHybrid",
        "Stabilizer simulator with automatic fallback to dense simulation\n"
        "when non-Clifford gates are applied. Stays in polynomial-memory mode\n"
        "for as long as the circuit is Clifford; switches to QHybrid (CPU+GPU)\n"
        "on the first non-Clifford gate.")
        .def("__init__",
            [](QrackStabHybrid* self, bitLenInt qubitCount,
               bool isCpuGpuHybrid, bool isOpenCL,
               bool isHostPointer, bool isSparse) {
                StabHybridConfig cfg{isCpuGpuHybrid, isOpenCL,
                                     isHostPointer, isSparse};
                new (self) QrackStabHybrid(qubitCount, cfg);
            },
            nb::arg("qubitCount")     = 0,
            nb::arg("isCpuGpuHybrid") = true,
            nb::arg("isOpenCL")       = true,
            nb::arg("isHostPointer")  = false,
            nb::arg("isSparse")       = false,
            "Create a stabilizer-hybrid simulator. Flags select the dense\n"
            "fallback engine — same semantics as on QrackSimulator.")
        .def("__repr__", &QrackStabHybrid::repr);

    add_clifford_gates(cls);
    add_clifford_two_qubit(cls);
    add_t_gates(cls);            // T, T† — trigger fallback
    add_rotation_gates(cls);     // RX, RY, RZ, R1 — trigger fallback
    add_u_gates(cls);            // U, U2 — trigger fallback
    add_matrix_gates(cls);       // Mtrx, MCMtrx, MACMtrx — may trigger fallback
    add_measurement(cls);
    add_pauli_methods(cls);
    add_state_access(cls);       // state_vector, probabilities, get_amplitude

    cls
        .def_prop_ro("num_qubits",
            [](const QrackStabHybrid& s) { return s.numQubits; })
        .def_prop_ro("is_clifford",
            [](const QrackStabHybrid& s) { return s.sim->isClifford(); },
            "True if the engine is currently in stabilizer (Clifford) mode.\n"
            "Becomes False after the first non-Clifford gate forces fallback.")
        .def("set_t_injection",
            [](QrackStabHybrid& s, bool useGadget) {
                s.sim->SetTInjection(useGadget);
            },
            nb::arg("use_gadget"),
            "Enable T-injection gadget for near-Clifford circuits.\n"
            "Trades simulation time for reduced memory growth.")
        .def("set_use_exact_near_clifford",
            [](QrackStabHybrid& s, bool exact) {
                s.sim->SetUseExactNearClifford(exact);
            },
            nb::arg("exact"))
        .def("reset_all", [](QrackStabHybrid& s) { s.sim->SetPermutation(0); })
        .def("set_permutation",
            [](QrackStabHybrid& s, bitCapInt p) { s.sim->SetPermutation(p); },
            nb::arg("permutation"))
        .def("__enter__", [](QrackStabHybrid& s) -> QrackStabHybrid& { return s; })
        .def("__exit__",
            [](QrackStabHybrid& s, nb::object, nb::object, nb::object) { s.sim.reset(); });
}
```

**Notes:**
- `state_vector` is exposed and works as on `QrackSimulator` — Qrack materialises amplitudes on demand if currently in stabilizer mode.
- `is_clifford` lets users check whether the cheap stabilizer representation is still in use, useful for VQE warm-ups, circuit budgeting, and integration tests.

---

## 6. `module.cpp` — Updated Registration

```cpp
void bind_exceptions(nb::module_& m);
void bind_pauli(nb::module_& m);
void bind_simulator(nb::module_& m);
void bind_circuit(nb::module_& m);
void bind_stabilizer(nb::module_& m);  // ← new

NB_MODULE(_qrackbind_core, m) {
    m.doc() = "qrackbind — nanobind bindings for the Qrack quantum simulator";

    bind_exceptions(m);
    bind_pauli(m);
    bind_simulator(m);
    bind_circuit(m);
    bind_stabilizer(m);  // after simulator and circuit
}

// In bindings/stabilizer.cpp:
void bind_stabilizer(nb::module_& m) {
    bind_stabilizer_class(m);
    bind_stabilizer_hybrid_class(m);
}
```

---

## 7. `__init__.py` — Exports

```python
from ._qrackbind_core import (
    QrackSimulator as _QrackSimulator,
    QrackCircuit, GateType, Pauli,
    QrackException, QrackQubitError, QrackArgumentError,
    QrackStabilizer, QrackStabilizerHybrid,  # ← new
)
from ._compat import _PyqrackAliasMixin

class QrackSimulator(_PyqrackAliasMixin, _QrackSimulator):
    __slots__ = ()

__all__ = [
    "QrackSimulator", "QrackCircuit", "GateType", "Pauli",
    "QrackException", "QrackQubitError", "QrackArgumentError",
    "QrackStabilizer", "QrackStabilizerHybrid",
]
```

---

## 8. Test Suite

```python
# tests/test_phase10.py
import math
import pytest
from qrackbind import (
    QrackStabilizer, QrackStabilizerHybrid, QrackSimulator, Pauli,
)


# ── QrackStabilizer ────────────────────────────────────────────────────────────

class TestStabilizerCore:
    def test_construction(self):
        s = QrackStabilizer(qubitCount=4)
        assert s.num_qubits == 4
        assert "4" in repr(s)

    def test_bell_state(self):
        s = QrackStabilizer(qubitCount=2)
        s.h(0); s.cnot(0, 1)
        assert s.prob(0) == pytest.approx(0.5, abs=1e-4)
        assert s.prob(1) == pytest.approx(0.5, abs=1e-4)

    def test_large_ghz_polynomial_memory(self):
        # 50-qubit GHZ — would OOM in a dense simulator; trivial here.
        s = QrackStabilizer(qubitCount=50)
        s.h(0)
        for q in range(1, 50):
            s.cnot(0, q)
        # Measure qubit 0; all others must agree.
        first = s.measure(0)
        for q in range(1, 50):
            assert s.measure(q) == first

    def test_no_non_clifford_methods(self):
        # QrackStabilizer must NOT expose non-Clifford gates.
        s = QrackStabilizer(qubitCount=2)
        assert not hasattr(s, "rx")
        assert not hasattr(s, "ry")
        assert not hasattr(s, "rz")
        assert not hasattr(s, "u")
        assert not hasattr(s, "t")
        assert not hasattr(s, "mtrx")

    def test_no_state_vector(self):
        # State vector is intentionally omitted from the stabilizer API.
        s = QrackStabilizer(qubitCount=2)
        assert not hasattr(s, "state_vector")
        assert not hasattr(s, "probabilities")

    def test_pauli_z_expectation(self):
        s = QrackStabilizer(qubitCount=1)
        assert s.exp_val(Pauli.PauliZ, 0) == pytest.approx(1.0, abs=1e-4)
        s.x(0)
        assert s.exp_val(Pauli.PauliZ, 0) == pytest.approx(-1.0, abs=1e-4)

    def test_context_manager(self):
        with QrackStabilizer(qubitCount=2) as s:
            s.h(0); s.cnot(0, 1)
            assert s.prob(0) == pytest.approx(0.5, abs=1e-4)


# ── QrackStabilizerHybrid ──────────────────────────────────────────────────────

class TestStabilizerHybridCore:
    def test_construction(self):
        s = QrackStabilizerHybrid(qubitCount=3)
        assert s.num_qubits == 3
        assert s.is_clifford is True   # starts in stabilizer mode

    def test_clifford_circuit_stays_clifford(self):
        s = QrackStabilizerHybrid(qubitCount=4)
        s.h(0); s.cnot(0, 1); s.cnot(1, 2); s.cnot(2, 3)
        assert s.is_clifford is True

    def test_non_clifford_gate_triggers_fallback(self):
        s = QrackStabilizerHybrid(qubitCount=2)
        s.h(0)
        assert s.is_clifford is True
        s.rx(0.5, 0)
        assert s.is_clifford is False

    def test_rz_engages_near_clifford_path(self):
        # RZ is the canonical near-Clifford gate. With T-injection enabled
        # the engine should defer falling out of stabilizer mode for as long
        # as it can; with T-injection off, RZ at an arbitrary angle generally
        # forces the dense-fallback path.
        #
        # The exact `is_clifford` value here depends on Qrack's near-Clifford
        # heuristics. The contract this test pins down is: the simulation
        # produces correct probabilities in both modes, and toggling T-injection
        # does not change the observable outcome — only the cost path.
        a = QrackStabilizerHybrid(qubitCount=1)
        a.set_t_injection(True)
        a.h(0); a.rz(math.pi / 3, 0); a.h(0)
        p_inj = a.prob(0)

        b = QrackStabilizerHybrid(qubitCount=1)
        b.set_t_injection(False)
        b.h(0); b.rz(math.pi / 3, 0); b.h(0)
        p_dense = b.prob(0)

        assert p_inj == pytest.approx(p_dense, abs=1e-4)

    def test_state_vector_after_fallback(self):
        s = QrackStabilizerHybrid(qubitCount=2)
        s.rx(math.pi, 0)            # forces fallback to dense
        sv = s.state_vector
        assert sv.shape == (4,)
        assert abs(sv[1]) == pytest.approx(1.0, abs=1e-3)  # |01>

    def test_matches_qrack_simulator(self):
        # Same Clifford circuit on both — probabilities must agree.
        a = QrackStabilizerHybrid(qubitCount=3)
        b = QrackSimulator(qubitCount=3, isStabilizerHybrid=True)
        for sim in (a, b):
            sim.h(0); sim.cnot(0, 1); sim.cnot(1, 2)
        for q in range(3):
            assert a.prob(q) == pytest.approx(b.prob(q), abs=1e-4)

    def test_t_injection_toggle(self):
        s = QrackStabilizerHybrid(qubitCount=2)
        s.set_t_injection(True)
        s.set_t_injection(False)   # idempotent toggle, just ensure no exception
```

---

## 9. PennyLane Plugin Update — Optional Phase 8 Extension

If [[qrackbind Phase 8]] is already shipped, this phase optionally adds two new device names that map onto the standalone classes:

```toml
# qrack.toml (qrackbind/pennylane/qrack.toml)
[device.qrackbind_stabilizer]
short_name = "qrackbind.stabilizer"
backend_class = "qrackbind.QrackStabilizer"
gates = ["Hadamard", "PauliX", "PauliY", "PauliZ", "S", "Adjoint(S)",
         "SX", "Adjoint(SX)", "CNOT", "CY", "CZ", "SWAP", "ISWAP",
         "MultiControlledX", "MultiControlledZ"]
# No rotation gates — PennyLane will reject non-Clifford ops at compile.

[device.qrackbind_stabilizer_hybrid]
short_name = "qrackbind.stabilizer_hybrid"
backend_class = "qrackbind.QrackStabilizerHybrid"
# Full gate set inherited from QrackSimulator.
```

The dispatch table in `_dispatch.py` is unchanged — both new devices route gate calls through the same gate-name → method-name table.

---

## 10. Phase 10 Completion Checklist

```
□ bindings/gate_helpers.h compiles standalone
□ bindings/stabilizer.cpp builds clean
□ Templated gate helpers extracted from simulator.cpp; Phase 1 tests still green
□ QrackStabilizer importable; constructs with qubitCount kwarg
□ QrackStabilizer.h, .x, .cnot, .swap, .mcx work
□ QrackStabilizer does NOT expose .rx, .ry, .rz, .u, .t, .mtrx (hasattr returns False)
□ QrackStabilizer does NOT expose .state_vector or .probabilities
□ QrackStabilizer 50-qubit GHZ runs in seconds with negligible memory
□ QrackStabilizer.exp_val(Pauli.PauliZ, q) returns ±1 for basis states
□ QrackStabilizerHybrid importable; constructs
□ QrackStabilizerHybrid.is_clifford starts True
□ QrackStabilizerHybrid.is_clifford becomes False after rx()
□ QrackStabilizerHybrid produces same probs as QrackSimulator(isStabilizerHybrid=True) on Clifford circuits
□ QrackStabilizerHybrid.state_vector works after fallback
□ set_t_injection / set_use_exact_near_clifford callable
□ just stubs regenerates _core.pyi with both new classes
□ pyright passes with zero new errors
□ uv run pytest tests/test_phase1.py … tests/test_phase10.py — all green
□ README "Development Phases" table updated with Phase 10 row
```

---

## 11. What Phase 10 Leaves Out (Deferred)

| Item | Reason |
|---|---|
| Catalyst QJIT support for `qrackbind.stabilizer*` devices | Requires a C++ `QuantumDevice` runtime adapter, not just nanobind Python bindings. Delivered in [[qrackbind Phase 11]] alongside DLPack and QIR serialisation |
| Zero-copy GPU state vector via DLPack (`state_vector_jax`, `state_vector_cuda`) | Same — Phase 11 adds an `add_state_access_dlpack<T>` helper wired into `QrackStabilizerHybrid` |
| `nb::call_guard<nb::gil_scoped_release>` on long-running operations | Phase 11 retroactively applies this to the templated helpers introduced here |
| Cross-class state transfer (`QrackStabilizer.to_simulator()`) | Requires a `Compose`/`Decompose` API across class boundaries; deferred |
| Stabilizer tableau export (`QrackStabilizer.tableau`) | Qrack's `QStabilizer` does not expose the raw tableau on `QInterface`; would need a downcast and an additional binding file |
| `QrackCircuit.run(QrackStabilizer)` | `QrackCircuit.run()` already accepts any `QInterface`-shaped sim; will accept these once the runtime check is loosened (one-line change in [[qrackbind Phase 6]]) |
| `QrackStabilizerNoisy` wrapper | Phase 11+ — depends on noise-channel binding work |
| Stabilizer-specific QASM output (Stim format) | Out of scope; only OpenQASM 2.0 via `QrackCircuit` is supported |

---

## Related

- [[qrackbind Project Phase Breakdown]]
- [[qrackbind Phase 1 Revised]]
- [[qrackbind Phase 6]]
- [[qrackbind Phase 7]]
- [[qrackbind Phase 8]]
- [[qrackbind Phase 11]]
- [[QrackSimulator API Method Categories]]
- [[Qrack and Hybrid Quantum Classical Computing]]
- [[qrack project/Reference/qinterface.hpp.md]]
- [[PennyLane Integration]]
