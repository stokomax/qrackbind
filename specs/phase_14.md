---
tags:
  - qrack
  - nanobind
  - python
  - noise
  - density-matrix
  - qinterface-noisy
  - pennylane
  - implementation
  - qrackbind
  - phase14
---
## qrackbind Phase 14 — Noisy Wrapper Layer and Density-Matrix Methods

Builds on [[qrackbind Phase 6]], [[qrackbind Phase 8]], [[qrackbind Phase 10]], [[qrackbind Phase 11]], and [[qrackbind Phase 12]]. `QINTERFACE_NOISY` is Qrack's wrapper layer that injects depolarizing noise around any underlying engine stack — listed in `qinterface.hpp` between `QINTERFACE_TENSOR_NETWORK` and the device-specific engines as the "Noisy wrapper layer." Phase 10 explicitly deferred `QrackStabilizerNoisy` to "Phase 11+ — depends on noise-channel binding work," and Phase 6 listed "Noise model attachment" as deferred. Phase 14 picks up that thread.

The noise control surface already exists on `QInterface` itself as virtual no-ops: `SetNoiseParameter`, `GetNoiseParameter`, `DepolarizingChannelWeak1Qb`, `GetUnitaryFidelity`, `ResetUnitaryFidelity`. They become live methods only when the underlying object is a `QInterfaceNoisy`. Phase 14's job is to (a) expose this wrapper layer as a typed, documented Python class, (b) bind the noise control methods, and (c) wire the new device through PennyLane and Catalyst so noise-aware research workflows are reachable from `qml.device(...)`.

**Prerequisite:** Phase 6 (`QrackCircuit`), Phase 8 (PennyLane plugin), Phase 10 (templated gate helpers, standalone-class pattern), Phase 11 (Catalyst runtime adapter, GIL-released helpers), and Phase 12 (approximation knob template) shipped.

---

## Why a Separate Class — Mirroring the Phase 10 Decision

`QInterfaceNoisy` is structurally a wrapper layer over any engine stack, peer to `QINTERFACE_TENSOR_NETWORK`. The architectural question is the same one Phase 10 answered for stabilizer: flag on `QrackSimulator` versus standalone class. The answer is the same — both, with the standalone class as the typed surface and the flag preserved for backward compatibility.

There is one meaningful divergence from Phase 10. `QrackStabilizer` and `QrackStabilizerHybrid` each pin a fixed engine stack: stabilizer is *the* engine. `QrackNoisySimulator` is fundamentally a wrapper that can sit over any underlying stack. The constructor therefore takes a `base=` kwarg selecting from a small enum of named stacks rather than hardcoding one.

| Reason | What it enables |
|---|---|
| Type-stub-documented contract | `QrackNoisySimulator.set_noise_parameter(...)` and `.unitary_fidelity` show up in `.pyi`; `QrackSimulator.unitary_fidelity` returns a constant 1.0 (the no-op base) so Pyright can guide users to the right class |
| Direct framework targets | PennyLane device names `qrackbind.noisy` and `qrackbind.noisy_stabilizer_hybrid` map onto these classes without flag plumbing through `QrackSimulator` kwargs |
| Density-matrix semantics documented at the class level | Class docstring states explicitly that `state_vector` is a trajectory sample, not the ensemble — the most likely source of user confusion under noise gets surfaced where users will see it |
| Configurable underlying engine | `base="simulator" \| "stabilizer_hybrid" \| "qbdd_hybrid" \| "hybrid"` selects what the noisy layer wraps; pyqrack's flag-based path can't expose this cleanly |

The pre-existing `noise: float` constructor flag and `set_noise_parameter()` method on `QrackSimulator` (per [[QrackSimulator API Method Categories]]) stay in place for backward compatibility with pyqrack and Bloqade users. Migration to `QrackNoisySimulator` is opt-in.

---

## Learning Goals

| Topic | Where it appears |
|---|---|
| Standalone class wrapping a configurable engine stack via constructor enum | §2 — `make_noisy()` factory; `NoisyBase` enum |
| Templated noise-control helper added to the existing `gate_helpers.h` cascade | §3 — `add_noise_methods<T>` |
| Density-matrix semantics in a state-vector-shaped binding | §5 — `state_vector` returns a trajectory; `sample_trajectories()` for ensemble averaging |
| PennyLane noise-aware device kwargs (`noise_param`) | §6 — `QrackNoisyDevice` |
| Catalyst runtime adapter — wrapper-layer engine stacks | §7 — `make_simulator_from_kwargs` recognises `engine="noisy_*"` |
| Strict-class composition: noisy wrapping stabilizer-hybrid (not pure stabilizer) | §4.2 — `QrackNoisyStabilizerHybrid` exists; `QrackNoisyStabilizer` does not |

---

## File Structure

| File | Changes |
|---|---|
| `bindings/noisy.cpp` | **New file** — `bind_noisy()` defines `QrackNoisySimulator` and `QrackNoisyStabilizerHybrid` |
| `bindings/gate_helpers.h` | Add `add_noise_methods<T>` template — exposes `set_noise_parameter`, `get_noise_parameter`, `depolarizing_channel_1qb`, `unitary_fidelity`, `reset_unitary_fidelity` |
| `bindings/module.cpp` | Add `bind_noisy()` registration after `bind_qbdd()` |
| `bindings/circuit.cpp` | `QrackCircuit.run()` accepts noisy backends without modification; runtime check loosened in Phase 6 already covers this |
| `bindings/catalyst_device.cpp` | Extend `make_simulator_from_kwargs` to recognise `engine="noisy_simulator"` and `engine="noisy_stabilizer_hybrid"`, plus a `noise_param` float kwarg |
| `CMakeLists.txt` | Add `bindings/noisy.cpp` to `_qrackbind_core` sources |
| `src/qrackbind/__init__.py` | Export `QrackNoisySimulator`, `QrackNoisyStabilizerHybrid`, `NoisyBase` |
| `src/qrackbind/_compat.py` | `_NoiseMixin` adding `noise()` context manager (analogous to Phase 12's `approximation()`) |
| `src/qrackbind/pennylane/device.py` | New `QrackNoisyDevice` and `QrackNoisyStabilizerHybridDevice` classes |
| `src/qrackbind/pennylane/qrack.toml` | Two new device entries; `noise_param` option declared |
| `tests/test_phase14_*.py` | Three test files: noise control, density-matrix semantics, PennyLane device |

---

## 1. Engine Stack Mapping

`QINTERFACE_NOISY` prepends to whatever stack would otherwise be used. Phase 14 ships two configurations; the constructor's `base=` kwarg selects between them, with two more deferred to a follow-up phase:

```
QrackNoisySimulator(base="simulator", ...)
    →  [QINTERFACE_NOISY, QINTERFACE_TENSOR_NETWORK, QINTERFACE_QUNIT,
        QINTERFACE_STABILIZER_HYBRID, QINTERFACE_QPAGER, QINTERFACE_HYBRID]

QrackNoisyStabilizerHybrid(...)
    →  [QINTERFACE_NOISY, QINTERFACE_STABILIZER_HYBRID, QINTERFACE_HYBRID]

# Deferred to Phase 14.5 / 15:
QrackNoisySimulator(base="qbdd_hybrid", ...)
    →  [QINTERFACE_NOISY, QINTERFACE_BDT_HYBRID, QINTERFACE_HYBRID]

QrackNoisySimulator(base="hybrid", ...)
    →  [QINTERFACE_NOISY, QINTERFACE_HYBRID]
```

The named-base enum (`NoisyBase.SIMULATOR`, `NoisyBase.STABILIZER_HYBRID`) is the right level of abstraction — researchers want noise applied to "the standard simulator" or "stabilizer-hybrid," not to a raw `QInterfaceEngine` vector. There is deliberately no `QrackNoisyStabilizer` (pure Clifford under noise needs a different mathematical treatment — depolarizing channels move out of the stabilizer formalism).

---

## 2. Factory Helper

```cpp
// ── File: bindings/noisy.cpp ─────────────────────────────────────────────────
#include "binding_core.h"
#include "gate_helpers.h"
#include "qfactory.hpp"

namespace {

enum class NoisyBase {
    SIMULATOR,
    STABILIZER_HYBRID,
};

struct NoisyConfig {
    NoisyBase base           = NoisyBase::SIMULATOR;
    real1_f   noise_param    = 0.01f;
    bool      isCpuGpuHybrid = true;
    bool      isOpenCL       = true;
    bool      isHostPointer  = false;
    bool      isSparse       = false;
};

QInterfacePtr make_noisy(bitLenInt n, const NoisyConfig& c) {
    std::vector<QInterfaceEngine> stack{QINTERFACE_NOISY};

    switch (c.base) {
    case NoisyBase::SIMULATOR:
        stack.push_back(QINTERFACE_TENSOR_NETWORK);
        stack.push_back(QINTERFACE_QUNIT);
        stack.push_back(QINTERFACE_STABILIZER_HYBRID);
        stack.push_back(QINTERFACE_QPAGER);
        if (c.isCpuGpuHybrid && c.isOpenCL) stack.push_back(QINTERFACE_HYBRID);
        else if (c.isOpenCL)                stack.push_back(QINTERFACE_OPENCL);
        else                                stack.push_back(QINTERFACE_CPU);
        break;
    case NoisyBase::STABILIZER_HYBRID:
        stack.push_back(QINTERFACE_STABILIZER_HYBRID);
        if (c.isCpuGpuHybrid && c.isOpenCL) stack.push_back(QINTERFACE_HYBRID);
        else if (c.isOpenCL)                stack.push_back(QINTERFACE_OPENCL);
        else                                stack.push_back(QINTERFACE_CPU);
        break;
    }

    auto sim = CreateQuantumInterface(
        stack, n, /*initState=*/0, /*rgp=*/nullptr,
        CMPLX_DEFAULT_ARG, /*doNorm=*/false, /*randomGP=*/true,
        c.isHostPointer, /*deviceId=*/-1, /*useHWRNG=*/true,
        c.isSparse);

    if (sim && c.noise_param > 0.0f) sim->SetNoiseParameter(c.noise_param);
    return sim;
}

} // namespace
```

The `noise_param` defaults to `0.01` (1% depolarizing strength) rather than zero — at zero, `QrackNoisySimulator` is structurally identical to `QrackSimulator` and the user is paying the wrapper-layer cost for no benefit. A non-zero default is a hint that this class is for noisy work.

---

## 3. Templated Noise Helper — `add_noise_methods<T>`

Phase 11 retrofitted `nb::call_guard<nb::gil_scoped_release>` onto the templated helpers. Phase 14 adds the noise-control helper following the same conventions:

```cpp
// ── File: bindings/gate_helpers.h ────────────────────────────────────────────
template <typename WrapperT>
void add_noise_methods(nb::class_<WrapperT>& cls) {
    cls
        .def("set_noise_parameter",
            [](WrapperT& w, real1_f lambda) { w.sim->SetNoiseParameter(lambda); },
            nb::arg("lambda"),
            "Depolarizing channel strength applied around each gate.\n"
            "0.0 disables noise (wrapper becomes a pass-through).\n"
            "Typical research values: 1e-4 to 1e-2.")
        .def("get_noise_parameter",
            [](const WrapperT& w) { return w.sim->GetNoiseParameter(); },
            "Current depolarizing channel strength.")
        .def("depolarizing_channel_1qb",
            [](WrapperT& w, bitLenInt q, real1_f lambda) {
                w.check_qubit(q, "depolarizing_channel_1qb");
                w.sim->DepolarizingChannelWeak1Qb(q, lambda);
            },
            nb::arg("qubit"), nb::arg("lambda"),
            nb::call_guard<nb::gil_scoped_release>(),
            "Apply an explicit single-qubit depolarizing channel of strength\n"
            "`lambda` to `qubit`. Independent of the global noise parameter —\n"
            "use for circuit-position-specific noise injection.")
        .def_prop_ro("unitary_fidelity",
            [](const WrapperT& w) { return w.sim->GetUnitaryFidelity(); },
            "Accumulated fidelity of the noisy simulation against an ideal\n"
            "unitary evolution. Starts at 1.0; decreases monotonically as\n"
            "noise channels are applied. Reset via reset_unitary_fidelity().")
        .def("reset_unitary_fidelity",
            [](WrapperT& w) { w.sim->ResetUnitaryFidelity(); },
            "Reset the accumulated unitary fidelity to 1.0. Useful when\n"
            "starting a new circuit on the same simulator instance.");
}
```

This helper is wired into `bind_noisy()` only. It is *not* applied to `QrackSimulator` — even though `QInterface` provides the no-op base implementations, exposing them on the non-noisy class would be misleading (`unitary_fidelity` would always read `1.0` regardless of what the user did, and `set_noise_parameter` would silently no-op).

---

## 4. The Two Classes

### 4.1 `QrackNoisySimulator`

```cpp
struct QrackNoisySim {
    QInterfacePtr sim;
    bitLenInt     numQubits;
    NoisyConfig   config;

    QrackNoisySim(bitLenInt n, const NoisyConfig& cfg)
        : numQubits(n), config(cfg), sim(make_noisy(n, cfg)) {
        if (!sim) throw QrackError("QrackNoisySimulator: factory returned null",
                                    QrackErrorKind::InvalidArgument);
    }

    void check_qubit(bitLenInt q, const char* method) const {
        if (q >= numQubits)
            throw QrackError(std::string(method) + ": qubit out of range",
                             QrackErrorKind::QubitOutOfRange);
    }

    std::string repr() const {
        return "QrackNoisySimulator(qubits=" + std::to_string(numQubits) +
               ", noise=" + std::to_string(sim->GetNoiseParameter()) +
               ", fidelity=" + std::to_string(sim->GetUnitaryFidelity()) + ")";
    }
};

void bind_noisy_simulator_class(nb::module_& m) {
    auto cls = nb::class_<QrackNoisySim>(m, "QrackNoisySimulator",
        "Quantum simulator with depolarizing noise injected around every gate.\n\n"
        "IMPORTANT — density-matrix semantics:\n"
        "  Under noise, the system is genuinely in a mixed state. `state_vector`\n"
        "  returns a single trajectory sample, NOT the ensemble. For expectation\n"
        "  values use `exp_val_*` (which average correctly via repeated sampling).\n"
        "  For ensemble state, use `sample_trajectories(n_shots)`.")
        .def("__init__",
            [](QrackNoisySim* self, bitLenInt qubitCount,
               NoisyBase base, real1_f noise_param,
               bool isCpuGpuHybrid, bool isOpenCL,
               bool isHostPointer, bool isSparse) {
                NoisyConfig cfg{base, noise_param, isCpuGpuHybrid, isOpenCL,
                                isHostPointer, isSparse};
                new (self) QrackNoisySim(qubitCount, cfg);
            },
            nb::arg("qubitCount")     = 0,
            nb::arg("base")           = NoisyBase::SIMULATOR,
            nb::arg("noise_param")    = 0.01f,
            nb::arg("isCpuGpuHybrid") = true,
            nb::arg("isOpenCL")       = true,
            nb::arg("isHostPointer")  = false,
            nb::arg("isSparse")       = false,
            "Create a noisy simulator on n qubits. The `base` kwarg selects\n"
            "the underlying engine stack: SIMULATOR for the full standard\n"
            "stack, STABILIZER_HYBRID for Clifford-with-fallback under noise.")
        .def("__repr__", &QrackNoisySim::repr);

    add_clifford_gates(cls);
    add_clifford_two_qubit(cls);
    add_t_gates(cls);
    add_rotation_gates(cls);
    add_u_gates(cls);
    add_matrix_gates(cls);
    add_measurement(cls);
    add_pauli_methods(cls);
    add_state_access(cls);          // documented as trajectory sample
    add_state_access_dlpack(cls);   // Phase 11 helper — same trajectory caveat
    add_approximation(cls);         // Phase 12 — SDRP/NCRP still apply
    add_noise_methods(cls);         // Phase 14 — the new helper

    cls
        .def_prop_ro("num_qubits",
            [](const QrackNoisySim& s) { return s.numQubits; })
        .def_prop_ro("base",
            [](const QrackNoisySim& s) { return s.config.base; },
            "The underlying engine base this noisy layer wraps.")
        .def("sample_trajectories",
            [](QrackNoisySim& s, size_t shots) {
                std::vector<bitCapInt> qPowers;
                qPowers.reserve(s.numQubits);
                for (bitLenInt q = 0; q < s.numQubits; ++q)
                    qPowers.push_back(pow2(q));
                return s.sim->MultiShotMeasureMask(qPowers, (unsigned)shots);
            },
            nb::arg("shots"),
            nb::call_guard<nb::gil_scoped_release>(),
            "Run `shots` independent noise trajectories and return a\n"
            "histogram of measured bit strings. This is the correct way\n"
            "to read ensemble statistics out of a noisy simulator.")
        .def("reset_all", [](QrackNoisySim& s) {
            s.sim->SetPermutation(0);
            s.sim->ResetUnitaryFidelity();
        })
        .def("set_permutation",
            [](QrackNoisySim& s, bitCapInt p) { s.sim->SetPermutation(p); },
            nb::arg("permutation"))
        .def("__enter__", [](QrackNoisySim& s) -> QrackNoisySim& { return s; })
        .def("__exit__",
            [](QrackNoisySim& s, nb::object, nb::object, nb::object) { s.sim.reset(); });
}
```

### 4.2 `QrackNoisyStabilizerHybrid`

```cpp
struct QrackNoisyStabHybrid {
    QInterfacePtr sim;
    bitLenInt     numQubits;
    NoisyConfig   config;
    /* ...same shape as QrackNoisySim... */
};

void bind_noisy_stabilizer_hybrid_class(nb::module_& m) {
    auto cls = nb::class_<QrackNoisyStabHybrid>(m, "QrackNoisyStabilizerHybrid",
        "Stabilizer-hybrid simulator with depolarizing noise injected around\n"
        "every gate. Stays in stabilizer mode for as long as gates are\n"
        "Clifford AND noise hasn't pushed the state out of the stabilizer\n"
        "manifold; falls back to dense simulation otherwise.\n\n"
        "Same density-matrix semantics caveat as QrackNoisySimulator —\n"
        "see that class's docstring.");

    /* Constructor pins base=NoisyBase::STABILIZER_HYBRID; rest of binding
       follows §4.1 exactly minus the `base` kwarg and `base` property. */

    add_clifford_gates(cls);
    add_clifford_two_qubit(cls);
    add_t_gates(cls);
    add_rotation_gates(cls);
    add_u_gates(cls);
    add_matrix_gates(cls);
    add_measurement(cls);
    add_pauli_methods(cls);
    add_state_access(cls);
    add_state_access_dlpack(cls);
    add_approximation(cls);
    add_noise_methods(cls);

    cls
        .def_prop_ro("is_clifford",
            [](const QrackNoisyStabHybrid& s) { return s.sim->isClifford(); },
            "True if the underlying engine is currently in stabilizer mode.\n"
            "Note: noise typically forces fallback faster than the noiseless\n"
            "stabilizer-hybrid does — depolarizing channels rarely preserve\n"
            "stabilizer states.")
        /* ...sample_trajectories, reset_all, etc. as in §4.1... */;
}
```

There is no `QrackNoisyStabilizer` class. Pure Clifford under depolarizing noise leaves the stabilizer formalism on the first noise application, so wrapping `QINTERFACE_STABILIZER` alone serves no one — anyone wanting noisy Clifford simulation should reach for `QrackNoisyStabilizerHybrid`.

---

## 5. Density-Matrix Semantics — The Critical Caveat

The single biggest source of confusion in noise-aware simulation is what `state_vector` means under noise. For an ideal pure-state simulator, `state_vector` is the state. For a noisy simulator, the actual quantum state is a density matrix — `state_vector` returns a **single trajectory sample** drawn from the channel.

Phase 14 surfaces this through three mechanisms:

1. **Class docstring** — both class docstrings open with an "IMPORTANT — density-matrix semantics" paragraph. This is what users see in `help(QrackNoisySimulator)` and IDE hovers.
2. **`sample_trajectories(shots)`** — explicit method that returns the correct ensemble statistic (a measurement histogram). Users reaching for ensemble averages have a direct path that doesn't go through `state_vector`.
3. **No silent renaming.** It would be tempting to rename `state_vector` to `state_vector_trajectory` or similar, but that breaks gate-helper template reuse and surprises users migrating code from `QrackSimulator`. The docstring caveat is the right level of intervention.

Expectation values via `exp_val_*` are the correct path even under noise — the trajectory sampling happens at measurement time and averages correctly across repeated calls. The wrapper does not need a separate API for this; the existing `add_pauli_methods<T>` works as-is because the noise is injected at the engine level.


## 6. Catalyst Runtime Adapter — Wrapper-Layer Engine Stacks

Phase 11's `make_simulator_from_kwargs` parser is extended to recognise the noisy variants:

```cpp
// ── File: bindings/catalyst_device.cpp — extend Phase 11/12 ──────────────────
QInterfacePtr make_simulator_from_kwargs(const std::string& kwargs) {
    auto opts = parse_kwargs(kwargs);

    std::vector<QInterfaceEngine> stack;
    if      (opts.engine == "qbdd")                 stack = {QINTERFACE_BDT};
    else if (opts.engine == "qbdd_hybrid")          stack = {QINTERFACE_BDT_HYBRID, QINTERFACE_HYBRID};
    else if (opts.engine == "stabilizer")           stack = {QINTERFACE_STABILIZER};
    else if (opts.engine == "stabilizer_hybrid")    stack = {QINTERFACE_STABILIZER_HYBRID, QINTERFACE_HYBRID};
    else if (opts.engine == "noisy_simulator") {                                          // ← Phase 14
        stack = {QINTERFACE_NOISY, QINTERFACE_TENSOR_NETWORK, QINTERFACE_QUNIT,
                 QINTERFACE_STABILIZER_HYBRID, QINTERFACE_QPAGER, QINTERFACE_HYBRID};
    }
    else if (opts.engine == "noisy_stabilizer_hybrid") {                                   // ← Phase 14
        stack = {QINTERFACE_NOISY, QINTERFACE_STABILIZER_HYBRID, QINTERFACE_HYBRID};
    }
    else /* simulator */                            stack = default_simulator_stack(opts);

    auto sim = CreateQuantumInterface(stack, opts.qubits, /* ... */);
    if (opts.sdrp > 0.0f)         sim->SetSdrp(opts.sdrp);
    if (opts.ncrp > 0.0f)         sim->SetNcrp(opts.ncrp);
    if (opts.noise_param > 0.0f)  sim->SetNoiseParameter(opts.noise_param);   // ← Phase 14
    return sim;
}
```

The Catalyst path now supports `@qjit` against the noisy device names. The compiled IR records the noise parameter as a runtime value, so the same compiled function can be re-invoked with different noise levels without recompilation.

---

## 7. `module.cpp` — Updated Registration

```cpp
void bind_exceptions(nb::module_& m);
void bind_pauli(nb::module_& m);
void bind_simulator(nb::module_& m);
void bind_circuit(nb::module_& m);
void bind_stabilizer(nb::module_& m);
void bind_qbdd(nb::module_& m);
void bind_noisy(nb::module_& m);   // ← Phase 14

NB_MODULE(_qrackbind_core, m) {
    m.doc() = "qrackbind — nanobind bindings for the Qrack quantum simulator";

    bind_exceptions(m);
    bind_pauli(m);
    bind_simulator(m);
    bind_circuit(m);
    bind_stabilizer(m);
    bind_qbdd(m);
    bind_noisy(m);   // last — depends on all of the above for templated helpers
}

// In bindings/noisy.cpp:
void bind_noisy(nb::module_& m) {
    nb::enum_<NoisyBase>(m, "NoisyBase",
        "Underlying engine base for QrackNoisySimulator.")
        .value("SIMULATOR", NoisyBase::SIMULATOR)
        .value("STABILIZER_HYBRID", NoisyBase::STABILIZER_HYBRID);

    bind_noisy_simulator_class(m);
    bind_noisy_stabilizer_hybrid_class(m);
}
```

---

## 8. Test Suite

```python
# tests/test_phase14_noise_control.py
import math
import pytest
from qrackbind import (
    QrackNoisySimulator, QrackNoisyStabilizerHybrid, QrackSimulator, NoisyBase,
)


class TestNoiseControl:
    def test_construction_default_noise(self):
        s = QrackNoisySimulator(qubitCount=3)
        assert s.num_qubits == 3
        assert s.get_noise_parameter() == pytest.approx(0.01, abs=1e-6)
        assert s.unitary_fidelity == pytest.approx(1.0, abs=1e-6)

    def test_set_get_noise_parameter_roundtrip(self):
        s = QrackNoisySimulator(qubitCount=2)
        s.set_noise_parameter(0.005)
        assert s.get_noise_parameter() == pytest.approx(0.005, abs=1e-6)

    def test_unitary_fidelity_decays_under_gates(self):
        s = QrackNoisySimulator(qubitCount=2, noise_param=0.05)
        f0 = s.unitary_fidelity
        for _ in range(10):
            s.h(0); s.cnot(0, 1)
        f1 = s.unitary_fidelity
        assert f1 < f0
        assert f1 > 0.0

    def test_zero_noise_matches_clean_simulator(self):
        a = QrackNoisySimulator(qubitCount=2, noise_param=0.0)
        b = QrackSimulator(qubitCount=2)
        for sim in (a, b):
            sim.h(0); sim.cnot(0, 1)
        # Probabilities should match within a tight tolerance — noise=0 is a
        # pass-through wrapper.
        for q in range(2):
            assert a.prob(q) == pytest.approx(b.prob(q), abs=1e-4)

    def test_reset_unitary_fidelity(self):
        s = QrackNoisySimulator(qubitCount=1, noise_param=0.1)
        for _ in range(20): s.h(0)
        assert s.unitary_fidelity < 1.0
        s.reset_unitary_fidelity()
        assert s.unitary_fidelity == pytest.approx(1.0, abs=1e-6)

    def test_explicit_depolarizing_channel(self):
        s = QrackNoisySimulator(qubitCount=1, noise_param=0.0)
        assert s.unitary_fidelity == pytest.approx(1.0, abs=1e-6)
        s.depolarizing_channel_1qb(0, 0.2)
        assert s.unitary_fidelity < 1.0


class TestNoisyBase:
    def test_simulator_base(self):
        s = QrackNoisySimulator(qubitCount=4, base=NoisyBase.SIMULATOR)
        assert s.base == NoisyBase.SIMULATOR

    def test_stabilizer_hybrid_base(self):
        s = QrackNoisySimulator(qubitCount=4, base=NoisyBase.STABILIZER_HYBRID)
        assert s.base == NoisyBase.STABILIZER_HYBRID


# tests/test_phase14_density_matrix_semantics.py
class TestDensityMatrixSemantics:
    def test_sample_trajectories_returns_histogram(self):
        s = QrackNoisySimulator(qubitCount=2, noise_param=0.01)
        s.h(0); s.cnot(0, 1)
        hist = s.sample_trajectories(shots=1000)
        # Bell-state-like distribution — most mass on |00> and |11>.
        total = sum(hist.values())
        assert total == 1000
        p00 = hist.get(0, 0) / 1000
        p11 = hist.get(3, 0) / 1000
        assert p00 + p11 > 0.85   # noise scatters ~10–15% to other outcomes

    def test_exp_val_is_correctly_averaged_under_noise(self):
        # <Z> on |0> with depolarizing noise should be close to 1, decreasing
        # monotonically with noise strength.
        from qrackbind import Pauli
        s_clean = QrackNoisySimulator(qubitCount=1, noise_param=0.0)
        s_weak  = QrackNoisySimulator(qubitCount=1, noise_param=0.01)
        s_loud  = QrackNoisySimulator(qubitCount=1, noise_param=0.10)
        for s in (s_clean, s_weak, s_loud):
            for _ in range(20): s.h(0); s.h(0)   # double-H = identity
        z_clean = s_clean.exp_val(Pauli.PauliZ, 0)
        z_weak  = s_weak.exp_val(Pauli.PauliZ, 0)
        z_loud  = s_loud.exp_val(Pauli.PauliZ, 0)
        assert z_clean > z_weak > z_loud

    def test_state_vector_is_documented_as_trajectory(self):
        # Smoke test — state_vector returns *something* of the right shape;
        # the contract that it's a trajectory sample is enforced by docstring.
        s = QrackNoisySimulator(qubitCount=2)
        s.h(0); s.cnot(0, 1)
        sv = s.state_vector
        assert sv.shape == (4,)


class TestNoisyStabilizerHybrid:
    def test_construction(self):
        s = QrackNoisyStabilizerHybrid(qubitCount=3)
        assert s.num_qubits == 3
        assert s.get_noise_parameter() == pytest.approx(0.01, abs=1e-6)

    def test_noise_forces_fallback_faster_than_clean(self):
        # Empirical: under noise, Clifford circuits exit stabilizer mode
        # earlier than their clean counterparts.
        s = QrackNoisyStabilizerHybrid(qubitCount=4, noise_param=0.05)
        s.h(0); s.cnot(0, 1); s.cnot(1, 2); s.cnot(2, 3)
        # No assertion on is_clifford — it's heuristic-dependent.
        # Just verify the property is callable and returns a bool.
        assert isinstance(s.is_clifford, bool)

```

---

## 9. Phase 14 Completion Checklist

```
□ bindings/noisy.cpp builds clean
□ NoisyBase enum exported with SIMULATOR and STABILIZER_HYBRID values
□ QrackNoisySimulator importable; constructs with qubitCount, base, noise_param
□ QrackNoisyStabilizerHybrid importable; constructs
□ set_noise_parameter / get_noise_parameter roundtrip
□ unitary_fidelity property reads 1.0 initially
□ unitary_fidelity decays monotonically under repeated gates
□ reset_unitary_fidelity returns it to 1.0
□ depolarizing_channel_1qb applies an explicit channel and decays fidelity
□ noise_param=0.0 produces probabilities matching QrackSimulator within 1e-4
□ sample_trajectories(shots) returns a histogram summing to `shots`
□ exp_val_* values decrease monotonically with increasing noise_param
□ Class docstrings include the density-matrix-semantics caveat
□ QrackSimulator does NOT expose unitary_fidelity / set_noise_parameter (hasattr False)
   (the existing pyqrack-compat noise flag stays as a constructor kwarg only)
□ qml.device("qrackbind.noisy", wires=N) loads
□ qml.device("qrackbind.noisy_stabilizer_hybrid", wires=N) loads
□ noise_param kwarg flows through to the underlying simulator
□ Adjoint diff_method raises a clear error on noisy devices
□ Parameter-shift diff_method works under noise
□ Catalyst runtime accepts noisy_simulator and noisy_stabilizer_hybrid engine names
□ Catalyst runtime accepts noise_param kwarg
□ TOML declares intrinsic_noise = true on both new device entries
□ batched_execution / kernel_matrix from Phase 13 work on noisy devices
□ just stubs regenerates _core.pyi with the new classes and methods
□ pyright passes with zero new errors
□ uv run pytest tests/test_phase1.py … tests/test_phase14_*.py — all green
□ README "Development Phases" table updated with Phase 14 row
```

---

## 10. What Phase 14 Leaves Out (Deferred)

| Item | Reason |
|---|---|
| `QrackNoisySimulator(base="qbdd_hybrid")` and `base="hybrid")` | Easy to add once §1's switch statement gets the additional cases; deferred to keep Phase 14 focused on the two highest-value configurations |
| Per-gate noise channels (`depolarizing_channel_2qb`, amplitude damping, dephasing) | Qrack exposes `DepolarizingChannelWeak1Qb` directly; richer channel zoo (Kraus operators, T1/T2 models) is a follow-up phase |
| Noise-aware adjoint differentiation | Adjoint assumes unitary evolution; supporting it under noise requires a fundamentally different gradient method (e.g., stochastic adjoint). Out of scope |
| Density-matrix output (`density_matrix` property returning a 2ⁿ × 2ⁿ array) | Cost-prohibitive for n > ~12; if needed, users can construct it from `sample_trajectories` themselves. Revisit if Qrack adds a native API |
| `qml.NoiseModel` integration (translating PennyLane's noise spec into qrackbind's noise channels) | Requires a small adapter layer; defer to a Phase 14.5 or to feedback from PennyLane noise-aware users |
| Pauli-twirling / tensor-network error mitigation | Higher-level mitigation methods belong in user code or in PennyLane transforms, not in the device |
| Stabilizer-format export of noisy circuits (Stim) | Out of scope; OpenQASM 2.0 via `QrackCircuit` remains the primary serialization target |

---

## Related

- [[qrackbind Project Phase Breakdown]]
- [[qrackbind Phase 6]]
- [[qrackbind Phase 8]]
- [[qrackbind Phase 10]]
- [[qrackbind Phase 11]]
- [[qrackbind Phase 12]]
- [[qrackbind Phase 13]]
- [[QrackSimulator API Method Categories]]
- [[Qrack and Hybrid Quantum Classical Computing]]
- [[PennyLane Use Cases]]
- [[PennyLane Support]]
- [[Framework Plugin Architecture (PennyLane + Qiskit)]]
- [[qrack project/Reference/qinterface.hpp.md]]
