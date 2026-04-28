---
tags:
  - qrack
  - nanobind
  - python
  - catalyst
  - qjit
  - dlpack
  - qir
  - implementation
  - qrackbind
  - phase11
---
## qrackbind Phase 11 — Catalyst QJIT, Zero-Copy GPU Data Flow, and QIR Serialization

Builds on [[qrackbind Phase 6]], [[qrackbind Phase 8]], and [[qrackbind Phase 10]]. Phase 8 delivers a working PennyLane device plugin for the standard Python execution path; Phase 10 adds the standalone `QrackStabilizer` / `QrackStabilizerHybrid` classes. Phase 11 fills the deeper integration gap that the [[PennyLane Support]] analysis identifies: enabling `@qjit` compilation through PennyLane Catalyst, exposing GPU state vectors via DLPack for zero-copy hybrid quantum-classical training, and giving `QrackCircuit` the QIR/MLIR serialization needed to feed Catalyst's compilation pipeline.

This is the most ambitious phase architecturally because it crosses three runtime boundaries — nanobind (Python/C++), Catalyst (LLVM/MLIR/QIR), and DLPack (cross-framework GPU memory). Each piece is independently useful; together they close the structural gaps that keep `pennylane-qrack` on a two-codebase architecture today.

**Prerequisite:** Phase 6 (`QrackCircuit` with `GateType` enum), Phase 8 (PennyLane plugin with `QrackDevice`), and Phase 10 (stabilizer classes) shipped.

---

## Why a Single Compiled Layer Matters

`pennylane-qrack` today maintains two completely separate codepaths: PyQrack ctypes for the standard Python device, and a separately-compiled C++ Catalyst runtime built from a Qrack git submodule when `@qjit` is used. The two paths can run different Qrack versions, exhibit divergent behaviour, and require building Qrack twice at install time. The [[PennyLane Support]] analysis frames this as the central architectural problem.

qrackbind's nanobind extension is already a single compiled `.so` for the standard path. Phase 11 adds a Catalyst `QuantumDevice` adapter that links against the **same** compiled Qrack library that the Python device uses, guaranteeing identical Qrack version, simulator state, and configuration in both paths. Install becomes one build; behaviour becomes one codepath.

```
┌─────────────────── Today (pennylane-qrack) ─────────────────┐
│  Python device path:                                         │
│    Python → PyQrack (ctypes) → libqrack.so (PPA build)       │
│                                                              │
│  QJIT path:                                                  │
│    Python → MLIR → Catalyst runtime → libqrack.so            │
│                                          (rebuilt from        │
│                                           submodule)          │
│                                                              │
│  Two codebases. Two builds. Two Qrack versions possible.     │
└──────────────────────────────────────────────────────────────┘

┌─────────── Phase 11 (qrackbind + Catalyst adapter) ──────────┐
│  Both paths:                                                 │
│    Python (nanobind) ──┐                                     │
│                        ├→ libqrack.so (one install)          │
│    Catalyst (C++ ABI) ─┘                                     │
│                                                              │
│  One Qrack. One behaviour.                                   │
└──────────────────────────────────────────────────────────────┘
```

---

## Learning Goals

| Topic | Where it appears |
|---|---|
| Implementing Catalyst's `QuantumDevice` C++ interface | §3 — `QrackBindCatalystDevice` |
| Device registration via `getCustomDevice` entry point | §3 — exported C symbol Catalyst loads via dlopen |
| `nb::ndarray` with device tags for DLPack export | §5 — `state_vector_jax`, `state_vector_cuda` |
| Cross-runtime memory ownership (capsule lifetime across nanobind→DLPack→JAX) | §5 — capsule deleter pattern |
| Optional CMake dependencies (Catalyst as optional build target) | §8 — `find_package(Catalyst CONFIG QUIET)` |
| `nb::call_guard<nb::gil_scoped_release>` for long-running operations | §6 — retroactive Phase 10 helper update |

---

## File Structure

| File | Changes |
|---|---|
| `bindings/catalyst_device.cpp` | **New file** — `QrackBindCatalystDevice` implementing `Catalyst::Runtime::QuantumDevice` |
| `bindings/circuit.cpp` | Add `to_qir()` and `to_mlir()` methods to `QrackCircuit` |
| `bindings/qir_lowering.cpp` | **New file** — `qcircuit_to_qir_bitcode` and `qcircuit_to_catalyst_mlir` helpers built against LLVM IRBuilder |
| `bindings/gate_helpers.h` | Add `add_state_access_dlpack<T>` template; retrofit existing helpers with `nb::call_guard<nb::gil_scoped_release>` |
| `bindings/simulator.cpp`, `bindings/stabilizer.cpp` | Wire up `add_state_access_dlpack` for `QrackSimulator` and `QrackStabilizerHybrid` (not `QrackStabilizer` — keeps the strict contract) |
| `CMakeLists.txt` | Optional `find_package(Catalyst CONFIG QUIET)`; build `_qrackbind_catalyst.so` only when Catalyst is present |
| `src/qrackbind/pennylane/device.py` | Implement `get_c_interface()` returning the Catalyst runtime path |
| `src/qrackbind/pennylane/qrack.toml` | Add `qjit_compatible = true` and adjoint declarations |
| `tests/test_phase11_catalyst.py` | Catalyst-only test suite (skip when Catalyst not installed) |
| `tests/test_phase11_dlpack.py` | DLPack tests (run unconditionally) |
| `tests/test_phase11_qir.py` | QIR/MLIR serialization tests |

---

## 1. Catalyst's `QuantumDevice` Interface

Catalyst's runtime defines an abstract C++ class that any QJIT-compatible device must implement. The contract (from Catalyst's `QuantumDevice.hpp`) covers qubit allocation, gate dispatch, measurement, observables, and tape recording. Phase 11 implements it as a thin shim over the same `QInterfacePtr` machinery used by the Python-facing classes.

```cpp
// Sketch — exact signatures should be confirmed against the installed
// Catalyst runtime headers at build time.
namespace Catalyst::Runtime {
class QuantumDevice {
public:
    virtual ~QuantumDevice() = default;

    virtual auto AllocateQubit() -> QubitIdType = 0;
    virtual auto AllocateQubits(size_t n) -> std::vector<QubitIdType> = 0;
    virtual void ReleaseQubit(QubitIdType q) = 0;

    virtual void StartTapeRecording() = 0;
    virtual void StopTapeRecording() = 0;

    virtual void NamedOperation(
        const std::string& name,
        const std::vector<double>& params,
        const std::vector<QubitIdType>& wires,
        bool inverse) = 0;
    virtual void MatrixOperation(
        const std::vector<std::complex<double>>& matrix,
        const std::vector<QubitIdType>& wires,
        bool inverse) = 0;

    virtual auto Measure(QubitIdType wire) -> Result = 0;
    virtual auto Expval(ObsIdType obs) -> double = 0;
    virtual auto Var(ObsIdType obs) -> double = 0;
};
}
```

---

## 2. `QrackBindCatalystDevice` — the Bridge Class

```cpp
// ── File: bindings/catalyst_device.cpp ───────────────────────────────────────
#include "QuantumDevice.hpp"   // From Catalyst runtime headers
#include "binding_core.h"
#include "qfactory.hpp"

class QrackBindCatalystDevice : public Catalyst::Runtime::QuantumDevice {
    QInterfacePtr sim_;
    bitLenInt nextWire_ = 0;
    std::unordered_map<QubitIdType, bitLenInt> idMap_;
    bool recording_ = false;
    std::shared_ptr<Qrack::QCircuit> tape_;

public:
    explicit QrackBindCatalystDevice(const std::string& kwargs)
        : sim_(make_simulator_from_kwargs(kwargs)) {}

    // ── Allocation ───────────────────────────────────────────────────────────
    auto AllocateQubit() -> QubitIdType override {
        bitLenInt q = sim_->Allocate(1);
        QubitIdType id = static_cast<QubitIdType>(nextWire_++);
        idMap_[id] = q;
        return id;
    }

    auto AllocateQubits(size_t n) -> std::vector<QubitIdType> override {
        std::vector<QubitIdType> ids;
        ids.reserve(n);
        for (size_t i = 0; i < n; ++i) ids.push_back(AllocateQubit());
        return ids;
    }

    void ReleaseQubit(QubitIdType q) override {
        auto it = idMap_.find(q);
        if (it != idMap_.end()) {
            sim_->Dispose(it->second, 1);
            idMap_.erase(it);
        }
    }

    // ── Gate dispatch ────────────────────────────────────────────────────────
    void NamedOperation(
        const std::string& name,
        const std::vector<double>& params,
        const std::vector<QubitIdType>& wires,
        bool inverse) override
    {
        std::vector<bitLenInt> q;
        q.reserve(wires.size());
        for (auto w : wires) q.push_back(idMap_.at(w));

        // Subset shown — full table mirrors Phase 8's _dispatch.py
        if      (name == "Hadamard") sim_->H(q[0]);
        else if (name == "PauliX")   sim_->X(q[0]);
        else if (name == "PauliY")   sim_->Y(q[0]);
        else if (name == "PauliZ")   sim_->Z(q[0]);
        else if (name == "S")        inverse ? sim_->IS(q[0]) : sim_->S(q[0]);
        else if (name == "T")        inverse ? sim_->IT(q[0]) : sim_->T(q[0]);
        else if (name == "SX")       inverse ? sim_->ISqrtX(q[0]) : sim_->SqrtX(q[0]);
        else if (name == "CNOT")     sim_->CNOT(q[0], q[1]);
        else if (name == "CZ")       sim_->CZ(q[0], q[1]);
        else if (name == "SWAP")     sim_->Swap(q[0], q[1]);
        else if (name == "RX")       sim_->RX(inverse ? -params[0] : params[0], q[0]);
        else if (name == "RY")       sim_->RY(inverse ? -params[0] : params[0], q[0]);
        else if (name == "RZ")       sim_->RZ(inverse ? -params[0] : params[0], q[0]);
        else if (name == "U3")       sim_->U(q[0], params[0], params[1], params[2]);
        else throw std::runtime_error("Catalyst dispatch: unknown gate " + name);
    }

    void MatrixOperation(
        const std::vector<std::complex<double>>& matrix,
        const std::vector<QubitIdType>& wires,
        bool inverse) override
    {
        std::vector<Qrack::complex> m32;
        m32.reserve(matrix.size());
        for (const auto& c : matrix)
            m32.emplace_back(static_cast<float>(c.real()),
                             static_cast<float>(c.imag()));
        if (inverse) {
            // Conjugate transpose for 2x2: swap m32[1] and m32[2], conjugate all
            std::swap(m32[1], m32[2]);
            for (auto& c : m32) c = std::conj(c);
        }
        sim_->Mtrx(m32.data(), idMap_.at(wires[0]));
    }

    // ── Measurement & observables ────────────────────────────────────────────
    auto Measure(QubitIdType wire) -> Catalyst::Runtime::Result override {
        return sim_->M(idMap_.at(wire));
    }

    auto Expval(Catalyst::Runtime::ObsIdType obs) -> double override {
        // Look up registered observable, dispatch to ExpectationPauliAll etc.
        return /* ... */;
    }

    // ── Tape recording for adjoint diff ──────────────────────────────────────
    void StartTapeRecording() override {
        recording_ = true;
        tape_ = std::make_shared<Qrack::QCircuit>();
    }
    void StopTapeRecording() override { recording_ = false; }
};

// Catalyst loads this entry point via dlopen at JIT time
extern "C" {
Catalyst::Runtime::QuantumDevice* getCustomDevice(const char* kwargs) {
    return new QrackBindCatalystDevice(kwargs ? kwargs : "");
}
}
```

The Python-side `QrackDevice` (Phase 8) gains a `get_c_interface()` method that returns the path to `_qrackbind_catalyst.so` and the entry-point name `getCustomDevice`. PennyLane Catalyst loads the device via `dlopen` at JIT time.

---

## 3. `QrackCircuit.to_qir()` and `.to_mlir()`

`QrackCircuit` from Phase 6 already records gates as a typed tape. Phase 11 lowers that tape into Catalyst's compilation IRs:

```cpp
// ── File: bindings/circuit.cpp — extend bind_circuit() ───────────────────────

.def("to_qir",
    [](const QrackCircuit& c) -> nb::bytes {
        std::string bitcode = qcircuit_to_qir_bitcode(c.circuit, c.numQubits);
        return nb::bytes(bitcode.data(), bitcode.size());
    },
    nb::sig("def to_qir(self) -> bytes"),
    "Serialise the circuit to QIR (Quantum Intermediate Representation)\n"
    "as LLVM bitcode. Used as input to Catalyst's MLIR compilation pipeline.")

.def("to_mlir",
    [](const QrackCircuit& c) -> std::string {
        return qcircuit_to_catalyst_mlir(c.circuit, c.numQubits);
    },
    nb::sig("def to_mlir(self) -> str"),
    "Serialise the circuit to Catalyst's quantum MLIR dialect as text.\n"
    "Useful for inspecting what Catalyst sees before lowering to QIR.")
```

The `qcircuit_to_qir_bitcode` helper (in `bindings/qir_lowering.cpp`) uses LLVM's IRBuilder to emit calls to QIR's standard gate function declarations: `__quantum__qis__h__body`, `__quantum__qis__cnot__body`, `__quantum__qis__rz__body`, etc. The translation is mechanical — `QrackCircuit`'s gate tape maps 1:1 to QIR named operations.

The `qcircuit_to_catalyst_mlir` helper emits text in Catalyst's `quantum.*` dialect — `quantum.alloc`, `quantum.custom "Hadamard"`, `quantum.measure`. This is human-readable and round-trips through `mlir-opt`.

---

## 4. DLPack-Aware State Vector — `add_state_access_dlpack<T>`

Phase 10's `add_state_access<T>` returns CPU NumPy. Phase 11 adds device-aware variants that avoid the GPU→CPU→GPU roundtrip when the consumer (PyTorch, JAX, CuPy) is on the same device as Qrack's simulator.

```cpp
// ── File: bindings/gate_helpers.h ────────────────────────────────────────────

template <typename WrapperT>
void add_state_access_dlpack(nb::class_<WrapperT>& cls) {
    cls.def_prop_ro("state_vector_jax",
        [](WrapperT& w) {
            size_t dim = 1ULL << w.numQubits;
            auto* buf = new std::vector<cf32>(dim);
            w.sim->GetQuantumState(buf->data());
            nb::capsule deleter(buf, [](void* p) noexcept {
                delete static_cast<std::vector<cf32>*>(p);
            });
            return nb::ndarray<nb::jax, cf32, nb::ndim<1>>(
                buf->data(), {dim}, deleter);
        },
        "State vector as a JAX-compatible DLPack array.\n"
        "Zero-copy when JAX is on the same device as Qrack.");

    cls.def_prop_ro("state_vector_cuda",
        [](WrapperT& w) {
            size_t dim = 1ULL << w.numQubits;
            auto* buf = new std::vector<cf32>(dim);
            w.sim->GetQuantumState(buf->data());
            nb::capsule deleter(buf, [](void* p) noexcept {
                delete static_cast<std::vector<cf32>*>(p);
            });
            return nb::ndarray<cf32, nb::ndim<1>, nb::device::cuda>(
                buf->data(), {dim}, deleter);
        },
        "State vector as a CUDA DLPack array.\n"
        "When Qrack runs on CUDA, this avoids the GPU→CPU→GPU roundtrip\n"
        "to PyTorch / JAX / CuPy. Falls back to a managed copy on CPU builds.");
}
```

Phase 11 wires this helper into `bind_simulator()` and `bind_stabilizer_hybrid_class()` only. `QrackStabilizer` deliberately does *not* get the DLPack export — keeping the strict-contract Clifford-only API and avoiding the expensive dense materialisation that DLPack would force.

> **Capsule lifetime model**: the `nb::capsule(buf, deleter)` pattern guarantees the C++ buffer survives at least as long as any DLPack consumer holds a reference. JAX's `jnp.asarray()` and PyTorch's `torch.from_dlpack()` both honour this.

---

## 5. GIL Release — Retroactive Phase 10 Helper Update

Phase 10's templated helpers in `bindings/gate_helpers.h` did not set a GIL policy. Phase 11 retroactively adds `nb::call_guard<nb::gil_scoped_release>()` to every long-running operation:

```cpp
// In add_clifford_gates<T>, add_rotation_gates<T>, etc. — apply to gate methods
.def(pyname, [](WrapperT& w, bitLenInt q) {
    w.check_qubit(q, pyname); w.sim->cppfn(q);
},
nb::arg("qubit"),
nb::call_guard<nb::gil_scoped_release>(),  // ← Phase 11 addition
doc)
```

Apply consistently to: gate methods, `measure*`, `prob*`, `state_vector*`, `exp_val*`, `measure_shots`. Trivial property accessors (`num_qubits`, `is_clifford`) keep the GIL — releasing for a constant-time read costs more than it saves.

The cost on small operations is a few nanoseconds per call. The benefit on large ones is significant for ML training pipelines that run gradient computation, data loading, and logging from separate Python threads. Phase 10's tests still pass — the guard is transparent to single-threaded code.

> **Mid-circuit measurement & Catalyst tracing**: `measure()` with the GIL released also satisfies Catalyst's mid-circuit measurement requirements for `cond` / `for_loop` / `while_loop` constructs. The Catalyst device's `Measure` (§2) calls the same `sim_->M(...)` underneath; both paths converge on identical semantics.

---

## 6. Adjoint Differentiation in the PennyLane TOML

With `QrackCircuit.inverse()` from Phase 6 and the Catalyst tape recording from §2, adjoint differentiation becomes possible. Phase 11 updates the device TOML:

```toml
# qrackbind/pennylane/qrack.toml

[device.qrackbind_simulator]
short_name = "qrackbind.simulator"
backend_class = "qrackbind.QrackSimulator"
qjit_compatible = true
supports_derivatives = ["parameter-shift", "adjoint"]

[device.qrackbind_stabilizer_hybrid]
short_name = "qrackbind.stabilizer_hybrid"
backend_class = "qrackbind.QrackStabilizerHybrid"
qjit_compatible = true
supports_derivatives = ["parameter-shift", "adjoint"]

[device.qrackbind_stabilizer]
short_name = "qrackbind.stabilizer"
backend_class = "qrackbind.QrackStabilizer"
qjit_compatible = true
supports_derivatives = []   # Pure stabilizer — no continuous parameters
```

The Python plugin's `execute()` selects adjoint when `supports_derivatives()` returns it and the circuit has more than ~10 parameters: adjoint is constant-cost vs. parameter-shift's O(N) cost. The crossover is roughly 10 parameters in practice.

---

## 7. CMake — Optional Catalyst Dependency

```cmake
find_package(Catalyst CONFIG QUIET)

if(Catalyst_FOUND)
    message(STATUS "Catalyst found — building catalyst_device.cpp")
    add_library(_qrackbind_catalyst SHARED
        bindings/catalyst_device.cpp)
    target_link_libraries(_qrackbind_catalyst PRIVATE
        ${QRACK_LIB}
        Catalyst::QuantumDevice)
    target_include_directories(_qrackbind_catalyst PRIVATE
        ${QRACK_INCLUDE}
        ${CMAKE_CURRENT_SOURCE_DIR}/bindings)
    install(TARGETS _qrackbind_catalyst LIBRARY DESTINATION qrackbind)
else()
    message(STATUS "Catalyst not found — skipping catalyst_device.cpp")
endif()

# QIR lowering needs LLVM regardless of Catalyst
find_package(LLVM CONFIG REQUIRED)
target_link_libraries(_qrackbind_core PRIVATE LLVMCore LLVMBitWriter)
target_sources(_qrackbind_core PRIVATE bindings/qir_lowering.cpp)
```

The Catalyst runtime is an optional build target. Phase 9 wheels ship the Catalyst adapter when Catalyst was available at build time; otherwise the standard PennyLane device works and `@qjit` raises a clear error directing the user to install Catalyst.

---

## 8. Test Suite

```python
# tests/test_phase11_catalyst.py
import pytest
catalyst = pytest.importorskip("catalyst")
import pennylane as qml
from catalyst import qjit


class TestCatalystDevice:
    def test_qjit_basic_circuit(self):
        dev = qml.device("qrackbind.simulator", wires=2)

        @qjit
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        probs = circuit()
        assert probs[0] == pytest.approx(0.5, abs=1e-4)
        assert probs[3] == pytest.approx(0.5, abs=1e-4)

    def test_qjit_stabilizer_clifford_only(self):
        dev = qml.device("qrackbind.stabilizer", wires=4)

        @qjit
        @qml.qnode(dev)
        def ghz():
            qml.Hadamard(wires=0)
            for q in range(1, 4):
                qml.CNOT(wires=[0, q])
            return qml.probs(wires=range(4))

        probs = ghz()
        assert probs[0]  == pytest.approx(0.5, abs=1e-4)
        assert probs[15] == pytest.approx(0.5, abs=1e-4)

    def test_qjit_stabilizer_rejects_non_clifford(self):
        dev = qml.device("qrackbind.stabilizer", wires=2)
        with pytest.raises(Exception):  # Catalyst should reject at compile
            @qjit
            @qml.qnode(dev)
            def bad():
                qml.RX(0.5, wires=0)
                return qml.expval(qml.PauliZ(0))
            bad()

    def test_adjoint_matches_parameter_shift(self):
        import jax.numpy as jnp
        dev = qml.device("qrackbind.simulator", wires=2)

        @qml.qnode(dev, diff_method="adjoint")
        def adj(x):
            qml.RX(x, wires=0); qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(1))

        @qml.qnode(dev, diff_method="parameter-shift")
        def psr(x):
            qml.RX(x, wires=0); qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(1))

        x = jnp.array(0.7)
        assert qml.grad(adj)(x) == pytest.approx(qml.grad(psr)(x), abs=1e-4)


# tests/test_phase11_dlpack.py
import numpy as np
import pytest
from qrackbind import QrackSimulator, QrackStabilizerHybrid


class TestDLPackStateVector:
    def test_state_vector_jax(self):
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp
        sim = QrackSimulator(qubitCount=3); sim.h(0)
        sv = jnp.asarray(sim.state_vector_jax)
        assert sv.shape == (8,)

    def test_state_vector_cuda_dlpack_roundtrip(self):
        sim = QrackSimulator(qubitCount=2)
        sv_dl = sim.state_vector_cuda
        sv_np = np.from_dlpack(sv_dl)
        assert sv_np.shape == (4,)
        assert abs(sv_np[0]) == pytest.approx(1.0, abs=1e-5)

    def test_capsule_lifetime_outlives_simulator(self):
        sim = QrackSimulator(qubitCount=2); sim.h(0)
        sv = sim.state_vector_jax
        del sim
        assert sv is not None  # Buffer must still be readable

    def test_stabilizer_hybrid_has_dlpack(self):
        sh = QrackStabilizerHybrid(qubitCount=2); sh.h(0); sh.cnot(0, 1)
        sv = np.from_dlpack(sh.state_vector_cuda)
        assert sv.shape == (4,)

    def test_pure_stabilizer_has_no_dlpack(self):
        from qrackbind import QrackStabilizer
        s = QrackStabilizer(qubitCount=2)
        assert not hasattr(s, "state_vector_jax")
        assert not hasattr(s, "state_vector_cuda")


# tests/test_phase11_qir.py
from qrackbind import QrackCircuit, GateType


class TestQirSerialization:
    def test_to_qir_returns_llvm_bitcode(self):
        c = QrackCircuit(2)
        c.append_gate(GateType.H, [0])
        c.append_gate(GateType.CNOT, [0, 1])
        bc = c.to_qir()
        assert isinstance(bc, bytes)
        assert bc[:4] == b"BC\xc0\xde"  # LLVM bitcode magic

    def test_to_mlir_contains_quantum_dialect(self):
        c = QrackCircuit(1)
        c.append_gate(GateType.H, [0])
        mlir = c.to_mlir()
        assert "quantum." in mlir
        assert "Hadamard" in mlir or "H" in mlir
```

---

## 9. Phase 11 Completion Checklist

```
□ Catalyst headers detected by CMake when present
□ catalyst_device.cpp compiles into _qrackbind_catalyst.so
□ QrackBindCatalystDevice registers via getCustomDevice entry point
□ qml.device("qrackbind.simulator", ...) works under @qjit
□ qml.device("qrackbind.stabilizer_hybrid", ...) works under @qjit
□ qml.device("qrackbind.stabilizer", ...) works under @qjit for Clifford circuits
□ qml.device("qrackbind.stabilizer", ...) rejects non-Clifford ops at compile time
□ Adjoint gradient matches parameter-shift to 1e-4 on RX-CNOT-expval circuit
□ QrackCircuit.to_qir() returns valid LLVM bitcode (magic bytes BC\xc0\xde)
□ QrackCircuit.to_mlir() emits Catalyst quantum dialect
□ state_vector_jax returns a JAX-compatible DLPack array
□ state_vector_cuda round-trips via np.from_dlpack
□ DLPack capsule lifetime survives simulator deletion
□ QrackStabilizer does NOT expose state_vector_jax / state_vector_cuda
□ nb::call_guard<nb::gil_scoped_release> applied to all long-running ops
□ Multi-threaded smoke test: gate calls from a worker thread don't block main GIL
□ qrack.toml declares qjit_compatible = true and adjoint support
□ Wheels build cleanly with and without Catalyst available
□ pyright passes with zero new errors after stub regeneration
□ uv run pytest tests/test_phase1.py … tests/test_phase11_*.py — all green
□ README "Development Phases" table updated with Phase 11 row
```

---

## 10. What Phase 11 Leaves Out (Deferred)

| Item | Reason |
|---|---|
| MLIR-level circuit optimisation passes (gate fusion at QIR layer) | Catalyst's existing passes already cover this; Qrack's `QrackCircuit.optimize()` runs at the higher level |
| Custom Catalyst gradient transforms beyond adjoint | Adjoint covers the dominant case; custom transforms deferred to Phase 12+ |
| `run_batch(params_batch)` for QSVM-style workloads | Phase 8 (PennyLane plugin) extension territory; not Catalyst-specific |
| Stim-format circuit export | Out of scope; QIR is the primary serialisation target |
| Multi-device sharding via Catalyst | Single-device only; multi-device deferred |
| Native CUDA Qrack wheels in CI | Phase 9 wheels remain CPU/OpenCL by default; CUDA wheels are a Phase 12 packaging concern |

---

## Related

- [[qrackbind Project Phase Breakdown]]
- [[qrackbind Phase 6]]
- [[qrackbind Phase 8]]
- [[qrackbind Phase 9]]
- [[qrackbind Phase 10]]
- [[PennyLane Support]]
- [[PennyLane Integration]]
- [[QrackSimulator API Method Categories]]
- [[Framework Plugin Architecture (PennyLane + Qiskit)]]
- [[qrack project/Reference/qinterface.hpp.md]]
