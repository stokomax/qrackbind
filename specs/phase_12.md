---
tags:
  - qrack
  - nanobind
  - python
  - qbdd
  - sdrp
  - pennylane
  - implementation
  - qrackbind
  - phase12
---
## qrackbind Phase 12 — Approximation Knobs, QBDD Engine, and Their PennyLane Devices

Builds on [[qrackbind Phase 8]], [[qrackbind Phase 10]], and [[qrackbind Phase 11]]. The [[PennyLane Use Cases]] analysis ranks three Qrack-distinctive capabilities as top-tier for the active research domains (QML, quantum chemistry, drug discovery, materials science): tunable approximation via SDRP/NCRP, the QBDD engine for structured Hamiltonian simulation, and batched parameter execution for kernel-matrix workloads. Phase 12 delivers the first two; the third ships in [[qrackbind Phase 13]].

For each capability, Phase 12 ships both the **nanobind binding** and the matching **PennyLane device feature**, so researchers using `qml.device(...)` reach the new functionality end-to-end without dropping to the qrackbind core API.

**Prerequisite:** Phase 8 (PennyLane plugin scaffolding), Phase 10 (templated gate helpers, standalone-class pattern), and Phase 11 (Catalyst runtime adapter) shipped.

---

## What Phase 12 Adds

| Capability | Nanobind binding | PennyLane device |
|---|---|---|
| SDRP / NCRP approximation knobs | `set_sdrp(float)`, `set_ncrp(float)` on `QrackSimulator` and `QrackStabilizerHybrid` | `sdrp` / `ncrp` kwargs on `qrackbind.simulator` and `qrackbind.stabilizer_hybrid` |
| QBDD engine | `QrackQBdd` and `QrackQBddHybrid` standalone classes | New `qrackbind.qbdd` and `qrackbind.qbdd_hybrid` device names |

---

## Learning Goals

| Topic | Where it appears |
|---|---|
| Runtime tunable approximation parameters via setter methods | §1 — `set_sdrp`, `set_ncrp`, `approximation` context manager |
| Third standalone engine class via the Phase 10 templated-helper pattern | §2 — `QrackQBdd`, `QrackQBddHybrid` |
| PennyLane device kwarg plumbing through to engine setters | §3.1 — `QrackDevice.__init__` SDRP wiring |
| Adding new short-name devices to an existing plugin | §3.2 — TOML entries for QBDD |
| Catalyst runtime adapter — engine stack and kwargs extension | §4 — `make_simulator_from_kwargs` recognises new options |

---

## File Structure

| File | Changes |
|---|---|
| `bindings/gate_helpers.h` | Add `add_approximation<T>` template — exposes `set_sdrp` / `set_ncrp` |
| `bindings/simulator.cpp` | Wire `add_approximation` into `bind_simulator()` |
| `bindings/stabilizer.cpp` | Wire `add_approximation` into `bind_stabilizer_hybrid_class()` (not `QrackStabilizer`) |
| `bindings/qbdd.cpp` | **New file** — `bind_qbdd()` defines `QrackQBdd` and `QrackQBddHybrid` |
| `bindings/module.cpp` | Add `bind_qbdd()` registration |
| `bindings/catalyst_device.cpp` | Extend `make_simulator_from_kwargs` to recognise QBDD engine stacks and SDRP/NCRP kwargs |
| `src/qrackbind/__init__.py` | Export `QrackQBdd`, `QrackQBddHybrid` |
| `src/qrackbind/_compat.py` | Add `_ApproximationMixin` providing the `approximation()` context manager |
| `src/qrackbind/pennylane/device.py` | New `QrackQBddDevice`, `QrackQBddHybridDevice` classes; SDRP / NCRP kwargs on existing devices |
| `src/qrackbind/pennylane/qrack.toml` | Two new device entries; SDRP / NCRP options declared on existing entries |

---

## 1. SDRP and NCRP — Python Surface

`SetSdrp(float)` and `SetNcrp(float)` are runtime-tunable approximation parameters on `QInterface`. SDRP (Schmidt Decomposition Rounding Parameter) trades fidelity for memory by rounding small singular values during Schmidt decomposition. NCRP (Near-Clifford Rounding Parameter) does the same for near-Clifford operations. Both default to zero (exact simulation).

The [[PennyLane Use Cases]] analysis identifies SDRP as the primary enabler for the QSVM kernel-matrix workload — accepting, say, 99.9% fidelity in exchange for 3× faster simulation makes previously infeasible kernel matrices computable. Phase 12 exposes the knob; Phase 13 adds the batched-execution path that turns the knob into measurable wall-clock savings.

```cpp
// ── File: bindings/gate_helpers.h ────────────────────────────────────────────
template <typename WrapperT>
void add_approximation(nb::class_<WrapperT>& cls) {
    cls
        .def("set_sdrp",
            [](WrapperT& w, float v) { w.sim->SetSdrp(v); },
            nb::arg("value"),
            "Schmidt Decomposition Rounding Parameter — rounds singular\n"
            "values smaller than `value` during Schmidt decomposition.\n"
            "Trades fidelity for memory. Default 0.0 (exact).")
        .def("set_ncrp",
            [](WrapperT& w, float v) { w.sim->SetNcrp(v); },
            nb::arg("value"),
            "Near-Clifford Rounding Parameter — analogous knob for\n"
            "near-Clifford operations. Default 0.0 (exact).");
}
```

The Python-side `QrackSimulator` class (built via `_PyqrackAliasMixin` from Phase 7) gains a context manager:

```python
# src/qrackbind/_compat.py
class _ApproximationMixin:
    @contextmanager
    def approximation(self, *, sdrp: float = 0.0, ncrp: float = 0.0):
        """Temporarily apply SDRP/NCRP approximation; restore on exit."""
        prev_sdrp, prev_ncrp = self._sdrp, self._ncrp
        self.set_sdrp(sdrp); self.set_ncrp(ncrp)
        self._sdrp, self._ncrp = sdrp, ncrp
        try:
            yield self
        finally:
            self.set_sdrp(prev_sdrp); self.set_ncrp(prev_ncrp)
            self._sdrp, self._ncrp = prev_sdrp, prev_ncrp
```

`QrackStabilizer` deliberately does **not** receive `add_approximation` — pure Clifford simulation has no continuous parameters to round.

---

## 2. `QrackQBdd` and `QrackQBddHybrid` — Standalone Engine Classes

QBDD (Quantum Binary Decision Diagrams) is well-suited to highly structured circuits — Heisenberg, Hubbard, and Ising lattice Hamiltonians — and runs only on CPU. The [[PennyLane Use Cases]] analysis flags QBDD as the practical path to materials-science simulation on GPU-less HPC clusters.

The class structure mirrors Phase 10 exactly: a strict variant and a hybrid variant. Engine stack mapping:

```
QrackQBdd        →  [QINTERFACE_BDT]
QrackQBddHybrid  →  [QINTERFACE_BDT_HYBRID, QINTERFACE_HYBRID]   (or _CPU per flags)
```

```cpp
// ── File: bindings/qbdd.cpp ──────────────────────────────────────────────────
struct QrackQBdd {
    QInterfacePtr sim;
    bitLenInt     numQubits;
    /* ...check_qubit, repr — same shape as QrackStab... */
};

struct QrackQBddHybrid {
    QInterfacePtr sim;
    bitLenInt     numQubits;
    /* ... */
};

void bind_qbdd_class(nb::module_& m) {
    auto cls = nb::class_<QrackQBdd>(m, "QrackQBdd",
        "Quantum Binary Decision Diagram simulator. Best suited to structured\n"
        "circuits with deep regularity — lattice Hamiltonians, QFT on basis\n"
        "states, structured oracle queries. CPU-only (no GPU acceleration).");

    add_clifford_gates(cls);
    add_clifford_two_qubit(cls);
    add_t_gates(cls);
    add_rotation_gates(cls);
    add_u_gates(cls);
    add_matrix_gates(cls);
    add_measurement(cls);
    add_pauli_methods(cls);
    add_state_access(cls);    // QBDD supports state vector access
    add_approximation(cls);   // SDRP applies to BDT path

    cls
        .def(nb::init<bitLenInt>(), nb::arg("qubitCount") = 0)
        .def_prop_ro("num_qubits", [](const QrackQBdd& s) { return s.numQubits; })
        /* ...standard methods... */;
}

void bind_qbdd_hybrid_class(nb::module_& m) {
    /* Same gate set; constructor accepts isCpuGpuHybrid (for fallback dense engine
       only — QBDD itself stays on CPU). add_state_access_dlpack from Phase 11
       wires up here too. */
}
```

> **Why standalone classes rather than a `useQBdd=True` flag**: same reasons as Phase 10. Typed contract for documentation and IDE; clean PennyLane device targets; benchmarking surface that doesn't sit under `QUnit`/tensor-network wrapping.

---

## 3. PennyLane Devices — End-to-End Coverage

This is the section that distinguishes Phase 12 from a pure-bindings phase. Each capability above is wired through a corresponding PennyLane device feature so researchers using `qml.device(...)` can reach it from a `@qml.qnode`.

### 3.1 SDRP / NCRP kwargs on existing devices

```python
# src/qrackbind/pennylane/device.py
class QrackDevice(qml.devices.Device):
    def __init__(
        self,
        wires,
        *,
        shots=None,
        is_stabilizer_hybrid=False,
        is_tensor_network=True,
        sdrp: float = 0.0,        # ← Phase 12 addition
        ncrp: float = 0.0,        # ← Phase 12 addition
        **kwargs,
    ):
        super().__init__(wires=wires, shots=shots)
        self._sim = QrackSimulator(
            qubitCount=len(wires),
            isStabilizerHybrid=is_stabilizer_hybrid,
            isTensorNetwork=is_tensor_network,
            **kwargs,
        )
        if sdrp > 0.0: self._sim.set_sdrp(sdrp)
        if ncrp > 0.0: self._sim.set_ncrp(ncrp)
```

Usage:

```python
# Tunable approximation in a research workflow
dev = qml.device("qrackbind.simulator", wires=20, sdrp=0.001)

@qml.qnode(dev)
def kernel_circuit(x, y):
    qml.AngleEmbedding(x, wires=range(20))
    qml.adjoint(qml.AngleEmbedding(y, wires=range(20)))
    return qml.probs(wires=range(20))
```

The same `sdrp` / `ncrp` kwargs also flow into Phase 11's Catalyst runtime via an extension to `make_simulator_from_kwargs` (see §4).

### 3.2 New `qrackbind.qbdd` and `qrackbind.qbdd_hybrid` devices

```python
# src/qrackbind/pennylane/device.py
class QrackQBddDevice(qml.devices.Device):
    """PennyLane device backed by Qrack's QBDD engine (CPU-only).

    Best for structured circuits — lattice Hamiltonians, QFT on basis states,
    Grover with structured oracles. Inherits the standard gate set from
    QrackQBdd; non-Clifford gates are supported."""

    name = "qrackbind.qbdd"

    def __init__(self, wires, *, shots=None, sdrp: float = 0.0, **kwargs):
        super().__init__(wires=wires, shots=shots)
        self._sim = QrackQBdd(qubitCount=len(wires), **kwargs)
        if sdrp > 0.0: self._sim.set_sdrp(sdrp)

    def execute(self, circuits, execution_config=None):
        # Dispatch via the shared gate-name → method-name table from _dispatch.py
        return _execute_circuits(self._sim, circuits)


class QrackQBddHybridDevice(qml.devices.Device):
    name = "qrackbind.qbdd_hybrid"
    # Same shape; constructs QrackQBddHybrid with isCpuGpuHybrid kwarg.
```

The dispatch table (`_dispatch.py` from Phase 8) is unchanged — the gate-name → method-name mapping is identical for `QrackSimulator`, `QrackStabilizerHybrid`, `QrackQBdd`, and `QrackQBddHybrid` because they all expose the same templated gate helpers.

### 3.3 Updated TOML

```toml
# src/qrackbind/pennylane/qrack.toml

[device.qrackbind_simulator]
short_name      = "qrackbind.simulator"
backend_class   = "qrackbind.QrackSimulator"
qjit_compatible = true
supports_derivatives = ["parameter-shift", "adjoint"]
options = ["sdrp", "ncrp", "is_stabilizer_hybrid", "is_tensor_network"]

[device.qrackbind_stabilizer]
short_name      = "qrackbind.stabilizer"
backend_class   = "qrackbind.QrackStabilizer"
qjit_compatible = true
supports_derivatives = []
# No SDRP — pure Clifford has no continuous parameters.

[device.qrackbind_stabilizer_hybrid]
short_name      = "qrackbind.stabilizer_hybrid"
backend_class   = "qrackbind.QrackStabilizerHybrid"
qjit_compatible = true
supports_derivatives = ["parameter-shift", "adjoint"]
options = ["sdrp", "ncrp", "is_cpu_gpu_hybrid"]

[device.qrackbind_qbdd]                              # ← Phase 12 addition
short_name      = "qrackbind.qbdd"
backend_class   = "qrackbind.QrackQBdd"
qjit_compatible = true
supports_derivatives = ["parameter-shift", "adjoint"]
options = ["sdrp"]
gpu_supported   = false

[device.qrackbind_qbdd_hybrid]                       # ← Phase 12 addition
short_name      = "qrackbind.qbdd_hybrid"
backend_class   = "qrackbind.QrackQBddHybrid"
qjit_compatible = true
supports_derivatives = ["parameter-shift", "adjoint"]
options = ["sdrp", "is_cpu_gpu_hybrid"]
```

Phase 13 extends these entries with `batched_execution = true` and `kernel_matrix = true` once the corresponding overrides land.

---

## 4. Catalyst Runtime Adapter — Engine Stack Extension

Phase 11's `make_simulator_from_kwargs` parses the `kwargs` string passed via `getCustomDevice`. Phase 12 extends the parser to recognise the new engine stacks and approximation knobs:

```cpp
// ── File: bindings/catalyst_device.cpp — extend Phase 11 ─────────────────────
QInterfacePtr make_simulator_from_kwargs(const std::string& kwargs) {
    auto opts = parse_kwargs(kwargs);

    std::vector<QInterfaceEngine> stack;
    if      (opts.engine == "qbdd")            stack = {QINTERFACE_BDT};
    else if (opts.engine == "qbdd_hybrid")     stack = {QINTERFACE_BDT_HYBRID, QINTERFACE_HYBRID};
    else if (opts.engine == "stabilizer")      stack = {QINTERFACE_STABILIZER};
    else if (opts.engine == "stabilizer_hybrid") stack = {QINTERFACE_STABILIZER_HYBRID, QINTERFACE_HYBRID};
    else /* simulator */                       stack = default_simulator_stack(opts);

    auto sim = CreateQuantumInterface(stack, opts.qubits, /* ... */);
    if (opts.sdrp > 0.0f) sim->SetSdrp(opts.sdrp);
    if (opts.ncrp > 0.0f) sim->SetNcrp(opts.ncrp);
    return sim;
}
```

This means `@qjit` works against all five Phase 12 device names — `qrackbind.simulator`, `qrackbind.stabilizer`, `qrackbind.stabilizer_hybrid`, `qrackbind.qbdd`, `qrackbind.qbdd_hybrid` — with SDRP/NCRP plumbed through identically to the standard Python path. Both layers respect the same Qrack instance configuration.

---

## 5. Test Suite

```python
# tests/test_phase12_approximation.py
import pytest
from qrackbind import QrackSimulator, QrackStabilizerHybrid, QrackStabilizer


class TestApproximation:
    def test_sdrp_zero_matches_exact(self):
        a = QrackSimulator(qubitCount=4); a.h(0); a.cnot(0, 1); a.rz(0.7, 1)
        b = QrackSimulator(qubitCount=4); b.set_sdrp(0.0)
        b.h(0); b.cnot(0, 1); b.rz(0.7, 1)
        for q in range(4):
            assert a.prob(q) == pytest.approx(b.prob(q), abs=1e-6)

    def test_sdrp_context_manager_restores(self):
        s = QrackSimulator(qubitCount=2)
        with s.approximation(sdrp=0.01):
            assert s._sdrp == 0.01
        assert s._sdrp == 0.0

    def test_pure_stabilizer_has_no_sdrp(self):
        s = QrackStabilizer(qubitCount=2)
        assert not hasattr(s, "set_sdrp")
        assert not hasattr(s, "set_ncrp")


# tests/test_phase12_qbdd.py
import numpy as np
import pytest
from qrackbind import QrackQBdd, QrackQBddHybrid


class TestQBdd:
    def test_construction(self):
        s = QrackQBdd(qubitCount=4)
        assert s.num_qubits == 4

    def test_heisenberg_chain_matches_state_vector(self):
        # 6-site Heisenberg evolution — QBDD vs QrackSimulator probabilities agree
        from qrackbind import QrackSimulator
        a = QrackQBdd(qubitCount=6)
        b = QrackSimulator(qubitCount=6, isTensorNetwork=False)
        for sim in (a, b):
            for q in range(6): sim.h(q)
            for q in range(5):
                sim.cnot(q, q + 1); sim.rz(0.3, q + 1); sim.cnot(q, q + 1)
        for q in range(6):
            assert a.prob(q) == pytest.approx(b.prob(q), abs=1e-4)

    def test_qbdd_hybrid_handles_full_gate_set(self):
        s = QrackQBddHybrid(qubitCount=3)
        s.h(0); s.rx(0.5, 0); s.u(1, 0.1, 0.2, 0.3)
        sv = s.state_vector
        assert sv.shape == (8,)
        assert np.isclose(np.sum(np.abs(sv)**2), 1.0, atol=1e-4)


# tests/test_phase12_pennylane_devices.py
import pytest
pennylane = pytest.importorskip("pennylane")
import pennylane as qml


class TestQBddDevices:
    def test_qbdd_device_loads(self):
        dev = qml.device("qrackbind.qbdd", wires=3)
        assert dev.name == "qrackbind.qbdd"

    def test_qbdd_hybrid_runs_qnode(self):
        dev = qml.device("qrackbind.qbdd_hybrid", wires=2)
        @qml.qnode(dev)
        def c():
            qml.Hadamard(wires=0); qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])
        probs = c()
        assert probs[0] == pytest.approx(0.5, abs=1e-4)
        assert probs[3] == pytest.approx(0.5, abs=1e-4)


class TestSdrpKwarg:
    def test_sdrp_zero_matches_no_kwarg(self):
        a = qml.device("qrackbind.simulator", wires=3)
        b = qml.device("qrackbind.simulator", wires=3, sdrp=0.0)
        @qml.qnode(a)
        def ca(): qml.Hadamard(wires=0); return qml.expval(qml.PauliZ(0))
        @qml.qnode(b)
        def cb(): qml.Hadamard(wires=0); return qml.expval(qml.PauliZ(0))
        assert ca() == pytest.approx(cb(), abs=1e-6)
```

---

## 6. Phase 12 Completion Checklist

```
□ set_sdrp / set_ncrp callable on QrackSimulator and QrackStabilizerHybrid
□ approximation() context manager restores prior values on exit
□ QrackStabilizer does NOT expose set_sdrp / set_ncrp (hasattr False)
□ QrackQBdd and QrackQBddHybrid importable; construct
□ QrackQBdd full gate set works (h, rx, u, mtrx, swap, mcx)
□ QrackQBdd 6-site Heisenberg evolution matches QrackSimulator within 1e-4
□ QrackQBddHybrid.state_vector returns valid normalized vector
□ qml.device("qrackbind.qbdd", wires=N) loads
□ qml.device("qrackbind.qbdd_hybrid", wires=N) loads
□ sdrp / ncrp kwargs accepted by simulator and stabilizer_hybrid devices
□ Catalyst runtime accepts sdrp/ncrp kwargs and qbdd/qbdd_hybrid engine names
□ All five qrackbind.* device names visible to PennyLane via the TOML
□ just stubs regenerates _core.pyi with the new methods and classes
□ pyright passes with zero new errors
□ uv run pytest tests/test_phase1.py … tests/test_phase12_*.py — all green
□ README "Development Phases" table updated with Phase 12 row
```

---

## 7. What Phase 12 Leaves Out (Deferred)

| Item | Reason |
|---|---|
| Batched parameter execution (`run_batch`, `kernel_matrix`) | Delivered in [[qrackbind Phase 13]] alongside the matching `batch_execute` and `compute_kernel_matrix` PennyLane device overrides |
| Distributed multi-node QBDD | Single-machine CPU only; multi-node deferred |
| GPU acceleration of QBDD | QBDD is CPU-only by design — no path forward without a Qrack-side change |
| Adaptive SDRP (auto-tuning the rounding parameter) | Out of scope; researcher dials manually |
| Per-circuit SDRP via `qml.execute` config | PennyLane's `execution_config` pathway is in flux; revisit when the API stabilises |

---

## Related

- [[qrackbind Project Phase Breakdown]]
- [[qrackbind Phase 6]]
- [[qrackbind Phase 8]]
- [[qrackbind Phase 10]]
- [[qrackbind Phase 11]]
- [[qrackbind Phase 13]]
- [[PennyLane Use Cases]]
- [[PennyLane Support]]
- [[QrackSimulator API Method Categories]]
- [[Framework Plugin Architecture (PennyLane + Qiskit)]]
- [[qrack project/Reference/qinterface.hpp.md]]
