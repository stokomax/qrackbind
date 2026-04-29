---
tags:
  - qrack
  - nanobind
  - python
  - batching
  - qsvm
  - kernel-matrix
  - pennylane
  - implementation
  - qrackbind
  - phase13
---
## qrackbind Phase 13 — Batched Parameter Execution and Kernel Matrix

Builds on [[qrackbind Phase 6]], [[qrackbind Phase 8]], and [[qrackbind Phase 12]]. The [[PennyLane Use Cases]] analysis ranks SDRP-tunable approximation, the QBDD engine, and batched parameter execution as the three Qrack-distinctive capabilities most relevant to active research domains. Phase 12 delivered the first two; Phase 13 closes the third — and like Phase 12, ships both the **nanobind binding** and the matching **PennyLane device override** so the capability is reachable through `qml.device(...)`.

The use cases are concrete: a QSVM kernel matrix for 1000 training samples requires ~500K circuit evaluations, parameter-shift gradients on a 50-parameter ansatz issue 101 circuits per gradient step, and barren-plateau diagnostics sweep thousands of random initialisations. Without batching, every one of these incurs a per-call Python boundary cost. With batching, they collapse to a single nanobind handoff with a C++ inner loop.

**Prerequisite:** Phase 6 (`QrackCircuit` with `GateType` enum), Phase 8 (PennyLane plugin scaffolding), and Phase 12 (QBDD devices, SDRP kwargs) shipped.

---

## What Phase 13 Adds

| Capability | Nanobind binding | PennyLane device |
|---|---|---|
| Parameter-batched circuit replay | `QrackSimulator.run_batch(circuit, params)` and same on `QrackStabilizerHybrid`, `QrackQBdd*` | `batch_execute` override detecting parameter-shift / kernel workloads |
| QSVM kernel matrix construction | `QrackSimulator.kernel_matrix(circuit, X1, X2)` and same on hybrid backends | `compute_kernel_matrix` override on `QrackDevice` |
| Parameter-slot circuit recording | `QrackCircuit.append_param_gate(gate, qubits, param_index)` and `QrackCircuit.run_with_params(sim, params)` | (internal — used by both overrides) |

---

## Learning Goals

| Topic | Where it appears |
|---|---|
| C++-loop batched execution with single nanobind handoff | §3 — `run_batch` accepting 2-D `nb::ndarray` |
| Quantum kernel matrix as an O(N²) inner loop in C++ | §4 — `kernel_matrix` with `U(a) U†(b)` structure |
| Parameter slots in the circuit tape | §2 — `append_param_gate` recording placeholder indices |
| `batch_execute` override for parameter-shift gradient batches | §5 — circuit-shape detection and dispatch |
| `compute_kernel_matrix` override short-circuiting `qml.kernels.kernel_matrix` | §6 |
| Catalyst non-applicability | §7 — when `@qjit` is on, batching is the compiler's job, not the runtime's |

---

## File Structure

| File | Changes |
|---|---|
| `bindings/circuit.cpp` | Add `append_param_gate(gate, qubits, param_index)` and `run_with_params(sim, params)` methods to `QrackCircuit` |
| `bindings/gate_helpers.h` | Add `add_batch_execution<T>` template — exposes `run_batch` and `kernel_matrix` |
| `bindings/simulator.cpp` | Wire `add_batch_execution` into `bind_simulator()` |
| `bindings/stabilizer.cpp` | Wire `add_batch_execution` into `bind_stabilizer_hybrid_class()` (not pure `QrackStabilizer`) |
| `bindings/qbdd.cpp` | Wire `add_batch_execution` into both QBDD classes |
| `src/qrackbind/pennylane/_batch.py` | **New file** — `_try_extract_param_batch` and `_to_qrack_circuit` helpers |
| `src/qrackbind/pennylane/device.py` | `execute` method gains the batch detection path; new `compute_kernel_matrix` method |
| `src/qrackbind/pennylane/qrack.toml` | Add `batched_execution = true` and `kernel_matrix = true` to all `qrackbind.*` device entries (except pure stabilizer) |

---

## 1. Where Batching Pays Off — and Where It Doesn't

The batching gain depends on the ratio of boundary-crossing cost to per-circuit simulation cost. The [[PennyLane Support]] analysis is precise about this:

- **Small circuits (<15 qubits), tight loops:** boundary cost dominates; batching delivers 2–5× wall-clock improvement. This covers the majority of QML / VQE prototyping work.
- **Large circuits (>25 qubits):** simulation compute dominates; batching is no faster than a Python loop.
- **Under `@qjit`:** Catalyst already compiles the gradient loop into LLVM IR — batching is irrelevant on that path.

Phase 13's batching target is therefore explicitly the standard Python execution path at small-to-medium qubit counts. This is where research workflows actually run today, and where Phase 13's gains are largest.

---

## 2. Parameter-Slot Circuit Recording

`QrackCircuit` from Phase 6 already records gates as a typed tape. Phase 13 extends it with parameter slots — placeholder indices that get substituted at replay time:

```cpp
// ── File: bindings/circuit.cpp — extend bind_circuit() ───────────────────────
.def("append_param_gate",
    [](QrackCircuit& self, GateType gate,
       const std::vector<bitLenInt>& qubits,
       size_t param_index) {
        self.param_gates.push_back({gate, qubits, param_index});
    },
    nb::arg("gate"), nb::arg("qubits"), nb::arg("param_index"),
    "Record a parameterised gate referencing slot `param_index` of the\n"
    "params array passed to run_with_params() or run_batch().")

.def("run_with_params",
    [](const QrackCircuit& self, /* sim wrapper */,
       nb::ndarray<float, nb::ndim<1>, nb::c_contig> params) {
        const float* p = params.data();
        for (const auto& g : self.gates) {
            if (g.is_param)
                apply_param_gate(sim, g.type, g.qubits, p[g.param_index]);
            else
                apply_static_gate(sim, g.type, g.qubits);
        }
    },
    nb::arg("sim"), nb::arg("params"),
    "Replay this circuit on `sim`, substituting parameter slots with\n"
    "values from the `params` array.")
```

The recorded circuit becomes a template that can be replayed thousands of times with different parameter vectors. `QrackCircuit` retains its existing static `append_gate` method — a circuit can mix concrete and parameterised gates freely.

---

## 3. `run_batch` — The C++ Inner Loop

```cpp
// ── File: bindings/gate_helpers.h ────────────────────────────────────────────
template <typename WrapperT>
void add_batch_execution(nb::class_<WrapperT>& cls) {
    cls.def("run_batch",
        [](WrapperT& w, const QrackCircuit& circuit,
           nb::ndarray<float, nb::ndim<2>, nb::c_contig> params) {
            size_t n_runs = params.shape(0);
            size_t n_params = params.shape(1);
            const float* p = params.data();

            // Output: (n_runs, n_qubits) array of single-qubit Z probabilities.
            // Heap-allocated so the capsule deleter can clean it up.
            auto* probs = new std::vector<float>(n_runs * w.numQubits);

            for (size_t i = 0; i < n_runs; ++i) {
                w.sim->SetPermutation(0);
                circuit.run_with_params_inner(w.sim, p + i * n_params, n_params);
                for (bitLenInt q = 0; q < w.numQubits; ++q)
                    (*probs)[i * w.numQubits + q] = w.sim->Prob(q);
            }

            nb::capsule del(probs, [](void* d) noexcept {
                delete static_cast<std::vector<float>*>(d);
            });
            return nb::ndarray<nb::numpy, float, nb::ndim<2>>(
                probs->data(), {n_runs, (size_t)w.numQubits}, del);
        },
        nb::arg("circuit"), nb::arg("params"),
        nb::call_guard<nb::gil_scoped_release>(),
        "Run `circuit` once for each row of `params`. Returns a\n"
        "(n_runs, n_qubits) array of single-qubit Z probabilities.\n"
        "The C++ inner loop avoids per-call Python boundary overhead.");

    /* kernel_matrix in §4 */
}
```

The template applies to `QrackSimulator`, `QrackStabilizerHybrid`, `QrackQBdd`, and `QrackQBddHybrid`. Pure `QrackStabilizer` does **not** receive batch execution — its restricted API has no continuous parameters to batch over.

---

## 4. `kernel_matrix` — QSVM in One Call

```cpp
cls.def("kernel_matrix",
    [](WrapperT& w, const QrackCircuit& circuit,
       nb::ndarray<float, nb::ndim<2>, nb::c_contig> a,
       nb::ndarray<float, nb::ndim<2>, nb::c_contig> b) {
        size_t n_a = a.shape(0), n_b = b.shape(0);
        size_t n_p = a.shape(1);
        if (b.shape(1) != n_p)
            throw QrackError("kernel_matrix: param dimension mismatch",
                             QrackErrorKind::InvalidArgument);

        auto* K = new std::vector<float>(n_a * n_b);

        // K[i,j] = |<0|U†(b[j]) U(a[i])|0>|^2 — standard fidelity kernel
        for (size_t i = 0; i < n_a; ++i) {
            for (size_t j = 0; j < n_b; ++j) {
                w.sim->SetPermutation(0);
                circuit.run_with_params_inner(w.sim, a.data() + i * n_p, n_p);
                circuit.run_inverse_with_params_inner(w.sim, b.data() + j * n_p, n_p);
                (*K)[i * n_b + j] = w.sim->ProbAll(0);  // |<0|psi>|^2
            }
        }

        nb::capsule del(K, [](void* d) noexcept {
            delete static_cast<std::vector<float>*>(d);
        });
        return nb::ndarray<nb::numpy, float, nb::ndim<2>>(
            K->data(), {n_a, n_b}, del);
    },
    nb::arg("circuit"), nb::arg("a"), nb::arg("b"),
    nb::call_guard<nb::gil_scoped_release>(),
    "Compute a quantum kernel matrix between two parameter-vector sets.\n"
    "K[i,j] = |<0|U†(b[j]) U(a[i])|0>|^2.");
```

`run_inverse_with_params_inner` uses `QrackCircuit.inverse()` from Phase 6 — the gate tape is already invertible, so the QSVM path reuses existing infrastructure rather than rebuilding it.

For 1000 training samples on 20 qubits with SDRP=0.001 (Phase 12), the kernel matrix becomes computable in minutes rather than hours — and that's the workflow the [[PennyLane Use Cases]] analysis identifies as Qrack's highest-value QML application.

---

## 5. PennyLane `batch_execute` Override

PennyLane's gradient transforms generate batches of structurally-identical circuits with shifted parameters. A naive device runs them in a Python loop; the qrackbind device detects the pattern and dispatches through `run_batch`:

```python
# src/qrackbind/pennylane/device.py
class QrackDevice(qml.devices.Device):
    # ...

    def execute(self, circuits, execution_config=None):
        if len(circuits) == 1:
            return _execute_circuits(self._sim, circuits)

        batch = _try_extract_param_batch(circuits)
        if batch is not None:
            template_circuit, params_array = batch
            qcirc = _to_qrack_circuit(template_circuit)
            results = self._sim.run_batch(qcirc, params_array)
            return _format_results(results, circuits)

        # Mixed batch — fall back to per-circuit execution
        return _execute_circuits(self._sim, circuits)
```

```python
# src/qrackbind/pennylane/_batch.py
def _try_extract_param_batch(circuits):
    """Return (template_script, params_2d_array) if all circuits share gate
    structure and differ only in trainable parameters; else None.

    Walk the QuantumScript ops in lockstep across the batch:
      - Gate types and wires must match exactly at each position
      - Static params (non-trainable) must match exactly
      - Trainable params get assigned consecutive slot indices
    """
    if not circuits:
        return None
    first = circuits[0]
    template_ops = list(first.operations)
    n_params = sum(1 for op in template_ops for p in op.data if _is_trainable(p))
    if n_params == 0:
        return None

    params = np.empty((len(circuits), n_params), dtype=np.float32)
    for i, c in enumerate(circuits):
        ops = list(c.operations)
        if len(ops) != len(template_ops):
            return None
        slot = 0
        for op_t, op in zip(template_ops, ops):
            if type(op_t) is not type(op) or op_t.wires != op.wires:
                return None
            for p_t, p in zip(op_t.data, op.data):
                if _is_trainable(p_t):
                    params[i, slot] = float(p); slot += 1
                elif p_t != p:
                    return None
    return _build_template(template_ops), params
```

For a 50-parameter parameter-shift gradient (101 circuits per gradient step), this collapses 101 nanobind calls into 1 with a single C-contiguous parameter array — the case where small-circuit boundary overhead is largest.

---

## 6. PennyLane `compute_kernel_matrix` Override

PennyLane's `qml.kernels.kernel_matrix` is the canonical entry point for QSVM. Without an override it issues N² circuit executions through `device.execute()`. Phase 13 short-circuits this:

```python
class QrackDevice(qml.devices.Device):
    # ...

    def compute_kernel_matrix(self, kernel_circuit, X1, X2):
        """Optional override that PennyLane checks before falling back to
        the generic per-pair execution. Available on QrackSimulator,
        QrackStabilizerHybrid, and both QBDD devices."""
        qcirc = _to_qrack_circuit(kernel_circuit)
        return self._sim.kernel_matrix(qcirc,
                                       np.asarray(X1, dtype=np.float32),
                                       np.asarray(X2, dtype=np.float32))
```

PennyLane's kernel utilities check for this method before falling back to per-pair execution. The pure stabilizer device doesn't get this method — it has no `kernel_matrix` binding underneath.

---

## 7. Catalyst Non-Applicability — and Why That's Fine

Under `@qjit`, Catalyst compiles the entire hybrid program (including the gradient loop and any kernel iteration) down to LLVM IR. There is no Python loop at runtime — the compiler has already inlined the parameter sweep into the IR. `run_batch` and `kernel_matrix` from Phase 13 add no value on the QJIT path because the Python-side overhead they eliminate doesn't exist there.

Phase 13 therefore makes **no changes** to the Catalyst runtime adapter from Phase 11. The capability targets the standard Python execution path explicitly, and the TOML's `batched_execution = true` flag does not affect QJIT compilation. This is the right division of labour: nanobind handles small-circuit Python overhead, Catalyst handles full-program compilation, and they don't compete.

---

## 8. Updated TOML

```toml
# src/qrackbind/pennylane/qrack.toml — Phase 13 modifications

[device.qrackbind_simulator]
# ...existing fields from Phase 12...
batched_execution = true              # ← Phase 13
kernel_matrix     = true              # ← Phase 13

[device.qrackbind_stabilizer]
# Pure Clifford — no continuous parameters, no batched_execution / kernel_matrix.

[device.qrackbind_stabilizer_hybrid]
# ...existing fields...
batched_execution = true              # ← Phase 13
kernel_matrix     = true              # ← Phase 13

[device.qrackbind_qbdd]
# ...existing fields...
batched_execution = true              # ← Phase 13
kernel_matrix     = true              # ← Phase 13

[device.qrackbind_qbdd_hybrid]
# ...existing fields...
batched_execution = true              # ← Phase 13
kernel_matrix     = true              # ← Phase 13
```

---

## 9. Test Suite

```python
# tests/test_phase13_run_batch.py
import math
import numpy as np
import pytest
from qrackbind import QrackSimulator, QrackCircuit, GateType


class TestRunBatch:
    def test_run_batch_matches_python_loop(self):
        circuit = QrackCircuit(2)
        circuit.append_param_gate(GateType.RX, [0], param_index=0)
        circuit.append_gate(GateType.CNOT, [0, 1])

        params = np.random.uniform(0, math.pi, (50, 1)).astype(np.float32)

        sim = QrackSimulator(qubitCount=2)
        batch_results = sim.run_batch(circuit, params)

        loop_results = np.zeros((50, 2), dtype=np.float32)
        for i, p in enumerate(params):
            sim.reset_all(); circuit.run_with_params(sim, p)
            loop_results[i, 0] = sim.prob(0); loop_results[i, 1] = sim.prob(1)

        np.testing.assert_allclose(batch_results, loop_results, atol=1e-4)

    def test_run_batch_releases_gil(self):
        # Smoke test: two threads can run_batch concurrently without deadlock
        import threading
        circuit = QrackCircuit(2)
        circuit.append_param_gate(GateType.RX, [0], param_index=0)
        params = np.random.uniform(0, math.pi, (10, 1)).astype(np.float32)

        results = [None, None]
        def worker(idx):
            sim = QrackSimulator(qubitCount=2)
            results[idx] = sim.run_batch(circuit, params)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for t in threads: t.start()
        for t in threads: t.join(timeout=30)
        assert all(r is not None for r in results)

    def test_pure_stabilizer_has_no_run_batch(self):
        from qrackbind import QrackStabilizer
        s = QrackStabilizer(qubitCount=2)
        assert not hasattr(s, "run_batch")
        assert not hasattr(s, "kernel_matrix")


# tests/test_phase13_kernel_matrix.py
class TestKernelMatrix:
    def test_kernel_matrix_matches_per_pair(self):
        circuit = QrackCircuit(3)
        for q in range(3):
            circuit.append_param_gate(GateType.RY, [q], param_index=q)
        for q in range(2):
            circuit.append_gate(GateType.CNOT, [q, q + 1])

        sim = QrackSimulator(qubitCount=3)
        X = np.random.uniform(0, math.pi, (5, 3)).astype(np.float32)
        K_fast = sim.kernel_matrix(circuit, X, X)

        # Per-pair reference
        K_ref = np.zeros((5, 5), dtype=np.float32)
        for i in range(5):
            for j in range(5):
                sim.reset_all()
                circuit.run_with_params(sim, X[i])
                circuit.run_inverse_with_params(sim, X[j])
                K_ref[i, j] = sim.prob_all(0)

        np.testing.assert_allclose(K_fast, K_ref, atol=1e-3)

    def test_kernel_matrix_diagonal_is_one(self):
        # K[i,i] = |<0|U†(x_i)U(x_i)|0>|^2 = 1
        circuit = QrackCircuit(2)
        circuit.append_param_gate(GateType.RX, [0], param_index=0)
        circuit.append_param_gate(GateType.RY, [1], param_index=1)
        sim = QrackSimulator(qubitCount=2)
        X = np.random.uniform(0, math.pi, (4, 2)).astype(np.float32)
        K = sim.kernel_matrix(circuit, X, X)
        for i in range(4):
            assert K[i, i] == pytest.approx(1.0, abs=1e-3)


# tests/test_phase13_pennylane_overrides.py
import pytest
pennylane = pytest.importorskip("pennylane")
import pennylane as qml


class TestBatchExecuteOverride:
    def test_parameter_shift_uses_batch_path(self, monkeypatch):
        dev = qml.device("qrackbind.simulator", wires=2)
        calls = []
        original = dev._sim.run_batch
        monkeypatch.setattr(dev._sim, "run_batch",
            lambda *a, **kw: calls.append(1) or original(*a, **kw))

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0); qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(1))

        qml.grad(circuit)(0.5)
        assert calls, "run_batch should have been called for parameter-shift"

    def test_mixed_batch_falls_back_to_per_circuit(self):
        # Two circuits with different gate structures should not trigger run_batch.
        # (Implementation: monkeypatch run_batch to raise; verify execution still works.)
        ...


class TestKernelMatrixOverride:
    def test_kernel_matrix_override_matches_default(self):
        dev = qml.device("qrackbind.simulator", wires=4)
        @qml.qnode(dev)
        def kernel(x, y):
            qml.AngleEmbedding(x, wires=range(4))
            qml.adjoint(qml.AngleEmbedding(y, wires=range(4)))
            return qml.probs(wires=range(4))

        X = np.random.uniform(0, math.pi, (5, 4))
        K_fast = qml.kernels.kernel_matrix(X, X, kernel)
        # Compare against a per-pair reference computed with default.qubit
        ref_dev = qml.device("default.qubit", wires=4)
        # ... build reference and assert agreement to ~1e-4 ...
```

---

## 10. Phase 13 Completion Checklist

```
□ QrackCircuit.append_param_gate records parameter slots
□ QrackCircuit.run_with_params replays with parameter substitution
□ QrackCircuit.run_inverse_with_params works for kernel_matrix
□ QrackSimulator.run_batch returns same probs as a Python loop (1e-4)
□ QrackStabilizerHybrid.run_batch works
□ QrackQBdd.run_batch and QrackQBddHybrid.run_batch work
□ QrackStabilizer does NOT expose run_batch (hasattr False)
□ run_batch releases the GIL (concurrent threads don't deadlock)
□ QrackSimulator.kernel_matrix matches per-pair reference (1e-3)
□ kernel_matrix diagonal entries equal 1 within 1e-3
□ batch_execute override fires for parameter-shift gradient batches
□ batch_execute falls back cleanly when circuits have mixed structure
□ compute_kernel_matrix override fires for qml.kernels.kernel_matrix
□ Pure stabilizer device does NOT advertise batched_execution / kernel_matrix
□ TOML updated for the four batching-eligible devices
□ Catalyst runtime adapter UNCHANGED (Phase 11 stays correct)
□ just stubs regenerates _core.pyi with the new methods
□ pyright passes with zero new errors
□ uv run pytest tests/test_phase1.py … tests/test_phase13_*.py — all green
□ README "Development Phases" table updated with Phase 13 row
```

---

## 11. What Phase 13 Leaves Out (Deferred)

| Item | Reason |
|---|---|
| Batching under `@qjit` | Catalyst already compiles the gradient loop — see §7. Not a Phase 13 concern, ever |
| Vectorised observable specification (multiple observables per run) | Single Z-basis prob vector for now; multi-observable batching deferred |
| Batched shot-noise sampling | `run_batch` returns analytic probabilities; finite-shot batching is a Phase 14+ extension |
| Sparse kernel matrices | Dense (N² entries) only; sparse kernel approximations deferred |
| GPU kernel matrix computation | Inherits whatever GPU the underlying simulator uses; no separate GPU kernel-matrix path |
| Auto-tuning batch size for memory pressure | Researcher controls batch size by chunking the input array manually |

---

## Related

- [[qrackbind Project Phase Breakdown]]
- [[qrackbind Phase 6]]
- [[qrackbind Phase 8]]
- [[qrackbind Phase 10]]
- [[qrackbind Phase 11]]
- [[qrackbind Phase 12]]
- [[PennyLane Use Cases]]
- [[PennyLane Support]]
- [[QrackSimulator API Method Categories]]
- [[Framework Plugin Architecture (PennyLane + Qiskit)]]
- [[qrack project/Reference/qinterface.hpp.md]]
