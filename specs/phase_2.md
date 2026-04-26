---
tags:
  - qrack
  - nanobind
  - python
  - implementation
  - qrackbind
  - phase2
---
## qrackbind Phase 2 — Dynamic Allocation, Simulator Registry, QFT, and Arithmetic

Builds directly on [[qrackbind Phase 1 Revised]]. Phase 1 delivered a complete static `QrackSimulator` — fixed qubit count, full gate set, measurement, NumPy state access, Pauli enum, and exceptions. Phase 2 addresses the three items explicitly deferred from Phase 1 and adds the QFT and arithmetic gate layers, which together unlock Bloqade's dynamic simulation mode and the quantum arithmetic algorithms (Shor, Grover oracle construction, quantum addition).

---

## What Phase 1 Deferred

| Deferred item | Why deferred | Phase 2 resolution |
|---|---|---|
| `cloneSid` constructor arg | Requires a global live-simulator registry | **Deprecated — superseded by `QrackSimulator.clone()` / `copy.deepcopy(sim)`**. See §1 for the historical design (no longer implemented). |
| `allocate` / `dispose` | Qubit count changes after construction | §2 — Dynamic allocation binding |
| `pyzxCircuit` / `qiskitCircuit` constructor args | External library dependencies | Still deferred — see §7 |
| Arithmetic gates (`mul`, `div`, `pown`, etc.) | Runtime-guarded by `isTensorNetwork` | §4 — Arithmetic gates with guard |
| QFT / IQFT | Algorithm-tier, not core | §3 — QFT binding |

---

## Nanobind Learning Goals

| Topic | Where it appears |
|---|---|
| Global C++ object registry exposed to Python | §1 — `cloneSid` registry |
| Binding methods that mutate the object's own structure | §2 — `allocate` / `dispose` |
| Runtime-conditional method behaviour and Python-side guards | §4 — arithmetic gate guards |
| `nb::keep_alive` and lifetime annotation | §1 — cloned simulator lifetime |
| Binding overloaded C++ methods | §3 — QFT `(start, length)` vs `(qubits_list)` variants |

---

## File Structure — No New Files Required

All bindings in this phase go into the existing `bindings/simulator.cpp` inside `bind_simulator()`. The registry (§1) adds a small amount of file-scope state to `simulator.cpp`. No new `.cpp` files are needed.

---

## 1. Simulator Registry and `cloneSid` — DEPRECATED

> **Status: DEPRECATED. Do not implement.**
>
> The original Phase 2 plan called for a global simulator registry indexed
> by integer SID, mirroring pyqrack's `cloneSid` constructor parameter.
> qrackbind has since taken a different — and simpler — approach: cloning
> is exposed as a method on the simulator itself, with no registry,
> no SIDs, and no cross-instance global state.
>
> **Use instead:**
>
> ```python
> import copy
> from qrackbind import QrackSimulator
>
> orig = QrackSimulator(qubitCount=4)
> orig.h(0); orig.cnot(0, 1)              # Bell state
>
> branch_a = orig.clone()                 # explicit
> branch_b = copy.deepcopy(orig)          # protocol-driven
> branch_c = copy.copy(orig)              # also a deep clone
> ```
>
> The clone is implemented in `bindings/simulator.cpp` as a copy
> constructor that delegates to `QInterface::Clone()` and is exposed via
> `.def("clone", ...)`, `.def("__copy__", ...)`, and
> `.def("__deepcopy__", ...)`. There is no `sid` property, no
> `cloneSid` keyword argument, and no live-simulator registry. New code
> should not depend on those mechanics. The remainder of §1 below is
> retained as historical record only.

### The problem

`cloneSid` in pyqrack lets you construct a new `QrackSimulator` that starts as an exact copy of an existing one. The C++ side achieves this via `QInterface::Clone()`. The Python side needs a way to look up a live `QrackSim` by integer ID so the constructor can call `Clone()` on it.

### File: `bindings/simulator.cpp` — add above `bind_simulator()`

```cpp
// ── Simulator registry ────────────────────────────────────────────────────────
// Maps integer SID → live QrackSim* for cloneSid support.
// Uses a plain mutex rather than nb::gil_scoped_release because insertions
// happen during Python object construction (GIL is held).
#include <mutex>
#include <unordered_map>

static std::mutex            g_registry_mutex;
static std::atomic<int>      g_next_sid{1};
static std::unordered_map<int, QrackSim*> g_registry;

static int register_sim(QrackSim* s)
{
    int sid = g_next_sid.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    g_registry[sid] = s;
    return sid;
}

static void unregister_sim(int sid)
{
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    g_registry.erase(sid);
}

static QrackSim* lookup_sim(int sid)
{
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    auto it = g_registry.find(sid);
    if (it == g_registry.end())
        throw std::out_of_range(
            "cloneSid: no live simulator with sid=" + std::to_string(sid));
    return it->second;
}
```

### Updated `QrackSim` struct — add `sid` field

```cpp
struct QrackSim {
    QInterfacePtr sim;
    bitLenInt     numQubits;
    SimConfig     config;
    int           sid;       // ← new: unique identity for cloneSid

    QrackSim(bitLenInt n, const SimConfig& cfg)
        : numQubits(n), config(cfg), sim(make_simulator(n, cfg))
    {
        if (!sim) throw std::runtime_error("QrackSimulator: factory returned null");
        sid = register_sim(this);
    }

    // Clone constructor — used by cloneSid
    explicit QrackSim(QrackSim* src)
        : numQubits(src->numQubits)
        , config(src->config)
        , sim(src->sim->Clone())
    {
        if (!sim) throw std::runtime_error("QrackSimulator: Clone() returned null");
        sid = register_sim(this);
    }

    ~QrackSim() { unregister_sim(sid); }

    void check_qubit(bitLenInt q, const char* method) const { /* unchanged */ }
    std::string repr() const {
        return "QrackSimulator(qubits=" + std::to_string(numQubits) +
               ", sid=" + std::to_string(sid) + ")";
    }
};
```

### Updated constructor binding — add `cloneSid` parameter

Add one parameter to the existing `__init__` lambda. When `cloneSid >= 0`, ignore all other construction args and clone instead:

```cpp
.def("__init__",
    [](QrackSim* self,
       int cloneSid,          // ← new: -1 means "don't clone"
       bitLenInt qubitCount,
       bool isTensorNetwork,
       /* … all existing kwargs … */
       real1_f noise)
    {
        if (cloneSid >= 0) {
            // Clone path — ignores all other args
            QrackSim* src = lookup_sim(cloneSid);
            new (self) QrackSim(src);
        } else {
            // Normal construction path — unchanged from Phase 1
            SimConfig cfg{ isTensorNetwork, /* … */ noise > 0.0f };
            new (self) QrackSim(static_cast<bitLenInt>(qubitCount), cfg);
        }
    },
    nb::arg("cloneSid")              = -1,    // ← new
    nb::arg("qubitCount")            = -1,
    nb::arg("isTensorNetwork")       = true,
    /* … all existing nb::arg() lines unchanged … */
    nb::arg("noise")                 = 0.0f,
    "Create a QrackSimulator. Pass cloneSid=other.sid to clone an existing simulator.")

// Expose sid as a read-only property
.def_prop_ro("sid",
    [](const QrackSim& s) { return s.sid; },
    "Unique integer ID for this simulator instance. Pass to cloneSid= to clone.")
```

---

## 2. Dynamic Qubit Allocation — `allocate` and `dispose`

Required for Bloqade's `DynamicMemorySimulator`. Without these, any Bloqade workflow that uses runtime qubit allocation will fail immediately.

`QInterface::Allocate(bitLenInt start, bitLenInt length)` inserts `length` fresh |0⟩ qubits at index `start`, shifting existing qubits upward. `Dispose(start, length)` removes qubits that are already in a separable |0⟩ or |1⟩ state. Both update the qubit count, so `numQubits` on the `QrackSim` struct must be kept in sync.

```cpp
// ── File: bindings/simulator.cpp, inside bind_simulator() ─────────────────────

.def("allocate",
    [](QrackSim& s, bitLenInt start, bitLenInt length) -> bitLenInt {
        bitLenInt offset = s.sim->Allocate(start, length);
        s.numQubits = s.sim->GetQubitCount();   // sync after allocation
        return offset;
    },
    nb::arg("start"), nb::arg("length"),
    "Allocate 'length' new |0> qubits at index 'start'. Returns the start offset.\n"
    "Existing qubits at >= start shift up. Updates num_qubits automatically.\n"
    "Incompatible with isTensorNetwork=True.")

.def("dispose",
    [](QrackSim& s, bitLenInt start, bitLenInt length) {
        s.sim->Dispose(start, length);
        s.numQubits = s.sim->GetQubitCount();   // sync after disposal
    },
    nb::arg("start"), nb::arg("length"),
    "Remove 'length' qubits starting at 'start'. Qubits must be separably |0> or |1>.\n"
    "Updates num_qubits automatically.")

// Convenience: allocate at end (most common case)
.def("allocate_qubits",
    [](QrackSim& s, bitLenInt n) -> bitLenInt {
        bitLenInt offset = s.sim->Allocate(n);   // Allocate(length) appends at end
        s.numQubits = s.sim->GetQubitCount();
        return offset;
    },
    nb::arg("n"),
    "Allocate n new |0> qubits at the end. Returns the index of the first new qubit.")
```

> **Note on `isTensorNetwork`:** Bloqade's docs explicitly state that `DynamicMemorySimulator` is incompatible with `isTensorNetwork=True`. This constraint lives entirely at the Python level — the C++ `Allocate`/`Dispose` methods exist on all `QInterface` implementations, but the tensor network layer doesn't support mid-circuit qubit count changes. The simplest handling is to let the C++ raise naturally if someone misuses it, and document the constraint.

---

## 3. QFT and IQFT

The quantum Fourier transform is algorithm-tier functionality needed for Shor's algorithm, quantum phase estimation, and the QASM2 QFT gate. Two calling conventions exist in `QInterface`: register-based (`start`, `length`) and random-access (`qubits: list[int]`).

```cpp
// ── File: bindings/simulator.cpp, inside bind_simulator() ─────────────────────

.def("qft",
    [](QrackSim& s, bitLenInt start, bitLenInt length, bool trySeparate) {
        s.sim->QFT(start, length, trySeparate);
    },
    nb::arg("start"), nb::arg("length"), nb::arg("try_separate") = false,
    "Quantum Fourier Transform on a contiguous register [start, start+length).\n"
    "try_separate: optimization hint for QUnit — set True if you expect a permutation\n"
    "basis eigenstate result; otherwise leave False.")

.def("iqft",
    [](QrackSim& s, bitLenInt start, bitLenInt length, bool trySeparate) {
        s.sim->IQFT(start, length, trySeparate);
    },
    nb::arg("start"), nb::arg("length"), nb::arg("try_separate") = false,
    "Inverse Quantum Fourier Transform on a contiguous register.")

// Random-access variants (non-contiguous qubit lists)
.def("qftr",
    [](QrackSim& s, std::vector<bitLenInt> qubits, bool trySeparate) {
        s.sim->QFTR(qubits, trySeparate);
    },
    nb::arg("qubits"), nb::arg("try_separate") = false,
    "Quantum Fourier Transform on an arbitrary list of qubit indices.")

.def("iqftr",
    [](QrackSim& s, std::vector<bitLenInt> qubits, bool trySeparate) {
        s.sim->IQFTR(qubits, trySeparate);
    },
    nb::arg("qubits"), nb::arg("try_separate") = false,
    "Inverse Quantum Fourier Transform on an arbitrary list of qubit indices.")
```

---

## 4. Arithmetic Gates

Arithmetic gates are runtime-guarded: `QInterface` raises if `isTensorNetwork=True` is in the simulation stack. The binding layer adds a Python-visible guard that raises `QrackException` with a helpful message before the C++ even sees the call — consistent with pyqrack's own behaviour.

```cpp
// ── File: bindings/simulator.cpp — add helper above bind_simulator() ──────────

static void check_arithmetic(const QrackSim& s, const char* method)
{
    if (s.config.isTensorNetwork)
        throw std::runtime_error(
            std::string(method) + ": isTensorNetwork=True is incompatible with "
            "arithmetic gates. Construct with isTensorNetwork=False.");
}
```

### Integer arithmetic (unsigned, without sign)

```cpp
.def("add",
    [](QrackSim& s, bitCapInt val, bitLenInt start, bitLenInt length) {
        check_arithmetic(s, "add");
        s.sim->INC(val, start, length);
    },
    nb::arg("value"), nb::arg("start"), nb::arg("length"),
    "Add classical integer 'value' to the quantum register [start, start+length).")

.def("sub",
    [](QrackSim& s, bitCapInt val, bitLenInt start, bitLenInt length) {
        check_arithmetic(s, "sub");
        s.sim->DEC(val, start, length);
    },
    nb::arg("value"), nb::arg("start"), nb::arg("length"),
    "Subtract classical integer 'value' from the quantum register.")

.def("mul",
    [](QrackSim& s, bitCapInt toMul, bitCapInt modN,
       bitLenInt inStart, bitLenInt outStart, bitLenInt length) {
        check_arithmetic(s, "mul");
        s.sim->MULModNOut(toMul, modN, inStart, outStart, length);
    },
    nb::arg("to_mul"), nb::arg("mod_n"),
    nb::arg("in_start"), nb::arg("out_start"), nb::arg("length"),
    "Modular multiplication: out = in * to_mul mod mod_n (out of place).")

.def("div",
    [](QrackSim& s, bitCapInt toDiv, bitCapInt modN,
       bitLenInt inStart, bitLenInt outStart, bitLenInt length) {
        check_arithmetic(s, "div");
        s.sim->IMULModNOut(toDiv, modN, inStart, outStart, length);
    },
    nb::arg("to_div"), nb::arg("mod_n"),
    nb::arg("in_start"), nb::arg("out_start"), nb::arg("length"),
    "Inverse modular multiplication (modular division, out of place).")

.def("pown",
    [](QrackSim& s, bitCapInt base, bitCapInt modN,
       bitLenInt inStart, bitLenInt outStart, bitLenInt length) {
        check_arithmetic(s, "pown");
        // pown: out = base^in mod modN — requires CMULModNOut with controlled input
        // pyqrack implements this as a controlled modular multiplication chain.
        // Bind via CMULModNOut with an empty controls list for the unconditional case.
        s.sim->CMULModNOut(base, modN, inStart, outStart, length, {});
    },
    nb::arg("base"), nb::arg("mod_n"),
    nb::arg("in_start"), nb::arg("out_start"), nb::arg("length"),
    "Modular exponentiation: out = base^in mod mod_n (out of place). "
    "Central operation of Shor's algorithm.")

// Controlled variants
.def("mcmul",
    [](QrackSim& s, bitCapInt toMul, bitCapInt modN,
       bitLenInt inStart, bitLenInt outStart, bitLenInt length,
       std::vector<bitLenInt> controls) {
        check_arithmetic(s, "mcmul");
        s.sim->CMULModNOut(toMul, modN, inStart, outStart, length, controls);
    },
    nb::arg("to_mul"), nb::arg("mod_n"),
    nb::arg("in_start"), nb::arg("out_start"), nb::arg("length"),
    nb::arg("controls"),
    "Controlled modular multiplication.")

.def("mcdiv",
    [](QrackSim& s, bitCapInt toDiv, bitCapInt modN,
       bitLenInt inStart, bitLenInt outStart, bitLenInt length,
       std::vector<bitLenInt> controls) {
        check_arithmetic(s, "mcdiv");
        s.sim->CIMULModNOut(toDiv, modN, inStart, outStart, length, controls);
    },
    nb::arg("to_div"), nb::arg("mod_n"),
    nb::arg("in_start"), nb::arg("out_start"), nb::arg("length"),
    nb::arg("controls"),
    "Controlled modular division (inverse modular multiplication).")
```

### Shift and rotate

```cpp
.def("lsl",
    [](QrackSim& s, bitLenInt shift, bitLenInt start, bitLenInt length) {
        s.sim->LSL(shift, start, length); },
    nb::arg("shift"), nb::arg("start"), nb::arg("length"),
    "Logical shift left — fills vacated bits with |0>.")

.def("lsr",
    [](QrackSim& s, bitLenInt shift, bitLenInt start, bitLenInt length) {
        s.sim->LSR(shift, start, length); },
    nb::arg("shift"), nb::arg("start"), nb::arg("length"),
    "Logical shift right — fills vacated bits with |0>.")

.def("rol",
    [](QrackSim& s, bitLenInt shift, bitLenInt start, bitLenInt length) {
        s.sim->ROL(shift, start, length); },
    nb::arg("shift"), nb::arg("start"), nb::arg("length"),
    "Circular rotate left.")

.def("ror",
    [](QrackSim& s, bitLenInt shift, bitLenInt start, bitLenInt length) {
        s.sim->ROR(shift, start, length); },
    nb::arg("shift"), nb::arg("start"), nb::arg("length"),
    "Circular rotate right.")
```

---

## 5. Register Measurement

Multi-qubit measurement returning a classical integer rather than a list of bools. Required for quantum arithmetic result readout.

```cpp
.def("measure_shots",
    [](QrackSim& s, std::vector<bitLenInt> qubits, unsigned shots)
        -> std::map<bitCapInt, int>
    {
        std::vector<bitCapInt> qpowers;
        qpowers.reserve(qubits.size());
        for (auto q : qubits) {
            s.check_qubit(q, "measure_shots");
            qpowers.push_back(bitCapInt(1) << q);
        }
        return s.sim->MultiShotMeasureMask(qpowers, shots);
    },
    nb::arg("qubits"), nb::arg("shots"),
    "Sample 'shots' measurements of 'qubits' without collapsing state.\n"
    "Returns dict[int, int]: measurement result → count.")

.def("m_reg",
    [](QrackSim& s, bitLenInt start, bitLenInt length) -> bitCapInt {
        return s.sim->MReg(start, length);
    },
    nb::arg("start"), nb::arg("length"),
    "Measure a contiguous register of 'length' qubits starting at 'start'.\n"
    "Collapses state. Returns result as a classical integer.")
```

---

## 6. Updated `__init__.py` Exports

No new Python files are needed. Document the new Phase 2 surface in `__init__.py`'s module docstring:

```python
# src/qrackbind/__init__.py  (module docstring addition only)
"""
qrackbind — nanobind-based Qrack quantum simulator binding.

New in Phase 2:
  sim.clone()                      — independent deep copy of this simulator
                                     (also via copy.copy / copy.deepcopy)
  sim.allocate(start, length)      — allocate qubits at runtime
  sim.dispose(start, length)       — release qubits
  sim.allocate_qubits(n)           — append n qubits at the end
  sim.qft(start, length)           — quantum Fourier transform
  sim.iqft(start, length)          — inverse QFT
  sim.qftr(qubits)                 — random-access QFT
  sim.add / sub / mul / div / pown — arithmetic gates (isTensorNetwork=False only)
  sim.mcmul / mcdiv                — controlled arithmetic
  sim.lsl / lsr / rol / ror        — shift and rotate
  sim.measure_shots(qubits, shots) — multi-shot sampling
  sim.m_reg(start, length)         — register measurement → int
"""
```

---

## 7. Still Deferred — Not Phase 2

| Item | Reason still deferred |
|---|---|
| `pyzxCircuit` constructor arg | Requires PyZX as a dependency; out of scope until Phase 6 (QrackCircuit) |
| `qiskitCircuit` constructor arg | Requires Qiskit as a dependency; out of scope until Qiskit integration phase |
| `try_separate` / `set_sdrp` / simulator tuning params | Low priority — advanced tuning methods; defer to a quality-of-life phase |
| `add_with_carry` / `sub_with_carry` (`INCC`/`DECC`) | Carry bit arithmetic; low demand — defer to arithmetic expansion phase |
| Hamiltonian time evolution (`time_evolve`) | High complexity, niche use — defer |

---

## 8. Test Suite

Tests are organised by **functional area**, not by phase. Phase 2 work
extends the existing functional test files rather than introducing a
single `test_phase2.py`. The mapping:

| Functional area | Test file | Phase 2 contributions |
|---|---|---|
| Construction, configuration, cloning | `tests/test_constructor.py` | `clone()`, `__copy__`, `__deepcopy__` (replaces the deprecated `cloneSid` design) |
| Dynamic qubit allocation | `tests/test_dynamic_alloc.py` | `allocate`, `dispose`, `allocate_qubits`, tensor-network compatibility |
| Measurement & register I/O | `tests/test_measurement.py` | `measure_shots`, `m_reg`, `set_permutation` |
| Arithmetic, shifts, QFT | `tests/test_arithmetic.py` | `add`/`sub`/`mul`/`div`/`pown`/`mcmul`/`mcdiv`/`mcpown`, `lsl`/`lsr`/`rol`/`ror`, `qft`/`iqft`/`qftr`/`iqftr`, `isTensorNetwork` runtime guards |

Shared fixtures live in `tests/conftest.py`:

```python
# tests/conftest.py
@pytest.fixture
def arith_sim():
    """Factory: build a simulator stack compatible with arithmetic gates.

    isTensorNetwork=False is required (runtime-guarded). isSchmidtDecompose
    MUST stay enabled — QUnit implements QAlu (used by pown/mcpown) and is
    what propagates single-qubit gates correctly down to QPager.
    """
    def _make(n: int) -> QrackSimulator:
        return QrackSimulator(qubitCount=n, isTensorNetwork=False)
    return _make

@pytest.fixture
def dyn_sim():
    """Factory: build a simulator that supports allocate / dispose."""
    def _make(n: int) -> QrackSimulator:
        return QrackSimulator(qubitCount=n, isTensorNetwork=False)
    return _make
```

Representative tests (one per area; the actual files contain more
exhaustive coverage):

```python
# tests/test_constructor.py — cloning
def test_clone_preserves_state():
    orig = QrackSimulator(qubitCount=2)
    orig.h(0); orig.cnot(0, 1)
    clone = orig.clone()
    assert clone.num_qubits == orig.num_qubits

def test_clone_is_independent():
    orig = QrackSimulator(qubitCount=1)
    orig.x(0)
    clone = orig.clone()
    clone.x(0)  # flip clone back to |0>
    assert orig.prob(0) == pytest.approx(1.0, abs=1e-5)
    assert clone.prob(0) == pytest.approx(0.0, abs=1e-5)

def test_deepcopy_protocol():
    import copy
    orig = QrackSimulator(qubitCount=2)
    clone = copy.deepcopy(orig)
    assert clone.num_qubits == orig.num_qubits
```

```python
# tests/test_dynamic_alloc.py — allocate / dispose
def test_allocate_qubits_increases_count(dyn_sim):
    sim = dyn_sim(2)
    sim.allocate_qubits(3)
    assert sim.num_qubits == 5

def test_dispose_decreases_count(dyn_sim):
    sim = dyn_sim(3)
    sim.dispose(2, 1)
    assert sim.num_qubits == 2

def test_allocate_with_tensor_network_succeeds():
    # Qrack supports Allocate() on the tensor-network stack.
    sim = QrackSimulator(qubitCount=2, isTensorNetwork=True)
    sim.allocate_qubits(1)
    assert sim.num_qubits == 3
```

```python
# tests/test_measurement.py — register I/O
def test_m_reg_after_set_permutation():
    sim = QrackSimulator(qubitCount=4)
    sim.set_permutation(7)
    assert int(sim.m_reg(0, 4)) == 7

def test_measure_shots_returns_dict():
    sim = QrackSimulator(qubitCount=2)
    sim.h(0)
    results = sim.measure_shots([0], 200)
    assert sum(results.values()) == 200
```

```python
# tests/test_arithmetic.py — arithmetic, shifts, QFT
def test_add_to_zero_register(arith_sim):
    sim = arith_sim(4)
    sim.add(3, 0, 4)
    assert int(sim.m_reg(0, 4)) == 3

def test_arithmetic_raises_with_tensor_network():
    sim = QrackSimulator(qubitCount=4, isTensorNetwork=True)
    with pytest.raises(Exception, match="isTensorNetwork"):
        sim.add(1, 0, 4)

def test_pown_smoke_executes_on_qalu_capable_stack(arith_sim):
    sim = arith_sim(8)
    sim.set_permutation(2)
    sim.pown(3, 11, 0, 4, 4)        # 3^2 mod 11 = 9
    assert int(sim.m_reg(4, 4)) == 9

@pytest.mark.xfail(strict=True, reason="Qrack CPOWModNOut bug on QUnit→QPager→QHybrid stack")
def test_mcpown_with_active_control_applies(arith_sim):
    sim = arith_sim(9)
    sim.set_permutation((2 << 1) | 1)
    sim.mcpown(3, 11, 1, 5, 4, [0])
    assert int(sim.m_reg(5, 4)) == 9

def test_qft_iqft_round_trip_basis_state():
    sim = QrackSimulator(qubitCount=3, isTensorNetwork=False)
    sim.x(0)
    sim.qft(0, 3); sim.iqft(0, 3)
    assert sim.prob(0) == pytest.approx(1.0, abs=1e-4)
```

---

## 9. Phase 2 Completion Checklist

```
□ sim.clone() / copy.deepcopy(sim) returns an independent simulator
□ Cloned simulator preserves quantum state and configuration
□ Cloned simulator is independent — mutations don't propagate back
□ sim.allocate_qubits(n) increases num_qubits
□ sim.allocate(start, length) inserts at correct index
□ sim.dispose(start, length) decreases num_qubits
□ sim.qft(start, length) and sim.iqft(start, length) round-trip correctly
□ sim.qftr(qubits) and sim.iqftr(qubits) round-trip correctly
□ sim.add(val, start, length) increments register correctly
□ sim.sub(val, start, length) decrements register correctly
□ sim.mul / div / pown execute without error (isTensorNetwork=False)
□ All arithmetic raises QrackException when isTensorNetwork=True
□ sim.lsl / lsr / rol / ror produce correct register values
□ sim.mcmul / mcdiv execute without error with empty and non-empty controls
□ sim.measure_shots(qubits, shots) returns dict summing to shots
□ sim.m_reg(start, length) returns correct classical integer
□ uv run pytest — all green (199 passed, 2 skipped, 1 xfailed as of 2026-04-26)
□ nanobind-stubgen produces .pyi with all new signatures
```

---

## Related

- [[qrackbind Phase 1 Revised]]
- [[qrackbind Project Phase Breakdown]]
- [[qrackbind Compatibility Review — April 2026]]
- [[QuEra Bloqade — pyqrack Dependency Analysis]]
- [[QrackSimulator API Method Categories]]
- [[qrack project/Reference/qinterface.hpp.md]]
