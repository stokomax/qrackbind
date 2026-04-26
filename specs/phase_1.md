---
tags:
  - qrack
  - nanobind
  - python
  - implementation
  - qrackbind
  - phase1
---
## qrackbind Phase 1 Revised — Complete QrackSimulator Core

Replaces the earlier Phase 1 Detailed note. Expanded scope reflects the revised plan: because the shared dispatch table (Phase 2) and PennyLane/Qiskit plugins (Phases 3–4) build directly on Phase 1, Phase 1 must now deliver a complete, usable `QrackSimulator` — not just basic gates.

---

## What Changed From the Previous Phase 1

| Area | Previous Phase 1 | Revised Phase 1 |
|---|---|---|
| Constructor | `QINTERFACE_OPTIMAL`, no flags | All pyqrack flags mapped to factory |
| Controlled gates | Phase 2 | Phase 1 |
| `state_vector` ndarray | Phase 3 | Phase 1 |
| `Pauli` enum | Phase 4 | Phase 1 |
| Exception handling | Phase 5 | Phase 1 |
| Deprecated aliases | Later / compat module | On the class from day one |
| Context manager | Mentioned, not built | Phase 1 |
| `sx`, `sxdg` gates | Missing | Phase 1 (needed for Qiskit basis) |

---

## File Structure

```
qrackbind/
├── pyproject.toml
├── CMakeLists.txt
├── justfile
├── src/
│   └── qrackbind/
│       ├── __init__.py
│       └── _compat.py          ← deprecated alias mixin (pure Python)
├── bindings/                   ← all C++ source files live here
│   ├── binding_core.h          ← shared nanobind header (include first everywhere)
│   ├── module.cpp              ← NB_MODULE entry point
│   ├── simulator.cpp           ← QrackSimulator binding
│   └── pauli.cpp               ← Pauli enum binding
└── tests/
    └── test_phase1.py
```

---

## C++ Types Reference

```cpp
typedef uint32_t bitLenInt;           // qubit index
typedef uint64_t bitCapInt;           // basis state index (e.g. for set_permutation)
typedef float    real1_f;             // rotation angle / probability
typedef std::complex<float> complex;  // amplitude
```

---

## 1. Constructor — Mapping pyqrack Flags to Qrack Factory

**File:** `bindings/simulator.cpp` — place `SimConfig`, `make_simulator`, and `QrackSim` above `bind_simulator()`. No new files required.

This is the largest gap in the previous note. The pyqrack flags select which layers of Qrack's simulation stack to use. The C++ factory function is `CreateQuantumInterface` from `qfactory.hpp`.

### Flag → Layer mapping

```
isTensorNetwork=True    →  QINTERFACE_TENSOR_NETWORK  (outermost)
isSchmidtDecompose=True →  QINTERFACE_QUNIT            (Schmidt decomp layer)
isStabilizerHybrid=True →  QINTERFACE_STABILIZER_HYBRID
isBinaryDecisionTree=T  →  QINTERFACE_BDT
isPaged=True            →  QINTERFACE_QPAGER
isCpuGpuHybrid=True     →  QINTERFACE_HYBRID           (CPU↔GPU auto)
isOpenCL=True           →  QINTERFACE_OPENCL
isOpenCL=False          →  QINTERFACE_CPU
isHostPointer=True      →  useHostMem=true parameter
isSparse=True           →  useSparseStateVec=true parameter
```

### Default stack (pyqrack defaults)

The default pyqrack constructor (`isTensorNetwork=True`, `isSchmidtDecompose=True`, `isPaged=True`, `isCpuGpuHybrid=True`, `isOpenCL=True`) produces:

```
QINTERFACE_TENSOR_NETWORK
    └─ QINTERFACE_QUNIT
           └─ QINTERFACE_STABILIZER_HYBRID  (if isStabilizerHybrid)
                  └─ QINTERFACE_QPAGER
                         └─ QINTERFACE_HYBRID / QINTERFACE_OPENCL
```

### Implementation strategy

Build a helper that constructs the layer vector, then call `CreateQuantumInterface`:

```cpp
#include "qfactory.hpp"

struct SimConfig {
    bool isTensorNetwork     = true;
    bool isSchmidtDecompose  = true;
    bool isSchmidtDecomposeMulti = false;
    bool isStabilizerHybrid  = false;
    bool isBinaryDecisionTree = false;
    bool isPaged             = true;
    bool isCpuGpuHybrid      = true;
    bool isOpenCL            = true;
    bool isHostPointer       = false;
    bool isSparse            = false;
    bool isNoise             = false;
};

QInterfacePtr make_simulator(bitLenInt n, const SimConfig& c) {
    std::vector<QInterfaceEngine> stack;

    if (c.isTensorNetwork)
        stack.push_back(QINTERFACE_TENSOR_NETWORK);

    if (c.isSchmidtDecompose)
        stack.push_back(QINTERFACE_QUNIT);

    if (c.isStabilizerHybrid)
        stack.push_back(QINTERFACE_STABILIZER_HYBRID);

    if (c.isBinaryDecisionTree) {
        stack.push_back(QINTERFACE_BDT);
    } else if (c.isPaged) {
        stack.push_back(QINTERFACE_QPAGER);
        if (c.isCpuGpuHybrid && c.isOpenCL)
            stack.push_back(QINTERFACE_HYBRID);
        else if (c.isOpenCL)
            stack.push_back(QINTERFACE_OPENCL);
        else
            stack.push_back(QINTERFACE_CPU);
    }

    if (stack.empty())
        stack.push_back(QINTERFACE_OPTIMAL);

    return CreateQuantumInterface(
        stack,
        n,
        /*initState=*/  0,
        /*rgp=*/        nullptr,
        /*phaseFac=*/   CMPLX_DEFAULT_ARG,
        /*doNorm=*/     false,
        /*randomGP=*/   true,
        /*useHostMem=*/ c.isHostPointer,
        /*deviceId=*/   -1,
        /*useHWRNG=*/   true,
        /*isSparse=*/   c.isSparse
    );
}
```

### The QrackSim struct

```cpp
struct QrackSim {
    QInterfacePtr sim;
    bitLenInt     numQubits;
    SimConfig     config;

    QrackSim(bitLenInt n, const SimConfig& cfg)
        : numQubits(n)
        , config(cfg)
        , sim(make_simulator(n, cfg))
    {
        if (!sim)
            throw std::runtime_error("QrackSimulator: factory returned null");
    }

    void check_qubit(bitLenInt q, const char* method) const {
        if (q >= numQubits)
            throw std::out_of_range(
                std::string(method) + ": qubit " + std::to_string(q) +
                " out of range [0, " + std::to_string(numQubits - 1) + "]");
    }

    std::string repr() const {
        return "QrackSimulator(qubits=" + std::to_string(numQubits) + ")";
    }
};
```

### The nanobind constructor binding

**File:** `bindings/simulator.cpp` — inside `bind_simulator()`, replacing the starter `nb::init<bitLenInt>()` call.

```cpp
nb::class_<QrackSim>(m, "QrackSimulator", "Qrack quantum simulator.")
    .def("__init__",
        [](QrackSim* self,
           bitLenInt qubitCount,
           bool isTensorNetwork,
           bool isSchmidtDecompose,
           bool isSchmidtDecomposeMulti,
           bool isStabilizerHybrid,
           bool isBinaryDecisionTree,
           bool isPaged,
           bool isCpuGpuHybrid,
           bool isOpenCL,
           bool isHostPointer,
           bool isSparse,
           real1_f noise)
        {
            SimConfig cfg{
                isTensorNetwork, isSchmidtDecompose, isSchmidtDecomposeMulti,
                isStabilizerHybrid, isBinaryDecisionTree,
                isPaged, isCpuGpuHybrid, isOpenCL, isHostPointer, isSparse,
                noise > 0.0f
            };
            new (self) QrackSim(qubitCount, cfg);
        },
        nb::arg("qubitCount")             = -1,
        nb::arg("isTensorNetwork")        = true,
        nb::arg("isSchmidtDecompose")     = true,
        nb::arg("isSchmidtDecomposeMulti")= false,
        nb::arg("isStabilizerHybrid")     = false,
        nb::arg("isBinaryDecisionTree")   = false,
        nb::arg("isPaged")                = true,
        nb::arg("isCpuGpuHybrid")         = true,
        nb::arg("isOpenCL")               = true,
        nb::arg("isHostPointer")          = false,
        nb::arg("isSparse")               = false,
        nb::arg("noise")                  = 0.0f,
        "Create a QrackSimulator. All keyword arguments match pyqrack's "
        "QrackSimulator constructor exactly.")
```

**Note:** `qubitCount=-1` means 0 qubits (matching pyqrack's default). `cloneSid` is deferred — it requires a lookup table of live simulator IDs.

**Deferred constructor kwargs — not in Phase 1:**
- `pyzxCircuit` — constructs and immediately runs a PyZX circuit; requires `pyzx` as a dependency and a circuit execution path not yet implemented
- `qiskitCircuit` — constructs and runs a Qiskit `QuantumCircuit`; requires `qiskit` as a dependency
- `cloneSid` — clones an existing simulator by ID; requires a global registry of live `QrackSim` instances

**Dynamic qubit allocation gap:** pyqrack supports dynamic qubit allocation at runtime via `Allocate`/`Dispose`. Bloqade's `DynamicMemorySimulator` mode depends on this. `QInterface::Allocate(start, length)` and `Dispose(start, length)` are bindable but are not included in Phase 1. Bloqade's `DynamicMemorySimulator` will not work until these are exposed in a later phase.

---

## 2. Single-Qubit Gates — Full Set

**File:** `bindings/simulator.cpp` — all `.def()` calls go inside `bind_simulator()`, after the constructor binding. No new files required.

### Clifford gates

```cpp
// Helper macro to reduce boilerplate
#define GATE1(pyname, cppfn, doc) \
    .def(pyname, [](QrackSim& s, bitLenInt q) { \
        s.check_qubit(q, pyname); s.sim->cppfn(q); }, \
        nb::arg("qubit"), doc)

GATE1("h",        H,      "Hadamard gate.")
GATE1("x",        X,      "Pauli X (bit flip) gate.")
GATE1("y",        Y,      "Pauli Y gate.")
GATE1("z",        Z,      "Pauli Z (phase flip) gate.")
GATE1("s",        S,      "S gate — phase shift π/2.")
GATE1("t",        T,      "T gate — phase shift π/4.")
GATE1("sdg",      IS,     "S† (inverse S) gate.")
GATE1("tdg",      IT,     "T† (inverse T) gate.")
GATE1("sx",       SqrtX,  "√X gate (half-X). Native Qiskit basis gate.")
GATE1("sxdg",     ISqrtX, "√X† gate (inverse √X).")
```

> **C++ name note:** `QInterface` uses `SqrtX` and `ISqrtX` — not `SX`/`ISX`. The Python-side names `sx`/`sxdg` are preserved to match pyqrack and Qiskit conventions.

**Note:** `sx`/`sxdg` are included because Qiskit's default transpiler basis uses `[sx, rz, cx]`. Without `sx`, the Qiskit plugin would need a longer decomposition chain.

### Rotation gates

```cpp
#define RGATE(pyname, cppfn, doc) \
    .def(pyname, [](QrackSim& s, real1_f angle, bitLenInt q) { \
        s.check_qubit(q, pyname); s.sim->cppfn(angle, q); }, \
        nb::arg("angle"), nb::arg("qubit"), doc)

RGATE("rx",  RX, "Rotate around X axis by angle radians. Equiv: exp(-i·angle/2·X).")
RGATE("ry",  RY, "Rotate around Y axis by angle radians.")
RGATE("rz",  RZ, "Rotate around Z axis by angle radians.")
RGATE("r1",  RT, "Phase rotation: exp(-i·angle/2) around |1> state. C++ name: RT.")
```

### U gates (PennyLane / QASM3 standard)

```cpp
.def("u",
    [](QrackSim& s, real1_f theta, real1_f phi, real1_f lam, bitLenInt q) {
        s.check_qubit(q, "u");
        s.sim->U(q, theta, phi, lam);
    },
    nb::arg("theta"), nb::arg("phi"), nb::arg("lam"), nb::arg("qubit"),
    "General single-qubit unitary: U(θ,φ,λ). Decomposes to RZ·RY·RZ.")

.def("u2",
    [](QrackSim& s, real1_f phi, real1_f lam, bitLenInt q) {
        s.check_qubit(q, "u2");
        s.sim->U(q, M_PI / 2.0f, phi, lam);
    },
    nb::arg("phi"), nb::arg("lam"), nb::arg("qubit"),
    "U2 gate: U(π/2, φ, λ).")
```

---

## 3. Two-Qubit and Controlled Gates

**File:** `bindings/simulator.cpp` — continue inside `bind_simulator()`. The `#include <nanobind/stl/vector.h>` shown here should already be present in `bindings/binding_core.h`; do not add it to `simulator.cpp` directly.

These were Phase 2 in the previous plan but are required in Phase 1 for the dispatch table.

```cpp
// Two-qubit convenience gates
.def("cnot",
    [](QrackSim& s, bitLenInt ctrl, bitLenInt tgt) {
        s.check_qubit(ctrl, "cnot"); s.check_qubit(tgt, "cnot");
        s.sim->CNOT(ctrl, tgt);
    },
    nb::arg("control"), nb::arg("target"),
    "Controlled-NOT (CNOT / CX) gate.")

.def("cy",
    [](QrackSim& s, bitLenInt ctrl, bitLenInt tgt) {
        s.check_qubit(ctrl, "cy"); s.check_qubit(tgt, "cy");
        s.sim->CY(ctrl, tgt);
    },
    nb::arg("control"), nb::arg("target"), "Controlled-Y gate.")

.def("cz",
    [](QrackSim& s, bitLenInt ctrl, bitLenInt tgt) {
        s.check_qubit(ctrl, "cz"); s.check_qubit(tgt, "cz");
        s.sim->CZ(ctrl, tgt);
    },
    nb::arg("control"), nb::arg("target"), "Controlled-Z gate.")

.def("swap",
    [](QrackSim& s, bitLenInt q1, bitLenInt q2) {
        s.check_qubit(q1, "swap"); s.check_qubit(q2, "swap");
        s.sim->Swap(q1, q2);
    },
    nb::arg("qubit1"), nb::arg("qubit2"), "SWAP gate.")

.def("iswap",
    [](QrackSim& s, bitLenInt q1, bitLenInt q2) {
        s.check_qubit(q1, "iswap"); s.check_qubit(q2, "iswap");
        s.sim->ISwap(q1, q2);
    },
    nb::arg("qubit1"), nb::arg("qubit2"), "iSWAP gate.")

.def("ccnot",
    [](QrackSim& s, bitLenInt c1, bitLenInt c2, bitLenInt tgt) {
        s.sim->CCNOT(c1, c2, tgt);
    },
    nb::arg("control1"), nb::arg("control2"), nb::arg("target"),
    "Toffoli (CCX / CCNOT) gate.")
```

### Multiply-controlled gates (vector control)

> **Critical finding from `qinterface.hpp`:** There are **no** `MCX`, `MCY`, `MCZ`, `MCH`, `MCRZ`, `MACX`, `MACY`, `MACZ` methods on `QInterface`. The multi-control primitive API is `MCInvert`, `MACInvert`, `MCPhase`, `MACPhase`, and `MCMtrx`/`MACMtrx`. All named multi-control gates are composed from these. All take `const std::vector<bitLenInt>&` — not a `(size, ptr)` pair.

The pyqrack-compatible Python names (`mcx`, `macy`, etc.) are implemented by delegating to the correct primitive:

```cpp
// ── Constants needed for multi-control gates ─────────────────────────────────
// Place these at file scope in simulator.cpp, above bind_simulator()
static const Qrack::complex ONE_C  = Qrack::complex(1.0f, 0.0f);
static const Qrack::complex NEG1_C = Qrack::complex(-1.0f, 0.0f);
static const Qrack::complex I_C    = Qrack::complex(0.0f, 1.0f);
static const Qrack::complex NEG_I_C= Qrack::complex(0.0f, -1.0f);
static const Qrack::real1 SQRT1_2  = (Qrack::real1)std::sqrt(0.5f);

// H gate matrix: [1/√2, 1/√2, 1/√2, -1/√2]
static const Qrack::complex H_MTRX[4] = {
    {SQRT1_2, 0}, {SQRT1_2, 0}, {SQRT1_2, 0}, {-SQRT1_2, 0}
};

// ── mcx / macx  (X = Invert with phase [1,1]) ────────────────────────────────
.def("mcx",
    [](QrackSim& s, std::vector<bitLenInt> controls, bitLenInt tgt) {
        s.sim->MCInvert(controls, ONE_C, ONE_C, tgt);
    },
    nb::arg("controls"), nb::arg("target"),
    "Multiply-controlled X. Fires when all controls are |1>.")

.def("macx",
    [](QrackSim& s, std::vector<bitLenInt> controls, bitLenInt tgt) {
        s.sim->MACInvert(controls, ONE_C, ONE_C, tgt);
    },
    nb::arg("controls"), nb::arg("target"),
    "Anti-controlled X. Fires when all controls are |0>.")

// ── mcy / macy  (Y = Invert with phase [-i, i]) ──────────────────────────────
.def("mcy",
    [](QrackSim& s, std::vector<bitLenInt> controls, bitLenInt tgt) {
        s.sim->MCInvert(controls, NEG_I_C, I_C, tgt);
    },
    nb::arg("controls"), nb::arg("target"), "Multiply-controlled Y.")

.def("macy",
    [](QrackSim& s, std::vector<bitLenInt> controls, bitLenInt tgt) {
        s.sim->MACInvert(controls, NEG_I_C, I_C, tgt);
    },
    nb::arg("controls"), nb::arg("target"), "Anti-controlled Y.")

// ── mcz / macz  (Z = Phase [1, -1]) ─────────────────────────────────────────
.def("mcz",
    [](QrackSim& s, std::vector<bitLenInt> controls, bitLenInt tgt) {
        s.sim->MCPhase(controls, ONE_C, NEG1_C, tgt);
    },
    nb::arg("controls"), nb::arg("target"), "Multiply-controlled Z.")

.def("macz",
    [](QrackSim& s, std::vector<bitLenInt> controls, bitLenInt tgt) {
        s.sim->MACPhase(controls, ONE_C, NEG1_C, tgt);
    },
    nb::arg("controls"), nb::arg("target"), "Anti-controlled Z.")

// ── mch  (H = MCMtrx with Hadamard matrix) ───────────────────────────────────
.def("mch",
    [](QrackSim& s, std::vector<bitLenInt> controls, bitLenInt tgt) {
        s.sim->MCMtrx(controls, H_MTRX, tgt);
    },
    nb::arg("controls"), nb::arg("target"), "Multiply-controlled H.")

// ── mcrz  (RZ = Phase [exp(-iθ/2), exp(iθ/2)]) ──────────────────────────────
.def("mcrz",
    [](QrackSim& s, real1_f angle, std::vector<bitLenInt> controls, bitLenInt tgt) {
        const Qrack::real1 half = (Qrack::real1)(angle / 2.0f);
        const Qrack::complex phase0 = std::exp(Qrack::complex(0.0f, -half));
        const Qrack::complex phase1 = std::exp(Qrack::complex(0.0f,  half));
        s.sim->MCPhase(controls, phase0, phase1, tgt);
    },
    nb::arg("angle"), nb::arg("controls"), nb::arg("target"),
    "Multiply-controlled RZ.")

// ── mcu  (arbitrary multi-controlled U gate) ─────────────────────────────────
.def("mcu",
    [](QrackSim& s, std::vector<bitLenInt> controls, bitLenInt tgt,
       real1_f theta, real1_f phi, real1_f lam) {
        s.sim->CU(controls, tgt, theta, phi, lam);
    },
    nb::arg("controls"), nb::arg("target"),
    nb::arg("theta"), nb::arg("phi"), nb::arg("lam"),
    "Multiply-controlled U(θ,φ,λ) gate.")
```

### Arbitrary matrix gate

```cpp
.def("mtrx",
    [](QrackSim& s,
       std::vector<std::complex<float>> m,
       bitLenInt q)
    {
        if (m.size() < 4)
            throw std::invalid_argument("mtrx: matrix must have 4 elements");
        s.check_qubit(q, "mtrx");
        s.sim->Mtrx(m.data(), q);
    },
    nb::arg("matrix"), nb::arg("qubit"),
    "Apply arbitrary 2x2 unitary. matrix is [m00, m01, m10, m11] row-major.")

.def("mcmtrx",
    [](QrackSim& s,
       std::vector<bitLenInt> controls,
       std::vector<std::complex<float>> m,
       bitLenInt q)
    {
        if (m.size() < 4)
            throw std::invalid_argument("mcmtrx: matrix must have 4 elements");
        // Signature: MCMtrx(const std::vector<bitLenInt>& controls,
        //                   const complex* mtrx, bitLenInt target)
        // Pass vector directly — do NOT use controls.size(), controls.data()
        s.sim->MCMtrx(controls, m.data(), q);
    },
    nb::arg("controls"), nb::arg("matrix"), nb::arg("qubit"),
    "Multiply-controlled arbitrary 2x2 unitary.")

.def("macmtrx",
    [](QrackSim& s,
       std::vector<bitLenInt> controls,
       std::vector<std::complex<float>> m,
       bitLenInt q)
    {
        if (m.size() < 4)
            throw std::invalid_argument("macmtrx: matrix must have 4 elements");
        s.sim->MACMtrx(controls, m.data(), q);
    },
    nb::arg("controls"), nb::arg("matrix"), nb::arg("qubit"),
    "Anti-controlled arbitrary 2x2 unitary.")

// ── multiplex1_mtrx  (uniformly-controlled arbitrary unitary) ────────────────
// Maps to QInterface::UniformlyControlledSingleBit.
// mtrxs is a flat array of 4 * 2^len(controls) complex values — one 2x2 matrix
// per control permutation, in row-major order. Used by Bloqade's QASM2 interpreter.
.def("multiplex1_mtrx",
    [](QrackSim& s,
       std::vector<bitLenInt> controls,
       std::vector<std::complex<float>> mtrxs,
       bitLenInt tgt)
    {
        const size_t expected = 4ULL << controls.size();
        if (mtrxs.size() < expected)
            throw std::invalid_argument(
                "multiplex1_mtrx: mtrxs must have at least 4 * 2^len(controls) = " +
                std::to_string(expected) + " elements");
        s.check_qubit(tgt, "multiplex1_mtrx");
        s.sim->UniformlyControlledSingleBit(controls, tgt,
            reinterpret_cast<const Qrack::complex*>(mtrxs.data()));
    },
    nb::arg("controls"), nb::arg("mtrxs"), nb::arg("target"),
    "Uniformly-controlled single-qubit gate. mtrxs is a flat list of "
    "4 * 2**len(controls) complex values — one 2x2 unitary per control permutation, "
    "in row-major order.")
```

---

## 4. Measurement

**File:** `bindings/simulator.cpp` — continue inside `bind_simulator()`.

```cpp
.def("measure",
    [](QrackSim& s, bitLenInt q) -> bool {
        s.check_qubit(q, "measure");
        return s.sim->M(q);
    },
    nb::arg("qubit"),
    "Measure qubit. Returns True=|1>, False=|0>. Collapses state.")

.def("measure_all",
    [](QrackSim& s) -> std::vector<bool> {
        std::vector<bool> out;
        out.reserve(s.numQubits);
        for (bitLenInt i = 0; i < s.numQubits; i++)
            out.push_back(s.sim->M(i));
        return out;
    },
    "Measure all qubits. Returns list[bool], LSB first.")

.def("force_measure",
    [](QrackSim& s, bitLenInt q, bool result) -> bool {
        s.check_qubit(q, "force_measure");
        return s.sim->ForceM(q, result);
    },
    nb::arg("qubit"), nb::arg("result"),
    "Force measurement outcome. Projects state to result without random draw.")

.def("prob",
    [](QrackSim& s, bitLenInt q) -> real1_f {
        s.check_qubit(q, "prob");
        return s.sim->Prob(q);
    },
    nb::arg("qubit"),
    "Probability of |1> for qubit. Does NOT collapse state.")

.def("prob_all",
    [](QrackSim& s) -> std::vector<real1_f> {
        std::vector<real1_f> out(s.numQubits);
        for (bitLenInt i = 0; i < s.numQubits; i++)
            out[i] = s.sim->Prob(i);
        return out;
    },
    "Per-qubit |1> probabilities for all qubits. Does NOT collapse state.")
```

---

## 5. State Vector and Probabilities — NumPy Properties

**File:** `bindings/simulator.cpp` — inside `bind_simulator()`. The `#include <nanobind/ndarray.h>` and `using cf32 = std::complex<float>` shown here belong in `bindings/binding_core.h`, not in `simulator.cpp` directly. Add them to `binding_core.h` if not already present.

```cpp
#include <nanobind/ndarray.h>
namespace nb = nanobind;
using cf32 = std::complex<float>;

.def_prop_ro("state_vector",
    [](QrackSim& s) {
        size_t dim = 1ULL << s.numQubits;
        // GetStateVector copies into a caller-provided buffer
        std::vector<cf32> buf(dim);
        s.sim->GetQuantumState(buf.data());

        // Wrap as a NumPy array (copy — Qrack owns its internal buffer)
        return nb::ndarray<nb::numpy, cf32, nb::ndim<1>>(
            buf.data(),
            {dim},
            nb::handle()  // no capsule: nanobind copies the data
        );
    },
    "Full state vector as np.ndarray[complex64], shape (2**n,).\n"
    "Returns a copy — Qrack's internal buffer is not exposed directly.")

.def_prop_ro("probabilities",
    [](QrackSim& s) {
        size_t dim = 1ULL << s.numQubits;
        std::vector<float> buf(dim);
        s.sim->GetProbs(buf.data());
        return nb::ndarray<nb::numpy, float, nb::ndim<1>>(
            buf.data(), {dim}, nb::handle());
    },
    "Probability of each basis state as np.ndarray[float32], shape (2**n,).")

.def("get_amplitude",
    [](QrackSim& s, bitCapInt idx) -> cf32 {
        size_t dim = 1ULL << s.numQubits;
        if (idx >= dim)
            throw std::out_of_range("get_amplitude: index out of range");
        return s.sim->GetAmplitude(idx);
    },
    nb::arg("index"),
    "Amplitude of a single basis state. O(1) for most simulation modes.")
```

---

## 6. Pauli Enum

**Files:** Two files are involved.
- The `bind_pauli()` function and enum registration go in `bindings/pauli.cpp` (existing file).
- The `measure_pauli`, `exp_val`, and `exp_val_pauli` method bindings go in `bindings/simulator.cpp` inside `bind_simulator()`, after the measurement section. Note: these use `static_cast<Qrack::Pauli>` because the local `Pauli` shadow enum in section 6 should be replaced with `Qrack::Pauli` from `pauli.hpp` directly — see [[qrackbind C++ Project Layout]] for the rationale.

Required in Phase 1 because both the dispatch table and PennyLane use it.

```cpp
// pauli.cpp
#include <nanobind/nanobind.h>
namespace nb = nanobind;

enum class Pauli { I = 0, X = 1, Y = 3, Z = 2 }; // matches Qrack's encoding

void bind_pauli(nb::module_& m) {
    nb::enum_<Pauli>(m, "Pauli", "Pauli operator basis for measurements.")
        .value("PauliI", Pauli::I, "Identity.")
        .value("PauliX", Pauli::X, "X basis.")
        .value("PauliY", Pauli::Y, "Y basis.")
        .value("PauliZ", Pauli::Z, "Z basis.")
        .export_values();
}
```

Add Pauli-basis methods to the simulator:

```cpp
.def("measure_pauli",
    [](QrackSim& s, Pauli basis, bitLenInt q) -> bool {
        s.check_qubit(q, "measure_pauli");
        return s.sim->MPauli(static_cast<Qrack::Pauli>(basis), q, 0);
    },
    nb::arg("basis"), nb::arg("qubit"),
    "Measure in Pauli basis. PauliX, PauliY, or PauliZ.")

.def("exp_val",
    [](QrackSim& s, Pauli basis, bitLenInt q) -> real1_f {
        s.check_qubit(q, "exp_val");
        return s.sim->ExpectationPauliAll({static_cast<Qrack::Pauli>(basis)}, {q});
    },
    nb::arg("basis"), nb::arg("qubit"),
    "Expectation value of a Pauli operator on one qubit. Does NOT collapse.")

.def("exp_val_pauli",
    [](QrackSim& s,
       std::vector<Pauli> bases,
       std::vector<bitLenInt> qubits) -> real1_f {
        std::vector<Qrack::Pauli> qbases(bases.size());
        for (size_t i = 0; i < bases.size(); i++)
            qbases[i] = static_cast<Qrack::Pauli>(bases[i]);
        return s.sim->ExpectationPauliAll(qbases, qubits);
    },
    nb::arg("bases"), nb::arg("qubits"),
    "Multi-qubit Pauli expectation value. bases and qubits must be same length.")
```

---

## 7. State Control and Properties

**File:** `bindings/simulator.cpp` — continue inside `bind_simulator()`.

```cpp
.def("reset_all",
    [](QrackSim& s) { s.sim->SetPermutation(0); },
    "Reset all qubits to |0...0>.")

.def("set_permutation",
    [](QrackSim& s, bitCapInt perm) { s.sim->SetPermutation(perm); },
    nb::arg("permutation"),
    "Initialize to computational basis state |perm>. "
    "E.g. set_permutation(0b101) on 3 qubits gives |101>.")

.def_prop_ro("num_qubits",
    [](const QrackSim& s) { return s.numQubits; },
    "Number of qubits in this simulator.")

.def_prop_ro("qubit_count",   // alias matching pyqrack
    [](const QrackSim& s) { return s.numQubits; })

.def("__repr__", &QrackSim::repr)
```

---

## 8. Context Manager

**File:** `bindings/simulator.cpp` — continue inside `bind_simulator()`. These are the last two `.def()` calls before the closing `;`.

```cpp
.def("__enter__",
    [](QrackSim& s) -> QrackSim& { return s; },
    "Enable use as a context manager: with QrackSimulator(n) as sim: ...")

.def("__exit__",
    [](QrackSim& s,
       nb::object /*exc_type*/,
       nb::object /*exc_val*/,
       nb::object /*exc_tb*/)
    {
        s.sim.reset();  // release the QInterfacePtr immediately
    })
```

---

## 9. Deprecated Aliases — Pure Python Mixin

These live in `src/qrackbind/_compat.py`, mixed into `QrackSimulator` at import time:

```python
# src/qrackbind/_compat.py
import warnings

class _PyqrackAliasMixin:
    """Deprecated method aliases for pyqrack backward compatibility."""

    def m(self, qubit: int) -> int:
        """Deprecated: use measure(qubit)."""
        warnings.warn("m() is deprecated, use measure()", DeprecationWarning, stacklevel=2)
        return int(self.measure(qubit))

    def m_all(self) -> list:
        """Deprecated: use measure_all()."""
        warnings.warn("m_all() is deprecated, use measure_all()", DeprecationWarning, stacklevel=2)
        return [int(b) for b in self.measure_all()]

    def get_state_vector(self) -> list:
        """Deprecated: use the state_vector property."""
        warnings.warn("get_state_vector() is deprecated, use state_vector", DeprecationWarning, stacklevel=2)
        return self.state_vector.tolist()

    def get_num_qubits(self) -> int:
        """Deprecated: use the num_qubits property."""
        warnings.warn("get_num_qubits() is deprecated, use num_qubits", DeprecationWarning, stacklevel=2)
        return self.num_qubits
```

Applied in `__init__.py`:

```python
# src/qrackbind/__init__.py
from ._qrackbind_core import QrackSimulator as _QrackSimulator, Pauli
from ._compat import _PyqrackAliasMixin

class QrackSimulator(_PyqrackAliasMixin, _QrackSimulator):
    """
    Qrack quantum simulator with strong typing and NumPy integration.

    Drop-in replacement for pyqrack.QrackSimulator with one import change.
    Constructor keyword arguments are identical to pyqrack.
    """
    __slots__ = ()

__all__ = ["QrackSimulator", "Pauli"]
__version__ = "0.1.0"
```

This pattern (mixin + subclass in Python) means the deprecated aliases never touch C++ and can be removed cleanly in a future version without touching any binding code.

---

## 10. Exception Registration

```cpp
// In module.cpp, before bind_simulator()
static PyObject* QrackExceptionType = nullptr;

NB_MODULE(_qrackbind_core, m) {
    // Register a custom Python exception class
    QrackExceptionType = PyErr_NewExceptionWithDoc(
        "qrackbind.QrackException",
        "Exception raised by the Qrack C++ library.",
        PyExc_RuntimeError,
        nullptr);
    if (QrackExceptionType)
        m.attr("QrackException") = QrackExceptionType;

    nb::register_exception_translator([](const std::exception& e, void*) {
        PyErr_SetString(QrackExceptionType, e.what());
    });

    bind_pauli(m);
    bind_simulator(m);
}
```

The `check_qubit()` helper in `QrackSim` throws `std::out_of_range`, which the translator converts to `QrackException`.

---

## 11. CMakeLists.txt (Updated)

```cmake
cmake_minimum_required(VERSION 3.15...3.27)
project(qrackbind LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.9 REQUIRED
    COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# Qrack — prefer env vars, fall back to system paths
find_library(QRACK_LIB qrack
    HINTS $ENV{QRACK_LIB_DIR} /usr/local/lib /usr/lib)
find_path(QRACK_INCLUDE qfactory.hpp
    HINTS $ENV{QRACK_INCLUDE_DIR} /usr/local/include/qrack /usr/include/qrack)

if(NOT QRACK_LIB)
    message(FATAL_ERROR "Qrack library not found. Set QRACK_LIB_DIR.")
endif()
if(NOT QRACK_INCLUDE)
    message(FATAL_ERROR "Qrack headers not found. Set QRACK_INCLUDE_DIR.")
endif()

nanobind_add_module(
    _qrackbind_core
    STABLE_ABI
    NB_STATIC
    NB_DOMAIN qrackbind
    bindings/module.cpp
    bindings/simulator.cpp
    bindings/pauli.cpp
)

target_include_directories(_qrackbind_core PRIVATE
    ${QRACK_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR}/bindings   # binding_core.h is here
)
target_link_libraries(_qrackbind_core PRIVATE ${QRACK_LIB})

install(TARGETS _qrackbind_core LIBRARY DESTINATION qrackbind)
```

---

## 12. Full Test Suite

```python
# tests/test_phase1.py
import math
import warnings
import pytest
import numpy as np
from qrackbind import QrackSimulator, Pauli

# --- Construction ---

class TestConstructor:
    def test_basic(self):
        sim = QrackSimulator(qubitCount=4)
        assert sim.num_qubits == 4

    def test_default_flags(self):
        # Should not raise with any combination of flags
        QrackSimulator(qubitCount=2, isTensorNetwork=True)
        QrackSimulator(qubitCount=2, isStabilizerHybrid=True, isSchmidtDecompose=False)
        QrackSimulator(qubitCount=2, isBinaryDecisionTree=True, isStabilizerHybrid=False)

    def test_repr_contains_qubit_count(self):
        sim = QrackSimulator(qubitCount=4)
        assert "4" in repr(sim)

    def test_context_manager(self):
        with QrackSimulator(qubitCount=2) as sim:
            sim.x(0)
            assert sim.measure(0) == True

# --- Single-qubit Clifford gates ---

class TestCliffordGates:
    def test_x_flips(self):
        sim = QrackSimulator(qubitCount=1)
        sim.x(0)
        assert sim.prob(0) == pytest.approx(1.0, abs=1e-5)

    def test_double_x_identity(self):
        sim = QrackSimulator(qubitCount=1)
        sim.x(0); sim.x(0)
        assert sim.prob(0) == pytest.approx(0.0, abs=1e-5)

    def test_h_superposition(self):
        results = [QrackSimulator(qubitCount=1) for _ in range(300)]
        for s in results: s.h(0)
        ratio = sum(s.measure(0) for s in results) / 300
        assert 0.38 < ratio < 0.62

    def test_sx_half_x(self):
        sim = QrackSimulator(qubitCount=1)
        sim.sx(0); sim.sx(0)  # two √X = X
        assert sim.prob(0) == pytest.approx(1.0, abs=1e-5)

    def test_s_sdg_identity(self):
        sim = QrackSimulator(qubitCount=1)
        sim.h(0); sim.s(0); sim.sdg(0); sim.h(0)
        assert sim.prob(0) == pytest.approx(0.0, abs=1e-5)

# --- Rotation gates ---

class TestRotationGates:
    def test_rx_pi_flips(self):
        sim = QrackSimulator(qubitCount=1)
        sim.rx(math.pi, 0)
        assert sim.prob(0) == pytest.approx(1.0, abs=1e-5)

    def test_rz_no_population_change(self):
        sim = QrackSimulator(qubitCount=1)
        sim.rz(math.pi / 3, 0)
        assert sim.prob(0) == pytest.approx(0.0, abs=1e-5)

    def test_u_gate_equivalent_to_rx(self):
        sim1 = QrackSimulator(qubitCount=1)
        sim1.rx(math.pi / 2, 0)
        sim2 = QrackSimulator(qubitCount=1)
        sim2.u(math.pi / 2, -math.pi / 2, math.pi / 2, 0)
        assert sim1.prob(0) == pytest.approx(sim2.prob(0), abs=1e-4)

# --- Two-qubit and controlled gates ---

class TestControlledGates:
    def test_cnot_bell_state(self):
        sim = QrackSimulator(qubitCount=2)
        sim.h(0); sim.cnot(0, 1)
        p = sim.prob(0)
        assert p == pytest.approx(0.5, abs=0.05)

    def test_ccnot_toffoli(self):
        sim = QrackSimulator(qubitCount=3)
        sim.x(0); sim.x(1)
        sim.ccnot(0, 1, 2)
        assert sim.prob(2) == pytest.approx(1.0, abs=1e-5)

    def test_mcx_multiply_controlled(self):
        sim = QrackSimulator(qubitCount=4)
        sim.x(0); sim.x(1); sim.x(2)
        sim.mcx([0, 1, 2], 3)
        assert sim.prob(3) == pytest.approx(1.0, abs=1e-5)

    def test_macx_anti_controlled(self):
        sim = QrackSimulator(qubitCount=2)
        # Control is |0>, so anti-control should fire
        sim.macx([0], 1)
        assert sim.prob(1) == pytest.approx(1.0, abs=1e-5)

    def test_swap(self):
        sim = QrackSimulator(qubitCount=2)
        sim.x(0)
        sim.swap(0, 1)
        assert sim.prob(0) == pytest.approx(0.0, abs=1e-5)
        assert sim.prob(1) == pytest.approx(1.0, abs=1e-5)

# --- Measurement ---

class TestMeasurement:
    def test_measure_collapses(self):
        sim = QrackSimulator(qubitCount=1)
        sim.h(0)
        result = sim.measure(0)
        after = sim.prob(0)
        expected = 1.0 if result else 0.0
        assert after == pytest.approx(expected, abs=1e-5)

    def test_measure_all_length_and_type(self):
        sim = QrackSimulator(qubitCount=5)
        results = sim.measure_all()
        assert len(results) == 5
        assert all(isinstance(r, bool) for r in results)

    def test_prob_all(self):
        sim = QrackSimulator(qubitCount=3)
        sim.x(1)
        probs = sim.prob_all()
        assert probs[0] == pytest.approx(0.0, abs=1e-5)
        assert probs[1] == pytest.approx(1.0, abs=1e-5)
        assert probs[2] == pytest.approx(0.0, abs=1e-5)

    def test_force_measure(self):
        sim = QrackSimulator(qubitCount=1)
        sim.h(0)
        result = sim.force_measure(0, True)
        assert result == True
        assert sim.prob(0) == pytest.approx(1.0, abs=1e-5)

# --- State vector ---

class TestStateVector:
    def test_state_vector_is_ndarray(self):
        sim = QrackSimulator(qubitCount=2)
        sv = sim.state_vector
        assert isinstance(sv, np.ndarray)
        assert sv.shape == (4,)

    def test_state_vector_ground_state(self):
        sim = QrackSimulator(qubitCount=2)
        sv = sim.state_vector
        assert abs(sv[0]) == pytest.approx(1.0, abs=1e-5)
        for i in range(1, 4):
            assert abs(sv[i]) == pytest.approx(0.0, abs=1e-5)

    def test_probabilities_normalized(self):
        sim = QrackSimulator(qubitCount=3)
        sim.h(0); sim.h(1)
        probs = sim.probabilities
        assert isinstance(probs, np.ndarray)
        assert probs.sum() == pytest.approx(1.0, abs=1e-4)

    def test_get_amplitude(self):
        sim = QrackSimulator(qubitCount=2)
        amp = sim.get_amplitude(0)
        assert abs(amp) == pytest.approx(1.0, abs=1e-5)

# --- Pauli ---

class TestPauli:
    def test_enum_values_exist(self):
        assert Pauli.PauliX is not None
        assert Pauli.PauliY is not None
        assert Pauli.PauliZ is not None
        assert Pauli.PauliI is not None

    def test_exp_val_z_ground(self):
        sim = QrackSimulator(qubitCount=1)
        ev = sim.exp_val(Pauli.PauliZ, 0)
        assert ev == pytest.approx(1.0, abs=1e-4)  # |0> is +1 eigenstate of Z

    def test_exp_val_z_excited(self):
        sim = QrackSimulator(qubitCount=1)
        sim.x(0)
        ev = sim.exp_val(Pauli.PauliZ, 0)
        assert ev == pytest.approx(-1.0, abs=1e-4)  # |1> is -1 eigenstate of Z

# --- Deprecated aliases ---

class TestDeprecatedAliases:
    def test_m_alias(self):
        sim = QrackSimulator(qubitCount=1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = sim.m(0)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "measure" in str(w[0].message)
        assert isinstance(result, int)

    def test_m_all_alias(self):
        sim = QrackSimulator(qubitCount=2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = sim.m_all()
        assert isinstance(results, list)
        assert all(isinstance(r, int) for r in results)

    def test_get_state_vector_alias(self):
        sim = QrackSimulator(qubitCount=1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sv = sim.get_state_vector()
        assert isinstance(sv, list)

    def test_get_num_qubits_alias(self):
        sim = QrackSimulator(qubitCount=3)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            n = sim.get_num_qubits()
        assert n == 3

# --- Exception handling ---

class TestExceptions:
    def test_out_of_range_qubit(self):
        sim = QrackSimulator(qubitCount=3)
        with pytest.raises(Exception):  # QrackException or IndexError
            sim.h(10)

    def test_mtrx_too_short(self):
        sim = QrackSimulator(qubitCount=1)
        with pytest.raises(Exception):
            sim.mtrx([1, 0, 0], 0)  # only 3 elements
```

---

## 13. Phase 1 Completion Checklist

```
□ _qrackbind_core module imports cleanly
□ QrackSimulator(qubitCount=n) constructs
□ All pyqrack constructor flags accepted without error
□ isStabilizerHybrid=True variant constructs and runs a circuit
□ isBinaryDecisionTree=True variant constructs
□ h, x, y, z, s, t, sdg, tdg, sx, sxdg bound
□ rx, ry, rz, r1, u, u2 bound with float args
□ cnot, cy, cz, swap, iswap, ccnot bound
□ mcx, mcy, mcz, mch, mcrz bound (list controls)
□ macx, macy, macz bound
□ mtrx, mcmtrx, macmtrx, multiplex1_mtrx bound
□ measure() returns bool, collapses state
□ measure_all() returns list[bool], correct length
□ force_measure() forces outcome
□ prob() returns float, no collapse
□ prob_all() returns list[float]
□ state_vector property returns np.ndarray[complex64], shape (2**n,)
□ probabilities property returns np.ndarray[float32], sums to 1.0
□ get_amplitude() returns complex
□ Pauli.PauliX, PauliY, PauliZ, PauliI defined
□ exp_val() and exp_val_pauli() return correct values
□ reset_all() and set_permutation() work
□ num_qubits property works
□ context manager (__enter__ / __exit__) works
□ m(), m_all(), get_state_vector(), get_num_qubits() emit DeprecationWarning
□ Out-of-range qubit raises exception (not segfault)
□ uv run pytest passes all tests
□ nanobind-stubgen produces .pyi with all signatures
```

---

## Related

- [[qrackbind Project Phase Breakdown]]
- [[qrackbind Python API Design]]
- [[Framework Plugin Architecture (PennyLane + Qiskit)]]
- [[pyqrack Compatibility Strategy]]
- [[QrackSimulator API Method Categories]]
