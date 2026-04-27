# qrackbind

[Qrack](https://github.com/unitaryfoundation/qrack) is a C++ quantum simulator supporting CPU, GPU (OpenCL and CUDA), tensor network, stabilizer hybrid, and binary decision tree simulation backends. `qrackbind` provides an alternative Python interface to this simulator alongside the existing [pyqrack](https://github.com/unitaryfoundation/pyqrack) binding.

Where pyqrack is a pure-Python wrapper that communicates with Qrack through a C shared library interface, `qrackbind` is a compiled extension module built with [nanobind](https://github.com/wjakob/nanobind). This approach provides static typing, generated `.pyi` stubs, native NumPy array integration via the DLPack protocol, and lower overhead at the Python/C++ boundary. The public API preserves the gate method names, constructor arguments, and `Pauli` enum from pyqrack, and includes deprecated aliases for the small number of methods that have been renamed, so existing pyqrack code requires minimal changes to run against `qrackbind`.

### Motivations

`qrackbind` addresses three primary motivations: providing a Python interface to Qrack that is compatible with AI code generation tooling, establishing a typed and performant substrate for framework integrations with PennyLane, Qiskit, and QuEra Bloqade, and improving on the runtime characteristics of the existing pyqrack binding through a compiled nanobind extension.

**AI code generation compatibility**

| Property | Description |
|---|---|
| Named parameters | All methods accept keyword arguments, producing call sites that are unambiguous regardless of argument order |
| `.pyi` type stubs | Generated stubs for every bound class and function provide type information accessible to code analysis tools and language model context windows |
| Typed exceptions | `QrackException`, `QrackQubitError`, and `QrackArgumentError` produce informative stack traces that identify the error source without requiring inspection of C++ internals |
| NumPy array returns | `state_vector` and `probabilities` return `np.ndarray` directly, compatible with the scientific Python ecosystem that generated post-processing code commonly targets |

**Framework ecosystem integration**

PennyLane, Qiskit, and QuEra Bloqade each interact with a quantum simulator through a small, well-defined dispatch interface rather than the full gate method surface. `qrackbind` provides the `QrackCircuit` class and `GateType` enum as a typed dispatch layer shared across all three frameworks. A single gate dispatch table maps PennyLane operation names, Qiskit gate names, and Bloqade IR calls onto the same C++ methods. Migration for existing Bloqade users consists of updating the package dependency and import statements; all gate method names, constructor keyword arguments, and the `Pauli` enum are preserved without change.

**Compiled extension performance**

pyqrack communicates with the Qrack C++ library through Python's `ctypes` module. At import time, `ctypes.CDLL` loads the Qrack shared library and resolves exported C symbols by name. Each gate call then follows a multi-step sequence at runtime: Python arguments are marshalled into ctypes-compatible types (`ctypes.c_int`, `ctypes.c_float`, `ctypes.c_double`, ctypes array types for vector arguments), the ctypes foreign function interface constructs a C call frame, the call crosses the Python/C boundary, and the return value is unmarshalled back into a Python object. For array arguments such as gate matrices or qubit index lists, pyqrack allocates an intermediate ctypes array on each call — for example, a complex matrix argument is unpacked into real/imaginary pairs and loaded into a `c_double` array before the call, then discarded afterward.

`qrackbind` replaces this with a compiled nanobind extension module. Type conversion between Python objects and C++ types is handled at compile time through nanobind's type caster infrastructure, and the call dispatch path is a direct C function call with no intermediate marshalling step. Vector arguments such as qubit index lists are converted from Python lists to `std::vector` by nanobind's STL casters without allocating a ctypes array. The `state_vector` and `probabilities` properties use nanobind's `nb::ndarray` with capsule-based lifetime management, transferring the C++ buffer to Python without a secondary copy. These characteristics are most relevant for workloads that apply large numbers of short gate operations from Python, pass array arguments frequently, or read the state vector during variational circuit optimisation.

## Migrating from pyqrack

`qrackbind` preserves the core surface of [pyqrack](https://github.com/unitaryfoundation/pyqrack): gate method names, constructor keyword arguments, and the `Pauli` enum are identical. A small number of method names have been updated to follow Python naming conventions. Deprecated aliases for the previous names are included and emit `DeprecationWarning` at runtime.

### API changes

| pyqrack | qrackbind | Notes |
|---|---|---|
| `from pyqrack import QrackSimulator` | `from qrackbind import QrackSimulator` | One line per file |
| `sim.m(q)` | `sim.measure(q)` | `m()` retained as a deprecated alias |
| `sim.m_all()` | `sim.measure_all()` | `m_all()` retained as a deprecated alias |
| `sim.get_state_vector()` | `sim.state_vector` | Property returning `np.ndarray[complex64]` |
| `sim.get_num_qubits()` | `sim.num_qubits` | Property |

### Preserved API

The following are unchanged and require no modification in existing code:

```python
# Constructor keyword arguments are identical to pyqrack
sim = QrackSimulator(
    qubitCount=12,
    isTensorNetwork=True,
    isStabilizerHybrid=False,
    isSchmidtDecompose=True,
    isPaged=True,
    isCpuGpuHybrid=True,
    isOpenCL=True,
)

# Gate method names are unchanged
sim.h(0)
sim.x(1)
sim.rx(0.5, 2)
sim.mcx([0, 1], 2)    # multiply-controlled X
sim.macx([0], 1)      # anti-controlled X
sim.swap(0, 1)

# Pauli enum values are unchanged
from qrackbind import Pauli
ev = sim.exp_val_pauli([Pauli.PauliZ, Pauli.PauliZ], [0, 1])
```

For downstream projects such as `bloqade-pyqrack`, migration consists of updating the package dependency declaration and import statements. Gate dispatch calls (`h`, `x`, `mcx`, etc.) are unchanged.

---

## API Design

### Named parameters

All methods accept named parameters in addition to positional ones:

```python
sim.rx(angle=0.5, qubit=0)
sim.mcx(controls=[0, 1], target=2)
sim.exp_val_pauli(paulis=[Pauli.PauliZ, Pauli.PauliZ], qubits=[0, 1])
```

### Type stubs

`qrackbind` ships `.pyi` stub files generated from the nanobind extension. The stubs reflect the actual C++ signatures and are compatible with Pyright, mypy, and IDE language servers:

```python
class QrackSimulator:
    def exp_val_pauli(
        self,
        paulis: list[Pauli],
        qubits: list[int],
    ) -> float: ...

    @property
    def state_vector(self) -> numpy.ndarray[numpy.complex64]: ...
```

### Typed exceptions

Errors raise typed exceptions from a project-specific hierarchy:

- `QrackException` — base class for all qrackbind errors (inherits `RuntimeError`)
- `QrackQubitError` — qubit index out of the valid range
- `QrackArgumentError` — invalid method arguments

```python
try:
    sim.h(99)
except QrackQubitError as e:
    print(e)  # QrackQubitError: qubit 99 out of range [0, 3]
```

### NumPy integration

`state_vector` and `probabilities` return NumPy arrays via nanobind's `nb::ndarray` mechanism. The C++ buffer is allocated by the binding layer, filled by Qrack, and transferred to Python under capsule-based lifetime management. The Python garbage collector is responsible for deallocation; no secondary copy is made.

```python
sv    = sim.state_vector    # np.ndarray[complex64], shape (2^n,)
probs = sim.probabilities   # np.ndarray[float32],  shape (2^n,)

fidelity = np.abs(np.dot(sv.conj(), target_sv)) ** 2
entropy  = -np.sum(probs * np.log2(probs + 1e-12))
```

## Project Phases and Status

| Phase | Title | Description | Status |
|---|---|---|---|
| 0 | Project Scaffold | uv + scikit-build-core project structure, CMakeLists.txt, justfile, smoke test | :white_check_mark: |
| 1 | QrackSimulator Core | Full gate set, constructor, measurement, properties, multi-control gates, `multiplex1_mtrx`, pyqrack compat aliases | :white_check_mark: |
| 2 | Dynamic Allocation, QFT, Arithmetic | Simulator registry, `cloneSid`, `allocate`/`dispose`, QFT, arithmetic gates, shift/rotate, `measure_shots` | :white_check_mark: |
| 3 | State Vector and NumPy | `state_vector`, `probabilities`, `set_state_vector`, `get_amplitude`, reduced density matrix, `prob_all` | :white_check_mark: |
| 4 | Enums and Pauli Operators | `Pauli` enum, `measure_pauli`, `exp_val`, `exp_val_pauli`, `variance_pauli`, `exp_val_floats` | :white_check_mark: |
| 5 | Exception Handling | `QrackException`, `QrackQubitError`, `QrackArgumentError`, C++ exception translator | :white_check_mark: |
| 6 | QrackCircuit | `QrackCircuit`, `GateType`, `append_gate`, `run`, `inverse`, `optimize`, `to_qasm`, `exp_val_unitary` | :construction: |
| 7 | Stub Generation and Type Annotations | `.pyi` stubs, docstrings on all bindings, `pyright` passing | :pending: |
| 8 | PennyLane Device Plugin | `qrackbind.pennylane` device, `execute()`, parameter-shift gradients, VQE | :pending: |
| 9 | Packaging and Distribution | PyPI wheel via cibuildwheel, CMake `FetchContent` auto-download, `scripts/install_qrack.sh`, `uv run` scripts | :pending: |


## Installation

Coming soon...

## Quick Start

```python
from qrackbind import QrackSimulator

sim = QrackSimulator(qubitCount=3)
sim.h(0)
sim.cnot(0, 1)
sim.x(2)
results = sim.measure_all()
print(results)  # [True, True, True]
```

## Usage

```python
from qrackbind import QrackSimulator
import math

# Create a simulator with 4 qubits
sim = QrackSimulator(qubitCount=4)

# Apply gates
sim.h(0)           # Hadamard on qubit 0
sim.cnot(0, 1)     # CNOT with control=0, target=1
sim.rx(math.pi, 2) # RX(π) on qubit 2

# Get probability of |1> for a qubit (does not collapse)
print(sim.prob(0))  # 0.5

# Measure a qubit (collapses state)
result = sim.measure(0)  # True or False

# Measure all qubits at once
results = sim.measure_all()  # [bool, bool, bool, bool]

# Reset all qubits to |0...0>
sim.reset_all()
```

## Available Gates

### Single-qubit gates

| Gate | Description |
|------|-------------|
| `h(qubit)` | Hadamard gate |
| `x(qubit)` | Pauli-X (bit flip) |
| `y(qubit)` | Pauli-Y |
| `z(qubit)` | Pauli-Z (phase flip) |
| `s(qubit)` | S gate (π/2 phase) |
| `t(qubit)` | T gate (π/4 phase) |
| `sdg(qubit)` | S† (inverse S) |
| `tdg(qubit)` | T† (inverse T) |
| `sx(qubit)` | √X gate |
| `sxdg(qubit)` | √X† gate |

### Rotation gates

| Gate | Description |
|------|-------------|
| `rx(angle, qubit)` | Rotate around X axis |
| `ry(angle, qubit)` | Rotate around Y axis |
| `rz(angle, qubit)` | Rotate around Z axis |
| `r1(angle, qubit)` | Phase rotation (global phase) |

### General unitary

| Gate | Description |
|------|-------------|
| `u(theta, phi, lam, qubit)` | General single-qubit U(θ, φ, λ) |
| `u2(phi, lam, qubit)` | U2(φ, λ) = U(π/2, φ, λ) |

### Multi-qubit gates

| Gate | Description |
|------|-------------|
| `cnot(control, target)` | Controlled-NOT |
| `cy(control, target)` | Controlled-Y |
| `cz(control, target)` | Controlled-Z |
| `swap(q1, q2)` | SWAP two qubits |
| `iswap(q1, q2)` | iSWAP gate |
| `ccnot(c1, c2, target)` | Toffoli / CCX gate |

### Multi-controlled & arbitrary unitaries

| Gate | Description |
|------|-------------|
| `mcx(controls, target)` / `macx(controls, target)` | Multi-controlled / anti-controlled X |
| `mcy` / `macy`, `mcz` / `macz` | Multi-controlled / anti-controlled Y, Z |
| `mch(controls, target)` | Multi-controlled Hadamard |
| `mcrz(angle, controls, target)` | Multi-controlled RZ |
| `mcu(controls, target, theta, phi, lam)` | Multi-controlled U(θ, φ, λ) |
| `mtrx(matrix, qubit)` | Apply arbitrary 2×2 unitary |
| `mcmtrx` / `macmtrx(controls, matrix, qubit)` | Controlled / anti-controlled 2×2 unitary |
| `multiplex1_mtrx(controls, mtrxs, target)` | Uniformly-controlled single-qubit gate |

### Arithmetic, shifts, and QFT

| Method | Description |
|--------|-------------|
| `add(value, start, length)` / `sub(...)` | Classical add/subtract on a quantum register |
| `mul(to_mul, mod_n, in_start, out_start, length)` / `div(...)` | Modular multiplication / inverse (out-of-place) |
| `pown(base, mod_n, in_start, out_start, length)` | Modular exponentiation — Shor's central op |
| `mcmul` / `mcdiv` / `mcpown(..., controls)` | Controlled variants |
| `lsl(shift, start, length)` / `lsr(...)` | Logical shift left / right |
| `rol(shift, start, length)` / `ror(...)` | Circular rotate left / right |
| `qft(start, length)` / `iqft(...)` | Quantum / inverse Fourier transform on a contiguous register |
| `qftr(qubits)` / `iqftr(qubits)` | QFT / IQFT on an arbitrary qubit list |

> Note: arithmetic and shift operations require `isTensorNetwork=False` at construction.

### Measurement & state control

| Method | Description |
|--------|-------------|
| `measure(qubit)` / `measure_all()` | Measure a qubit / all qubits, collapsing the state |
| `force_measure(qubit, result)` | Project the state to the given outcome (no random draw) |
| `prob(qubit)` / `prob_all` | Per-qubit \|1⟩ probability (does not collapse) |
| `prob_perm(index)` / `prob_mask(mask, permutation)` | Probability of a specific basis state / masked permutation |
| `m_reg(start, length)` | Measure a contiguous register, return the integer outcome |
| `measure_shots(qubits, shots)` | Sample `shots` measurements without collapsing — returns `dict[int, int]` |
| `reset_all()` / `set_permutation(value)` | Reset to \|0…0⟩ / to the basis state \|value⟩ |

### State vector access (NumPy)

| Method / property | Description |
|-------------------|-------------|
| `state_vector` | 1-D NumPy array of complex amplitudes (`complex64` by default; `complex128` on a double-precision Qrack build) |
| `probabilities` | 1-D NumPy array of per-basis-state probabilities (`float32` / `float64`) |
| `set_state_vector(state)` | Set the state from a 1-D complex NumPy array (length `2**num_qubits`) |
| `get_amplitude(index)` / `set_amplitude(index, amplitude)` | Per-permutation read / write |
| `get_reduced_density_matrix(qubits)` | Reduced density matrix as a 2-D `(2**k, 2**k)` complex NumPy array |
| `update_running_norm()` | Recompute normalisation after a manual `set_amplitude` / `set_state_vector` |
| `first_nonzero_phase()` | Phase of the lowest-index nonzero amplitude (rad) |

### Pauli observables

The `Pauli` enum (`PauliI`, `PauliX`, `PauliY`, `PauliZ`) is `IntEnum`-compatible — integer codes are accepted everywhere a `Pauli` is expected. Qrack's underlying values are non-sequential: `PauliI=0, PauliX=1, PauliZ=2, PauliY=3`.

| Method | Description |
|--------|-------------|
| `measure_pauli(basis, qubit)` | Measure a qubit in a Pauli basis (rotates → measures → rotates back). Returns the post-rotation computational-basis bit (`True=\|1⟩`). `PauliI` is a no-op. |
| `exp_val(basis, qubit)` | Single-qubit Pauli expectation value, in `[-1.0, 1.0]`. Does not collapse. |
| `exp_val_pauli(paulis, qubits)` | Tensor-product Pauli expectation value `<ψ\|P₀⊗P₁⊗…\|ψ>`. |
| `variance_pauli(paulis, qubits)` | Variance of a Pauli tensor-product observable, in `[0.0, 1.0]`. For Paulis, `Var(P) = 1 − <P>²`. |
| `exp_val_all(basis)` | Broadcast a single basis across every qubit. |
| `exp_val_floats(qubits, weights)` | Weighted-sum diagonal-observable expectation value. `weights` has length `2 * len(qubits)`: `[w₀_for_\|0⟩, w₀_for_\|1⟩, w₁_for_\|0⟩, w₁_for_\|1⟩, …]`. |
| `variance_floats(qubits, weights)` | Variance counterpart to `exp_val_floats`. |

```python
from qrackbind import Pauli, QrackSimulator

sim = QrackSimulator(qubitCount=2)
sim.h(0); sim.cnot(0, 1)                              # Bell state
print(sim.exp_val_pauli([Pauli.PauliZ, Pauli.PauliZ], [0, 1]))  # → 1.0
```

### Dynamic qubit allocation

| Method | Description |
|--------|-------------|
| `allocate(start, length)` | Insert `length` new \|0⟩ qubits at index `start`; existing qubits shift up |
| `allocate_qubits(n)` | Append `n` new \|0⟩ qubits at the end |
| `dispose(start, length)` | Remove `length` qubits — they must be separable \|0⟩ or \|1⟩ |

> Note: dynamic allocation requires `isTensorNetwork=False`.

### Cloning

`QrackSimulator` supports deep copy via `clone()` and the `copy` module:

```python
import copy
branch_a = sim.clone()
branch_b = copy.deepcopy(sim)
```

The clone is fully independent — gates applied to one have no effect on the other.

## Features

- **Complete gate coverage**: All standard quantum gates including Pauli (`x`, `y`, `z`), Hadamard (`h`), phase (`s`, `t`), rotations (`rx`, `ry`, `rz`, `r1`), general unitary (`u`, `u2`), multi-qubit gates (`cnot`, `cy`, `cz`, `swap`, `iswap`, `ccnot`), and multi-controlled / arbitrary-matrix variants
- **Quantum arithmetic & QFT**: in-place add / sub, modular mul / div / pown (with controlled variants), logical shifts and rotations, QFT / IQFT
- **Pauli observables**: `measure_pauli`, single- and multi-qubit `exp_val` / `variance_pauli`, weighted `exp_val_floats` / `variance_floats`, `IntEnum`-compatible `Pauli` enum
- **NumPy state-vector access**: full state vector, probability vector, and reduced density matrix as zero-copy NumPy ndarrays; per-amplitude read / write
- **Dynamic qubit allocation**: grow and shrink the register at runtime
- **High performance**: Near-native C++ performance through nanobind
- **Type safety**: Full type stubs for mypy/pyright
- **GPU acceleration**: OpenCL and CUDA backends when available
- **Python 3.12+**: Leverages nanobind's stable ABI

## Configuration notes

`QrackSimulator(...)` exposes Qrack's full simulator-stack configuration as keyword arguments
(`isTensorNetwork`, `isSchmidtDecompose`, `isStabilizerHybrid`, `isBinaryDecisionTree`, `isPaged`,
`isCpuGpuHybrid`, `isOpenCL`, `isHostPointer`, `isSparse`, `noise`).

A few caveats specific to the current Qrack release (10.6.2):

- **`isOpenCL=True`** is silently downgraded to a CPU-only stack on hosts without an OpenCL runtime.
- **`isPaged=True`** is currently overridden to `False` internally — Qrack 10.6.2's QPager produces zero
  amplitudes for entangled states, breaks `CPOWModNOut`, and segfaults from `SetAmplitude`. The default
  is preserved for forward compatibility but is not active until upstream fixes QPager.
- **Arithmetic / shifts** require `isTensorNetwork=False`.
- **Dynamic `allocate` / `dispose`** require `isTensorNetwork=False`.

## License

MIT
>>>>>>> 03274ef (feat: initial qrackbind — Phase 1 + Phase 2 surface)
