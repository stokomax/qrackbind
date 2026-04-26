# qrackbind

Python bindings for the [Qrack](https://github.com/vm6502q/qrack) quantum computer simulator library.

## Why nanobind?

qrackbind uses [nanobind](https://github.com/wjakob/nanobind) for C++ bindings instead of ctypes, giving you:

- **No manual marshaling** — nanobind handles Python ↔ C++ type conversion automatically (classes, methods, exceptions). ctypes requires explicit `argtypes`/`restype` declarations, manual struct packing, and pointer juggling.
- **Near-native performance** — direct C++ function calls with minimal overhead. ctypes adds function call overhead and often copies data across the boundary.
- **Stable ABI** — compile once and the wheel works across CPython 3.12+ minor versions without recompilation. ctypes bindings can break when the underlying library changes or on different platforms.
- **Type safety** — nanobind generates accurate type stubs that mypy/pyright understand. ctypes has no static type information; errors surface at runtime.
- **Clean, Pythonic API** — C++ classes and methods appear as native Python objects with readable exceptions. ctypes exposes raw C signatures and opaque pointers.

## Installation

### Prerequisites

- Qrack library must be installed on your system
- C++17 compiler
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended)

### Build

```bash
# Install build dependencies
just setup

# Build and install (CPU-only)
just build-cpu

# Or with GPU acceleration (requires OpenCL)
just build

# Or with CUDA
just build-cuda
```

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

## Build Variants

| Command | Description |
|---------|-------------|
| `just build` | Default (OpenCL ON) |
| `just build-cpu` | CPU-only (OpenCL OFF) |
| `just build-cuda` | CUDA backend |
| `just build-double` | Double precision floats |
| `just build-no-simd` | Disable AVX/SSE3 |

## Development

```bash
# Run all tests
just test

# Run just the gate tests
uv run pytest tests/test_qrack_gates.py -v

# Format code
just fmt

# Lint
just lint

# Type check
just typecheck

# Generate type stubs
just stubs
```

## Architecture

```
Python API (src/qrackbind/)
    ↓
nanobind C++ bindings (bindings/)
    ↓
Qrack C++ library (system-installed)
    ↓
OpenCL / CUDA drivers (optional GPU backend)
```

## License

MIT
>>>>>>> 03274ef (feat: initial qrackbind — Phase 1 + Phase 2 surface)
