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

## Features

- **Complete gate coverage**: All standard quantum gates including Pauli (`x`, `y`, `z`), Hadamard (`h`), phase (`s`, `t`), rotations (`rx`, `ry`, `rz`, `r1`), general unitary (`u`, `u2`), and multi-qubit gates (`cnot`, `cy`, `cz`, `swap`, `iswap`, `ccnot`)
- **High performance**: Near-native C++ performance through nanobind
- **Type safety**: Full type stubs for mypy/pyright
- **GPU acceleration**: OpenCL and CUDA backends when available
- **Python 3.12+**: Leverages nanobind's stable ABI

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
