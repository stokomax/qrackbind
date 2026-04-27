---
tags:
  - qrack
  - nanobind
  - python
  - stubs
  - typing
  - pyright
  - implementation
  - qrackbind
  - phase7
---
## qrackbind Phase 7 — Stub Generation and Type Annotations

Builds directly on [[qrackbind Phase 6]]. Phases 1–6 produce a fully functional extension module. Phase 7 ensures that module is properly typed — with `.pyi` stub files that make IDE autocompletion, Pyright, and mypy accurate, and with consistent docstrings on every binding that make `help()` useful and appear in generated stub documentation.

This phase contains no new C++ bindings. It is entirely about making the existing bindings legible to type checkers, language servers, and users.

**Prerequisite:** All Phase 6 checklist items passing. `uv run pytest tests/test_phase6.py` green.

---

## Nanobind Learning Goals

| Topic | Where it appears |
|---|---|
| `nanobind.stubgen` — how it works and what triggers `bad_cast` | §1 — stub generation pipeline |
| `nb::sig()` — manually overriding generated stub signatures | §2 — fixing problematic signatures |
| `NB_TYPE_CASTER` and `noexcept` rules for custom casters | §3 — Qrack typedef handling |
| `py.typed` marker — PEP 561 compliance | §4 — marker file |
| Pyright configuration for a nanobind project | §5 — `pyrightconfig.json` |
| Docstring conventions for nanobind — what appears in stubs vs `help()` | §6 — docstring audit |

---

## How nanobind Stub Generation Works

nanobind's `stubgen.py` generates `.pyi` files by importing the compiled extension and introspecting every registered type and function using nanobind's internal metadata. It does **not** parse C++ source files — it inspects the live Python module.

The generation pipeline triggered by scikit-build-core is:

```
cmake --build  →  _core.so compiled
                  ↓
nanobind.stubgen -m _core -o _core.pyi
                  ↓
scikit-build-core copies _core.pyi into the wheel
```

`stubgen` runs automatically at the end of `pip install` (step 16 in the build log). If it fails with `bad_cast` or `ImportError`, the build fails even though the `.so` compiled successfully. This is why Phase 7 must audit every signature for stubgen compatibility before declaring the project complete.

---

## 1. File Structure

| File | Role |
|---|---|
| `src/qrackbind/_core.pyi` | Generated — do not edit by hand |
| `src/qrackbind/py.typed` | PEP 561 marker — checked into source control |
| `pyrightconfig.json` | Pyright configuration at project root |
| `justfile` | `just stubs` target added |
| `pyproject.toml` | `[tool.uv.scripts] stubs` entry added |

---

## 2. Identifying and Fixing Problematic Signatures

Run stubgen manually to find all failures before relying on the build to catch them:

```bash
cd build/cp314-cp314-linux_x86_64
python -W error -m nanobind.stubgen -q -i . -M py.typed -m _core -o _core.pyi 2>&1
```

### Known signature issues in qrackbind

**`bitCapInt` on 64-bit Linux with `ENABLE_UINT128`:**

If Qrack is compiled with `ENABLE_UINT128`, `bitCapInt` resolves to `unsigned __int128`. stubgen cannot emit a Python annotation for this type and throws `bad_cast`.

Check first:
```bash
grep -E "typedef|using" /usr/include/qrack/qrack_types.hpp | grep bitCapInt
```

If `__int128` appears, add `nb::sig()` to every affected method:

```cpp
// Affected methods: get_amplitude, set_amplitude, prob_all, prob_mask,
//                   m_reg, set_permutation, exp_val_bits_factorized,
//                   measure_shots, allocate, dispose

// Example fix:
.def("m_reg",
    [](QrackSim& s, bitLenInt start, bitLenInt length) -> bitCapInt {
        return s.sim->MReg(start, length);
    },
    nb::arg("start"), nb::arg("length"),
    nb::sig("def m_reg(self, start: int, length: int) -> int"),
    "Measure a contiguous register, returning result as a classical integer.")

.def("prob_all",
    [](QrackSim& s, bitCapInt perm) -> float {
        return static_cast<float>(s.sim->ProbAll(perm));
    },
    nb::arg("index"),
    nb::sig("def prob_all(self, index: int) -> float"),
    "Probability of a specific basis state by integer index.")
```

The pattern is the same for every `bitCapInt` parameter or return: add `nb::sig()` and map `bitCapInt` → `int`.

**`nb::arg().doc()` version requirement:**

`nb::arg().doc()` requires nanobind ≥ 2.0. Confirm the installed version:

```bash
python -c "import nanobind; print(nanobind.__version__)"
```

If below 2.0, remove all `.doc()` calls from `nb::arg()` and rely on the function-level docstring and `nb::sig()` instead.

**`nb::sig()` and `nb::arg().doc()` conflict:**

These two cannot coexist on the same `.def()`. When `nb::sig()` is present it overrides the entire signature; `.doc()` on `nb::arg()` is redundant and causes conflicting internal metadata. Use one or the other:

```cpp
// ✓ sig() only — full signature control
.def("measure_pauli",
    [...],
    nb::arg("basis"), nb::arg("qubit"),
    nb::sig("def measure_pauli(self, basis: Pauli, qubit: int) -> bool"),
    "Measure in the given Pauli basis.")

// ✓ arg().doc() only — no sig override
.def("h",
    [...],
    nb::arg("qubit").doc("Index of the qubit to apply Hadamard to."),
    "Hadamard gate.")
```

---

## 3. Auditing All Bindings for `nb::sig()` Completeness

Every method that takes or returns a non-primitive Qrack type needs a `nb::sig()`. The following table covers all methods across all phases:

### `simulator.cpp` — `nb::sig()` audit

| Method | Return type | Needs sig? | Reason |
|---|---|---|---|
| `state_vector` | `nb::ndarray<...>` | Yes | ndarray type needs explicit NumPy annotation |
| `probabilities` | `nb::ndarray<...>` | Yes | same |
| `set_state_vector` | `void` | Yes | ndarray input type |
| `get_amplitude` | `cf32` | No | `complex` caster handles it |
| `set_amplitude` | `void` | No | `complex` input handled by caster |
| `m_reg` | `bitCapInt` | **Yes if __int128** | maps to `int` |
| `prob_all` | `float` from `bitCapInt` arg | **Yes if __int128** | arg maps to `int` |
| `prob_mask` | `float` from `bitCapInt` args | **Yes if __int128** | args map to `int` |
| `set_permutation` | `void` from `bitCapInt` | **Yes if __int128** | arg maps to `int` |
| `get_reduced_density_matrix` | `nb::ndarray<...>` | Yes | 2-D ndarray |
| `exp_val_pauli` | `float` | Yes | `list[Pauli]` param |
| `variance_pauli` | `float` | Yes | `list[Pauli]` param |
| `measure_pauli` | `bool` | Yes | `Pauli` param |
| `exp_val` | `float` | Yes | `Pauli` param |
| `exp_val_unitary` | `float` | Yes | `list[complex]` param |
| `measure_shots` | `dict` from `bitCapInt` | **Yes if __int128** | qubit powers map to `int` |
| `__exit__` | `bool` | Yes | exception type params |

### `circuit.cpp` — `nb::sig()` audit

| Method | Needs sig? | Reason |
|---|---|---|
| `append_gate` | Yes | `GateType` param, `list[float]` default |
| `run` | Yes | `QrackSimulator` param |
| `inverse` | Yes | `QrackCircuit` return |
| `append` | Yes | `QrackCircuit` param |
| `to_qasm` | No | `str` return inferred correctly |
| `gate_count` | No | `int` inferred correctly |

---

## 4. `py.typed` Marker

PEP 561 requires a `py.typed` marker file in the package directory to signal to type checkers that the package supports typing. Create it as an empty file:

```bash
touch src/qrackbind/py.typed
```

Check it into source control. scikit-build-core and nanobind's stubgen both handle this file automatically during the wheel build — nanobind generates it in the build directory, and scikit-build-core includes it in the wheel. The source-controlled copy ensures it is present during editable installs.

Verify it is included in the wheel manifest by adding to `pyproject.toml`:

```toml
[tool.scikit-build]
wheel.packages = ["src/qrackbind"]
# py.typed is picked up automatically as it's inside the package directory
```

---

## 5. Pyright Configuration

Create `pyrightconfig.json` at the project root:

```json
{
    "include": ["src", "tests"],
    "exclude": ["build", "_skbuild", ".venv"],
    "venvPath": ".",
    "venv": ".venv",
    "pythonVersion": "3.12",
    "typeCheckingMode": "standard",
    "reportMissingModuleSource": "none",
    "reportUnknownMemberType": "none",
    "reportUnknownVariableType": "none"
}
```

**`reportMissingModuleSource: none`** — suppresses the warning that `_qrackbind_core` has no Python source (it's a compiled extension — expected).

**`reportUnknownMemberType: none`** — nanobind stubs occasionally leave some return types as `Unknown` for overloaded functions. Setting this to none prevents false positives until the stubs are fully annotated.

Run Pyright:

```bash
uv run pyright
```

The target for Phase 7 completion is zero errors. Warnings about `Unknown` types in the extension itself (not in user code) are acceptable if the underlying type genuinely can't be annotated (e.g. a bitCapInt overload with no `nb::sig()`).

---

## 6. Docstring Audit

Every `.def()`, `.def_prop_ro()`, `.def_prop_rw()`, and `nb::class_<>` call must have a non-empty docstring. These appear in:
- The generated `.pyi` stub as inline comments
- Python `help()` output
- IDE hover text (via the language server reading the stub)

### Docstring format convention

```cpp
// Short one-line summary.
//
// Extended description if needed. Use \n\n for paragraphs.
// Reference related methods by name.
//
// Example:
//   sim.h(0)
//   sim.h(qubit=0)   # keyword form

"Short one-line summary.\n\n"
"Extended description if needed.\n\n"
"Example:\n"
"  sim.h(0)"
```

### Minimum docstring requirements

| Binding type | Minimum content |
|---|---|
| Class | One-line purpose + list of key methods |
| `__init__` | All non-obvious parameters explained |
| Gate methods | What the gate does, whether it collapses state |
| Property | What it returns, units if applicable |
| Exception class | When it is raised, with an example |

### grep audit — find empty docstrings

```bash
# Find .def() calls with no trailing string literal
grep -n '\.def("' bindings/*.cpp | grep -v '"$' | grep -v '//'
```

Any `.def("method_name", lambda, ...)` that ends without a string literal argument has no docstring.

---

## 7. `justfile` and `uv` Script Targets

Add a `stubs` target to the justfile:

```makefile
# Regenerate .pyi stubs from the compiled extension
stubs:
    #!/usr/bin/env bash
    set -euo pipefail
    BUILD_DIR=$(find build -name "_core.cpython-*.so" 2>/dev/null | head -1 | xargs dirname)
    if [ -z "${BUILD_DIR}" ]; then
        echo "Extension not found. Run 'just build' first."
        exit 1
    fi
    echo "Generating stubs from ${BUILD_DIR}..."
    uv run python -m nanobind.stubgen \
        -q \
        -i "${BUILD_DIR}" \
        -M py.typed \
        -m _core \
        -o src/qrackbind/
    echo "Stubs written to src/qrackbind/_core.pyi"

# Run pyright type checker
typecheck:
    uv run pyright

# Full quality check: test + stubs + typecheck
check:
    just test
    just stubs
    just typecheck
```

Add the equivalent `uv run` scripts to `pyproject.toml`:

```toml
[tool.uv.scripts]
stubs     = "python -m nanobind.stubgen -q -i build/ -M py.typed -m _core -o src/qrackbind/"
typecheck = "pyright"
check     = { cmd = "bash -c 'pytest tests/ -v && python -m nanobind.stubgen -q -i build/ -M py.typed -m _core -o src/qrackbind/ && pyright'" }
```

---

## 8. Expected Stub Output

After a successful `just stubs`, `src/qrackbind/_core.pyi` should contain approximately:

```python
from __future__ import annotations
import numpy
import numpy.typing
from typing import overload

class Pauli(enum.IntEnum):
    PauliI = 0
    PauliX = 1
    PauliZ = 2
    PauliY = 3

class QrackException(RuntimeError): ...
class QrackQubitError(QrackException): ...
class QrackArgumentError(QrackException): ...

class QrackSimulator:
    def __init__(
        self,
        cloneSid: int = -1,
        qubitCount: int = -1,
        isTensorNetwork: bool = True,
        isSchmidtDecompose: bool = True,
        isSchmidtDecomposeMulti: bool = False,
        isStabilizerHybrid: bool = False,
        isBinaryDecisionTree: bool = False,
        isPaged: bool = True,
        isCpuGpuHybrid: bool = True,
        isOpenCL: bool = True,
        isHostPointer: bool = False,
        isSparse: bool = False,
        noise: float = 0.0,
    ) -> None: ...

    def h(self, qubit: int) -> None: ...
    def x(self, qubit: int) -> None: ...
    def y(self, qubit: int) -> None: ...
    def z(self, qubit: int) -> None: ...
    # ... all gates ...

    def measure(self, qubit: int) -> bool: ...
    def measure_all(self) -> list[bool]: ...
    def prob(self, qubit: int) -> float: ...

    @property
    def state_vector(self) -> numpy.ndarray[numpy.complex64]: ...

    @property
    def probabilities(self) -> numpy.ndarray[numpy.float32]: ...

    def set_state_vector(self, state: numpy.ndarray[numpy.complex64]) -> None: ...
    def get_amplitude(self, index: int) -> complex: ...
    def set_amplitude(self, index: int, amplitude: complex) -> None: ...

    def exp_val_pauli(self, paulis: list[Pauli], qubits: list[int]) -> float: ...
    def measure_pauli(self, basis: Pauli, qubit: int) -> bool: ...

    @property
    def num_qubits(self) -> int: ...
    @property
    def sid(self) -> int: ...

    def __enter__(self) -> QrackSimulator: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> bool: ...
    def __repr__(self) -> str: ...

class GateType(enum.Enum):
    H = ...
    X = ...
    # ...

class QrackCircuit:
    def __init__(self, qubitCount: int) -> None: ...
    def append_gate(
        self,
        gate: GateType,
        qubits: list[int],
        params: list[float] = [],
    ) -> None: ...
    def run(self, simulator: QrackSimulator) -> None: ...
    def inverse(self) -> QrackCircuit: ...
    def optimize(self) -> None: ...
    def append(self, other: QrackCircuit) -> None: ...
    def to_qasm(self) -> str: ...
    @property
    def gate_count(self) -> int: ...
    def __repr__(self) -> str: ...
```

---

## 9. Test Suite

Phase 7 tests verify that the stubs exist, are valid Python, and that Pyright passes — rather than testing runtime behaviour.

```python
# tests/test_phase7.py
import ast
import importlib
import subprocess
import sys
from pathlib import Path
import pytest


STUB_PATH = Path("src/qrackbind/_core.pyi")
PYTYPED_PATH = Path("src/qrackbind/py.typed")


# ── Stub file existence ───────────────────────────────────────────────────────

class TestStubFiles:
    def test_stub_file_exists(self):
        assert STUB_PATH.exists(), \
            f"{STUB_PATH} not found. Run 'just stubs' to generate."

    def test_pytyped_marker_exists(self):
        assert PYTYPED_PATH.exists(), \
            f"{PYTYPED_PATH} not found. Create it with: touch src/qrackbind/py.typed"

    def test_stub_is_valid_python(self):
        source = STUB_PATH.read_text()
        try:
            ast.parse(source)
        except SyntaxError as e:
            pytest.fail(f"_core.pyi is not valid Python: {e}")

    def test_stub_is_non_empty(self):
        assert STUB_PATH.stat().st_size > 100, "_core.pyi appears to be empty"


# ── Stub content completeness ──────────────────────────────────────────────────

class TestStubContent:
    @pytest.fixture(autouse=True)
    def stub_source(self):
        self.source = STUB_PATH.read_text()

    def test_qrack_simulator_in_stub(self):
        assert "class QrackSimulator" in self.source

    def test_pauli_in_stub(self):
        assert "class Pauli" in self.source

    def test_pauli_values_in_stub(self):
        assert "PauliI" in self.source
        assert "PauliX" in self.source
        assert "PauliY" in self.source
        assert "PauliZ" in self.source

    def test_exceptions_in_stub(self):
        assert "QrackException" in self.source
        assert "QrackQubitError" in self.source
        assert "QrackArgumentError" in self.source

    def test_qrack_circuit_in_stub(self):
        assert "class QrackCircuit" in self.source

    def test_gate_type_in_stub(self):
        assert "class GateType" in self.source

    def test_state_vector_annotated(self):
        assert "state_vector" in self.source
        assert "numpy" in self.source or "ndarray" in self.source

    def test_exp_val_pauli_in_stub(self):
        assert "exp_val_pauli" in self.source

    def test_measure_pauli_in_stub(self):
        assert "measure_pauli" in self.source

    def test_no_unknown_in_public_methods(self):
        # nanobind emits Unknown when a type can't be resolved.
        # This is acceptable inside the stub for internal types but
        # should not appear on publicly documented method signatures.
        lines_with_unknown = [
            line for line in self.source.splitlines()
            if "Unknown" in line and not line.strip().startswith("#")
        ]
        # Allow some Unknown — flag if more than a threshold
        assert len(lines_with_unknown) < 5, \
            f"Too many Unknown types in stub ({len(lines_with_unknown)}):\n" + \
            "\n".join(lines_with_unknown[:10])


# ── Runtime docstrings ────────────────────────────────────────────────────────

class TestDocstrings:
    def setup_method(self):
        from qrackbind import QrackSimulator, Pauli, QrackCircuit, GateType
        self.QrackSimulator = QrackSimulator
        self.Pauli = Pauli
        self.QrackCircuit = QrackCircuit
        self.GateType = GateType

    def test_simulator_class_has_docstring(self):
        assert self.QrackSimulator.__doc__ is not None
        assert len(self.QrackSimulator.__doc__) > 10

    def test_pauli_has_docstring(self):
        assert self.Pauli.__doc__ is not None

    def test_gate_type_has_docstring(self):
        assert self.GateType.__doc__ is not None

    def test_circuit_class_has_docstring(self):
        assert self.QrackCircuit.__doc__ is not None

    def test_key_methods_have_docstrings(self):
        from qrackbind import QrackSimulator
        sim = QrackSimulator(qubitCount=1)
        for method_name in ["h", "x", "measure", "prob", "exp_val_pauli",
                            "measure_pauli", "set_state_vector"]:
            method = getattr(QrackSimulator, method_name, None)
            assert method is not None, f"Method {method_name} not found"
            assert method.__doc__ is not None and len(method.__doc__) > 5, \
                f"Method {method_name} has no docstring"


# ── Pyright ───────────────────────────────────────────────────────────────────

class TestPyright:
    def test_pyright_passes(self):
        result = subprocess.run(
            [sys.executable, "-m", "pyright", "--outputjson"],
            capture_output=True,
            text=True,
        )
        # pyright exits with 1 on errors, 0 on success
        # Parse the JSON to get error count
        import json
        try:
            data = json.loads(result.stdout)
            error_count = data.get("summary", {}).get("errorCount", 0)
            assert error_count == 0, \
                f"Pyright reported {error_count} errors:\n{result.stdout}"
        except json.JSONDecodeError:
            # Fall back to exit code check if JSON parsing fails
            assert result.returncode == 0, \
                f"Pyright failed:\n{result.stdout}\n{result.stderr}"
```

---

## 10. Phase 7 Completion Checklist

```
□ just stubs runs without error
□ src/qrackbind/_core.pyi exists and is non-empty
□ src/qrackbind/py.typed marker file exists
□ _core.pyi passes ast.parse() — valid Python syntax
□ QrackSimulator, Pauli, QrackException, QrackCircuit, GateType all appear in stub
□ state_vector and probabilities annotated as numpy.ndarray in stub
□ exp_val_pauli signature shows list[Pauli] parameter
□ measure_pauli signature shows Pauli parameter
□ __exit__ signature shows correct exception type parameters
□ No more than 5 Unknown types in stub (acceptable threshold for nanobind)
□ bitCapInt parameters all have nb::sig() overrides if ENABLE_UINT128
□ No nb::sig() + nb::arg().doc() conflicts on any .def()
□ All classes have non-empty docstrings
□ All public methods have non-empty docstrings
□ pyright runs with zero errors
□ just typecheck passes
□ uv run stubs equivalent works
□ uv run typecheck equivalent works
□ uv run pytest tests/test_phase7.py — all green
□ uv run pytest tests/test_phase1.py … tests/test_phase7.py — all green
```

---

## 11. What Phase 7 Leaves Out (Deferred)

| Item | Reason deferred |
|---|---|
| mypy compatibility | mypy has different stub behaviour from Pyright for nanobind extensions; addressing both adds complexity disproportionate to the benefit at this stage |
| `__all__` in `_core.pyi` | nanobind does not generate `__all__` — adding it manually risks going stale; Pyright handles re-exports correctly without it |
| Sphinx / pdoc documentation generation | Documentation tooling is a separate concern from type correctness; deferred to a documentation phase |
| Stub testing in CI on every PR | The stub regeneration step is slow; CI runs it on tag builds only until a caching strategy is in place |

---

## Related

- [[qrackbind Phase 6]]
- [[qrackbind Phase 5]]
- [[qrackbind Phase 4]]
- [[qrackbind Phase 9]]
- [[qrackbind Project Phase Breakdown]]
- [[nanobind Type Casting and std-bad_cast Troubleshooting]]
