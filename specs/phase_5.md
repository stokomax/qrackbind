---
tags:
  - qrack
  - nanobind
  - python
  - exceptions
  - implementation
  - qrackbind
  - phase5
---
## qrackbind Phase 5 — Exception Handling and Error Propagation

Builds directly on [[qrackbind Phase 4]]. Phases 1–4 used ad-hoc `std::out_of_range` and `std::invalid_argument` throws with no consistent Python exception type. Phase 5 installs a proper exception hierarchy, registers it with nanobind, retrofits all existing error paths, and ensures every C++ failure surfaces in Python as a catchable, typed exception with a readable message — not a segfault, abort, or opaque `RuntimeError`.

This is a short phase (2–3 days) but a safety-critical one. Shipping without it means the first time a user passes a wrong qubit index in a framework integration they get an unrecoverable crash rather than a Python exception.

**Prerequisite:** All Phase 4 checklist items passing. `uv run pytest tests/test_phase4.py` green.

---

## Nanobind Learning Goals

| Topic | Where it appears |
|---|---|
| `nb::register_exception<T>()` — basic registration | §1 — `QrackException` |
| Exception hierarchy — chaining Python base classes | §1 — subclass hierarchy |
| Translating multiple C++ exception types to one Python type | §2 — exception translator |
| `nb::exception` vs `nb::register_exception` — when to use each | §2 — design note |
| Thread safety of exception translation | §3 — GIL and exception state |
| Retrofitting existing `.def()` lambdas — no API change required | §4 — mechanical retrofit |

---

## File Structure

| File | Changes |
|---|---|
| `bindings/exceptions.cpp` | **New file** — `bind_exceptions()` registers `QrackException` and its subclasses |
| `bindings/binding_core.h` | Add `QrackError` C++ exception class declaration |
| `bindings/simulator.cpp` | Retrofit `check_qubit` to throw `QrackError` instead of `std::out_of_range` |
| `bindings/module.cpp` | Add `bind_exceptions(m)` call — must be first, before all other bind calls |
| `CMakeLists.txt` | Add `bindings/exceptions.cpp` to sources |
| `src/qrackbind/__init__.py` | Export `QrackException`, `QrackQubitError`, `QrackArgumentError` |

---

## 1. Python Exception Hierarchy Design

Three exception classes cover all qrackbind error conditions:

```
QrackException (base)           ← catch-all for any qrackbind error
├── QrackQubitError              ← qubit index out of range
└── QrackArgumentError           ← invalid method arguments (length mismatch, bad value)
```

All inherit from Python's `Exception`. `QrackException` is the base — user code that only wants to catch qrackbind errors catches this. Subclasses allow more specific handling when needed.

This mirrors the pattern used by NumPy (`numpy.exceptions.UFuncTypeError` → `TypeError`) and PyTorch (`torch.TorchRuntimeError` → `RuntimeError`): a project-specific base class that inherits from a standard Python exception so it's automatically handled by broad `except Exception:` clauses.

---

## 2. C++ Exception Class

Define a single C++ exception type in `binding_core.h` that nanobind will translate to `QrackException` (and subclasses). Using one C++ type with a kind enum is simpler than a full C++ hierarchy and produces cleaner nanobind registration code.

```cpp
// ── File: bindings/binding_core.h — add after includes ───────────────────────

enum class QrackErrorKind { Generic, QubitOutOfRange, InvalidArgument };

class QrackError : public std::exception {
public:
    explicit QrackError(std::string msg,
                        QrackErrorKind kind = QrackErrorKind::Generic)
        : msg_(std::move(msg)), kind_(kind) {}

    const char* what() const noexcept override { return msg_.c_str(); }
    QrackErrorKind kind() const noexcept { return kind_; }

private:
    std::string    msg_;
    QrackErrorKind kind_;
};
```

---

## 3. `exceptions.cpp` — Registration

```cpp
// ── File: bindings/exceptions.cpp ────────────────────────────────────────────
#include "binding_core.h"

void bind_exceptions(nb::module_& m) {

    // Base: QrackException
    // Inherits from Python's RuntimeError so broad except clauses catch it.
    auto exc_base = nb::exception<QrackError>(m, "QrackException",
        PyExc_RuntimeError,
        "Base class for all qrackbind exceptions.\n\n"
        "Catch this to handle any error raised by QrackSimulator or\n"
        "QrackCircuit. Use the subclasses for more specific handling.\n\n"
        "Subclasses:\n"
        "  QrackQubitError    — qubit index out of valid range\n"
        "  QrackArgumentError — invalid method arguments");

    // QrackQubitError — qubit index violations
    // Inherits from QrackException (Python side) and IndexError (standard).
    nb::exception<QrackError>(m, "QrackQubitError",
        exc_base.ptr(),
        "Raised when a qubit index is out of the valid range [0, num_qubits).\n\n"
        "Example:\n"
        "  sim = QrackSimulator(qubitCount=2)\n"
        "  sim.h(5)  # raises QrackQubitError: qubit 5 out of range [0, 1]");

    // QrackArgumentError — argument validation failures
    nb::exception<QrackError>(m, "QrackArgumentError",
        exc_base.ptr(),
        "Raised when method arguments are invalid — mismatched lengths,\n"
        "wrong array sizes, or incompatible configuration.\n\n"
        "Example:\n"
        "  sim.exp_val_pauli([Pauli.PauliZ], [0, 1])  # raises: length mismatch");

    // Register the C++ → Python translator.
    // nanobind calls this for every QrackError that escapes a .def() lambda.
    nb::register_exception_translator(
        [](const std::exception_ptr& p, void*) {
            try {
                if (p) std::rethrow_exception(p);
            } catch (const QrackError& e) {
                // Route to the appropriate Python subclass based on kind
                switch (e.kind()) {
                    case QrackErrorKind::QubitOutOfRange:
                        PyErr_SetString(
                            PyObject_GetAttrString(
                                PyImport_ImportModule("qrackbind"),
                                "QrackQubitError"),
                            e.what());
                        break;
                    case QrackErrorKind::InvalidArgument:
                        PyErr_SetString(
                            PyObject_GetAttrString(
                                PyImport_ImportModule("qrackbind"),
                                "QrackArgumentError"),
                            e.what());
                        break;
                    default:
                        PyErr_SetString(
                            PyObject_GetAttrString(
                                PyImport_ImportModule("qrackbind"),
                                "QrackException"),
                            e.what());
                        break;
                }
            }
        },
        nullptr);
}
```

> **Registration order matters.** `bind_exceptions(m)` must be called **first** in `NB_MODULE`, before `bind_pauli`, `bind_simulator`, or any other `bind_*`. nanobind processes exception translators in LIFO order — the most recently registered translator runs first. Registering exceptions first means the QrackError translator runs before nanobind's default `std::exception` translator, so errors don't fall through to a generic `RuntimeError`.

---

## 4. `module.cpp` — Updated Registration Order

```cpp
// ── File: bindings/module.cpp ─────────────────────────────────────────────────
#include "binding_core.h"

void bind_exceptions(nb::module_& m);
void bind_pauli(nb::module_& m);
void bind_simulator(nb::module_& m);

NB_MODULE(_qrackbind_core, m) {
    m.doc() = "qrackbind — nanobind bindings for the Qrack quantum simulator";
    m.attr("__version__") = "0.1.0";

    bind_exceptions(m);    // ← must be first
    bind_pauli(m);
    bind_simulator(m);
}
```

---

## 5. Retrofitting `check_qubit` in `simulator.cpp`

The existing `check_qubit` helper throws `std::out_of_range`. Replace with `QrackError`:

```cpp
// ── File: bindings/simulator.cpp — update check_qubit ────────────────────────

// BEFORE (remove this):
void check_qubit(bitLenInt q, const char* fn) const {
    if (q >= numQubits)
        throw std::out_of_range(
            std::string(fn) + ": qubit " + std::to_string(q) +
            " out of range [0, " + std::to_string(numQubits - 1) + "]");
}

// AFTER (replace with this):
void check_qubit(bitLenInt q, const char* fn) const {
    if (q >= numQubits)
        throw QrackError(
            std::string(fn) + ": qubit " + std::to_string(q) +
            " out of range [0, " + std::to_string(numQubits - 1) + "]"
            + " (simulator has " + std::to_string(numQubits) + " qubits)",
            QrackErrorKind::QubitOutOfRange);
}
```

Also update all `throw std::invalid_argument(...)` calls to use `QrackError` with `QrackErrorKind::InvalidArgument`. A grep finds them all:

```bash
grep -n "throw std::invalid_argument\|throw std::out_of_range" bindings/simulator.cpp
```

Replace each one with `throw QrackError("...", QrackErrorKind::InvalidArgument)` or `QrackErrorKind::QubitOutOfRange` as appropriate.

---

## 6. Guard for Qrack's Own `std::runtime_error`

Qrack's C++ layer itself throws `std::runtime_error` when:
- The factory returns null (bad construction)
- `isTensorNetwork=True` is used with an incompatible operation
- OpenCL device initialisation fails

These should also surface as `QrackException` rather than a generic Python `RuntimeError`. Add a second translator in `exceptions.cpp`:

```cpp
// In bind_exceptions(), after the QrackError translator:
nb::register_exception_translator(
    [](const std::exception_ptr& p, void*) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const std::runtime_error& e) {
            // Only catch runtime_errors from Qrack — check message prefix
            std::string msg = e.what();
            if (msg.find("QrackSimulator") != std::string::npos ||
                msg.find("isTensorNetwork") != std::string::npos ||
                msg.find("factory returned null") != std::string::npos) {
                PyErr_SetString(
                    PyObject_GetAttrString(
                        PyImport_ImportModule("qrackbind"), "QrackException"),
                    e.what());
            }
            // Otherwise let nanobind's default translator handle it
        }
    },
    nullptr);
```

> **Why message-prefix filtering?** nanobind's exception translators form a chain — if a translator doesn't set a Python error, the next translator in the chain runs. Filtering on message content ensures we only intercept Qrack's own runtime errors, not unrelated `std::runtime_error` throws from STL or other libraries.

---

## 7. `__init__.py` — Export the Exceptions

```python
# src/qrackbind/__init__.py
from ._qrackbind_core import QrackSimulator, Pauli
from ._qrackbind_core import QrackException, QrackQubitError, QrackArgumentError

__all__ = [
    "QrackSimulator", "Pauli",
    "QrackException", "QrackQubitError", "QrackArgumentError",
]
```

Module docstring addition:

```python
"""
New in Phase 5:
  QrackException       — base class for all qrackbind errors
  QrackQubitError      — qubit index out of range (subclass of QrackException)
  QrackArgumentError   — invalid arguments (subclass of QrackException)

All errors from QrackSimulator and QrackCircuit now raise QrackException
or a subclass rather than generic RuntimeError or crashing.
"""
```

---

## 8. Test Suite

```python
# tests/test_phase5.py
import pytest
from qrackbind import (
    QrackSimulator, Pauli,
    QrackException, QrackQubitError, QrackArgumentError,
)


# ── Exception hierarchy ────────────────────────────────────────────────────────

class TestExceptionHierarchy:
    def test_qubit_error_is_qrack_exception(self):
        assert issubclass(QrackQubitError, QrackException)

    def test_argument_error_is_qrack_exception(self):
        assert issubclass(QrackArgumentError, QrackException)

    def test_qrack_exception_is_runtime_error(self):
        assert issubclass(QrackException, RuntimeError)

    def test_importable(self):
        import qrackbind
        assert hasattr(qrackbind, "QrackException")
        assert hasattr(qrackbind, "QrackQubitError")
        assert hasattr(qrackbind, "QrackArgumentError")


# ── Qubit out of range ─────────────────────────────────────────────────────────

class TestQubitOutOfRange:
    def _sim(self, n=3):
        return QrackSimulator(qubitCount=n)

    def test_h_raises_qubit_error(self):
        sim = self._sim()
        with pytest.raises(QrackQubitError):
            sim.h(99)

    def test_x_raises_qubit_error(self):
        sim = self._sim()
        with pytest.raises(QrackQubitError):
            sim.x(3)

    def test_measure_raises_qubit_error(self):
        sim = self._sim()
        with pytest.raises(QrackQubitError):
            sim.measure(10)

    def test_prob_raises_qubit_error(self):
        sim = self._sim()
        with pytest.raises(QrackQubitError):
            sim.prob(5)

    def test_rx_raises_qubit_error(self):
        sim = self._sim()
        with pytest.raises(QrackQubitError):
            sim.rx(1.0, 99)

    def test_cnot_control_raises(self):
        sim = self._sim()
        with pytest.raises(QrackQubitError):
            sim.cnot(99, 0)

    def test_cnot_target_raises(self):
        sim = self._sim()
        with pytest.raises(QrackQubitError):
            sim.cnot(0, 99)

    def test_exp_val_raises_qubit_error(self):
        sim = self._sim()
        with pytest.raises(QrackQubitError):
            sim.exp_val(Pauli.PauliZ, 99)

    def test_error_caught_as_base_class(self):
        sim = self._sim()
        with pytest.raises(QrackException):
            sim.h(99)

    def test_error_message_contains_qubit_index(self):
        sim = self._sim(n=3)
        with pytest.raises(QrackQubitError, match="5"):
            sim.h(5)

    def test_error_message_contains_valid_range(self):
        sim = self._sim(n=3)
        with pytest.raises(QrackQubitError, match="3"):
            sim.h(5)

    def test_boundary_valid(self):
        # Qubit at index numQubits-1 is valid — should not raise
        sim = self._sim(n=4)
        sim.h(3)   # index 3 is valid for a 4-qubit sim

    def test_boundary_invalid(self):
        # Qubit at index numQubits is invalid
        sim = self._sim(n=4)
        with pytest.raises(QrackQubitError):
            sim.h(4)


# ── Invalid arguments ─────────────────────────────────────────────────────────

class TestInvalidArguments:
    def test_exp_val_pauli_length_mismatch(self):
        sim = QrackSimulator(qubitCount=2)
        with pytest.raises(QrackArgumentError):
            sim.exp_val_pauli([Pauli.PauliZ], [0, 1])

    def test_variance_pauli_length_mismatch(self):
        sim = QrackSimulator(qubitCount=2)
        with pytest.raises(QrackArgumentError):
            sim.variance_pauli([Pauli.PauliZ, Pauli.PauliX], [0])

    def test_set_state_vector_wrong_size(self):
        import numpy as np
        sim = QrackSimulator(qubitCount=2)
        with pytest.raises((QrackArgumentError, ValueError)):
            sim.set_state_vector(np.zeros(3, dtype=np.complex64))

    def test_exp_val_floats_length_mismatch(self):
        sim = QrackSimulator(qubitCount=2)
        with pytest.raises(QrackArgumentError):
            sim.exp_val_floats([0], [1.0, 2.0])

    def test_argument_error_caught_as_base(self):
        sim = QrackSimulator(qubitCount=2)
        with pytest.raises(QrackException):
            sim.exp_val_pauli([Pauli.PauliZ], [0, 1])


# ── Error recovery — exceptions don't corrupt state ──────────────────────────

class TestStatePreservedAfterException:
    def test_state_intact_after_qubit_error(self):
        sim = QrackSimulator(qubitCount=2)
        sim.h(0)
        prob_before = sim.prob(0)
        try:
            sim.h(99)
        except QrackQubitError:
            pass
        prob_after = sim.prob(0)
        assert prob_before == pytest.approx(prob_after, abs=1e-5)

    def test_multiple_errors_dont_accumulate(self):
        sim = QrackSimulator(qubitCount=1)
        for _ in range(10):
            try:
                sim.h(99)
            except QrackQubitError:
                pass
        # Simulator should still work normally
        sim.h(0)
        assert sim.prob(0) == pytest.approx(0.5, abs=1e-4)
```

---

## 9. Phase 5 Completion Checklist

```
□ QrackException, QrackQubitError, QrackArgumentError importable from qrackbind
□ QrackQubitError is a subclass of QrackException
□ QrackArgumentError is a subclass of QrackException
□ QrackException is a subclass of RuntimeError
□ sim.h(99) raises QrackQubitError for a 3-qubit sim
□ sim.h(99) can also be caught as QrackException
□ Error message contains the invalid index value
□ Error message contains the valid range (num_qubits)
□ sim.h(numQubits - 1) does NOT raise — boundary is valid
□ sim.h(numQubits) raises QrackQubitError — boundary is invalid
□ exp_val_pauli length mismatch raises QrackArgumentError
□ variance_pauli length mismatch raises QrackArgumentError
□ State is not corrupted after a caught QrackQubitError
□ Ten consecutive caught errors don't affect simulator
□ bind_exceptions registered BEFORE bind_simulator in module.cpp
□ uv run pytest tests/test_phase1.py … tests/test_phase5.py — all green
```

---

## 10. What Phase 5 Leaves Out (Deferred)

| Item | Reason deferred |
|---|---|
| `QrackOpenCLError` subclass | OpenCL device errors are rare and platform-specific — message-prefix filtering covers them adequately for now |
| Exception chaining (`raise X from Y`) | nanobind's translator doesn't support `__cause__` natively; would require custom Python exception class; low priority |
| Warning system (`warnings.warn`) for deprecated aliases | Phase 7 (compat layer cleanup) |

---

## Related

- [[qrackbind Phase 4]]
- [[qrackbind Phase 3]]
- [[qrackbind Phase 1 Revised]]
- [[qrackbind Project Phase Breakdown]]
- [[nanobind Type Casting and std-bad_cast Troubleshooting]]
