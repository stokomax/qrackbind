"""qrackbind — nanobind bindings for the Qrack quantum simulator.

Element dtypes for ``state_vector`` / ``probabilities`` follow the Qrack
build precision (complex64 / float32 by default; complex128 / float64 with
FPPOW=6). Note that ``prob_perm`` queries a single full-register basis
state, while the existing ``prob_all`` (Phase 1) returns per-qubit |1>
probabilities.

Phase 4 — Pauli observables
---------------------------
The :class:`Pauli` enum (``PauliI``, ``PauliX``, ``PauliY``, ``PauliZ``) is
``IntEnum``-compatible — integer codes are accepted everywhere a ``Pauli``
is expected. Qrack's underlying values are non-sequential
(``PauliI=0, PauliX=1, PauliZ=2, PauliY=3``).

Pauli-aware methods on :class:`QrackSimulator`:

* ``measure_pauli(basis, qubit)`` — measure in a Pauli basis (collapses)
* ``exp_val(basis, qubit)`` — single-qubit Pauli expectation value
* ``exp_val_pauli(paulis, qubits)`` — tensor-product Pauli expectation
* ``variance_pauli(paulis, qubits)`` — Pauli observable variance
* ``exp_val_all(basis)`` — broadcast a single basis across all qubits
* ``exp_val_floats(qubits, weights)`` — weighted-sum expectation value
* ``variance_floats(qubits, weights)`` — weighted-sum variance

Phase 5 — Exception hierarchy
-----------------------------
All errors raised by :class:`QrackSimulator` are instances of
:class:`QrackException` (itself a subclass of ``RuntimeError``):

* :class:`QrackException` — base class for any qrackbind error
* :class:`QrackQubitError` — qubit index out of range ``[0, num_qubits)``
* :class:`QrackArgumentError` — invalid arguments (length mismatch,
  wrong array size, etc.)

Catch :class:`QrackException` to handle any qrackbind-specific failure;
the subclasses allow narrower handling when needed.
"""

# Import from _qrackbind_core — the underscore marks it as a private
# implementation detail; users import from qrackbind, not _qrackbind_core.
import warnings

import numpy as np

from ._core import (
    GateType,
    QrackCircuit,
    QrackSimulator as _QrackSimulator,
    Pauli,
    QrackException,
    QrackQubitError,
    QrackArgumentError,
    QrackStabilizer,                                    # Phase 10: imported directly — no state access to wrap
    QrackStabilizerHybrid as _QrackStabilizerHybrid,   # Phase 10: wrapped below for state_vector property
)


class QrackSimulator(_QrackSimulator):
    """
    Qrack quantum simulator with strong typing and NumPy integration.

    Drop-in replacement for pyqrack.QrackSimulator (with one notable
    departure: cloning is done via ``sim.clone()`` or ``copy.deepcopy(sim)``
    rather than the ``cloneSid=`` constructor kwarg).
    """

    __slots__ = ()

    @property
    def state_vector(self) -> "np.ndarray":
        """Full state vector snapshot as a 1-D complex NumPy array of length 2**num_qubits.

        Element dtype follows the Qrack build (complex64 by default). Returns a
        copy — modifying the array does not affect the simulator. To inject a
        state, use :meth:`set_state_vector`.
        """
        return self._state_vector_impl()

    @property
    def probabilities(self) -> "np.ndarray":
        """Probability of each basis state as a 1-D float NumPy array of length 2**num_qubits.

        Equivalent to ``abs(state_vector)**2``. Does not collapse the state.
        Element dtype follows the Qrack build (float32 by default).
        """
        return self._probabilities_impl()

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


class QrackStabilizerHybrid(_QrackStabilizerHybrid):
    """
    Stabilizer-hybrid simulator with automatic fallback to dense simulation.

    Exposes state_vector and probabilities as Python properties (the C++
    binding uses the _impl suffix pattern, same as QrackSimulator).
    """

    __slots__ = ()

    @property
    def state_vector(self) -> "np.ndarray":
        """Full state vector snapshot. Available before and after dense fallback.

        Before a non-Clifford gate is applied, Qrack materialises amplitudes
        from the stabilizer tableau. After fallback, returns the dense vector
        directly. Returns a copy; does not collapse the state.
        """
        return self._state_vector_impl()

    @property
    def probabilities(self) -> "np.ndarray":
        """Probability of each basis state as a 1-D float NumPy array.

        Equivalent to ``abs(state_vector)**2``. Does not collapse the state.
        """
        return self._probabilities_impl()


__all__ = [
    "GateType",
    "Pauli",
    "QrackArgumentError",
    "QrackCircuit",
    "QrackException",
    "QrackQubitError",
    "QrackSimulator",
    "QrackStabilizer",
    "QrackStabilizerHybrid",
]
try:
    from importlib.metadata import version as _version, PackageNotFoundError as _PackageNotFoundError
    __version__ = _version("qrackbind")
except _PackageNotFoundError:
    __version__ = "unknown"
