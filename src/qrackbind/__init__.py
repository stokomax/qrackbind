# Import from _qrackbind_core — the underscore marks it as a private
# implementation detail; users import from qrackbind, not _qrackbind_core.
import warnings

from ._core import QrackSimulator as _QrackSimulator, Pauli, QrackException


class QrackSimulator(_QrackSimulator):
    """
    Qrack quantum simulator with strong typing and NumPy integration.

    Drop-in replacement for pyqrack.QrackSimulator (with one notable
    departure: cloning is done via ``sim.clone()`` or ``copy.deepcopy(sim)``
    rather than the ``cloneSid=`` constructor kwarg).
    """

    __slots__ = ()

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


__all__ = ["Pauli", "QrackException", "QrackSimulator"]
__version__ = "0.1.0"
