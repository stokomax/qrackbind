"""qrackbind — nanobind bindings for the Qrack quantum simulator"""

from collections.abc import Sequence
import enum

from qrackbind import QrackException as QrackException


class Pauli(enum.IntEnum):
    PauliI = 0

    PauliX = 1

    PauliY = 3

    PauliZ = 2

class QrackSimulator:
    """
    Qrack quantum simulator.

    Simulator Cloning
    -----------------
    A simulator can be deep-copied — producing an independent simulator with
    the same quantum state and configuration — using either the ``clone()``
    method or the standard ``copy.deepcopy`` / ``copy.copy`` protocol::

        import copy
        original = QrackSimulator(qubitCount=4)
        original.h(0)
        original.cnot(0, 1)              # Bell state on qubits 0 and 1

        branch_a = original.clone()      # explicit
        branch_b = copy.deepcopy(original)   # protocol-driven

    After construction, the clone is fully independent — gates applied to one
    have no effect on the other. The clone inherits the source simulator's
    qubit count and configuration.

    Typical use case is mid-circuit branching — capturing the state at a
    decision point so that multiple continuations can be explored without
    re-running the expensive state preparation.
    """

    def __init__(self, qubitCount: int, isTensorNetwork: bool = True, isSchmidtDecompose: bool = True, isSchmidtDecomposeMulti: bool = False, isStabilizerHybrid: bool = False, isBinaryDecisionTree: bool = False, isPaged: bool = True, isCpuGpuHybrid: bool = True, isOpenCL: bool = True, isHostPointer: bool = False, isSparse: bool = False, noise: float = 0.0) -> None: ...

    def __repr__(self) -> str: ...

    def h(self, qubit: int) -> None:
        """Hadamard gate."""

    def x(self, qubit: int) -> None:
        """Pauli X (bit flip) gate."""

    def y(self, qubit: int) -> None:
        """Pauli Y gate."""

    def z(self, qubit: int) -> None:
        """Pauli Z (phase flip) gate."""

    def s(self, qubit: int) -> None:
        """S gate — phase shift π/2."""

    def t(self, qubit: int) -> None:
        """T gate — phase shift π/4."""

    def sdg(self, qubit: int) -> None:
        """S† (inverse S) gate."""

    def tdg(self, qubit: int) -> None:
        """T† (inverse T) gate."""

    def sx(self, qubit: int) -> None:
        """√X gate (half-X). Native Qiskit basis gate."""

    def sxdg(self, qubit: int) -> None:
        """√X† gate (inverse √X)."""

    def rx(self, angle: float, qubit: int) -> None:
        """Rotate around X axis by angle radians. Equiv: exp(-i·angle/2·X)."""

    def ry(self, angle: float, qubit: int) -> None:
        """Rotate around Y axis by angle radians."""

    def rz(self, angle: float, qubit: int) -> None:
        """Rotate around Z axis by angle radians."""

    def r1(self, angle: float, qubit: int) -> None:
        """Phase rotation: apply e^(i·angle) to |1> state."""

    def u(self, theta: float, phi: float, lam: float, qubit: int) -> None:
        """General single-qubit unitary: U(θ,φ,λ). Decomposes to RZ·RY·RZ."""

    def u2(self, phi: float, lam: float, qubit: int) -> None:
        """U2 gate: U(π/2, φ, λ)."""

    def cnot(self, control: int, target: int) -> None:
        """Controlled-NOT (CNOT / CX) gate."""

    def cy(self, control: int, target: int) -> None:
        """Controlled-Y gate."""

    def cz(self, control: int, target: int) -> None:
        """Controlled-Z gate."""

    def swap(self, qubit1: int, qubit2: int) -> None:
        """SWAP gate."""

    def iswap(self, qubit1: int, qubit2: int) -> None:
        """iSWAP gate."""

    def ccnot(self, control1: int, control2: int, target: int) -> None:
        """Toffoli (CCX / CCNOT) gate."""

    def mcx(self, controls: Sequence[int], target: int) -> None:
        """Multiply-controlled X. Fires when all controls are |1>."""

    def macx(self, controls: Sequence[int], target: int) -> None:
        """Anti-controlled X. Fires when all controls are |0>."""

    def mcy(self, controls: Sequence[int], target: int) -> None:
        """Multiply-controlled Y."""

    def macy(self, controls: Sequence[int], target: int) -> None:
        """Anti-controlled Y."""

    def mcz(self, controls: Sequence[int], target: int) -> None:
        """Multiply-controlled Z."""

    def macz(self, controls: Sequence[int], target: int) -> None:
        """Anti-controlled Z."""

    def mch(self, controls: Sequence[int], target: int) -> None:
        """Multiply-controlled H."""

    def mcrz(self, angle: float, controls: Sequence[int], target: int) -> None:
        """Multiply-controlled RZ."""

    def mcu(self, controls: Sequence[int], target: int, theta: float, phi: float, lam: float) -> None:
        """Multiply-controlled U(θ,φ,λ) gate."""

    def mtrx(self, matrix: Sequence[complex], qubit: int) -> None:
        """Apply arbitrary 2x2 unitary. matrix is [m00, m01, m10, m11] row-major."""

    def mcmtrx(self, controls: Sequence[int], matrix: Sequence[complex], qubit: int) -> None:
        """Multiply-controlled arbitrary 2x2 unitary."""

    def macmtrx(self, controls: Sequence[int], matrix: Sequence[complex], qubit: int) -> None:
        """Anti-controlled arbitrary 2x2 unitary."""

    def multiplex1_mtrx(self, controls: Sequence[int], mtrxs: Sequence[complex], target: int) -> None:
        """
        Uniformly-controlled single-qubit gate. mtrxs is a flat list of 4 * 2**len(controls) complex values — one 2x2 unitary per control permutation, in row-major order.
        """

    def measure(self, qubit: int) -> bool:
        """Measure qubit. Returns True=|1>, False=|0>. Collapses state."""

    def measure_all(self) -> list[bool]:
        """Measure all qubits. Returns list[bool], LSB first."""

    def force_measure(self, qubit: int, result: bool) -> bool:
        """
        Force measurement outcome. Projects state to result without random draw.
        """

    def prob(self, qubit: int) -> float:
        """Probability of |1> for qubit. Does NOT collapse state."""

    def prob_all(self) -> list[float]:
        """Per-qubit |1> probabilities for all qubits. Does NOT collapse state."""

    def allocate(self, start: int, length: int) -> int:
        """
        Allocate 'length' new |0> qubits at index 'start'. Returns the start offset.
        Existing qubits at >= start shift up. Updates num_qubits automatically.
        Incompatible with isTensorNetwork=True.
        """

    def dispose(self, start: int, length: int) -> None:
        """
        Remove 'length' qubits starting at 'start'. Qubits must be separably |0> or |1>.
        Updates num_qubits automatically.
        """

    def allocate_qubits(self, n: int) -> int:
        """
        Allocate n new |0> qubits at the end. Returns the index of the first new qubit.
        """

    def qft(self, start: int, length: int, try_separate: bool = False) -> None:
        """
        Quantum Fourier Transform on a contiguous register [start, start+length).
        try_separate: optimization hint for QUnit — set True if you expect a permutation
        basis eigenstate result; otherwise leave False.
        """

    def iqft(self, start: int, length: int, try_separate: bool = False) -> None:
        """Inverse Quantum Fourier Transform on a contiguous register."""

    def qftr(self, qubits: Sequence[int], try_separate: bool = False) -> None:
        """Quantum Fourier Transform on an arbitrary list of qubit indices."""

    def iqftr(self, qubits: Sequence[int], try_separate: bool = False) -> None:
        """
        Inverse Quantum Fourier Transform on an arbitrary list of qubit indices.
        """

    def reset_all(self) -> None:
        """Reset all qubits to |0...0>."""

    def add(self, value: int, start: int, length: int) -> None:
        """
        Add classical integer 'value' to the quantum register [start, start+length).
        """

    def sub(self, value: int, start: int, length: int) -> None:
        """Subtract classical integer 'value' from the quantum register."""

    def mul(self, to_mul: int, mod_n: int, in_start: int, out_start: int, length: int) -> None:
        """Modular multiplication: out = in * to_mul mod mod_n (out of place)."""

    def div(self, to_div: int, mod_n: int, in_start: int, out_start: int, length: int) -> None:
        """Inverse modular multiplication (modular division, out of place)."""

    def pown(self, base: int, mod_n: int, in_start: int, out_start: int, length: int) -> None:
        """
        Modular exponentiation: out = base^in mod mod_n (out of place). Central operation of Shor's algorithm.
        """

    def mcmul(self, to_mul: int, mod_n: int, in_start: int, out_start: int, length: int, controls: Sequence[int]) -> None:
        """Controlled modular multiplication."""

    def mcdiv(self, to_div: int, mod_n: int, in_start: int, out_start: int, length: int, controls: Sequence[int]) -> None:
        """Controlled modular division (inverse modular multiplication)."""

    def lsl(self, shift: int, start: int, length: int) -> None:
        """Logical shift left — fills vacated bits with |0>."""

    def lsr(self, shift: int, start: int, length: int) -> None:
        """Logical shift right — fills vacated bits with |0>."""

    def rol(self, shift: int, start: int, length: int) -> None:
        """Circular rotate left."""

    def ror(self, shift: int, start: int, length: int) -> None:
        """Circular rotate right."""

    @property
    def num_qubits(self) -> int:
        """Number of qubits in this simulator."""

    def clone(self) -> QrackSimulator:
        """
        Return an independent deep copy of this simulator. The clone
        starts with a full copy of this simulator's quantum state and
        configuration; subsequent gates applied to either simulator
        have no effect on the other.
        """

    def __copy__(self) -> QrackSimulator:
        """Support for ``copy.copy(sim)``. Equivalent to ``sim.clone()``."""

    def __deepcopy__(self, memo: object) -> QrackSimulator:
        """Support for ``copy.deepcopy(sim)``. Equivalent to ``sim.clone()``."""

    def __enter__(self) -> QrackSimulator: ...

    def __exit__(self, exc_type: object | None, exc_val: object | None, exc_tb: object | None) -> bool: ...
