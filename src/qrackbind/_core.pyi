"""qrackbind — nanobind bindings for the Qrack quantum simulator"""

from collections.abc import Sequence
import enum
from typing import Annotated

import numpy
from numpy.typing import NDArray

class QrackException(RuntimeError):
    """
    Base class for all qrackbind errors.

    Inherits RuntimeError so it can be caught by generic error handlers.
    """

class QrackQubitError(QrackException):
    """
    Qubit index out of the valid range [0, num_qubits).

    Raised when a gate or measurement method is called with an index
    that exceeds the simulator's qubit count.
    """

class QrackArgumentError(QrackException):
    """
    Invalid method arguments.

    Raised when arguments have the wrong type, length mismatch,
    out-of-range values, or other argument validation failures.
    """

class Pauli(enum.IntEnum):
    """
    Pauli operator basis for single-qubit observables.

    Used by measure_pauli(), exp_val(), exp_val_pauli(), and
    variance_pauli(). Integer codes are accepted wherever a Pauli
    is expected (IntEnum semantics).

    Qrack's underlying values are non-sequential:
        PauliI = 0, PauliX = 1, PauliZ = 2, PauliY = 3.
    """

    PauliI = 0
    """Identity operator — no rotation applied."""

    PauliX = 1
    """Pauli X basis — measures in the X (Hadamard) basis."""

    PauliY = 3
    """Pauli Y basis — measures in the Y basis (S†H rotation)."""

    PauliZ = 2
    """Pauli Z basis — computational basis, no rotation needed."""

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

    def measure_pauli(self, basis: Pauli, qubit: int) -> bool:
        """
        Measure a qubit in the specified Pauli basis.

        Rotates the qubit into the computational basis, measures, and
        rotates back. Returns the same bit-valued bool as :meth:`measure`:
        ``True`` if the rotated qubit collapsed to ``|1>``, ``False`` for
        ``|0>``. For Pauli Z, this means ``True`` ↔ −1 eigenvalue and
        ``False`` ↔ +1 eigenvalue. The state is collapsed in the chosen
        basis.

        Example::

            sim.x(0)
            sim.measure_pauli(Pauli.PauliZ, 0)  # → True (|1>)
        """

    def exp_val(self, basis: Pauli, qubit: int) -> float:
        """
        Single-qubit Pauli expectation value.

        Equivalent to ``exp_val_pauli([basis], [qubit])``. Result is
        in [-1.0, +1.0]. Does not collapse the state.

        Example::

            sim.h(0)
            print(sim.exp_val(Pauli.PauliX, 0))  # → 1.0
        """

    def exp_val_pauli(self, paulis: Sequence[Pauli], qubits: Sequence[int]) -> float:
        """
        Expectation value of a Pauli tensor product observable.

        Returns <ψ|P₀⊗P₁⊗…⊗Pₙ|ψ> where each Pᵢ is a Pauli operator
        acting on the corresponding qubit. Result is in [-1.0, +1.0].
        Does not collapse the state.

        ``paulis`` and ``qubits`` must have equal length.

        Example::

            # Measure <ZZ> on a Bell state — should be +1
            sim.h(0); sim.cnot(0, 1)
            sim.exp_val_pauli([Pauli.PauliZ, Pauli.PauliZ], [0, 1])
        """

    def variance_pauli(self, paulis: Sequence[Pauli], qubits: Sequence[int]) -> float:
        """
        Variance of a Pauli tensor product observable.

        For a Pauli operator P (P² = I), Var(P) = 1 − <P>².
        Result is in [0.0, 1.0]. Does not collapse the state.

        Eigenstates have variance 0; maximally-mixed states have
        variance 1.
        """

    def exp_val_all(self, basis: Pauli) -> float:
        """
        Expectation value of the same Pauli operator applied to every
        qubit. Equivalent to::

            sim.exp_val_pauli([basis] * sim.num_qubits,
                              list(range(sim.num_qubits)))
        """

    def exp_val_floats(self, qubits: Sequence[int], weights: Sequence[float]) -> float:
        """
        Expectation value of a weighted single-qubit observable.

        Each qubit gets two classical eigenvalues — one for ``|0>`` and
        one for ``|1>``. ``weights`` must have length ``2 * len(qubits)``::

            weights = [w0_for_|0>, w0_for_|1>,
                       w1_for_|0>, w1_for_|1>,
                       ...]

        Returns ``Σᵢ (wᵢ⁰ · P(qᵢ=|0>) + wᵢ¹ · P(qᵢ=|1>))``. Used by
        PennyLane's Hamiltonian expectation-value path.
        """

    def variance_floats(self, qubits: Sequence[int], weights: Sequence[float]) -> float:
        """
        Variance of a weighted single-qubit observable. Symmetric
        counterpart to :meth:`exp_val_floats`. ``weights`` must have
        length ``2 * len(qubits)`` (see :meth:`exp_val_floats`).
        """

    def exp_val_unitary(self, qubits: Sequence[int], basis_ops: Sequence[complex], eigen_vals: Sequence[float] = []) -> float:
        """
        Expectation value of a tensor product of arbitrary 2x2 unitary
        observables. ``basis_ops`` is a flat list of ``4 * len(qubits)``
        complex values — one 2x2 matrix per qubit, in row-major order.
        """

    def variance_unitary(self, qubits: Sequence[int], basis_ops: Sequence[complex], eigen_vals: Sequence[float] = []) -> float:
        """
        Variance of a tensor product of arbitrary 2x2 unitary observables.
        See :meth:`exp_val_unitary` for parameter conventions.
        """

    def exp_val_bits_factorized(self, qubits: Sequence[int], perms: Sequence[int]) -> float:
        """
        Per-qubit weighted expectation value using bitCapInt permutation
        weights. Low-level API used by Shor's and arithmetic expectation
        paths.
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

    def m_reg(self, start: int, length: int) -> int:
        """
        Measure a contiguous register of 'length' qubits starting at 'start'. Collapses state. Returns result as a classical integer.
        """

    def set_permutation(self, value: int) -> None:
        """
        Reset state to the computational basis state |value>. Bit i of value sets qubit i.
        """

    def measure_shots(self, qubits: Sequence[int], shots: int) -> dict[int, int]:
        """
        Sample 'shots' measurements of 'qubits' without collapsing state. Returns dict[int, int]: measurement result -> count.
        """

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

    def mcpown(self, base: int, mod_n: int, in_start: int, out_start: int, length: int, controls: Sequence[int]) -> None:
        """Controlled modular exponentiation."""

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

    def set_state_vector(self, state: Annotated[NDArray[numpy.complex64], dict(shape=(None,), order='C', writable=False)]) -> None:
        """
        Set the simulator's quantum state from a 1-D complex NumPy array.
        Array must be C-contiguous, have length 2**num_qubits, and use the
        build's complex dtype (complex64 by default). The array is copied
        into the simulator. SetQuantumState does NOT renormalise — call
        update_running_norm() afterwards if the input may not be unit-norm.
        """

    def get_amplitude(self, index: int) -> complex:
        """
        Get the complex amplitude of a specific basis state by integer index.
        index must be in [0, 2**num_qubits). Does not collapse the state.
        """

    def set_amplitude(self, index: int, amplitude: complex) -> None:
        """
        Set the complex amplitude of a specific basis state.
        Does NOT re-normalise — call update_running_norm() if the resulting
        state may not be unit-norm.
        """

    def get_reduced_density_matrix(self, qubits: Sequence[int]) -> Annotated[NDArray[numpy.complex64], dict(shape=(None, None))]:
        """
        Reduced density matrix of the specified qubits as a 2-D complex
        NumPy array of shape (2**k, 2**k), where k = len(qubits). All other
        qubits are traced out. The result is Hermitian, positive semi-definite,
        and has trace 1.
        """

    def prob_perm(self, index: int) -> float:
        """
        Probability of a specific full-register basis state by integer index.
        More efficient than ``probabilities[index]`` for sparse queries.
        Does not collapse the state.

        Note: distinct from the ``prob_all`` property, which returns the
        per-qubit |1> probabilities (length num_qubits).
        """

    def prob_mask(self, mask: int, permutation: int) -> float:
        """
        Probability that the masked qubits match the given permutation.
        mask selects which qubits to check; permutation gives their expected
        values. Bits not in mask should be 0 in permutation.
        """

    def update_running_norm(self) -> None:
        """
        Recompute and apply the state vector normalisation factor.
        Call after set_amplitude() or set_state_vector() if the injected
        state may not be exactly unit-norm.
        """

    def first_nonzero_phase(self) -> float:
        """
        Return the phase angle of the lowest-index nonzero amplitude, in
        radians. Useful for global phase normalisation before state
        comparison.
        """

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

class GateType(enum.Enum):
    """
    Gate type identifier for ``QrackCircuit.append_gate()``.

    Used to specify which gate to add to the circuit without
    immediately applying it to a simulator.
    """

    H = 0
    """Hadamard gate"""

    X = 1
    """Pauli X (bit flip)"""

    Y = 2
    """Pauli Y"""

    Z = 3
    """Pauli Z (phase flip)"""

    S = 4
    """S gate (phase π/2)"""

    T = 5
    """T gate (phase π/4)"""

    IS = 6
    """S† (inverse S)"""

    IT = 7
    """T† (inverse T)"""

    SqrtX = 8
    """√X gate"""

    ISqrtX = 9
    """√X† gate"""

    RX = 10
    """X rotation — 1 angle param"""

    RY = 11
    """Y rotation — 1 angle param"""

    RZ = 12
    """Z rotation — 1 angle param"""

    R1 = 13
    """Phase rotation — 1 angle param"""

    CNOT = 14
    """Controlled NOT — 2 qubits"""

    CY = 15
    """Controlled Y — 2 qubits"""

    CZ = 16
    """Controlled Z — 2 qubits"""

    CH = 17
    """Controlled H — 2 qubits"""

    MCX = 18
    """Multi-controlled X — last qubit is target"""

    MCY = 19
    """Multi-controlled Y — last qubit is target"""

    MCZ = 20
    """Multi-controlled Z — last qubit is target"""

    SWAP = 21
    """SWAP gate — 2 qubits"""

    ISWAP = 22
    """iSWAP gate — not yet implemented"""

    U = 23
    """Arbitrary unitary U(θ, φ, λ) — 3 angle params"""

    Mtrx = 24
    """Arbitrary 2x2 unitary — 8 float params (4 complex)"""

    MCMtrx = 25
    """Multi-controlled arbitrary 2x2 — 8 float params"""

class QrackCircuit:
    """
    A replayable, optimisable quantum circuit.

    Records gate operations that can be executed on any
    :class:`QrackSimulator` via :meth:`run`. Circuits can be
    inverted, combined, and (in future) serialised to QASM.

    Example::

        circ = QrackCircuit(2)
        circ.append_gate(GateType.H, [0])
        circ.append_gate(GateType.CNOT, [0, 1])
        sim = QrackSimulator(qubitCount=2)
        circ.run(sim)  # Bell state prepared
    """

    def __init__(self, qubitCount: int) -> None:
        """Construct a circuit with the given number of qubits."""

    def __repr__(self) -> str: ...

    def append_gate(self, gate: GateType, qubits: Sequence[int], params: Sequence[float] = []) -> None:
        """
        Append a gate to the circuit without executing it.

        Gates are accumulated and can be optimised before running.
        ``params`` carries angle values for rotation gates, or complex
        components (real, imag pairs in row-major order) for matrix gates.

        Multi-controlled gates (MCX, MCY, MCZ, MCMtrx) treat the *last*
        qubit as the target and all others as controls.
        """

    def run(self, simulator: QrackSimulator) -> None:
        """
        Apply the circuit to the given simulator.

        The simulator's state is updated in place. The circuit itself
        is not consumed — it can be run on multiple simulators.
        The simulator must have at least as many qubits as the circuit.
        """

    def inverse(self) -> QrackCircuit:
        """
        Return a new circuit that is the adjoint (inverse) of this circuit.

        Applies all gates in reverse order with conjugate-transposed matrices.
        Useful for uncomputation and ansatz construction.

        Example::

            circ = QrackCircuit(2)
            circ.append_gate(GateType.H, [0])
            circ_inv = circ.inverse()   # applies H† = H
            circ.run(sim)
            circ_inv.run(sim)            # net effect: identity
        """

    def append(self, other: QrackCircuit) -> None:
        """
        Append all gates from another circuit to the end of this circuit.
        The other circuit's qubit count must be <= this circuit's qubit count.
        """

    @property
    def gate_count(self) -> int:
        """Number of gates currently recorded in the circuit."""

    @property
    def num_qubits(self) -> int:
        """Number of qubits this circuit operates on."""

class QrackStabilizer:
    """
    Pure Clifford-only quantum simulator. Polynomial memory in qubit count.

    Supports H, X, Y, Z, S, S†, √X, √X† single-qubit gates; CNOT, CY, CZ,
    SWAP, iSWAP two-qubit gates; and their multiply-controlled forms (1-control
    only for MCX on the underlying QINTERFACE_STABILIZER engine).

    Non-Clifford gates (RX, RY, RZ, U, T, T†, arbitrary matrices) are NOT
    exposed — use QrackStabilizerHybrid or QrackSimulator for those.

    State vector and probabilities are also intentionally omitted — the
    stabilizer engine stores a tableau, not amplitudes. Use
    QrackStabilizerHybrid if you need state access.
    """

    def __init__(self, qubitCount: int = 0) -> None:
        """Create a stabilizer simulator on n qubits, initialised to |0...0>."""

    def __repr__(self) -> str: ...
    def __enter__(self) -> QrackStabilizer: ...
    def __exit__(self, exc_type: object | None, exc_val: object | None, exc_tb: object | None) -> None: ...

    # ── Clifford single-qubit gates ──────────────────────────────────────────

    def h(self, qubit: int) -> None:
        """Hadamard gate."""

    def x(self, qubit: int) -> None:
        """Pauli X."""

    def y(self, qubit: int) -> None:
        """Pauli Y."""

    def z(self, qubit: int) -> None:
        """Pauli Z."""

    def s(self, qubit: int) -> None:
        """S gate (phase π/2)."""

    def sdg(self, qubit: int) -> None:
        """S† (inverse S)."""

    def sx(self, qubit: int) -> None:
        """√X gate."""

    def sxdg(self, qubit: int) -> None:
        """√X†."""

    # ── Clifford two-qubit gates ─────────────────────────────────────────────

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

    def mcx(self, controls: Sequence[int], target: int) -> None:
        """Multiply-controlled X. On QINTERFACE_STABILIZER, only 1 control is supported."""

    def mcy(self, controls: Sequence[int], target: int) -> None:
        """Multiply-controlled Y."""

    def mcz(self, controls: Sequence[int], target: int) -> None:
        """Multiply-controlled Z."""

    def macx(self, controls: Sequence[int], target: int) -> None:
        """Anti-controlled X."""

    def macy(self, controls: Sequence[int], target: int) -> None:
        """Anti-controlled Y."""

    def macz(self, controls: Sequence[int], target: int) -> None:
        """Anti-controlled Z."""

    # ── Measurement ─────────────────────────────────────────────────────────

    def measure(self, qubit: int) -> bool:
        """Measure qubit. Returns True=|1>, False=|0>. Collapses state."""

    def measure_all(self) -> list[bool]:
        """Measure all qubits. Returns list[bool], LSB first."""

    def force_measure(self, qubit: int, result: bool) -> bool:
        """Force measurement outcome without random draw."""

    def prob(self, qubit: int) -> float:
        """Probability of |1> for qubit. Does NOT collapse state."""

    def prob_all(self) -> list[float]:
        """Per-qubit |1> probabilities for all qubits."""

    # ── Pauli observables ────────────────────────────────────────────────────

    def measure_pauli(self, basis: Pauli, qubit: int) -> bool:
        """Measure a qubit in the specified Pauli basis."""

    def exp_val(self, basis: Pauli, qubit: int) -> float:
        """Single-qubit Pauli expectation value."""

    def exp_val_pauli(self, paulis: Sequence[Pauli], qubits: Sequence[int]) -> float:
        """Expectation value of a Pauli tensor product observable."""

    def variance_pauli(self, paulis: Sequence[Pauli], qubits: Sequence[int]) -> float:
        """Variance of a Pauli tensor product observable."""

    # ── State management ─────────────────────────────────────────────────────

    @property
    def num_qubits(self) -> int:
        """Number of qubits in this stabilizer simulator."""

    def reset_all(self) -> None:
        """Reset all qubits to |0...0>."""

    def set_permutation(self, permutation: int) -> None:
        """Reset state to the computational basis state |permutation>."""


class QrackStabilizerHybrid:
    """
    Stabilizer simulator with automatic fallback to dense simulation
    when non-Clifford gates are applied. Stays in polynomial-memory
    stabilizer mode for as long as the circuit is Clifford; switches to
    a QHybrid (CPU+GPU) dense backend on the first non-Clifford gate.

    The is_clifford property reflects QInterface.isClifford() — a type-level
    flag that is always True for this engine class (it IS a Clifford-type
    interface). It does NOT toggle to False on dense fallback.

    set_t_injection enables the near-Clifford T-injection gadget for
    circuits with few T gates (Clifford+T / RZ workloads).
    """

    def __init__(
        self,
        qubitCount: int = 0,
        isCpuGpuHybrid: bool = True,
        isOpenCL: bool = True,
        isHostPointer: bool = False,
        isSparse: bool = False,
    ) -> None:
        """Create a stabilizer-hybrid simulator. Flags select the dense fallback engine."""

    def __repr__(self) -> str: ...
    def __enter__(self) -> QrackStabilizerHybrid: ...
    def __exit__(self, exc_type: object | None, exc_val: object | None, exc_tb: object | None) -> None: ...

    # ── Clifford single-qubit gates ──────────────────────────────────────────

    def h(self, qubit: int) -> None:
        """Hadamard gate."""

    def x(self, qubit: int) -> None:
        """Pauli X."""

    def y(self, qubit: int) -> None:
        """Pauli Y."""

    def z(self, qubit: int) -> None:
        """Pauli Z."""

    def s(self, qubit: int) -> None:
        """S gate (phase π/2)."""

    def sdg(self, qubit: int) -> None:
        """S† (inverse S)."""

    def sx(self, qubit: int) -> None:
        """√X gate."""

    def sxdg(self, qubit: int) -> None:
        """√X†."""

    # ── Non-Clifford single-qubit gates (trigger dense fallback) ────────────

    def t(self, qubit: int) -> None:
        """T gate — phase shift π/4. Triggers dense fallback if not near-Clifford."""

    def tdg(self, qubit: int) -> None:
        """T† gate. Triggers dense fallback if not near-Clifford."""

    def rx(self, angle: float, qubit: int) -> None:
        """Rotate around X axis by angle radians."""

    def ry(self, angle: float, qubit: int) -> None:
        """Rotate around Y axis by angle radians."""

    def rz(self, angle: float, qubit: int) -> None:
        """Rotate around Z axis by angle radians."""

    def r1(self, angle: float, qubit: int) -> None:
        """Phase rotation: apply e^(i·angle) to |1> state."""

    def u(self, theta: float, phi: float, lam: float, qubit: int) -> None:
        """General single-qubit unitary U(θ,φ,λ)."""

    def u2(self, phi: float, lam: float, qubit: int) -> None:
        """U2 gate: U(π/2, φ, λ)."""

    # ── Two-qubit gates ──────────────────────────────────────────────────────

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

    def mcx(self, controls: Sequence[int], target: int) -> None:
        """Multiply-controlled X."""

    def mcy(self, controls: Sequence[int], target: int) -> None:
        """Multiply-controlled Y."""

    def mcz(self, controls: Sequence[int], target: int) -> None:
        """Multiply-controlled Z."""

    def macx(self, controls: Sequence[int], target: int) -> None:
        """Anti-controlled X."""

    def macy(self, controls: Sequence[int], target: int) -> None:
        """Anti-controlled Y."""

    def macz(self, controls: Sequence[int], target: int) -> None:
        """Anti-controlled Z."""

    def mtrx(self, matrix: Sequence[complex], qubit: int) -> None:
        """Apply arbitrary 2x2 unitary. matrix is [m00, m01, m10, m11] row-major."""

    def mcmtrx(self, controls: Sequence[int], matrix: Sequence[complex], qubit: int) -> None:
        """Multiply-controlled arbitrary 2x2 unitary."""

    def macmtrx(self, controls: Sequence[int], matrix: Sequence[complex], qubit: int) -> None:
        """Anti-controlled arbitrary 2x2 unitary."""

    def multiplex1_mtrx(self, controls: Sequence[int], mtrxs: Sequence[complex], target: int) -> None:
        """Uniformly-controlled single-qubit gate."""

    # ── Measurement ─────────────────────────────────────────────────────────

    def measure(self, qubit: int) -> bool:
        """Measure qubit. Returns True=|1>, False=|0>. Collapses state."""

    def measure_all(self) -> list[bool]:
        """Measure all qubits. Returns list[bool], LSB first."""

    def force_measure(self, qubit: int, result: bool) -> bool:
        """Force measurement outcome without random draw."""

    def prob(self, qubit: int) -> float:
        """Probability of |1> for qubit. Does NOT collapse state."""

    def prob_all(self) -> list[float]:
        """Per-qubit |1> probabilities for all qubits."""

    # ── Pauli observables ────────────────────────────────────────────────────

    def measure_pauli(self, basis: Pauli, qubit: int) -> bool:
        """Measure a qubit in the specified Pauli basis."""

    def exp_val(self, basis: Pauli, qubit: int) -> float:
        """Single-qubit Pauli expectation value."""

    def exp_val_pauli(self, paulis: Sequence[Pauli], qubits: Sequence[int]) -> float:
        """Expectation value of a Pauli tensor product observable."""

    def variance_pauli(self, paulis: Sequence[Pauli], qubits: Sequence[int]) -> float:
        """Variance of a Pauli tensor product observable."""

    # ── State access ─────────────────────────────────────────────────────────

    def _state_vector_impl(self) -> NDArray[numpy.complex64]:
        """Raw state vector snapshot (use the state_vector property instead)."""

    def _probabilities_impl(self) -> NDArray[numpy.float32]:
        """Raw probabilities snapshot (use the probabilities property instead)."""

    def get_amplitude(self, index: int) -> complex:
        """Get the complex amplitude of a specific basis state."""

    def set_amplitude(self, index: int, amplitude: complex) -> None:
        """Set the complex amplitude of a specific basis state."""

    # ── Properties and configuration ─────────────────────────────────────────

    @property
    def num_qubits(self) -> int:
        """Number of qubits in this stabilizer-hybrid simulator."""

    @property
    def is_clifford(self) -> bool:
        """
        Type-level Clifford flag (always True for QrackStabilizerHybrid).

        Reflects QInterface.isClifford() which returns True for any stabilizer-
        type engine interface regardless of whether the internal dense fallback
        has been triggered by a non-Clifford gate.
        """

    def set_t_injection(self, use_gadget: bool) -> None:
        """
        Enable or disable the T-injection gadget for near-Clifford circuits.

        With the gadget enabled (default), T gates and small-angle rotations
        are deferred using a Clifford+T approach before the dense fallback
        is triggered.
        """

    def set_use_exact_near_clifford(self, exact: bool) -> None:
        """Toggle exact near-Clifford simulation path."""

    def reset_all(self) -> None:
        """Reset all qubits to |0...0>."""

    def set_permutation(self, permutation: int) -> None:
        """Reset state to the computational basis state |permutation>."""
