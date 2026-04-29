"""Gate dispatch table — PennyLane operation name → QrackSimulator method.

Exports two dispatch tables:
- ``GATE_DISPATCH``: full gate surface for QrackSimulator / QrackStabilizerHybrid
- ``CLIFFORD_GATE_DISPATCH``: Clifford-only subset for QrackStabilizer
"""

from __future__ import annotations

import numpy as np

from qrackbind import QrackSimulator, Pauli

# ── Helpers ───────────────────────────────────────────────────────────────────


def _basis_state_int(bits: list[int]) -> int:
    """Convert a list of 0/1 bit values to an integer (MSB first)."""
    # Qrack basis indexing treats wire 0 as the least-significant bit.  This
    # keeps qml.BasisState bit order aligned with qml.probs wire ordering in
    # the device's marginal-probability routine.
    return int(sum(int(b) << i for i, b in enumerate(bits)))


# ── GATE_DISPATCH ─────────────────────────────────────────────────────────────
# PennyLane operation name → (sim, wires, params) → None
#
# PennyLane decomposes unsupported operations before reaching execute().
# All operations reaching dispatch_gate() are guaranteed to have their
# qubit indices already remapped to integers [0, N).

GATE_DISPATCH: dict[str, callable] = {
    # ── Single-qubit Clifford ─────────────────────────────────────────────
    "Hadamard": lambda sim, w, p: sim.h(w[0]),
    "PauliX": lambda sim, w, p: sim.x(w[0]),
    "PauliY": lambda sim, w, p: sim.y(w[0]),
    "PauliZ": lambda sim, w, p: sim.z(w[0]),
    "S": lambda sim, w, p: sim.s(w[0]),
    "T": lambda sim, w, p: sim.t(w[0]),
    "SX": lambda sim, w, p: sim.sx(w[0]),
    # Adjoint variants — PennyLane's decompose transform expands
    # Adjoint(S) → S†, but we keep these as fallback
    "Adjoint(S)": lambda sim, w, p: sim.sdg(w[0]),
    "Adjoint(T)": lambda sim, w, p: sim.tdg(w[0]),
    "Adjoint(SX)": lambda sim, w, p: sim.sxdg(w[0]),

    # ── Single-qubit rotations ────────────────────────────────────────────
    "RX": lambda sim, w, p: sim.rx(p[0], w[0]),
    "RY": lambda sim, w, p: sim.ry(p[0], w[0]),
    "RZ": lambda sim, w, p: sim.rz(p[0], w[0]),
    # Qrack's bound r1/RT behaves as a global phase in this stack; implement
    # PennyLane's relative phase diag(1, exp(iφ)) via an explicit matrix.
    "PhaseShift": lambda sim, w, p: sim.mtrx(_phase_matrix(p[0]), w[0]),
    "Rot": lambda sim, w, p: sim.u(p[0], p[1], p[2], w[0]),  # U(θ,φ,λ)
    "U": lambda sim, w, p: sim.u(p[0], p[1], p[2], w[0]),
    "U3": lambda sim, w, p: sim.u(p[0], p[1], p[2], w[0]),
    "U2": lambda sim, w, p: sim.u2(p[1], p[2], w[0]),

    # ── Two-qubit gates ───────────────────────────────────────────────────
    "CNOT": lambda sim, w, p: sim.cnot(w[0], w[1]),
    "CY": lambda sim, w, p: sim.cy(w[0], w[1]),
    "CZ": lambda sim, w, p: sim.cz(w[0], w[1]),
    "CH": lambda sim, w, p: sim.mcmtrx([w[0]],
                                       _hadamard_matrix(), w[1]),
    "CRX": lambda sim, w, p: sim.mcmtrx([w[0]],
                                        _rx_matrix(p[0]), w[1]),
    "CRY": lambda sim, w, p: sim.mcmtrx([w[0]],
                                        _ry_matrix(p[0]), w[1]),
    "CRZ": lambda sim, w, p: sim.mcrz(p[0], [w[0]], w[1]),
    "ControlledPhaseShift": lambda sim, w, p: sim.mcz([w[0]], w[1])
                               if p[0] == np.pi
                               else sim.mcmtrx([w[0]],
                                               _phase_matrix(p[0]), w[1]),
    "SWAP": lambda sim, w, p: sim.swap(w[0], w[1]),
    "ISWAP": lambda sim, w, p: sim.iswap(w[0], w[1]),
    "PSWAP": lambda sim, w, p: sim.iswap(w[0], w[1]),  # approximate with ISWAP

    # ── Multi-qubit gates ─────────────────────────────────────────────────
    "Toffoli": lambda sim, w, p: sim.mcx([w[0], w[1]], w[2]),
    "MultiControlledX": lambda sim, w, p: sim.mcx(w[:-1], w[-1]),
    "MultiControlledY": lambda sim, w, p: sim.mcy(w[:-1], w[-1]),
    "MultiControlledZ": lambda sim, w, p: sim.mcz(w[:-1], w[-1]),

    # ── Arbitrary single-qubit unitary ────────────────────────────────────
    "QubitUnitary": lambda sim, w, p: sim.mtrx(
        _flatten_matrix(p[0], np.complex64), w[0]),
    "DiagonalQubitUnitary": lambda sim, w, p: sim.mtrx(
        _flatten_matrix(p[0], np.complex64), w[0]),

    # ── Controlled arbitrary unitary ──────────────────────────────────────
    "ControlledQubitUnitary": lambda sim, w, p: sim.mcmtrx(
        w[:-1], _flatten_matrix(p[0], np.complex64), w[-1]),
    "ControlledUnitary": lambda sim, w, p: sim.mcmtrx(
        w[:-1], _flatten_matrix(p[0], np.complex64), w[-1]),

    # ── State preparation ─────────────────────────────────────────────────
    "BasisState": lambda sim, w, p: sim.set_permutation(
        _basis_state_int(list(p[0]))),
    "StatePrep": lambda sim, w, p: sim.set_state_vector(
        np.array(p[0], dtype=np.complex64)),
}


# ── Matrix helpers ─────────────────────────────────────────────────────────────


def _flatten_matrix(matrix, dtype) -> list[complex]:
    """Flatten a PennyLane matrix to a flat list of complex values."""
    arr = np.asarray(matrix, dtype=dtype)
    # PennyLane gives shape (2, 2) for single-qubit gates
    if arr.ndim == 3:
        # Batch of matrices — use first
        arr = arr[0]
    return list(arr.flatten())


def _hadamard_matrix() -> list[complex]:
    h = 1 / np.sqrt(2)
    return [h, h, h, -h]


def _rx_matrix(angle: float) -> list[complex]:
    c, s = np.cos(angle / 2), np.sin(angle / 2)
    return [complex(c), complex(0, -s), complex(0, -s), complex(c)]


def _ry_matrix(angle: float) -> list[complex]:
    c, s = np.cos(angle / 2), np.sin(angle / 2)
    return [complex(c), -complex(s), complex(s), complex(c)]


def _rz_matrix(angle: float) -> list[complex]:
    e = np.exp(1j * angle / 2)
    return [complex(np.cos(-angle / 2)), 0,
            0, complex(np.cos(angle / 2))]


def _phase_matrix(angle: float) -> list[complex]:
    return [1 + 0j, 0 + 0j, 0 + 0j, complex(np.exp(1j * angle))]


# ── CLIFFORD_GATE_DISPATCH ────────────────────────────────────────────────────
# Clifford-only subset for QrackStabilizer.
#
# Key differences from GATE_DISPATCH:
#   - T, Adjoint(T) omitted — non-Clifford
#   - RX, RY, RZ, PhaseShift, Rot, U, U2, U3 omitted — rotations are non-Clifford
#   - CRX, CRY, CRZ, ControlledPhaseShift omitted — non-Clifford
#   - CH uses mch() instead of mcmtrx() — mch is natively bound on QrackStabilizer
#   - Toffoli, MultiControlledX/Y/Z omitted — 2+ control MCInvert not Clifford on
#     the pure stabilizer engine
#   - QubitUnitary, DiagonalQubitUnitary, ControlledQubitUnitary, ControlledUnitary
#     omitted — arbitrary matrices not bound on QrackStabilizer
#   - StatePrep omitted — needs set_state_vector, not available on QrackStabilizer
#   - BasisState retained — set_permutation IS available on QrackStabilizer

CLIFFORD_GATE_DISPATCH: dict[str, callable] = {
    # ── Single-qubit Clifford ─────────────────────────────────────────────
    "Hadamard": lambda sim, w, p: sim.h(w[0]),
    "PauliX": lambda sim, w, p: sim.x(w[0]),
    "PauliY": lambda sim, w, p: sim.y(w[0]),
    "PauliZ": lambda sim, w, p: sim.z(w[0]),
    "S": lambda sim, w, p: sim.s(w[0]),
    "SX": lambda sim, w, p: sim.sx(w[0]),
    "Adjoint(S)": lambda sim, w, p: sim.sdg(w[0]),
    "Adjoint(SX)": lambda sim, w, p: sim.sxdg(w[0]),

    # ── Two-qubit Clifford ────────────────────────────────────────────────
    "CNOT": lambda sim, w, p: sim.cnot(w[0], w[1]),
    "CY": lambda sim, w, p: sim.cy(w[0], w[1]),
    "CZ": lambda sim, w, p: sim.cz(w[0], w[1]),
    # CH: uses mch() — natively bound on QrackStabilizer via add_clifford_two_qubit
    "CH": lambda sim, w, p: sim.mch([w[0]], w[1]),
    "SWAP": lambda sim, w, p: sim.swap(w[0], w[1]),
    "ISWAP": lambda sim, w, p: sim.iswap(w[0], w[1]),
    "PSWAP": lambda sim, w, p: sim.iswap(w[0], w[1]),

    # ── State preparation ─────────────────────────────────────────────────
    "BasisState": lambda sim, w, p: sim.set_permutation(
        _basis_state_int(list(p[0]))),
}


def dispatch_clifford_gate(sim, op) -> None:
    """Apply a PennyLane operation to a QrackStabilizer (Clifford-only).

    Args:
        sim: QrackStabilizer instance
        op: pennylane.Operation with .name, .wires, .parameters

    Raises:
        NotImplementedError: if the operation name is not in CLIFFORD_GATE_DISPATCH
    """
    name = op.name
    wires = list(op.wires)
    params = list(op.parameters)

    handler = CLIFFORD_GATE_DISPATCH.get(name)
    if handler is None:
        raise NotImplementedError(
            f"QrackStabilizerDevice: operation '{name}' is not supported "
            f"on the pure Clifford stabilizer. Use QrackStabilizerHybridDevice "
            f"or QrackDevice for non-Clifford operations.")
    handler(sim, wires, params)


# ── Dispatch ───────────────────────────────────────────────────────────────────


def dispatch_gate(sim: QrackSimulator, op) -> None:
    """Apply a PennyLane operation to the simulator.

    Args:
        sim: QrackSimulator instance
        op: pennylane.Operation with .name, .wires, .parameters

    Raises:
        KeyError: if the operation name is not in GATE_DISPATCH and
                  should have been decomposed by preprocess_transforms().
    """
    name = op.name
    wires = list(op.wires)
    params = list(op.parameters)

    handler = GATE_DISPATCH.get(name)
    if handler is None:
        raise NotImplementedError(
            f"QrackDevice: operation '{name}' is not supported. "
            f"It should have been decomposed by preprocess_transforms().")
    handler(sim, wires, params)
