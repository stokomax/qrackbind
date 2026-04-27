"""Unit tests for exception handling.

Phase 1–4 tested error paths via generic ``pytest.raises(Exception)``;
Phase 5 tightens the contract: every error path now raises a typed
subclass of :class:`qrackbind.QrackException` so callers can catch
qrackbind-specific failures without overshooting (e.g. masking real
bugs by catching ``Exception``).

Hierarchy under test::

    RuntimeError
      └── QrackException
            ├── QrackQubitError      (qubit index out of [0, num_qubits))
            └── QrackArgumentError   (length mismatch / bad value / etc.)
"""

import math

import numpy as np
import pytest

from qrackbind import (
    Pauli,
    QrackArgumentError,
    QrackException,
    QrackQubitError,
    QrackSimulator,
)


def oob(sim):
    """Return an out-of-range qubit index for the given simulator."""
    return sim.num_qubits


# ── Single-qubit gates ───────────────────────────────────────────────────────

def test_h_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError):
        sim.h(oob(sim))


def test_x_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError):
        sim.x(oob(sim))


def test_y_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError):
        sim.y(oob(sim))


def test_z_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError):
        sim.z(oob(sim))


def test_s_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError):
        sim.s(oob(sim))


def test_t_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError):
        sim.t(oob(sim))


def test_sdg_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError):
        sim.sdg(oob(sim))


def test_tdg_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError):
        sim.tdg(oob(sim))


def test_sx_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError):
        sim.sx(oob(sim))


def test_sxdg_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError):
        sim.sxdg(oob(sim))


# ── Rotation gates ───────────────────────────────────────────────────────────

def test_rx_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError):
        sim.rx(math.pi, oob(sim))


def test_ry_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError):
        sim.ry(math.pi, oob(sim))


def test_rz_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError):
        sim.rz(math.pi, oob(sim))


def test_r1_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError):
        sim.r1(math.pi, oob(sim))


# ── General unitary ──────────────────────────────────────────────────────────

def test_u_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError):
        sim.u(math.pi, 0.0, math.pi, oob(sim))


def test_u2_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError):
        sim.u2(0.0, math.pi, oob(sim))


# ── Two-qubit gates ──────────────────────────────────────────────────────────

def test_cnot_control_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError):
        sim.cnot(oob(sim), 0)


def test_cnot_target_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError):
        sim.cnot(0, oob(sim))


def test_cy_control_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError):
        sim.cy(oob(sim), 0)


def test_cy_target_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError):
        sim.cy(0, oob(sim))


def test_cz_control_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError):
        sim.cz(oob(sim), 0)


def test_cz_target_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError):
        sim.cz(0, oob(sim))


def test_swap_q1_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError):
        sim.swap(oob(sim), 0)


def test_swap_q2_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError):
        sim.swap(0, oob(sim))


def test_iswap_q1_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError):
        sim.iswap(oob(sim), 0)


def test_iswap_q2_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError):
        sim.iswap(0, oob(sim))


# ── Measurement & state ──────────────────────────────────────────────────────

def test_measure_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError):
        sim.measure(oob(sim))


def test_force_measure_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError):
        sim.force_measure(oob(sim), True)


def test_prob_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError):
        sim.prob(oob(sim))


# ── Matrix gates ─────────────────────────────────────────────────────────────

def test_mtrx_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    identity = [1.0, 0.0, 0.0, 1.0]
    with pytest.raises(QrackQubitError):
        sim.mtrx(identity, oob(sim))


def test_mtrx_too_few_elements():
    sim = QrackSimulator(qubitCount=1)
    with pytest.raises(QrackArgumentError):
        sim.mtrx([1.0, 0.0], 0)


def test_mcmtrx_too_few_elements():
    sim = QrackSimulator(qubitCount=1)
    with pytest.raises(QrackArgumentError):
        sim.mcmtrx([], [1.0, 0.0], 0)


def test_macmtrx_too_few_elements():
    sim = QrackSimulator(qubitCount=1)
    with pytest.raises(QrackArgumentError):
        sim.macmtrx([], [1.0, 0.0], 0)


def test_multiplex1_mtrx_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError):
        sim.multiplex1_mtrx([], [1.0, 0.0, 0.0, 1.0], oob(sim))


def test_multiplex1_mtrx_too_few_elements():
    sim = QrackSimulator(qubitCount=1)
    with pytest.raises(QrackArgumentError):
        sim.multiplex1_mtrx([], [1.0, 0.0], 0)


# ── Error message contents ───────────────────────────────────────────────────

def test_error_message_contains_qubit_index():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError) as exc_info:
        sim.h(5)
    assert "5" in str(exc_info.value)


def test_error_message_contains_method_name():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(QrackQubitError) as exc_info:
        sim.h(5)
    assert "h" in str(exc_info.value)


def test_error_message_contains_valid_range():
    sim = QrackSimulator(qubitCount=3)
    with pytest.raises(QrackQubitError) as exc_info:
        sim.h(7)
    msg = str(exc_info.value)
    assert "3" in msg  # the simulator has 3 qubits


# ── Phase 5: typed exception hierarchy ───────────────────────────────────────

class TestExceptionHierarchy:
    """The three Python exception classes are wired correctly."""

    def test_qubit_error_is_qrack_exception(self):
        assert issubclass(QrackQubitError, QrackException)

    def test_argument_error_is_qrack_exception(self):
        assert issubclass(QrackArgumentError, QrackException)

    def test_qrack_exception_is_runtime_error(self):
        assert issubclass(QrackException, RuntimeError)

    def test_all_importable_from_top_level(self):
        import qrackbind

        assert hasattr(qrackbind, "QrackException")
        assert hasattr(qrackbind, "QrackQubitError")
        assert hasattr(qrackbind, "QrackArgumentError")

    def test_qubit_error_caught_as_base_class(self):
        sim = QrackSimulator(qubitCount=2)
        with pytest.raises(QrackException):
            sim.h(99)

    def test_argument_error_caught_as_base_class(self):
        sim = QrackSimulator(qubitCount=2)
        with pytest.raises(QrackException):
            sim.exp_val_pauli([Pauli.PauliZ], [0, 1])

    def test_qubit_error_still_a_runtime_error(self):
        # Old code that catches RuntimeError keeps working post-Phase 5.
        sim = QrackSimulator(qubitCount=2)
        with pytest.raises(RuntimeError):
            sim.h(99)


# ── Phase 5: qubit index boundary handling ───────────────────────────────────

class TestQubitIndexBoundaries:
    """Index `numQubits - 1` is valid; `numQubits` is the first OOB index."""

    def test_max_valid_index_does_not_raise(self):
        sim = QrackSimulator(qubitCount=4)
        sim.h(3)  # last legal index — should be a no-op exception-wise

    def test_one_past_max_raises(self):
        sim = QrackSimulator(qubitCount=4)
        with pytest.raises(QrackQubitError):
            sim.h(4)

    def test_zero_index_valid(self):
        sim = QrackSimulator(qubitCount=1)
        sim.h(0)


# ── Phase 5: Pauli / weighted-observable argument validation ─────────────────

class TestPauliArgumentErrors:
    """`exp_val_pauli`, `variance_pauli`, `exp_val_floats`, `variance_floats`
    all raise `QrackArgumentError` (not generic Exception) on bad input."""

    def test_exp_val_pauli_length_mismatch(self):
        sim = QrackSimulator(qubitCount=2)
        with pytest.raises(QrackArgumentError):
            sim.exp_val_pauli([Pauli.PauliZ], [0, 1])

    def test_variance_pauli_length_mismatch(self):
        sim = QrackSimulator(qubitCount=2)
        with pytest.raises(QrackArgumentError):
            sim.variance_pauli([Pauli.PauliZ, Pauli.PauliX], [0])

    def test_exp_val_floats_wrong_weight_count(self):
        sim = QrackSimulator(qubitCount=2)
        # weights must have 2 * len(qubits) entries — passing only 2 is wrong
        # for a single qubit it would be valid (2*1=2), so use 1 qubit + 1 weight
        with pytest.raises(QrackArgumentError):
            sim.exp_val_floats([0], [1.0])

    def test_variance_floats_wrong_weight_count(self):
        sim = QrackSimulator(qubitCount=2)
        with pytest.raises(QrackArgumentError):
            sim.variance_floats([0, 1], [1.0, 2.0])  # need 4, given 2

    def test_exp_val_qubit_out_of_range(self):
        sim = QrackSimulator(qubitCount=2)
        with pytest.raises(QrackQubitError):
            sim.exp_val(Pauli.PauliZ, 99)

    def test_set_state_vector_wrong_size(self):
        sim = QrackSimulator(qubitCount=2)
        bad = np.zeros(3, dtype=np.complex64)  # need length 4
        with pytest.raises(QrackArgumentError):
            sim.set_state_vector(bad)


# ── Phase 5: state preservation after caught exceptions ──────────────────────

class TestStatePreservedAfterException:
    """A caught exception must not corrupt the simulator state."""

    def test_state_intact_after_qubit_error(self):
        sim = QrackSimulator(qubitCount=2)
        sim.h(0)
        prob_before = sim.prob(0)
        with pytest.raises(QrackQubitError):
            sim.h(99)
        prob_after = sim.prob(0)
        assert prob_before == pytest.approx(prob_after, abs=1e-5)

    def test_repeated_caught_errors_dont_break_simulator(self):
        sim = QrackSimulator(qubitCount=1)
        for _ in range(10):
            with pytest.raises(QrackQubitError):
                sim.h(99)
        # Simulator should still operate normally afterwards
        sim.h(0)
        assert sim.prob(0) == pytest.approx(0.5, abs=1e-4)

    def test_state_intact_after_argument_error(self):
        sim = QrackSimulator(qubitCount=2)
        sim.h(0)
        prob_before = sim.prob(0)
        with pytest.raises(QrackArgumentError):
            sim.exp_val_pauli([Pauli.PauliZ], [0, 1])  # length mismatch
        prob_after = sim.prob(0)
        assert prob_before == pytest.approx(prob_after, abs=1e-5)
