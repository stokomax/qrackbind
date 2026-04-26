"""Unit tests for exception handling (out-of-range qubits and invalid arguments)."""

import math

import pytest

from qrackbind import QrackSimulator


def oob(sim):
    """Return an out-of-range qubit index for the given simulator."""
    return sim.num_qubits


# ── Single-qubit gates ───────────────────────────────────────────────────────

def test_h_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception):
        sim.h(oob(sim))


def test_x_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception):
        sim.x(oob(sim))


def test_y_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception):
        sim.y(oob(sim))


def test_z_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception):
        sim.z(oob(sim))


def test_s_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception):
        sim.s(oob(sim))


def test_t_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception):
        sim.t(oob(sim))


def test_sdg_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception):
        sim.sdg(oob(sim))


def test_tdg_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception):
        sim.tdg(oob(sim))


def test_sx_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception):
        sim.sx(oob(sim))


def test_sxdg_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception):
        sim.sxdg(oob(sim))


# ── Rotation gates ───────────────────────────────────────────────────────────

def test_rx_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception):
        sim.rx(math.pi, oob(sim))


def test_ry_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception):
        sim.ry(math.pi, oob(sim))


def test_rz_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception):
        sim.rz(math.pi, oob(sim))


def test_r1_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception):
        sim.r1(math.pi, oob(sim))


# ── General unitary ──────────────────────────────────────────────────────────

def test_u_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception):
        sim.u(math.pi, 0.0, math.pi, oob(sim))


def test_u2_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception):
        sim.u2(0.0, math.pi, oob(sim))


# ── Two-qubit gates ──────────────────────────────────────────────────────────

def test_cnot_control_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception):
        sim.cnot(oob(sim), 0)


def test_cnot_target_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception):
        sim.cnot(0, oob(sim))


def test_cy_control_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception):
        sim.cy(oob(sim), 0)


def test_cy_target_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception):
        sim.cy(0, oob(sim))


def test_cz_control_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception):
        sim.cz(oob(sim), 0)


def test_cz_target_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception):
        sim.cz(0, oob(sim))


def test_swap_q1_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception):
        sim.swap(oob(sim), 0)


def test_swap_q2_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception):
        sim.swap(0, oob(sim))


def test_iswap_q1_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception):
        sim.iswap(oob(sim), 0)


def test_iswap_q2_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception):
        sim.iswap(0, oob(sim))


# ── Measurement & state ──────────────────────────────────────────────────────

def test_measure_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception):
        sim.measure(oob(sim))


def test_force_measure_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception):
        sim.force_measure(oob(sim), True)


def test_prob_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception):
        sim.prob(oob(sim))


# ── Matrix gates ─────────────────────────────────────────────────────────────

def test_mtrx_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    identity = [1.0, 0.0, 0.0, 1.0]
    with pytest.raises(Exception):
        sim.mtrx(identity, oob(sim))


def test_mtrx_too_few_elements():
    sim = QrackSimulator(qubitCount=1)
    with pytest.raises(Exception):
        sim.mtrx([1.0, 0.0], 0)


def test_mcmtrx_too_few_elements():
    sim = QrackSimulator(qubitCount=1)
    with pytest.raises(Exception):
        sim.mcmtrx([], [1.0, 0.0], 0)


def test_macmtrx_too_few_elements():
    sim = QrackSimulator(qubitCount=1)
    with pytest.raises(Exception):
        sim.macmtrx([], [1.0, 0.0], 0)


def test_multiplex1_mtrx_out_of_range():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception):
        sim.multiplex1_mtrx([], [1.0, 0.0, 0.0, 1.0], oob(sim))


def test_multiplex1_mtrx_too_few_elements():
    sim = QrackSimulator(qubitCount=1)
    with pytest.raises(Exception):
        sim.multiplex1_mtrx([], [1.0, 0.0], 0)


# ── Error message contents ───────────────────────────────────────────────────

def test_error_message_contains_qubit_index():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception) as exc_info:
        sim.h(5)
    assert "5" in str(exc_info.value)


def test_error_message_contains_method_name():
    sim = QrackSimulator(qubitCount=2)
    with pytest.raises(Exception) as exc_info:
        sim.h(5)
    assert "h" in str(exc_info.value)
