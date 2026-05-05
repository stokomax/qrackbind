"""Unit tests for every gate binding defined in simulator.cpp.

Every gate gets at least two focused tests:
  • a population test (prob() after the gate)
  • an involution / identity / interference test

Phase-only gates are exercised via Hadamard sandwiches so that
phase changes become measurable population shifts.
"""

import math
import pytest
from qrackbind import QrackSimulator

ABS = 1e-5  # default tolerance for floating-point assertions


# ═══════════════════════════════════════════════════════════════════════════════
# Single-qubit gates (no parameters)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Hadamard ──────────────────────────────────────────────────────────────────

def test_h_superposition_from_zero():
    sim = QrackSimulator(qubitCount=1)
    sim.h(0)
    assert sim.prob(0) == pytest.approx(0.5, abs=ABS)


def test_h_involution():
    sim = QrackSimulator(qubitCount=1)
    sim.h(0)
    sim.h(0)
    assert sim.prob(0) == pytest.approx(0.0, abs=ABS)


def test_h_on_one():
    sim = QrackSimulator(qubitCount=1)
    sim.x(0)
    sim.h(0)
    assert sim.prob(0) == pytest.approx(0.5, abs=ABS)


# ── Pauli X ───────────────────────────────────────────────────────────────────

def test_x_flips_zero():
    sim = QrackSimulator(qubitCount=1)
    sim.x(0)
    assert sim.prob(0) == pytest.approx(1.0, abs=ABS)


def test_x_involution():
    sim = QrackSimulator(qubitCount=1)
    sim.x(0)
    sim.x(0)
    assert sim.prob(0) == pytest.approx(0.0, abs=ABS)


# ── Pauli Y ───────────────────────────────────────────────────────────────────

def test_y_flips_zero():
    sim = QrackSimulator(qubitCount=1)
    sim.y(0)
    assert sim.prob(0) == pytest.approx(1.0, abs=ABS)


def test_y_involution():
    sim = QrackSimulator(qubitCount=1)
    sim.y(0)
    sim.y(0)
    assert sim.prob(0) == pytest.approx(0.0, abs=ABS)


# ── Pauli Z ───────────────────────────────────────────────────────────────────

def test_z_no_pop_change_on_zero():
    sim = QrackSimulator(qubitCount=1)
    sim.z(0)
    assert sim.prob(0) == pytest.approx(0.0, abs=ABS)


def test_z_no_pop_change_on_one():
    sim = QrackSimulator(qubitCount=1)
    sim.x(0)
    sim.z(0)
    assert sim.prob(0) == pytest.approx(1.0, abs=ABS)


def test_z_via_hadamard_sandwich():
    """H·Z·H = X, so HZH|0> = |1>."""
    sim = QrackSimulator(qubitCount=1)
    sim.h(0)
    sim.z(0)
    sim.h(0)
    assert sim.prob(0) == pytest.approx(1.0, abs=ABS)


# ── S gate ────────────────────────────────────────────────────────────────────

def test_s_no_pop_change():
    sim = QrackSimulator(qubitCount=1)
    sim.s(0)
    assert sim.prob(0) == pytest.approx(0.0, abs=ABS)


def test_s_squared_equals_z_via_interference():
    """S^2 = Z, therefore H·S·S·H = H·Z·H = X."""
    sim = QrackSimulator(qubitCount=1)
    sim.h(0)
    sim.s(0)
    sim.s(0)
    sim.h(0)
    assert sim.prob(0) == pytest.approx(1.0, abs=ABS)


# ── T gate ────────────────────────────────────────────────────────────────────

def test_t_no_pop_change():
    sim = QrackSimulator(qubitCount=1)
    sim.t(0)
    assert sim.prob(0) == pytest.approx(0.0, abs=ABS)


def test_t_fourth_power_equals_z_via_interference():
    """T^4 = Z, therefore H·T^4·H = X."""
    sim = QrackSimulator(qubitCount=1)
    sim.h(0)
    sim.t(0)
    sim.t(0)
    sim.t(0)
    sim.t(0)
    sim.h(0)
    assert sim.prob(0) == pytest.approx(1.0, abs=ABS)


# ── S† (sdg) ──────────────────────────────────────────────────────────────────

def test_sdg_no_pop_change():
    sim = QrackSimulator(qubitCount=1)
    sim.sdg(0)
    assert sim.prob(0) == pytest.approx(0.0, abs=ABS)


def test_s_sdg_identity():
    """S·S† = I, therefore H·S·S†·H = I."""
    sim = QrackSimulator(qubitCount=1)
    sim.h(0)
    sim.s(0)
    sim.sdg(0)
    sim.h(0)
    assert sim.prob(0) == pytest.approx(0.0, abs=ABS)


def test_sdg_squared_equals_z_via_interference():
    """(S†)^2 = Z, therefore H·sdg·sdg·H = X."""
    sim = QrackSimulator(qubitCount=1)
    sim.h(0)
    sim.sdg(0)
    sim.sdg(0)
    sim.h(0)
    assert sim.prob(0) == pytest.approx(1.0, abs=ABS)


# ── T† (tdg) ──────────────────────────────────────────────────────────────────

def test_tdg_no_pop_change():
    sim = QrackSimulator(qubitCount=1)
    sim.tdg(0)
    assert sim.prob(0) == pytest.approx(0.0, abs=ABS)


def test_t_tdg_identity():
    """T·T† = I, therefore H·T·T†·H = I."""
    sim = QrackSimulator(qubitCount=1)
    sim.h(0)
    sim.t(0)
    sim.tdg(0)
    sim.h(0)
    assert sim.prob(0) == pytest.approx(0.0, abs=ABS)


# ── √X (sx) ───────────────────────────────────────────────────────────────────

def test_sx_creates_superposition():
    sim = QrackSimulator(qubitCount=1)
    sim.sx(0)
    assert sim.prob(0) == pytest.approx(0.5, abs=ABS)


def test_sx_squared_is_x():
    sim = QrackSimulator(qubitCount=1)
    sim.sx(0)
    sim.sx(0)
    assert sim.prob(0) == pytest.approx(1.0, abs=ABS)


# ── √X† (sxdg) ────────────────────────────────────────────────────────────────

def test_sxdg_creates_superposition():
    sim = QrackSimulator(qubitCount=1)
    sim.sxdg(0)
    assert sim.prob(0) == pytest.approx(0.5, abs=ABS)


def test_sxdg_squared_is_x():
    sim = QrackSimulator(qubitCount=1)
    sim.sxdg(0)
    sim.sxdg(0)
    assert sim.prob(0) == pytest.approx(1.0, abs=ABS)


def test_sx_sxdg_identity():
    sim = QrackSimulator(qubitCount=1)
    sim.sx(0)
    sim.sxdg(0)
    assert sim.prob(0) == pytest.approx(0.0, abs=ABS)


# ═══════════════════════════════════════════════════════════════════════════════
# Rotation gates
# ═══════════════════════════════════════════════════════════════════════════════

# ── RX ────────────────────────────────────────────────────────────────────────

def test_rx_pi_is_x():
    sim = QrackSimulator(qubitCount=1)
    sim.rx(math.pi, 0)
    assert sim.prob(0) == pytest.approx(1.0, abs=ABS)


def test_rx_zero_is_identity():
    sim = QrackSimulator(qubitCount=1)
    sim.rx(0.0, 0)
    assert sim.prob(0) == pytest.approx(0.0, abs=ABS)


def test_rx_half_pi_superposition():
    sim = QrackSimulator(qubitCount=1)
    sim.rx(math.pi / 2, 0)
    assert sim.prob(0) == pytest.approx(0.5, abs=ABS)


def test_rx_two_pi_is_identity():
    """RX(2π) = -I (global phase only)."""
    sim = QrackSimulator(qubitCount=1)
    sim.rx(2 * math.pi, 0)
    assert sim.prob(0) == pytest.approx(0.0, abs=ABS)


# ── RY ────────────────────────────────────────────────────────────────────────

def test_ry_pi_flips():
    sim = QrackSimulator(qubitCount=1)
    sim.ry(math.pi, 0)
    assert sim.prob(0) == pytest.approx(1.0, abs=ABS)


def test_ry_zero_is_identity():
    sim = QrackSimulator(qubitCount=1)
    sim.ry(0.0, 0)
    assert sim.prob(0) == pytest.approx(0.0, abs=ABS)


def test_ry_half_pi_superposition():
    sim = QrackSimulator(qubitCount=1)
    sim.ry(math.pi / 2, 0)
    assert sim.prob(0) == pytest.approx(0.5, abs=ABS)


# ── RZ ────────────────────────────────────────────────────────────────────────

def test_rz_no_pop_change():
    sim = QrackSimulator(qubitCount=1)
    sim.rz(math.pi, 0)
    assert sim.prob(0) == pytest.approx(0.0, abs=ABS)


def test_rz_via_interference():
    """RZ(π) ≈ Z (global phase), therefore H·RZ(π)·H ≈ X."""
    sim = QrackSimulator(qubitCount=1)
    sim.h(0)
    sim.rz(math.pi, 0)
    sim.h(0)
    assert sim.prob(0) == pytest.approx(1.0, abs=ABS)


# ── R1 (phase rotation) ───────────────────────────────────────────────────────

def test_r1_no_pop_change_on_zero():
    sim = QrackSimulator(qubitCount=1)
    sim.r1(math.pi, 0)
    assert sim.prob(0) == pytest.approx(0.0, abs=ABS)


def test_r1_no_pop_change_on_one():
    sim = QrackSimulator(qubitCount=1)
    sim.x(0)
    sim.r1(math.pi, 0)
    assert sim.prob(0) == pytest.approx(1.0, abs=ABS)


def test_r1_global_phase_no_pop_change():
    """R1 is a global phase rotation; it never changes measurement probabilities."""
    sim = QrackSimulator(qubitCount=1)
    sim.h(0)
    sim.r1(math.pi, 0)
    assert sim.prob(0) == pytest.approx(0.5, abs=ABS)


# ═══════════════════════════════════════════════════════════════════════════════
# General unitary gates
# ═══════════════════════════════════════════════════════════════════════════════

# ── U (U3) ────────────────────────────────────────────────────────────────────

def test_u_as_x():
    """U(π, 0, π) = Pauli-X."""
    sim = QrackSimulator(qubitCount=1)
    sim.u(math.pi, 0.0, math.pi, 0)
    assert sim.prob(0) == pytest.approx(1.0, abs=ABS)


def test_u_as_hadamard():
    """U(π/2, 0, π) = Hadamard."""
    sim = QrackSimulator(qubitCount=1)
    sim.u(math.pi / 2, 0.0, math.pi, 0)
    assert sim.prob(0) == pytest.approx(0.5, abs=ABS)


def test_u_identity():
    """U(0, 0, 0) = Identity."""
    sim = QrackSimulator(qubitCount=1)
    sim.u(0.0, 0.0, 0.0, 0)
    assert sim.prob(0) == pytest.approx(0.0, abs=ABS)


# ── U2 ────────────────────────────────────────────────────────────────────────

def test_u2_as_hadamard():
    """U2(0, π) = U(π/2, 0, π) = Hadamard."""
    sim = QrackSimulator(qubitCount=1)
    sim.u2(0.0, math.pi, 0)
    assert sim.prob(0) == pytest.approx(0.5, abs=ABS)


def test_u2_always_superposition():
    """U2 fixes θ = π/2, so it always creates a superposition from |0>."""
    sim = QrackSimulator(qubitCount=1)
    sim.u2(0.0, 0.0, 0)
    assert sim.prob(0) == pytest.approx(0.5, abs=ABS)


# ═══════════════════════════════════════════════════════════════════════════════
# Two-qubit gates
# ═══════════════════════════════════════════════════════════════════════════════

# ── CNOT ──────────────────────────────────────────────────────────────────────

def test_cnot_control_zero_no_flip():
    sim = QrackSimulator(qubitCount=2)
    sim.cnot(0, 1)
    assert sim.prob(1) == pytest.approx(0.0, abs=ABS)


def test_cnot_control_one_flips_target():
    sim = QrackSimulator(qubitCount=2)
    sim.x(0)
    sim.cnot(0, 1)
    assert sim.prob(1) == pytest.approx(1.0, abs=ABS)


def test_cnot_involution():
    sim = QrackSimulator(qubitCount=2)
    sim.x(0)
    sim.cnot(0, 1)
    sim.cnot(0, 1)
    assert sim.prob(1) == pytest.approx(0.0, abs=ABS)


def test_cnot_creates_bell_state():
    """H|0> on control, then CNOT creates entanglement: measuring control
    then target should show perfect correlation."""
    for _ in range(50):
        sim = QrackSimulator(qubitCount=2)
        sim.h(0)
        sim.cnot(0, 1)
        m0 = sim.measure(0)
        m1 = sim.measure(1)
        assert m0 == m1, f"Entanglement broken: {m0} != {m1}"


# ── CY ────────────────────────────────────────────────────────────────────────

def test_cy_control_zero_no_effect():
    sim = QrackSimulator(qubitCount=2)
    sim.cy(0, 1)
    assert sim.prob(1) == pytest.approx(0.0, abs=ABS)


def test_cy_control_one_applies_y():
    sim = QrackSimulator(qubitCount=2)
    sim.x(0)
    sim.cy(0, 1)
    # Y|0> = i|1>, so target should be |1>
    assert sim.prob(1) == pytest.approx(1.0, abs=ABS)


def test_cy_control_one_flips_target_from_one():
    sim = QrackSimulator(qubitCount=2)
    sim.x(0)
    sim.x(1)
    sim.cy(0, 1)
    # Y|1> = -i|0>, so target should be |0>
    assert sim.prob(1) == pytest.approx(0.0, abs=ABS)


# ── CZ ────────────────────────────────────────────────────────────────────────

def test_cz_no_pop_change_on_zero_zero():
    sim = QrackSimulator(qubitCount=2)
    sim.cz(0, 1)
    assert sim.prob(0) == pytest.approx(0.0, abs=ABS)
    assert sim.prob(1) == pytest.approx(0.0, abs=ABS)


def test_cz_no_pop_change_on_one_one():
    sim = QrackSimulator(qubitCount=2)
    sim.x(0)
    sim.x(1)
    sim.cz(0, 1)
    assert sim.prob(0) == pytest.approx(1.0, abs=ABS)
    assert sim.prob(1) == pytest.approx(1.0, abs=ABS)


def test_cz_via_interference():
    """
    ctrl=|1>, h(tgt), cz, h(tgt)  ==  ctrl=|1>, x(tgt).
    CZ with ctrl=|1> applies Z to target; sandwiched in H gives X.
    """
    sim = QrackSimulator(qubitCount=2)
    sim.x(0)
    sim.h(1)
    sim.cz(0, 1)
    sim.h(1)
    assert sim.prob(1) == pytest.approx(1.0, abs=ABS)


# ── SWAP ──────────────────────────────────────────────────────────────────────

def test_swap_exchanges_qubits():
    sim = QrackSimulator(qubitCount=2)
    sim.x(0)
    sim.swap(0, 1)
    assert sim.prob(0) == pytest.approx(0.0, abs=ABS)
    assert sim.prob(1) == pytest.approx(1.0, abs=ABS)


def test_swap_involution():
    sim = QrackSimulator(qubitCount=2)
    sim.x(0)
    sim.swap(0, 1)
    sim.swap(0, 1)
    assert sim.prob(0) == pytest.approx(1.0, abs=ABS)
    assert sim.prob(1) == pytest.approx(0.0, abs=ABS)


def test_swap_symmetric():
    sim = QrackSimulator(qubitCount=2)
    sim.x(1)
    sim.swap(0, 1)
    assert sim.prob(0) == pytest.approx(1.0, abs=ABS)
    assert sim.prob(1) == pytest.approx(0.0, abs=ABS)


# ── iSWAP ─────────────────────────────────────────────────────────────────────

def test_iswap_exchanges_populations():
    sim = QrackSimulator(qubitCount=2)
    sim.x(0)
    sim.iswap(0, 1)
    assert sim.prob(0) == pytest.approx(0.0, abs=ABS)
    assert sim.prob(1) == pytest.approx(1.0, abs=ABS)


def test_iswap_exchanges_reverse():
    sim = QrackSimulator(qubitCount=2)
    sim.x(1)
    sim.iswap(0, 1)
    assert sim.prob(0) == pytest.approx(1.0, abs=ABS)
    assert sim.prob(1) == pytest.approx(0.0, abs=ABS)


def test_iswap_on_both_one():
    """iSWAP|11> = -|11> (phase only), so populations stay at 1."""
    sim = QrackSimulator(qubitCount=2)
    sim.x(0)
    sim.x(1)
    sim.iswap(0, 1)
    assert sim.prob(0) == pytest.approx(1.0, abs=ABS)
    assert sim.prob(1) == pytest.approx(1.0, abs=ABS)


# ═══════════════════════════════════════════════════════════════════════════════
# Three-qubit gates
# ═══════════════════════════════════════════════════════════════════════════════

# ── CCNOT (Toffoli) ───────────────────────────────────────────────────────────

def test_ccnot_both_controls_one_flips_target():
    sim = QrackSimulator(qubitCount=3)
    sim.x(0)
    sim.x(1)
    sim.ccnot(0, 1, 2)
    assert sim.prob(2) == pytest.approx(1.0, abs=ABS)


def test_ccnot_only_c1_no_flip():
    sim = QrackSimulator(qubitCount=3)
    sim.x(0)
    sim.ccnot(0, 1, 2)
    assert sim.prob(2) == pytest.approx(0.0, abs=ABS)


def test_ccnot_only_c2_no_flip():
    sim = QrackSimulator(qubitCount=3)
    sim.x(1)
    sim.ccnot(0, 1, 2)
    assert sim.prob(2) == pytest.approx(0.0, abs=ABS)


def test_ccnot_neither_control_no_flip():
    sim = QrackSimulator(qubitCount=3)
    sim.ccnot(0, 1, 2)
    assert sim.prob(2) == pytest.approx(0.0, abs=ABS)


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-controlled Hadamard: mch and mach
# ═══════════════════════════════════════════════════════════════════════════════

# ── mch — multiply-controlled H ───────────────────────────────────────────────

def test_mch_control_zero_no_effect():
    """mch with control |0> leaves target unchanged."""
    sim = QrackSimulator(qubitCount=2)
    sim.mch([0], 1)
    assert sim.prob(1) == pytest.approx(0.0, abs=ABS)


def test_mch_control_one_applies_h():
    """mch with control |1> puts target into superposition."""
    sim = QrackSimulator(qubitCount=2)
    sim.x(0)
    sim.mch([0], 1)
    assert sim.prob(1) == pytest.approx(0.5, abs=ABS)


def test_mch_two_controls_both_one_applies_h():
    """mch with two controls, both |1>, applies H to target."""
    sim = QrackSimulator(qubitCount=3)
    sim.x(0)
    sim.x(1)
    sim.mch([0, 1], 2)
    assert sim.prob(2) == pytest.approx(0.5, abs=ABS)


def test_mch_two_controls_one_missing_no_effect():
    """mch with two controls, only one |1>, does not fire."""
    sim = QrackSimulator(qubitCount=3)
    sim.x(0)
    sim.mch([0, 1], 2)
    assert sim.prob(2) == pytest.approx(0.0, abs=ABS)


def test_mch_involution_via_double_application():
    """mch·mch = I (H is self-inverse), so applying twice leaves |0>."""
    sim = QrackSimulator(qubitCount=2)
    sim.x(0)
    sim.mch([0], 1)
    sim.mch([0], 1)
    assert sim.prob(1) == pytest.approx(0.0, abs=ABS)


# ── mach — anti-controlled H ──────────────────────────────────────────────────

def test_mach_control_zero_applies_h():
    """mach (anti-controlled H) fires when control is |0>, putting target into superposition."""
    sim = QrackSimulator(qubitCount=2)
    sim.mach([0], 1)
    assert sim.prob(1) == pytest.approx(0.5, abs=ABS)


def test_mach_control_one_no_effect():
    """mach does not fire when control is |1>."""
    sim = QrackSimulator(qubitCount=2)
    sim.x(0)
    sim.mach([0], 1)
    assert sim.prob(1) == pytest.approx(0.0, abs=ABS)


def test_mach_involution_via_double_application():
    """mach·mach = I, so applying twice leaves |0> when control is |0>."""
    sim = QrackSimulator(qubitCount=2)
    sim.mach([0], 1)
    sim.mach([0], 1)
    assert sim.prob(1) == pytest.approx(0.0, abs=ABS)


def test_ccnot_involution():
    sim = QrackSimulator(qubitCount=3)
    sim.x(0)
    sim.x(1)
    sim.ccnot(0, 1, 2)
    sim.ccnot(0, 1, 2)
    assert sim.prob(2) == pytest.approx(0.0, abs=ABS)
