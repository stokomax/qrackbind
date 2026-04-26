"""Unit tests for measurement, probability, and reset functions."""

import math

import pytest

from qrackbind import QrackSimulator

ABS = 1e-5


# ── prob / prob_all ──────────────────────────────────────────────────────────

def test_prob_ground_state_is_zero():
    sim = QrackSimulator(qubitCount=1)
    assert sim.prob(0) == pytest.approx(0.0, abs=ABS)


def test_prob_excited_state_is_one():
    sim = QrackSimulator(qubitCount=1)
    sim.x(0)
    assert sim.prob(0) == pytest.approx(1.0, abs=ABS)


def test_prob_superposition():
    sim = QrackSimulator(qubitCount=1)
    sim.h(0)
    assert sim.prob(0) == pytest.approx(0.5, abs=ABS)


def test_prob_does_not_collapse():
    sim = QrackSimulator(qubitCount=1)
    sim.h(0)
    before = sim.prob(0)
    after = sim.prob(0)
    assert before == pytest.approx(after, abs=ABS)


def test_prob_all_length():
    sim = QrackSimulator(qubitCount=5)
    probs = sim.prob_all()
    assert len(probs) == 5


def test_prob_all_zero_state():
    sim = QrackSimulator(qubitCount=3)
    probs = sim.prob_all()
    assert all(p == pytest.approx(0.0, abs=ABS) for p in probs)


def test_prob_all_after_x():
    sim = QrackSimulator(qubitCount=3)
    sim.x(1)
    probs = sim.prob_all()
    assert probs[0] == pytest.approx(0.0, abs=ABS)
    assert probs[1] == pytest.approx(1.0, abs=ABS)
    assert probs[2] == pytest.approx(0.0, abs=ABS)


# ── measure ──────────────────────────────────────────────────────────────────

def test_measure_ground_state_is_false():
    sim = QrackSimulator(qubitCount=1)
    assert sim.measure(0) is False


def test_measure_excited_state_is_true():
    sim = QrackSimulator(qubitCount=1)
    sim.x(0)
    assert sim.measure(0) is True


def test_measure_returns_bool():
    sim = QrackSimulator(qubitCount=1)
    result = sim.measure(0)
    assert isinstance(result, bool)


def test_measure_collapses_to_zero():
    for _ in range(100):
        sim = QrackSimulator(qubitCount=1)
        sim.h(0)
        if sim.measure(0) is False:
            assert sim.prob(0) == pytest.approx(0.0, abs=ABS)
            return
    pytest.fail("Never measured False after 100 attempts")


def test_measure_collapses_to_one():
    for _ in range(100):
        sim = QrackSimulator(qubitCount=1)
        sim.h(0)
        if sim.measure(0) is True:
            assert sim.prob(0) == pytest.approx(1.0, abs=ABS)
            return
    pytest.fail("Never measured True after 100 attempts")


# ── measure_all ──────────────────────────────────────────────────────────────

def test_measure_all_length_matches():
    sim = QrackSimulator(qubitCount=5)
    results = sim.measure_all()
    assert len(results) == 5


def test_measure_all_types_are_bool():
    sim = QrackSimulator(qubitCount=3)
    results = sim.measure_all()
    assert all(isinstance(r, bool) for r in results)


def test_measure_all_ground_state():
    sim = QrackSimulator(qubitCount=3)
    results = sim.measure_all()
    assert all(r is False for r in results)


def test_measure_all_excited_state():
    sim = QrackSimulator(qubitCount=3)
    sim.x(0)
    sim.x(1)
    sim.x(2)
    results = sim.measure_all()
    assert all(r is True for r in results)


def test_measure_all_collapses_state():
    sim = QrackSimulator(qubitCount=3)
    sim.h(0)
    results = sim.measure_all()
    for i, r in enumerate(results):
        assert sim.prob(i) == pytest.approx(1.0 if r else 0.0, abs=ABS)


# ── force_measure ────────────────────────────────────────────────────────────

def test_force_measure_false():
    sim = QrackSimulator(qubitCount=1)
    result = sim.force_measure(0, False)
    assert result is False


def test_force_measure_true_collapses():
    sim = QrackSimulator(qubitCount=1)
    sim.force_measure(0, True)
    assert sim.prob(0) == pytest.approx(1.0, abs=ABS)


def test_force_measure_returns_bool():
    sim = QrackSimulator(qubitCount=1)
    result = sim.force_measure(0, False)
    assert isinstance(result, bool)


def test_force_measure_overrides_superposition():
    sim = QrackSimulator(qubitCount=1)
    sim.h(0)
    sim.force_measure(0, False)
    assert sim.prob(0) == pytest.approx(0.0, abs=ABS)


# ── reset_all ────────────────────────────────────────────────────────────────

def test_reset_all_from_ground():
    sim = QrackSimulator(qubitCount=2)
    sim.reset_all()
    for i in range(2):
        assert sim.prob(i) == pytest.approx(0.0, abs=ABS)


def test_reset_all_from_excited():
    sim = QrackSimulator(qubitCount=2)
    sim.x(0)
    sim.x(1)
    sim.reset_all()
    for i in range(2):
        assert sim.prob(i) == pytest.approx(0.0, abs=ABS)


def test_reset_all_multi_qubit_partial():
    sim = QrackSimulator(qubitCount=4)
    sim.x(0)
    sim.x(2)
    sim.reset_all()
    for i in range(4):
        assert sim.prob(i) == pytest.approx(0.0, abs=ABS)


# ── Bell-state correlation ───────────────────────────────────────────────────

def test_bell_state_correlation():
    """Measuring control then target of a Bell state should always agree."""
    for _ in range(50):
        sim = QrackSimulator(qubitCount=2)
        sim.h(0)
        sim.cnot(0, 1)
        m0 = sim.measure(0)
        m1 = sim.measure(1)
        assert m0 == m1, f"Entanglement broken: {m0} != {m1}"


# ── set_permutation ──────────────────────────────────────────────────────────

def test_set_permutation_zero():
    sim = QrackSimulator(qubitCount=4)
    sim.x(0)
    sim.x(2)
    sim.set_permutation(0)
    for i in range(4):
        assert sim.prob(i) == pytest.approx(0.0, abs=ABS)


def test_set_permutation_lsb_bit_pattern():
    # value=5 = 0b0101 -> qubit 0 and qubit 2 are |1>
    sim = QrackSimulator(qubitCount=4)
    sim.set_permutation(5)
    assert sim.prob(0) == pytest.approx(1.0, abs=ABS)
    assert sim.prob(1) == pytest.approx(0.0, abs=ABS)
    assert sim.prob(2) == pytest.approx(1.0, abs=ABS)
    assert sim.prob(3) == pytest.approx(0.0, abs=ABS)


def test_set_permutation_clears_superposition():
    sim = QrackSimulator(qubitCount=2)
    sim.h(0)
    sim.h(1)
    sim.set_permutation(3)  # |11>
    assert sim.prob(0) == pytest.approx(1.0, abs=ABS)
    assert sim.prob(1) == pytest.approx(1.0, abs=ABS)


# ── m_reg ────────────────────────────────────────────────────────────────────

def test_m_reg_ground_state_is_zero():
    sim = QrackSimulator(qubitCount=4)
    assert int(sim.m_reg(0, 4)) == 0


def test_m_reg_after_set_permutation():
    sim = QrackSimulator(qubitCount=4)
    sim.set_permutation(11)  # 0b1011
    assert int(sim.m_reg(0, 4)) == 11


def test_m_reg_partial_register():
    sim = QrackSimulator(qubitCount=6)
    sim.set_permutation(0b110100)
    # bits [2..5) are 0b1101 = 13
    assert int(sim.m_reg(2, 4)) == 0b1101


def test_m_reg_collapses_state():
    sim = QrackSimulator(qubitCount=2)
    sim.h(0)
    sim.h(1)
    val = int(sim.m_reg(0, 2))
    # After m_reg, qubits should be in a definite computational state
    assert int(sim.m_reg(0, 2)) == val


# ── measure_shots ────────────────────────────────────────────────────────────

def test_measure_shots_returns_dict():
    sim = QrackSimulator(qubitCount=2)
    sim.h(0)
    out = sim.measure_shots([0], 200)
    assert isinstance(out, dict)
    assert sum(out.values()) == 200


def test_measure_shots_ground_state_single_outcome():
    sim = QrackSimulator(qubitCount=3)
    out = sim.measure_shots([0, 1, 2], 50)
    assert out == {0: 50}


def test_measure_shots_excited_state_outcome_value():
    # qubit 0 = |1>, qubit 2 = |1> -> mask uses qpowers [1<<0, 1<<2] -> packed bits
    sim = QrackSimulator(qubitCount=3)
    sim.x(0)
    sim.x(2)
    out = sim.measure_shots([0, 2], 25)
    # both qubits |1> -> result mask has both bits set in their qpower positions
    assert sum(out.values()) == 25
    # Only one possible outcome since state is a permutation eigenstate
    assert len(out) == 1


def test_measure_shots_does_not_collapse():
    sim = QrackSimulator(qubitCount=1)
    sim.h(0)
    sim.measure_shots([0], 10)
    # Probability should still be ~0.5 — measure_shots samples without collapse
    assert sim.prob(0) == pytest.approx(0.5, abs=ABS)


def test_measure_shots_superposition_distribution():
    sim = QrackSimulator(qubitCount=1)
    sim.h(0)
    out = sim.measure_shots([0], 1000)
    # roughly 50/50
    zeros = out.get(0, 0)
    ones = out.get(1, 0)
    assert zeros + ones == 1000
    # very loose bound — just sanity check
    assert 300 < zeros < 700
    assert 300 < ones < 700
