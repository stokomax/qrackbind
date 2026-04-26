"""Phase 3 — per-basis-state amplitude access and prob_perm / prob_mask."""

import numpy as np
import pytest

from qrackbind import QrackSimulator


class TestGetAmplitude:
    def test_ground_state(self):
        sim = QrackSimulator(qubitCount=2)
        amp = sim.get_amplitude(0)
        assert abs(amp - 1.0) < 1e-5
        for i in range(1, 4):
            assert abs(sim.get_amplitude(i)) < 1e-5

    def test_after_x(self):
        sim = QrackSimulator(qubitCount=1)
        sim.x(0)
        assert abs(sim.get_amplitude(0)) < 1e-5
        assert abs(sim.get_amplitude(1) - 1.0) < 1e-5

    def test_returns_complex(self):
        sim = QrackSimulator(qubitCount=1)
        sim.h(0)
        amp = sim.get_amplitude(0)
        assert isinstance(amp, complex)


class TestSetAmplitude:
    def test_set_then_read_back(self):
        sim = QrackSimulator(qubitCount=2)
        # Move amplitude from |00> to |10>
        sim.set_amplitude(0, 0 + 0j)
        sim.set_amplitude(2, 1 + 0j)
        sim.update_running_norm()
        assert abs(sim.get_amplitude(2) - 1.0) < 1e-4
        assert abs(sim.get_amplitude(0)) < 1e-4


class TestProbPerm:
    def test_ground_state(self):
        sim = QrackSimulator(qubitCount=2)
        assert sim.prob_perm(0) == pytest.approx(1.0, abs=1e-5)
        for i in range(1, 4):
            assert sim.prob_perm(i) == pytest.approx(0.0, abs=1e-5)

    def test_matches_probabilities(self):
        sim = QrackSimulator(qubitCount=3)
        sim.h(0)
        sim.h(2)
        p_arr = sim.probabilities
        for i in range(8):
            assert sim.prob_perm(i) == pytest.approx(float(p_arr[i]), abs=1e-5)

    def test_distinct_from_prob_all_property(self):
        # `prob_all` is the per-qubit |1> probability vector (length n)
        # `prob_perm` is per-basis-state probability (scalar)
        sim = QrackSimulator(qubitCount=3)
        sim.h(0)
        per_qubit = sim.prob_all()
        assert isinstance(per_qubit, list)
        assert len(per_qubit) == 3
        scalar = sim.prob_perm(1)
        assert isinstance(scalar, float)


class TestProbMask:
    def test_selects_subset(self):
        sim = QrackSimulator(qubitCount=3)
        sim.x(1)  # state is |010>
        # mask = 0b010 = 2 (check qubit 1), permutation = 0b010 (qubit 1 is 1)
        assert sim.prob_mask(0b010, 0b010) == pytest.approx(1.0, abs=1e-5)
        assert sim.prob_mask(0b010, 0b000) == pytest.approx(0.0, abs=1e-5)

    def test_marginalises_unmasked_qubits(self):
        sim = QrackSimulator(qubitCount=2)
        sim.h(0)  # qubit 0 superposition; qubit 1 still |0>
        # Mask only qubit 1 (= bit 1 = 2). Qubit 1 is definitely 0.
        assert sim.prob_mask(0b10, 0b00) == pytest.approx(1.0, abs=1e-5)
        assert sim.prob_mask(0b10, 0b10) == pytest.approx(0.0, abs=1e-5)


class TestUpdateRunningNormAndPhase:
    def test_update_running_norm_runs(self):
        sim = QrackSimulator(qubitCount=2)
        sim.update_running_norm()  # smoke test

    def test_first_nonzero_phase_ground_state(self):
        sim = QrackSimulator(qubitCount=1)
        # Default global phase is undefined per build; just confirm scalar float.
        phase = sim.first_nonzero_phase()
        assert isinstance(phase, float)
