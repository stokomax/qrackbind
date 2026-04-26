"""Phase 3 — state_vector / probabilities / set_state_vector."""

import math

import numpy as np
import pytest

from qrackbind import QrackSimulator


# ── state_vector ───────────────────────────────────────────────────────────────


class TestStateVector:
    def test_returns_ndarray(self):
        sim = QrackSimulator(qubitCount=2)
        sv = sim.state_vector
        assert isinstance(sv, np.ndarray)
        assert sv.dtype == np.complex64

    def test_ground_state_shape(self):
        sim = QrackSimulator(qubitCount=3)
        sv = sim.state_vector
        assert sv.shape == (8,)  # 2^3

    def test_ground_state_first_amplitude_is_one(self):
        sim = QrackSimulator(qubitCount=2)
        sv = sim.state_vector
        assert abs(sv[0] - 1.0) < 1e-5
        assert np.allclose(sv[1:], 0.0, atol=1e-5)

    def test_normalised(self):
        sim = QrackSimulator(qubitCount=3)
        sim.h(0)
        sim.h(1)
        sim.h(2)
        sv = sim.state_vector
        norm = float(np.sum(np.abs(sv) ** 2))
        assert abs(norm - 1.0) < 1e-5

    def test_bell_state_amplitudes(self):
        sim = QrackSimulator(qubitCount=2)
        sim.h(0)
        sim.cnot(0, 1)
        sv = sim.state_vector
        # Bell state: (|00> + |11>) / sqrt(2)
        invsqrt2 = 1.0 / math.sqrt(2)
        assert abs(abs(sv[0]) - invsqrt2) < 1e-5
        assert abs(sv[1]) < 1e-5
        assert abs(sv[2]) < 1e-5
        assert abs(abs(sv[3]) - invsqrt2) < 1e-5
        # The two non-zero amplitudes must share a global phase
        assert abs(sv[3] / sv[0] - 1.0) < 1e-4

    def test_snapshot_independence(self):
        sim = QrackSimulator(qubitCount=1)
        sv1 = sim.state_vector
        sim.x(0)
        sv2 = sim.state_vector
        # sv1 should still show |0> — proving snapshot semantics
        assert abs(sv1[0] - 1.0) < 1e-5
        assert abs(sv1[1]) < 1e-5
        # sv2 should show |1>
        assert abs(sv2[0]) < 1e-5
        assert abs(sv2[1] - 1.0) < 1e-5

    def test_after_dispose_shape_shrinks(self):
        sim = QrackSimulator(qubitCount=3, isTensorNetwork=False)
        assert sim.state_vector.shape == (8,)
        sim.dispose(2, 1)
        assert sim.state_vector.shape == (4,)


# ── probabilities ──────────────────────────────────────────────────────────────


class TestProbabilities:
    def test_returns_float32_ndarray(self):
        sim = QrackSimulator(qubitCount=2)
        p = sim.probabilities
        assert isinstance(p, np.ndarray)
        assert p.dtype == np.float32

    def test_sums_to_one(self):
        sim = QrackSimulator(qubitCount=4)
        sim.h(0)
        sim.h(1)
        p = sim.probabilities
        assert abs(float(np.sum(p)) - 1.0) < 1e-5

    def test_ground_state(self):
        sim = QrackSimulator(qubitCount=2)
        p = sim.probabilities
        assert abs(p[0] - 1.0) < 1e-5
        assert np.allclose(p[1:], 0.0, atol=1e-5)

    def test_equal_superposition(self):
        sim = QrackSimulator(qubitCount=2)
        sim.h(0)
        sim.h(1)
        p = sim.probabilities
        assert np.allclose(p, 0.25, atol=1e-5)


# ── set_state_vector ──────────────────────────────────────────────────────────


class TestSetStateVector:
    def test_roundtrip(self):
        sim = QrackSimulator(qubitCount=2)
        # Prepare a Bell state via gates, capture, reset, reinject
        sim.h(0)
        sim.cnot(0, 1)
        sv_orig = sim.state_vector.copy()
        sim.reset_all()
        sim.set_state_vector(sv_orig)
        sv_after = sim.state_vector
        assert np.allclose(sv_orig, sv_after, atol=1e-5)

    def test_wrong_size_raises(self):
        sim = QrackSimulator(qubitCount=2)
        with pytest.raises(Exception):
            sim.set_state_vector(np.zeros(3, dtype=np.complex64))

    def test_excite_via_injection(self):
        sim = QrackSimulator(qubitCount=1)
        psi = np.array([0, 1], dtype=np.complex64)  # |1>
        sim.set_state_vector(psi)
        assert sim.prob(0) == pytest.approx(1.0, abs=1e-5)

    def test_accepts_readonly_array(self):
        sim = QrackSimulator(qubitCount=1)
        psi = np.array([1, 0], dtype=np.complex64)
        psi.setflags(write=False)
        sim.set_state_vector(psi)  # must not raise


# ── consistency between state_vector / probabilities / prob_perm ─────────────


class TestConsistency:
    def test_state_vector_and_probabilities_consistent(self):
        sim = QrackSimulator(qubitCount=3)
        sim.h(0)
        sim.h(1)
        sim.cnot(0, 2)
        sv = sim.state_vector
        p_from_sv = np.abs(sv) ** 2
        p_direct = sim.probabilities
        assert np.allclose(p_from_sv, p_direct, atol=1e-5)
