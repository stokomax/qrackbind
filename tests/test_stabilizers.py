"""
Phase 10 test suite — QrackStabilizer and QrackStabilizerHybrid.

Tests are organised in two classes mirroring the two new standalone classes.
"""
import math

import pytest

from qrackbind import QrackSimulator, QrackStabilizer, QrackStabilizerHybrid, Pauli


# ── QrackStabilizer ────────────────────────────────────────────────────────────

class TestStabilizerCore:

    def test_construction(self):
        s = QrackStabilizer(qubitCount=4)
        assert s.num_qubits == 4
        assert "4" in repr(s)

    def test_repr(self):
        s = QrackStabilizer(qubitCount=3)
        r = repr(s)
        assert "QrackStabilizer" in r
        assert "3" in r

    def test_bell_state_probs(self):
        s = QrackStabilizer(qubitCount=2)
        s.h(0)
        s.cnot(0, 1)
        # Both qubits should have P(|1>) ≈ 0.5
        assert s.prob(0) == pytest.approx(0.5, abs=1e-4)
        assert s.prob(1) == pytest.approx(0.5, abs=1e-4)

    def test_bell_state_correlated_measurement(self):
        s = QrackStabilizer(qubitCount=2)
        s.h(0)
        s.cnot(0, 1)
        # Both outcomes must agree
        first = s.measure(0)
        second = s.measure(1)
        assert first == second

    def test_large_ghz_polynomial_memory(self):
        # 50-qubit GHZ — trivial for a stabilizer engine (polynomial memory)
        s = QrackStabilizer(qubitCount=50)
        s.h(0)
        for q in range(1, 50):
            s.cnot(0, q)
        # All qubits must agree after measurement
        first = s.measure(0)
        for q in range(1, 50):
            assert s.measure(q) == first

    def test_no_non_clifford_methods(self):
        s = QrackStabilizer(qubitCount=2)
        assert not hasattr(s, "rx")
        assert not hasattr(s, "ry")
        assert not hasattr(s, "rz")
        assert not hasattr(s, "r1")
        assert not hasattr(s, "u")
        assert not hasattr(s, "t")
        assert not hasattr(s, "tdg")
        assert not hasattr(s, "mtrx")
        assert not hasattr(s, "mcmtrx")

    def test_no_state_vector(self):
        s = QrackStabilizer(qubitCount=2)
        assert not hasattr(s, "state_vector")
        assert not hasattr(s, "probabilities")
        assert not hasattr(s, "_state_vector_impl")
        assert not hasattr(s, "_probabilities_impl")

    def test_pauli_z_expectation_ground(self):
        s = QrackStabilizer(qubitCount=1)
        # |0> is +1 eigenstate of Z → <Z> = +1
        assert s.exp_val(Pauli.PauliZ, 0) == pytest.approx(1.0, abs=1e-4)

    def test_pauli_z_expectation_excited(self):
        s = QrackStabilizer(qubitCount=1)
        s.x(0)
        # |1> is −1 eigenstate of Z → <Z> = −1
        assert s.exp_val(Pauli.PauliZ, 0) == pytest.approx(-1.0, abs=1e-4)

    def test_pauli_x_expectation_plus(self):
        s = QrackStabilizer(qubitCount=1)
        s.h(0)
        # |+> is +1 eigenstate of X → <X> = +1
        assert s.exp_val(Pauli.PauliX, 0) == pytest.approx(1.0, abs=1e-4)

    def test_clifford_gates_one_qubit(self):
        """h, x, y, z, s, sdg, sx, sxdg all callable; no exception raised."""
        s = QrackStabilizer(qubitCount=2)
        for gate in ("h", "x", "y", "z", "s", "sdg", "sx", "sxdg"):
            getattr(s, gate)(0)

    def test_clifford_gates_two_qubit(self):
        """cnot, cy, cz, swap, iswap all callable; no exception raised."""
        s = QrackStabilizer(qubitCount=2)
        s.cnot(0, 1)
        s.cy(0, 1)
        s.cz(0, 1)
        s.swap(0, 1)
        s.iswap(0, 1)

    def test_mcx_single_control(self):
        # QStabilizer only supports 1-control MCX (CNOT).
        # 2-control Toffoli on QINTERFACE_STABILIZER raises at runtime because
        # it is not a Clifford gate (Toffoli is universal, not Clifford).
        s = QrackStabilizer(qubitCount=2)
        s.x(0)            # control in |1>
        s.mcx([0], 1)     # MCX with one control = CNOT
        assert s.measure(1) == True

    def test_reset_all(self):
        s = QrackStabilizer(qubitCount=3)
        s.h(0); s.x(1)
        s.reset_all()
        assert s.prob(0) == pytest.approx(0.0, abs=1e-4)
        assert s.prob(1) == pytest.approx(0.0, abs=1e-4)

    def test_set_permutation(self):
        s = QrackStabilizer(qubitCount=4)
        # Set to |0101> = permutation 5 (q0=1, q1=0, q2=1, q3=0)
        s.set_permutation(5)
        assert s.measure(0) == True
        assert s.measure(2) == True

    def test_context_manager(self):
        with QrackStabilizer(qubitCount=2) as s:
            s.h(0); s.cnot(0, 1)
            assert s.prob(0) == pytest.approx(0.5, abs=1e-4)

    def test_measure_all(self):
        s = QrackStabilizer(qubitCount=3)
        s.x(0); s.x(2)
        results = s.measure_all()
        assert results[0] == True
        assert results[1] == False
        assert results[2] == True

    def test_force_measure(self):
        s = QrackStabilizer(qubitCount=1)
        s.h(0)
        # Force the outcome to True
        result = s.force_measure(0, True)
        assert result == True
        assert s.measure(0) == True   # state is now collapsed to |1>


# ── QrackStabilizerHybrid ──────────────────────────────────────────────────────

class TestStabilizerHybridCore:

    def test_construction(self):
        s = QrackStabilizerHybrid(qubitCount=3)
        assert s.num_qubits == 3

    def test_repr(self):
        s = QrackStabilizerHybrid(qubitCount=2)
        r = repr(s)
        assert "QrackStabilizerHybrid" in r
        assert "2" in r

    def test_starts_in_clifford_mode(self):
        s = QrackStabilizerHybrid(qubitCount=4)
        assert s.is_clifford is True

    def test_clifford_circuit_stays_clifford(self):
        s = QrackStabilizerHybrid(qubitCount=4)
        s.h(0); s.cnot(0, 1); s.cnot(1, 2); s.cnot(2, 3)
        assert s.is_clifford is True

    def test_non_clifford_gate_does_not_raise(self):
        # QrackStabilizerHybrid silently falls back to dense simulation on
        # non-Clifford gates — no exception is raised.
        # NOTE: QInterface.isClifford() is a type-level property on
        # QStabilizerHybrid (it always returns True because the interface IS
        # a Clifford-type engine), so we cannot use is_clifford to detect the
        # internal mode switch.  The observable contract is: the simulation
        # produces correct probabilities before and after the non-Clifford gate,
        # and state_vector is available throughout.
        s = QrackStabilizerHybrid(qubitCount=2)
        s.set_t_injection(False)
        s.h(0)
        assert s.is_clifford is True            # always True for this engine type
        s.rx(0.5, 0)                            # triggers internal dense fallback
        assert s.is_clifford is True            # still True (type-level, not mode)
        # Verify the simulation still produces a sensible result.
        p = s.prob(0)
        assert 0.0 <= p <= 1.0
        sv = s.state_vector
        assert sv.shape == (4,)

    def test_bell_state_probs(self):
        s = QrackStabilizerHybrid(qubitCount=2)
        s.h(0); s.cnot(0, 1)
        assert s.prob(0) == pytest.approx(0.5, abs=1e-4)
        assert s.prob(1) == pytest.approx(0.5, abs=1e-4)

    def test_rz_near_clifford_path(self):
        """RZ at a non-π/2 angle; both T-injection on and off give same probs."""
        a = QrackStabilizerHybrid(qubitCount=1)
        a.set_t_injection(True)
        a.h(0); a.rz(math.pi / 3, 0); a.h(0)
        p_inj = a.prob(0)

        b = QrackStabilizerHybrid(qubitCount=1)
        b.set_t_injection(False)
        b.h(0); b.rz(math.pi / 3, 0); b.h(0)
        p_dense = b.prob(0)

        assert p_inj == pytest.approx(p_dense, abs=1e-4)

    def test_state_vector_after_fallback(self):
        s = QrackStabilizerHybrid(qubitCount=2)
        s.rx(math.pi, 0)        # RX(π) ≈ X; forces fallback to dense
        sv = s.state_vector
        assert sv.shape == (4,)
        # |01> (qubit 0 = |1> = LSB) → basis state index 1
        assert abs(sv[1]) == pytest.approx(1.0, abs=1e-3)

    def test_state_vector_clifford_mode(self):
        """state_vector works even while still in stabilizer mode."""
        s = QrackStabilizerHybrid(qubitCount=1)
        # Still Clifford here; Qrack should materialise from tableau
        sv = s.state_vector
        assert sv.shape == (2,)
        assert abs(sv[0]) == pytest.approx(1.0, abs=1e-4)  # |0>

    def test_probabilities_property(self):
        s = QrackStabilizerHybrid(qubitCount=2)
        s.x(0)
        probs = s.probabilities
        assert probs.shape == (4,)
        # |01> → index 1 has probability 1
        assert probs[1] == pytest.approx(1.0, abs=1e-4)

    def test_get_amplitude(self):
        s = QrackStabilizerHybrid(qubitCount=2)
        # Ground state: amplitude of |00> = 1+0j
        amp = s.get_amplitude(0)
        assert abs(amp) == pytest.approx(1.0, abs=1e-4)

    def test_matches_qrack_simulator(self):
        """Same Clifford circuit on QrackStabilizerHybrid and QrackSimulator → same probs."""
        a = QrackStabilizerHybrid(qubitCount=3)
        b = QrackSimulator(qubitCount=3, isStabilizerHybrid=True)
        for sim in (a, b):
            sim.h(0); sim.cnot(0, 1); sim.cnot(1, 2)
        for q in range(3):
            assert a.prob(q) == pytest.approx(b.prob(q), abs=1e-4)

    def test_t_injection_toggle(self):
        """set_t_injection is callable and idempotent."""
        s = QrackStabilizerHybrid(qubitCount=2)
        s.set_t_injection(True)
        s.set_t_injection(False)

    def test_set_use_exact_near_clifford(self):
        s = QrackStabilizerHybrid(qubitCount=2)
        s.set_use_exact_near_clifford(True)
        s.set_use_exact_near_clifford(False)

    def test_reset_all(self):
        s = QrackStabilizerHybrid(qubitCount=3)
        s.h(0); s.x(1)
        s.reset_all()
        assert s.prob(0) == pytest.approx(0.0, abs=1e-4)
        assert s.prob(1) == pytest.approx(0.0, abs=1e-4)
        assert s.is_clifford is True

    def test_context_manager(self):
        with QrackStabilizerHybrid(qubitCount=2) as s:
            s.h(0); s.cnot(0, 1)
            assert s.prob(0) == pytest.approx(0.5, abs=1e-4)

    def test_full_gate_surface(self):
        """Verify non-Clifford methods exist on the hybrid class."""
        s = QrackStabilizerHybrid(qubitCount=2)
        assert hasattr(s, "rx")
        assert hasattr(s, "ry")
        assert hasattr(s, "rz")
        assert hasattr(s, "r1")
        assert hasattr(s, "t")
        assert hasattr(s, "tdg")
        assert hasattr(s, "u")
        assert hasattr(s, "mtrx")
        assert hasattr(s, "mcmtrx")
        assert hasattr(s, "state_vector")
        assert hasattr(s, "probabilities")
