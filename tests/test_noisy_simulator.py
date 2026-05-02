"""
Noisy simulator tests — QrackNoisySimulator and QrackNoisyStabilizerHybrid.
"""
import pytest
from qrackbind import (
    QrackNoisySimulator, QrackNoisyStabilizerHybrid,
    QrackSimulator, NoisyBase, Pauli,
)


class TestNoisyBase:
    def test_values_exist(self):
        assert NoisyBase.SIMULATOR is not None
        assert NoisyBase.STABILIZER_HYBRID is not None

    def test_values_are_distinct(self):
        assert NoisyBase.SIMULATOR != NoisyBase.STABILIZER_HYBRID

    def test_base_property_simulator(self):
        s = QrackNoisySimulator(qubitCount=2, base=NoisyBase.SIMULATOR)
        assert s.base == NoisyBase.SIMULATOR

    def test_base_property_stabilizer_hybrid(self):
        s = QrackNoisySimulator(qubitCount=2, base=NoisyBase.STABILIZER_HYBRID)
        assert s.base == NoisyBase.STABILIZER_HYBRID


class TestNoisySimulatorConstruction:
    def test_default_num_qubits(self):
        s = QrackNoisySimulator(qubitCount=3)
        assert s.num_qubits == 3

    def test_default_noise_param(self):
        s = QrackNoisySimulator(qubitCount=2)
        assert s.get_noise_parameter() == pytest.approx(0.01, abs=1e-6)

    def test_custom_noise_param(self):
        s = QrackNoisySimulator(qubitCount=2, noise_param=0.05)
        assert s.get_noise_parameter() == pytest.approx(0.05, abs=1e-6)

    def test_zero_noise_param(self):
        s = QrackNoisySimulator(qubitCount=2, noise_param=0.0)
        assert s.get_noise_parameter() == pytest.approx(0.0, abs=1e-6)

    def test_initial_fidelity_is_one(self):
        s = QrackNoisySimulator(qubitCount=2)
        assert s.unitary_fidelity == pytest.approx(1.0, abs=1e-6)

    def test_repr_contains_qubits(self):
        s = QrackNoisySimulator(qubitCount=4)
        assert "4" in repr(s)

    def test_context_manager(self):
        with QrackNoisySimulator(qubitCount=2) as s:
            s.h(0)
            assert s.num_qubits == 2

    def test_stabilizer_hybrid_base(self):
        s = QrackNoisySimulator(qubitCount=3, base=NoisyBase.STABILIZER_HYBRID)
        assert s.num_qubits == 3
        assert s.base == NoisyBase.STABILIZER_HYBRID


class TestNoiseParameter:
    def test_set_get_roundtrip(self):
        s = QrackNoisySimulator(qubitCount=2)
        s.set_noise_parameter(0.005)
        assert s.get_noise_parameter() == pytest.approx(0.005, abs=1e-6)

    def test_set_to_zero(self):
        s = QrackNoisySimulator(qubitCount=2, noise_param=0.1)
        s.set_noise_parameter(0.0)
        assert s.get_noise_parameter() == pytest.approx(0.0, abs=1e-6)


class TestUnitaryFidelity:
    def test_fidelity_decays_under_gates(self):
        s = QrackNoisySimulator(qubitCount=2, noise_param=0.05)
        f0 = s.unitary_fidelity
        for _ in range(10):
            s.h(0); s.cnot(0, 1)
        assert s.unitary_fidelity < f0
        assert s.unitary_fidelity > 0.0

    def test_reset_unitary_fidelity(self):
        s = QrackNoisySimulator(qubitCount=1, noise_param=0.1)
        for _ in range(20): s.h(0)
        assert s.unitary_fidelity < 1.0
        s.reset_unitary_fidelity()
        assert s.unitary_fidelity == pytest.approx(1.0, abs=1e-6)

    def test_reset_all_resets_fidelity(self):
        s = QrackNoisySimulator(qubitCount=2, noise_param=0.1)
        for _ in range(10): s.h(0)
        s.reset_all()
        assert s.unitary_fidelity == pytest.approx(1.0, abs=1e-6)

    def test_zero_noise_fidelity_stays_one(self):
        # At noise_param=0.0 the noisy wrapper is a pass-through.
        # Fidelity should remain at 1.0 — no noise channel fires.
        # NOTE: Qrack's fidelity tracker is only meaningful when noise_param>0;
        # when 0.0 is set, it returns 1.0 unconditionally.
        s = QrackNoisySimulator(qubitCount=2, noise_param=0.0)
        for _ in range(5):
            s.h(0); s.cnot(0, 1)
        assert s.unitary_fidelity == pytest.approx(1.0, abs=1e-4)


class TestDepolarizingChannel:
    def test_explicit_channel_decays_fidelity(self):
        # DepolarizingChannelWeak1Qb reduces fidelity when the engine is in
        # noisy mode (noise_param > 0). Use the default 0.01 so the fidelity
        # tracker is active.
        s = QrackNoisySimulator(qubitCount=1, noise_param=0.01)
        f0 = s.unitary_fidelity
        for _ in range(5):
            s.depolarizing_channel_1qb(0, 0.2)
        assert s.unitary_fidelity <= f0

    def test_qubit_out_of_range(self):
        from qrackbind import QrackQubitError
        s = QrackNoisySimulator(qubitCount=2, noise_param=0.0)
        with pytest.raises(QrackQubitError):
            s.depolarizing_channel_1qb(5, 0.1)


class TestZeroNoiseMirror:
    def test_zero_noise_matches_clean_simulator(self):
        noisy = QrackNoisySimulator(qubitCount=2, noise_param=0.0)
        clean = QrackSimulator(qubitCount=2)
        for s in (noisy, clean):
            s.h(0); s.cnot(0, 1)
        for q in range(2):
            assert noisy.prob(q) == pytest.approx(clean.prob(q), abs=1e-4)

    def test_qrack_simulator_has_no_noise_methods(self):
        s = QrackSimulator(qubitCount=2)
        assert not hasattr(s, "unitary_fidelity")
        assert not hasattr(s, "set_noise_parameter")
        assert not hasattr(s, "get_noise_parameter")
        assert not hasattr(s, "reset_unitary_fidelity")
        assert not hasattr(s, "depolarizing_channel_1qb")


class TestNoisyStateAccess:
    def test_state_vector_shape(self):
        s = QrackNoisySimulator(qubitCount=2)
        sv = s.state_vector
        assert sv.shape == (4,)

    def test_probabilities_shape(self):
        s = QrackNoisySimulator(qubitCount=2)
        probs = s.probabilities
        assert probs.shape == (4,)

    def test_get_amplitude_returns_complex(self):
        s = QrackNoisySimulator(qubitCount=1, noise_param=0.0)
        amp = s.get_amplitude(0)
        assert isinstance(amp, complex)


class TestSampleTrajectories:
    def test_histogram_sums_to_shots(self):
        s = QrackNoisySimulator(qubitCount=2, noise_param=0.01)
        s.h(0); s.cnot(0, 1)
        hist = s.sample_trajectories(shots=200)
        assert sum(hist.values()) == 200

    def test_histogram_keys_are_ints(self):
        s = QrackNoisySimulator(qubitCount=2, noise_param=0.01)
        s.h(0)
        hist = s.sample_trajectories(shots=100)
        for k in hist:
            assert isinstance(k, int)

    def test_bell_state_concentrated(self):
        s = QrackNoisySimulator(qubitCount=2, noise_param=0.01)
        s.h(0); s.cnot(0, 1)
        hist = s.sample_trajectories(shots=500)
        p00 = hist.get(0, 0) / 500
        p11 = hist.get(3, 0) / 500
        assert p00 + p11 > 0.8

    def test_exp_val_decreases_with_noise(self):
        """Z expectation on |0> should be closer to 1 with lower noise."""
        results = []
        for strength in (0.0, 0.05, 0.2):
            s = QrackNoisySimulator(qubitCount=1, noise_param=strength)
            for _ in range(10):
                s.h(0); s.h(0)   # double-H = identity (noiseless)
            results.append(s.exp_val(Pauli.PauliZ, 0))
        # Each step should be less than or equal the previous (allowing tolerance)
        assert results[0] >= results[1] - 0.15
        assert results[1] >= results[2] - 0.15


class TestNoisyStabilizerHybrid:
    def test_construction(self):
        s = QrackNoisyStabilizerHybrid(qubitCount=3)
        assert s.num_qubits == 3
        assert s.get_noise_parameter() == pytest.approx(0.01, abs=1e-6)

    def test_initial_fidelity_is_one(self):
        s = QrackNoisyStabilizerHybrid(qubitCount=2)
        assert s.unitary_fidelity == pytest.approx(1.0, abs=1e-6)

    def test_set_noise_parameter(self):
        s = QrackNoisyStabilizerHybrid(qubitCount=2)
        s.set_noise_parameter(0.03)
        assert s.get_noise_parameter() == pytest.approx(0.03, abs=1e-6)

    def test_fidelity_decays(self):
        s = QrackNoisyStabilizerHybrid(qubitCount=2, noise_param=0.05)
        for _ in range(8):
            s.h(0); s.cnot(0, 1)
        assert s.unitary_fidelity < 1.0

    def test_is_clifford_is_bool(self):
        s = QrackNoisyStabilizerHybrid(qubitCount=3)
        assert isinstance(s.is_clifford, bool)

    def test_sample_trajectories_sums(self):
        s = QrackNoisyStabilizerHybrid(qubitCount=2, noise_param=0.01)
        s.h(0); s.cnot(0, 1)
        hist = s.sample_trajectories(shots=100)
        assert sum(hist.values()) == 100

    def test_reset_all_resets_fidelity(self):
        s = QrackNoisyStabilizerHybrid(qubitCount=2, noise_param=0.1)
        for _ in range(10): s.h(0)
        s.reset_all()
        assert s.unitary_fidelity == pytest.approx(1.0, abs=1e-6)

    def test_state_vector_shape(self):
        s = QrackNoisyStabilizerHybrid(qubitCount=2)
        assert s.state_vector.shape == (4,)

    def test_context_manager(self):
        with QrackNoisyStabilizerHybrid(qubitCount=2) as s:
            s.h(0)
            assert s.num_qubits == 2

    def test_no_base_kwarg(self):
        # QrackNoisyStabilizerHybrid pins STABILIZER_HYBRID; passing `base=`
        # should raise TypeError since the kwarg doesn't exist.
        with pytest.raises(TypeError):
            QrackNoisyStabilizerHybrid(qubitCount=2, base=0)
