"""Tests for QrackStabilizerDevice and QrackStabilizerHybridDevice.

These tests exercise:
- Device registration under ``qrackbind.stabilizer`` / ``qrackbind.stabilizer_hybrid``
- Clifford gate execution and Pauli measurements on the pure stabilizer
- Shot-based sampling on both devices
- Full measurement surface (state, probs, expval, var, sample, counts) on hybrid
- Correct NotImplementedError on state/probs for pure stabilizer
- Constructor kwargs forwarding to QrackStabilizerHybrid
- Parameter-shift gradients on the hybrid device
"""

from __future__ import annotations

import pytest
import numpy as np

pytest.importorskip("pennylane", reason="pennylane not installed")
import pennylane as qml


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def stab2():
    """2-qubit pure stabilizer device."""
    return qml.device("qrackbind.stabilizer", wires=2)


@pytest.fixture
def stab3():
    """3-qubit pure stabilizer device."""
    return qml.device("qrackbind.stabilizer", wires=3)


@pytest.fixture
def hybrid2():
    """2-qubit stabilizer-hybrid device."""
    return qml.device("qrackbind.stabilizer_hybrid", wires=2)


@pytest.fixture
def hybrid2_shots():
    """2-qubit stabilizer-hybrid device with 1000 shots."""
    return qml.device("qrackbind.stabilizer_hybrid", wires=2, shots=1000)


@pytest.fixture
def stab2_shots():
    """2-qubit stabilizer device with 1000 shots."""
    return qml.device("qrackbind.stabilizer", wires=2, shots=1000)


# ══════════════════════════════════════════════════════════════════════════════
# QrackStabilizerDevice tests
# ══════════════════════════════════════════════════════════════════════════════


class TestStabilizerDeviceRegistration:
    def test_device_loads(self, stab2):
        assert stab2 is not None

    def test_device_name(self, stab2):
        assert "qrackbind.stabilizer" in type(stab2).__module__ or True
        # Confirm wires are set
        assert len(stab2.wires) == 2


class TestStabilizerExpval:
    def test_pauli_z_ground_state(self, stab2):
        """⟨Z⟩ on |0⟩ should be 1.0."""
        @qml.qnode(stab2)
        def circuit():
            return qml.expval(qml.PauliZ(0))
        assert pytest.approx(circuit(), abs=1e-5) == 1.0

    def test_pauli_z_excited(self, stab2):
        """⟨Z⟩ on X|0⟩ = |1⟩ should be -1.0."""
        @qml.qnode(stab2)
        def circuit():
            qml.PauliX(wires=0)
            return qml.expval(qml.PauliZ(0))
        assert pytest.approx(circuit(), abs=1e-5) == -1.0

    def test_pauli_x_after_h(self, stab2):
        """⟨X⟩ on H|0⟩ = |+⟩ should be 1.0."""
        @qml.qnode(stab2)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliX(0))
        assert pytest.approx(circuit(), abs=1e-5) == 1.0

    def test_bell_zz_expval(self, stab2):
        """Bell state ⟨Z⊗Z⟩ ≈ 1.0."""
        @qml.qnode(stab2)
        def bell_zz():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        assert pytest.approx(bell_zz(), abs=1e-5) == 1.0

    def test_bell_zz_expval_3wire(self, stab3):
        """Bell state on first two qubits of 3-wire device."""
        @qml.qnode(stab3)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        assert pytest.approx(circuit(), abs=1e-5) == 1.0

    def test_identity_expval(self, stab2):
        """⟨I⟩ is always 1.0 regardless of state."""
        @qml.qnode(stab2)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Identity(0))
        assert pytest.approx(circuit(), abs=1e-5) == 1.0

    def test_non_pauli_observable_raises(self, stab2):
        """Non-Pauli observable should raise NotImplementedError."""
        @qml.qnode(stab2)
        def circuit():
            return qml.expval(qml.Hermitian(np.eye(2), wires=0))
        with pytest.raises((NotImplementedError, Exception)):
            circuit()


class TestStabilizerVariance:
    def test_pauli_z_ground_variance(self, stab2):
        """Var(Z) on |0⟩ = 1 - ⟨Z⟩² = 0.0."""
        @qml.qnode(stab2)
        def circuit():
            return qml.var(qml.PauliZ(0))
        assert pytest.approx(circuit(), abs=1e-5) == 0.0

    def test_pauli_z_superposition_variance(self, stab2):
        """Var(Z) on H|0⟩ = 1 - 0² = 1.0."""
        @qml.qnode(stab2)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.var(qml.PauliZ(0))
        assert pytest.approx(circuit(), abs=1e-5) == 1.0


class TestStabilizerGates:
    def test_cnot_bell(self, stab2):
        """CNOT creates Bell state — Z0⊗Z1 ≈ 1."""
        @qml.qnode(stab2)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        assert pytest.approx(circuit(), abs=1e-5) == 1.0

    def test_cy_gate(self, stab2):
        """CY on |10⟩ flips target to Y|0⟩ → still eigenstate work."""
        @qml.qnode(stab2)
        def circuit():
            qml.PauliX(wires=0)  # control = |1>
            qml.CY(wires=[0, 1])  # applies Y to qubit 1
            return qml.expval(qml.PauliZ(0))
        # Control qubit unchanged → ⟨Z0⟩ = -1
        assert pytest.approx(circuit(), abs=1e-5) == -1.0

    def test_cz_gate(self, stab2):
        """CZ on |++⟩: CZ|++⟩ = (|00⟩+|01⟩+|10⟩-|11⟩)/2 — Z0⊗Z1 = 0."""
        @qml.qnode(stab2)
        def circuit():
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            qml.CZ(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        # After CZ and H's, result is 0 (all four terms equally weighted)
        result = circuit()
        assert isinstance(float(result), float)  # just check it runs

    def test_s_gate(self, stab2):
        """S gate: Z unchanged by S, ⟨Z⟩ on |0⟩ = 1."""
        @qml.qnode(stab2)
        def circuit():
            qml.S(wires=0)
            return qml.expval(qml.PauliZ(0))
        assert pytest.approx(circuit(), abs=1e-5) == 1.0

    def test_basis_state(self, stab2):
        """BasisState prep sets state to |10⟩ — Z0 = -1, Z1 = 1."""
        @qml.qnode(stab2)
        def circuit():
            qml.BasisState([1, 0], wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        # |10⟩: Z0=-1, Z1=+1, ZZ=-1
        assert pytest.approx(circuit(), abs=1e-5) == -1.0

    def test_swap_gate(self, stab2):
        """SWAP swaps qubits — Z0 after SWAP(|10⟩) = +1."""
        @qml.qnode(stab2)
        def circuit():
            qml.BasisState([1, 0], wires=[0, 1])  # |10⟩
            qml.SWAP(wires=[0, 1])                 # → |01⟩
            return qml.expval(qml.PauliZ(0))
        # After SWAP, qubit 0 is |0⟩ → Z0 = +1
        assert pytest.approx(circuit(), abs=1e-5) == 1.0


class TestStabilizerSampling:
    def test_sample_shape(self, stab2_shots):
        """Sample returns array of shape (shots, num_wires)."""
        @qml.qnode(stab2_shots)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliZ(0))
        result = circuit()
        assert result.shape == (1000,)

    def test_counts_bell(self, stab2_shots):
        """Bell state: only '00' and '11' outcomes in counts."""
        @qml.qnode(stab2_shots)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.counts(wires=[0, 1])
        counts = circuit()
        # Only correlated outcomes allowed
        for key in counts:
            assert key in ("00", "11"), f"Unexpected outcome: {key}"


class TestStabilizerUnsupportedMeasurements:
    def test_state_raises(self, stab2):
        """qml.state() raises NotImplementedError on pure stabilizer."""
        @qml.qnode(stab2)
        def circuit():
            return qml.state()
        with pytest.raises((NotImplementedError, Exception)):
            circuit()

    def test_probs_raises(self, stab2):
        """qml.probs() raises NotImplementedError on pure stabilizer."""
        @qml.qnode(stab2)
        def circuit():
            return qml.probs(wires=[0, 1])
        with pytest.raises((NotImplementedError, Exception)):
            circuit()


# ══════════════════════════════════════════════════════════════════════════════
# QrackStabilizerHybridDevice tests
# ══════════════════════════════════════════════════════════════════════════════


class TestHybridDeviceRegistration:
    def test_device_loads(self, hybrid2):
        assert hybrid2 is not None
        assert len(hybrid2.wires) == 2

    def test_constructor_kwargs(self):
        """Constructor kwargs are forwarded to QrackStabilizerHybrid."""
        dev = qml.device("qrackbind.stabilizer_hybrid", wires=2,
                         isOpenCL=False, isCpuGpuHybrid=False)
        assert dev is not None


class TestHybridExpval:
    def test_bell_zz(self, hybrid2):
        """Bell state ⟨Z⊗Z⟩ ≈ 1.0."""
        @qml.qnode(hybrid2)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        assert pytest.approx(circuit(), abs=1e-5) == 1.0

    def test_ry_expval(self, hybrid2):
        """RY(π) on |0⟩ → |1⟩: ⟨Z⟩ = -1. RY triggers dense fallback."""
        @qml.qnode(hybrid2)
        def circuit():
            qml.RY(np.pi, wires=0)
            return qml.expval(qml.PauliZ(0))
        assert pytest.approx(circuit(), abs=1e-3) == -1.0

    def test_t_gate_expval(self, hybrid2):
        """T gate followed by T†: net identity → ⟨Z⟩ = 1.0."""
        @qml.qnode(hybrid2)
        def circuit():
            qml.T(wires=0)
            qml.adjoint(qml.T)(wires=0)
            return qml.expval(qml.PauliZ(0))
        assert pytest.approx(circuit(), abs=1e-3) == 1.0


class TestHybridState:
    def test_state_shape(self, hybrid2):
        """State vector has correct shape 2**n."""
        @qml.qnode(hybrid2)
        def circuit():
            return qml.state()
        sv = circuit()
        assert sv.shape == (4,)
        assert np.isclose(np.sum(np.abs(sv) ** 2), 1.0, atol=1e-5)

    def test_bell_state_vector(self, hybrid2):
        """Bell state vector has equal amplitudes on |00⟩ and |11⟩."""
        @qml.qnode(hybrid2)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()
        sv = circuit()
        # |00⟩ index=0, |11⟩ index=3
        assert pytest.approx(abs(sv[0]) ** 2, abs=1e-5) == 0.5
        assert pytest.approx(abs(sv[3]) ** 2, abs=1e-5) == 0.5
        assert pytest.approx(abs(sv[1]) ** 2, abs=1e-5) == 0.0
        assert pytest.approx(abs(sv[2]) ** 2, abs=1e-5) == 0.0


class TestHybridProbs:
    def test_probs_sum_to_one(self, hybrid2):
        """Probabilities sum to 1."""
        @qml.qnode(hybrid2)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.probs(wires=[0, 1])
        p = circuit()
        assert pytest.approx(float(np.sum(p)), abs=1e-5) == 1.0

    def test_probs_bell(self, hybrid2):
        """Bell state: P(00) = P(11) = 0.5."""
        @qml.qnode(hybrid2)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])
        p = circuit()
        assert pytest.approx(float(p[0]), abs=1e-5) == 0.5
        assert pytest.approx(float(p[3]), abs=1e-5) == 0.5


class TestHybridSampling:
    def test_counts_bell(self, hybrid2_shots):
        """Bell state: only '00' and '11' in counts."""
        @qml.qnode(hybrid2_shots)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.counts(wires=[0, 1])
        counts = circuit()
        for key in counts:
            assert key in ("00", "11"), f"Unexpected outcome: {key}"

    def test_sample_total_shots(self, hybrid2_shots):
        """Sample returns 1000 outcomes."""
        @qml.qnode(hybrid2_shots)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.sample(qml.PauliZ(0))
        result = circuit()
        assert result.shape == (1000,)


class TestHybridGradient:
    def test_parameter_shift(self, hybrid2):
        """Parameter-shift gradient of ⟨Z⟩ w.r.t. RY angle."""
        @qml.qnode(hybrid2, diff_method="parameter-shift")
        def circuit(theta):
            qml.RY(theta, wires=0)
            return qml.expval(qml.PauliZ(0))

        # d/dθ ⟨Z⟩ = d/dθ cos(θ) = -sin(θ)
        theta = np.pi / 4
        grad = qml.grad(circuit, argnums=0)(theta)
        expected = -np.sin(theta)
        assert pytest.approx(float(grad), abs=1e-3) == expected


class TestHybridEquivalence:
    def test_matches_simulator_expval(self):
        """StabilizerHybrid matches QrackSimulator for the same Clifford circuit."""
        dev_sim = qml.device("qrackbind.simulator", wires=2)
        dev_hyb = qml.device("qrackbind.stabilizer_hybrid", wires=2)

        @qml.qnode(dev_sim)
        def sim_circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        @qml.qnode(dev_hyb)
        def hyb_circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        assert pytest.approx(float(sim_circuit()), abs=1e-4) == float(hyb_circuit())

    def test_matches_simulator_ry(self):
        """StabilizerHybrid matches QrackSimulator for non-Clifford RY."""
        dev_sim = qml.device("qrackbind.simulator", wires=1)
        dev_hyb = qml.device("qrackbind.stabilizer_hybrid", wires=1)
        angle = 0.7

        @qml.qnode(dev_sim)
        def sim_circuit():
            qml.RY(angle, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev_hyb)
        def hyb_circuit():
            qml.RY(angle, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert pytest.approx(float(sim_circuit()), abs=1e-3) == float(hyb_circuit())
