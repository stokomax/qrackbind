"""Phase 8 tests — PennyLane device integration."""

import math

import numpy as np
import pytest

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not PENNYLANE_AVAILABLE,
    reason="PennyLane not installed")


@pytest.fixture
def dev2():
    return qml.device("qrackbind.simulator", wires=2)


@pytest.fixture
def dev4():
    return qml.device("qrackbind.simulator", wires=4)


# ── Device registration ────────────────────────────────────────────────────────

class TestDeviceRegistration:
    def test_device_importable(self):
        from qrackbind.pennylane.device import QrackDevice
        assert QrackDevice is not None

    def test_qml_device_creates_qrack_device(self):
        from qrackbind.pennylane.device import QrackDevice
        dev = qml.device("qrackbind.simulator", wires=2)
        assert isinstance(dev, QrackDevice)

    def test_device_has_correct_num_wires(self, dev2):
        assert len(dev2.wires) == 2


# ── Basic circuit execution ────────────────────────────────────────────────────

class TestCircuitExecution:
    def test_hadamard_expval(self, dev2):
        @qml.qnode(dev2)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))
        assert circuit() == pytest.approx(0.0, abs=1e-4)

    def test_pauli_x_expval(self, dev2):
        @qml.qnode(dev2)
        def circuit():
            qml.PauliX(wires=0)
            return qml.expval(qml.PauliZ(0))
        assert circuit() == pytest.approx(-1.0, abs=1e-5)

    def test_ground_state_expval(self, dev2):
        @qml.qnode(dev2)
        def circuit():
            return qml.expval(qml.PauliZ(0))
        assert circuit() == pytest.approx(1.0, abs=1e-5)

    def test_bell_state_zz_expval(self, dev2):
        @qml.qnode(dev2)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        assert circuit() == pytest.approx(1.0, abs=1e-4)

    def test_rx_expval(self, dev2):
        @qml.qnode(dev2)
        def circuit(angle):
            qml.RX(angle, wires=0)
            return qml.expval(qml.PauliZ(0))
        angle = 0.7
        assert circuit(angle) == pytest.approx(math.cos(angle), abs=1e-4)

    def test_state_measurement(self, dev2):
        @qml.qnode(dev2)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.state()
        sv = circuit()
        # State vector has length 2^(device_wires)
        assert len(sv) == 4
        # All amplitudes should be real
        assert all(abs(amp.imag) < 1e-4 for amp in sv)
        # Non-zero amplitudes should be near ±1/√2
        nonzero = [abs(amp.real) for amp in sv if abs(amp) > 1e-4]
        assert all(abs(val - 1 / math.sqrt(2)) < 1e-3 for val in nonzero)
        assert len(nonzero) == 2

    def test_probs_measurement(self, dev2):
        @qml.qnode(dev2)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.probs(wires=[0])
        probs = circuit()
        assert probs == pytest.approx([0.5, 0.5], abs=1e-4)


# ── Variance ──────────────────────────────────────────────────────────────────

class TestVariance:
    def test_z_variance_eigenstate(self, dev2):
        @qml.qnode(dev2)
        def circuit():
            return qml.var(qml.PauliZ(0))
        assert circuit() == pytest.approx(0.0, abs=1e-5)

    def test_z_variance_superposition(self, dev2):
        @qml.qnode(dev2)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.var(qml.PauliZ(0))
        assert circuit() == pytest.approx(1.0, abs=1e-4)


# ── Parametric circuits and gradients ────────────────────────────────────────

class TestGradients:
    def test_parameter_shift_gradient(self, dev2):
        @qml.qnode(dev2, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        x = np.array(0.5)
        grad = qml.grad(circuit, argnums=0)(x)
        expected = -math.sin(0.5)
        assert float(grad) == pytest.approx(expected, abs=1e-4)

    def test_two_parameter_gradient(self, dev2):
        @qml.qnode(dev2, diff_method="parameter-shift")
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        x = np.array(0.4)
        y = np.array(0.6)
        grads = qml.grad(circuit, argnums=(0, 1))(x, y)
        grad_x, grad_y = float(grads[0]), float(grads[1])
        assert isinstance(grad_x, float)
        assert isinstance(grad_y, float)

    def test_gradient_matches_finite_difference(self, dev2):
        @qml.qnode(dev2, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        x0 = np.array(0.8)
        grad_ps = float(qml.grad(circuit, argnums=0)(x0))

        eps = 1e-4
        fd = (circuit(x0 + eps) - circuit(x0 - eps)) / (2 * eps)
        assert grad_ps == pytest.approx(fd, abs=1e-3)


# ── Gate coverage ──────────────────────────────────────────────────────────────

class TestGateCoverage:
    def test_ry_gate(self, dev2):
        @qml.qnode(dev2)
        def circuit(angle):
            qml.RY(angle, wires=0)
            return qml.expval(qml.PauliZ(0))
        assert circuit(math.pi) == pytest.approx(-1.0, abs=1e-4)

    def test_toffoli_gate(self, dev4):
        @qml.qnode(dev4)
        def circuit():
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            qml.Toffoli(wires=[0, 1, 2])
            return qml.expval(qml.PauliZ(2))
        assert circuit() == pytest.approx(-1.0, abs=1e-4)

    def test_rot_gate(self, dev2):
        @qml.qnode(dev2)
        def circuit():
            qml.Rot(math.pi, 0, 0, wires=0)
            return qml.expval(qml.PauliZ(0))
        assert circuit() == pytest.approx(-1.0, abs=1e-4)

    def test_phase_shift_gate(self, dev2):
        @qml.qnode(dev2)
        def circuit(phi):
            qml.Hadamard(wires=0)
            qml.PhaseShift(phi, wires=0)
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))
        assert circuit(math.pi) == pytest.approx(0.0, abs=1e-4)


# ── VQE smoke test ────────────────────────────────────────────────────────────

class TestVQE:
    def test_vqe_converges(self):
        """VQE minimisation reduces energy over iterations.

        Uses manual gradient descent with qml.grad since PennyLane's
        GradientDescentOptimizer doesn't detect NumPy array parameters
        as trainable without requires_grad.
        """
        dev = qml.device("qrackbind.simulator", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        params = np.array([0.5, 0.5])
        energy_start = circuit(params)

        for _ in range(30):
            grad = qml.grad(circuit, argnums=0)(params)
            params = params - 0.5 * grad

        energy_final = circuit(params)
        assert energy_final < energy_start
