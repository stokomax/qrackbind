# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Operation-application tests for the qrackbind PennyLane device.

These tests are adapted from the Unitary Foundation PennyLane-Qrack plugin
tests:
https://github.com/unitaryfoundation/pennylane-qrack/tree/master/tests

The legacy tests exercised old device internals directly via ``dev.apply`` and
``dev.state``. qrackbind implements PennyLane's modern ``Device`` API, so these
tests exercise operations through QNodes and public measurements.
"""

import numpy as np
import pennylane as qml
import pytest

from conftest import U

np.random.seed(42)


# ── Reference matrices ────────────────────────────────────────────────────────

I = np.identity(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
S = np.diag([1, 1j])
T = np.diag([1, np.exp(1j * np.pi / 4)])
SX = np.array([[(1 + 1j) / 2, (1 - 1j) / 2], [(1 - 1j) / 2, (1 + 1j) / 2]])
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
CNOT = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
CZ = np.diag([1, 1, 1, -1])
ISWAP = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])

toffoli = np.diag([1 for _ in range(8)])
toffoli = qml.matrix(qml.Toffoli(wires=[2, 1, 0]), wire_order=[0, 1, 2])
multix4 = qml.matrix(qml.MultiControlledX(wires=[3, 2, 1, 0]), wire_order=[0, 1, 2, 3])


def phase_shift(phi):
    return np.diag([1, np.exp(1j * phi)])


def c_phase_shift(phi):
    return np.diag([1, 1, 1, np.exp(1j * phi)])


def rx(theta):
    return np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * X


def ry(theta):
    return np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * Y


def rz(theta):
    return np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * Z


def u3(theta, phi, delta):
    return np.array(
        [
            [np.cos(theta / 2), -np.exp(1j * delta) * np.sin(theta / 2)],
            [
                np.exp(1j * phi) * np.sin(theta / 2),
                np.exp(1j * (phi + delta)) * np.cos(theta / 2),
            ],
        ]
    )


def crx(theta):
    return qml.matrix(qml.CRX(theta, wires=[1, 0]), wire_order=[0, 1])


def cry(theta):
    return qml.matrix(qml.CRY(theta, wires=[1, 0]), wire_order=[0, 1])


def crz(theta):
    return qml.matrix(qml.CRZ(theta, wires=[1, 0]), wire_order=[0, 1])


def crot(phi, theta, omega):
    return qml.matrix(qml.CRot(phi, theta, omega, wires=[1, 0]), wire_order=[0, 1])

SINGLE_QUBIT = [
    (lambda: qml.PauliX(wires=0), X),
    (lambda: qml.PauliY(wires=0), Y),
    (lambda: qml.PauliZ(wires=0), Z),
    (lambda: qml.Hadamard(wires=0), H),
    (lambda: qml.S(wires=0), S),
    (lambda: qml.T(wires=0), T),
    (lambda: qml.SX(wires=0), SX),
]

SINGLE_QUBIT_PARAM = [
    (lambda theta: qml.RX(theta, wires=0), rx),
    (lambda theta: qml.RY(theta, wires=0), ry),
    (lambda theta: qml.RZ(theta, wires=0), rz),
    (lambda theta: qml.PhaseShift(theta, wires=0), phase_shift),
]

TWO_QUBIT = [
    (lambda: qml.CNOT(wires=[0, 1]), CNOT),
    (lambda: qml.SWAP(wires=[0, 1]), SWAP),
    (lambda: qml.CZ(wires=[0, 1]), CZ),
    (lambda: qml.ISWAP(wires=[0, 1]), ISWAP),
]

TWO_QUBIT_PARAM = [
    (lambda theta: qml.CRX(theta, wires=[0, 1]), crx),
    (lambda theta: qml.CRY(theta, wires=[0, 1]), cry),
    (lambda theta: qml.CRZ(theta, wires=[0, 1]), crz),
    (lambda theta: qml.ControlledPhaseShift(theta, wires=[0, 1]), c_phase_shift),
]


def _probs(state):
    return np.abs(state) ** 2


class TestStateApply:
    """Test the device's state after application of gates."""

    @pytest.mark.parametrize(
        "state",
        [
            np.array([0, 0, 1, 0]),
            np.array([1, 0, 1, 0]),
            np.array([1, 1, 1, 1]),
        ],
    )
    def test_basis_state(self, state, tol):
        dev = qml.device("qrackbind.simulator", wires=4)

        @qml.qnode(dev)
        def circuit():
            qml.BasisState(state, wires=[0, 1, 2, 3])
            return qml.probs()

        expected = np.zeros([2**4])
        expected[np.ravel_multi_index(state, [2] * 4)] = 1
        assert np.allclose(circuit(), expected, atol=tol)

    def test_state_prep(self, init_state, tol):
        dev = qml.device("qrackbind.simulator", wires=1)
        state = init_state(1)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(state, wires=[0])
            return qml.state()

        assert np.allclose(circuit(), state, atol=tol)

    @pytest.mark.parametrize("op_factory, mat", SINGLE_QUBIT)
    def test_single_qubit_no_parameters(self, init_state, op_factory, mat, tol):
        dev = qml.device("qrackbind.simulator", wires=1)
        state = init_state(1)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(state, wires=[0])
            op_factory()
            return qml.state()

        assert np.allclose(circuit(), mat @ state, atol=tol)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("op_factory, func", SINGLE_QUBIT_PARAM)
    def test_single_qubit_parameters(self, init_state, op_factory, func, theta, tol):
        dev = qml.device("qrackbind.simulator", wires=1)
        state = init_state(1)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(state, wires=[0])
            op_factory(theta)
            return qml.state()

        assert np.allclose(circuit(), func(theta) @ state, atol=tol)

    @pytest.mark.parametrize("phi", [0.126, -0.721])
    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("omega", [1.213, -0.221])
    def test_single_qubit_three_parameters(self, init_state, phi, theta, omega, tol):
        dev = qml.device("qrackbind.simulator", wires=1)
        state = init_state(1)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(state, wires=[0])
            qml.U3(phi, theta, omega, wires=0)
            return qml.state()

        assert np.allclose(circuit(), u3(phi, theta, omega) @ state, atol=tol)

    @pytest.mark.parametrize("op_factory, mat", TWO_QUBIT)
    def test_two_qubit_no_parameters(self, init_state, op_factory, mat, tol):
        dev = qml.device("qrackbind.simulator", wires=2)
        state = init_state(2)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(state, wires=[0, 1])
            op_factory()
            return qml.state()

        assert np.allclose(circuit(), mat @ state, atol=tol)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("op_factory, func", TWO_QUBIT_PARAM)
    def test_two_qubit_parameters(self, init_state, op_factory, func, theta, tol):
        dev = qml.device("qrackbind.simulator", wires=2)
        state = init_state(2)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(state, wires=[0, 1])
            op_factory(theta)
            return qml.state()

        assert np.allclose(circuit(), func(theta) @ state, atol=tol)

    @pytest.mark.parametrize("phi", [0.126, -0.721])
    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("omega", [1.213, -0.221])
    def test_two_qubit_three_parameters(self, init_state, phi, theta, omega, tol):
        dev = qml.device("qrackbind.simulator", wires=2)
        state = init_state(2)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(state, wires=[0, 1])
            qml.CRot(phi, theta, omega, wires=[0, 1])
            return qml.state()

        assert np.allclose(circuit(), crot(phi, theta, omega) @ state, atol=tol)

    def test_qubit_unitary(self, init_state, tol):
        dev = qml.device("qrackbind.simulator", wires=1)
        state = init_state(1)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(state, wires=[0])
            qml.QubitUnitary(U, wires=[0])
            return qml.state()

        assert np.allclose(circuit(), U @ state, atol=tol)

    def test_toffoli(self, init_state, tol):
        dev = qml.device("qrackbind.simulator", wires=3)
        state = init_state(3)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(state, wires=[0, 1, 2])
            qml.Toffoli(wires=[0, 1, 2])
            return qml.state()

        assert np.allclose(circuit(), toffoli @ state, atol=tol)

    def test_multi_controlled_x(self, init_state, tol):
        dev = qml.device("qrackbind.simulator", wires=4)
        state = init_state(4)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(state, wires=[0, 1, 2, 3])
            qml.MultiControlledX(wires=[0, 1, 2, 3])
            return qml.state()

        assert np.allclose(circuit(), multix4 @ state, atol=tol)

    def test_invalid_state_prep(self):
        dev = qml.device("qrackbind.simulator", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(np.array([0, 123.432]), wires=[0, 1])
            return qml.state()

        with pytest.raises(ValueError, match="State must be of length"):
            circuit()

    def test_invalid_qubit_unitary(self):
        state = np.array([[0, 123.432], [-0.432, 23.4]])

        with pytest.raises(ValueError, match=r"Input unitary must be of shape"):
            qml.QubitUnitary(state, wires=[0, 1])

    def test_apply_errors_basis_state(self):
        dev = qml.device("qrackbind.simulator", wires=2)

        @qml.qnode(dev)
        def invalid_values():
            qml.BasisState(np.array([-0.2, 4.2]), wires=[0, 1])
            return qml.probs()

        with pytest.raises(ValueError, match="Basis state must only consist of 0s and 1s"):
            invalid_values()

    def test_state_probabilities_match_reference_after_rotations(self, init_state, tol):
        """Compare probabilities for a small mixed gate circuit.

        Probability comparison avoids over-constraining global phases while still
        exercising operation application through the qrackbind PennyLane device.
        """
        dev = qml.device("qrackbind.simulator", wires=2)
        state = np.array([1, 0, 0, 0], dtype=np.complex128)
        theta = 0.412
        phi = -0.221

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(state, wires=[0, 1])
            qml.RX(theta, wires=0)
            qml.RY(phi, wires=1)
            return qml.probs()

        default = qml.device("default.qubit", wires=2)

        @qml.qnode(default)
        def expected_circuit():
            qml.StatePrep(state, wires=[0, 1])
            qml.RX(theta, wires=0)
            qml.RY(phi, wires=1)
            return qml.probs()

        assert np.allclose(circuit(), expected_circuit(), atol=tol)
