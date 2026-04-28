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
"""
Integration tests for the qrackbind PennyLane device.

Adapted from the Unitary Foundation PennyLane-Qrack test suite for use with
qrackbind:
https://github.com/unitaryfoundation/pennylane-qrack/tree/master/tests
"""
import numpy as np

import pennylane as qml


class TestIntegration:
    """Some basic integration tests."""

    def test_load_device(self):
        """Test that the qrackbind device loads correctly."""
        from qrackbind.pennylane.device import QrackDevice

        dev = qml.device("qrackbind.simulator", wires=2, shots=int(1e6))

        assert len(dev.wires) == 2
        assert dev.shots.total_shots == int(1e6)

        # Check that the device is registered correctly
        assert isinstance(dev, QrackDevice)

    def test_multiple_circuit_executions(self):
        """Test that the device works correctly with multiple circuit executions."""
        dev = qml.device("qrackbind.simulator", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliY(0))

        # Run multiple times to ensure reallocation works
        for _ in range(3):
            res = circuit()
            assert isinstance(res, float)

    def test_expectation(self):
        """Test that expectation of a non-trivial circuit is correct."""
        dev = qml.device("qrackbind.simulator", wires=2, shots=int(1e6))

        theta = 0.432
        phi = 0.123

        @qml.qnode(dev)
        def circuit():
            qml.adjoint(qml.RY(theta, wires=[0]))
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliX(wires=0)), qml.expval(qml.PauliX(wires=1))

        res = circuit()
        expected = np.array([np.sin(-theta) * np.sin(phi), np.sin(phi)])
        assert np.allclose(res, expected, atol=0.05)

    def test_expectation_with_gradients(self):
        """Test that expectation with gradients works correctly."""
        dev = qml.device("qrackbind.simulator", wires=2)

        theta = 0.432
        phi = 0.123

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x, y):
            qml.adjoint(qml.RY(x, wires=[0]))
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliX(wires=0))

        # Test that gradients work
        x = np.array(theta)
        y = np.array(phi)

        grad = qml.grad(circuit, argnums=(0, 1))(x, y)
        assert len(grad) == 2
        assert np.isscalar(grad[0]) or np.shape(grad[0]) == ()
        assert np.isscalar(grad[1]) or np.shape(grad[1]) == ()

    def test_variance(self):
        """Test that variance of a non-trivial circuit is correct."""
        dev = qml.device("qrackbind.simulator", wires=2, shots=int(1e6))

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.var(qml.PauliZ(0))

        res = circuit()
        # For a Hadamard state, variance of Z should be 1.0
        assert np.allclose(res, 1.0, atol=0.05)

    def test_probs(self):
        """Test that probabilities are computed correctly."""
        dev = qml.device("qrackbind.simulator", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.probs(wires=[0])

        res = circuit()
        assert np.allclose(res, [0.5, 0.5], atol=1e-4)

    def test_bell_state(self):
        """Test creation and measurement of a Bell state."""
        dev = qml.device("qrackbind.simulator", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        res = circuit()
        # Bell state should have expectation 1.0 for ZZ
        assert np.allclose(res, 1.0, atol=1e-4)
