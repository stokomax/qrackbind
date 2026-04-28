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
Unit tests for the qrackbind PennyLane device.

Adapted from the Unitary Foundation PennyLane-Qrack test suite for use with
qrackbind:
https://github.com/unitaryfoundation/pennylane-qrack/tree/master/tests
"""
import pytest
import numpy as np

# Import fixtures
from conftest import U, U2, A


class TestDeviceUnits:
    """Unit tests for the qrackbind device."""

    @pytest.mark.parametrize("num_wires, shots", [(1, None), (2, 184), (3, 1)])
    def test_device_attributes(self, num_wires, shots):
        """Test that attributes are set as expected."""
        import pennylane as qml

        dev = qml.device("qrackbind.simulator", wires=num_wires, shots=shots)

        assert len(dev.wires) == num_wires
        assert dev.shots.total_shots == shots

    @pytest.mark.parametrize(
        "wires, prob",
        [
            ([0], [1.0, 0.0]),
            ([0, 1], [0.0, 1.0, 0.0, 0.0]),
            ([1, 3], [0.0, 0.0, 0.0, 1.0]),
        ],
    )
    def test_analytic_probability(self, wires, prob, tol):
        """Test the analytic probability for a basis state preparation."""
        import pennylane as qml

        dev = qml.device("qrackbind.simulator", wires=4)

        @qml.qnode(dev)
        def circuit():
            qml.BasisState(np.array([0, 1, 0, 1]), wires=[0, 1, 2, 3])
            return qml.probs(wires=wires)

        res = circuit()
        assert np.allclose(list(res), prob, atol=tol)

    def test_reset(self, tol):
        """Test the reset by comparing state before and after."""
        import pennylane as qml

        dev = qml.device("qrackbind.simulator", wires=4)

        @qml.qnode(dev)
        def circuit_with_x():
            qml.PauliX(wires=0)
            return qml.state()

        @qml.qnode(dev)
        def circuit_ground():
            return qml.state()

        # After X on wire 0, the state should be |1000...⟩
        state_with_x = circuit_with_x()
        # Ground state should be |0000...⟩
        state_ground = circuit_ground()

        # Verify they are different
        assert not np.allclose(state_with_x, state_ground, atol=tol)

        # Ground state should have all amplitude in first basis state
        expected_ground = np.zeros(16, dtype=np.complex128)
        expected_ground[0] = 1.0
        assert np.allclose(state_ground, expected_ground, atol=tol)
