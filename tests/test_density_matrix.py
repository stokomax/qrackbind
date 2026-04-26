"""Phase 3 — get_reduced_density_matrix (partial trace)."""

import numpy as np
import pytest

from qrackbind import QrackSimulator


class TestReducedDensityMatrix:
    def test_shape_and_dtype(self):
        sim = QrackSimulator(qubitCount=4)
        rho = sim.get_reduced_density_matrix([0, 1])
        assert rho.shape == (4, 4)  # 2^2 x 2^2
        assert rho.dtype == np.complex64

    def test_single_qubit_shape(self):
        sim = QrackSimulator(qubitCount=3)
        rho = sim.get_reduced_density_matrix([0])
        assert rho.shape == (2, 2)

    def test_trace_is_one(self):
        sim = QrackSimulator(qubitCount=3)
        sim.h(0)
        sim.cnot(0, 1)
        rho = sim.get_reduced_density_matrix([0, 1])
        trace = complex(np.trace(rho))
        assert abs(trace - 1.0) < 1e-4

    def test_pure_state_purity(self):
        # For a pure subsystem, Tr(rho^2) = 1
        sim = QrackSimulator(qubitCount=2)
        sim.h(0)  # product state — each qubit is pure on its own
        rho = sim.get_reduced_density_matrix([0])
        rho_sq = rho @ rho
        purity = complex(np.trace(rho_sq))
        assert abs(purity - 1.0) < 1e-4

    def test_maximally_mixed_qubit_from_bell_state(self):
        # Tracing out half of a Bell state gives a maximally mixed qubit:
        # rho = I/2 = [[0.5, 0], [0, 0.5]]
        sim = QrackSimulator(qubitCount=2)
        sim.h(0)
        sim.cnot(0, 1)
        rho = sim.get_reduced_density_matrix([0])
        assert abs(rho[0, 0] - 0.5) < 1e-4
        assert abs(rho[1, 1] - 0.5) < 1e-4
        assert abs(rho[0, 1]) < 1e-4
        assert abs(rho[1, 0]) < 1e-4

    def test_hermitian(self):
        sim = QrackSimulator(qubitCount=3)
        sim.h(0)
        sim.cnot(0, 1)
        sim.h(2)
        rho = sim.get_reduced_density_matrix([0, 2])
        assert np.allclose(rho, rho.conj().T, atol=1e-5)

    def test_invalid_qubit_raises(self):
        sim = QrackSimulator(qubitCount=2)
        with pytest.raises(Exception):
            sim.get_reduced_density_matrix([0, 5])
