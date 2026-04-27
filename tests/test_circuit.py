"""Phase 6 tests — QrackCircuit, GateType, circuit-level operations."""

import math

import pytest

from qrackbind import (
    GateType,
    QrackArgumentError,
    QrackCircuit,
    QrackQubitError,
    QrackSimulator,
)


# ── Construction ─────────────────────────────────────────────────────────────


class TestConstruction:
    def test_basic_construction(self):
        circ = QrackCircuit(3)
        assert repr(circ) == "QrackCircuit(qubits=3, gates=0)"

    def test_empty_gate_count(self):
        circ = QrackCircuit(2)
        assert circ.gate_count == 0

    def test_num_qubits_property(self):
        circ = QrackCircuit(5)
        assert circ.num_qubits == 5


# ── GateType enum ────────────────────────────────────────────────────────────


class TestGateType:
    def test_members_exist(self):
        assert hasattr(GateType, "H")
        assert hasattr(GateType, "X")
        assert hasattr(GateType, "CNOT")
        assert hasattr(GateType, "RZ")
        assert hasattr(GateType, "Mtrx")

    def test_gatetype_is_enum(self):
        assert isinstance(GateType.H, GateType)


# ── append_gate ──────────────────────────────────────────────────────────────


class TestAppendGate:
    def test_append_h(self):
        circ = QrackCircuit(2)
        circ.append_gate(GateType.H, [0])
        assert circ.gate_count == 1

    def test_append_multiple(self):
        circ = QrackCircuit(2)
        circ.append_gate(GateType.H, [0])
        circ.append_gate(GateType.CNOT, [0, 1])
        assert circ.gate_count == 2

    def test_append_rotation(self):
        circ = QrackCircuit(1)
        circ.append_gate(GateType.RZ, [0], [math.pi / 2])
        assert circ.gate_count == 1

    def test_append_x(self):
        circ = QrackCircuit(1)
        circ.append_gate(GateType.X, [0])
        assert circ.gate_count == 1

    def test_append_z(self):
        circ = QrackCircuit(1)
        circ.append_gate(GateType.Z, [0])
        assert circ.gate_count == 1

    def test_qubit_out_of_range_raises(self):
        circ = QrackCircuit(2)
        with pytest.raises(QrackQubitError):
            circ.append_gate(GateType.H, [5])

    def test_missing_params_raises(self):
        circ = QrackCircuit(1)
        with pytest.raises(Exception):
            circ.append_gate(GateType.RZ, [0])  # no params

    def test_cnot_requires_2_qubits(self):
        circ = QrackCircuit(2)
        with pytest.raises(Exception):
            circ.append_gate(GateType.CNOT, [0])


# ── run ──────────────────────────────────────────────────────────────────────


class TestRun:
    def test_h_circuit_creates_superposition(self):
        circ = QrackCircuit(1)
        circ.append_gate(GateType.H, [0])
        sim = QrackSimulator(qubitCount=1)
        circ.run(sim)
        assert sim.prob(0) == pytest.approx(0.5, abs=1e-4)

    def test_bell_state_circuit(self):
        circ = QrackCircuit(2)
        circ.append_gate(GateType.H, [0])
        circ.append_gate(GateType.CNOT, [0, 1])
        sim = QrackSimulator(qubitCount=2)
        circ.run(sim)
        assert sim.prob(0) == pytest.approx(0.5, abs=1e-4)
        assert sim.prob(1) == pytest.approx(0.5, abs=1e-4)

    def test_x_flips_to_one(self):
        circ = QrackCircuit(1)
        circ.append_gate(GateType.X, [0])
        sim = QrackSimulator(qubitCount=1)
        circ.run(sim)
        assert sim.prob(0) == pytest.approx(1.0, abs=1e-5)

    def test_run_multiple_times(self):
        circ = QrackCircuit(1)
        circ.append_gate(GateType.X, [0])
        for _ in range(5):
            sim = QrackSimulator(qubitCount=1)
            circ.run(sim)
            assert sim.prob(0) == pytest.approx(1.0, abs=1e-5)

    def test_run_on_larger_simulator(self):
        circ = QrackCircuit(1)
        circ.append_gate(GateType.X, [0])
        sim = QrackSimulator(qubitCount=4)
        circ.run(sim)
        assert sim.prob(0) == pytest.approx(1.0, abs=1e-5)

    def test_run_on_too_small_simulator_raises(self):
        circ = QrackCircuit(4)
        sim = QrackSimulator(qubitCount=2)
        with pytest.raises(QrackArgumentError):
            circ.run(sim)


# ── inverse ──────────────────────────────────────────────────────────────────


class TestInverse:
    def test_h_inverse_is_h(self):
        circ = QrackCircuit(1)
        circ.append_gate(GateType.H, [0])
        circ_inv = circ.inverse()
        sim = QrackSimulator(qubitCount=1)
        circ.run(sim)
        circ_inv.run(sim)
        assert sim.prob(0) == pytest.approx(0.0, abs=1e-4)

    def test_inverse_returns_new_circuit(self):
        circ = QrackCircuit(2)
        circ.append_gate(GateType.H, [0])
        circ_inv = circ.inverse()
        assert circ_inv is not circ

    def test_rz_inverse(self):
        circ = QrackCircuit(1)
        circ.append_gate(GateType.RZ, [0], [0.7])
        circ_inv = circ.inverse()
        sim = QrackSimulator(qubitCount=1)
        sim.h(0)
        circ.run(sim)
        circ_inv.run(sim)
        sim.h(0)
        assert sim.prob(0) == pytest.approx(0.0, abs=1e-4)

    def test_x_inverse_is_x(self):
        circ = QrackCircuit(1)
        circ.append_gate(GateType.X, [0])
        circ_inv = circ.inverse()
        sim = QrackSimulator(qubitCount=1)
        circ.run(sim)
        circ_inv.run(sim)
        assert sim.prob(0) == pytest.approx(0.0, abs=1e-4)


# ── append (combine circuits) ────────────────────────────────────────────────


class TestAppend:
    def test_append_combines_circuits(self):
        circ1 = QrackCircuit(1)
        circ1.append_gate(GateType.H, [0])
        circ2 = QrackCircuit(1)
        circ2.append_gate(GateType.X, [0])
        circ1.append(circ2)
        assert circ1.gate_count == 2

    def test_append_and_run(self):
        circ1 = QrackCircuit(1)
        circ1.append_gate(GateType.X, [0])
        circ2 = QrackCircuit(1)
        circ2.append_gate(GateType.X, [0])
        circ1.append(circ2)  # X·X = I
        sim = QrackSimulator(qubitCount=1)
        circ1.run(sim)
        assert sim.prob(0) == pytest.approx(0.0, abs=1e-4)

    def test_append_larger_circuit_raises(self):
        circ1 = QrackCircuit(2)
        circ2 = QrackCircuit(4)
        with pytest.raises(QrackArgumentError):
            circ1.append(circ2)
