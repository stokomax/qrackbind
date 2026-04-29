"""QrackStabilizerDevice and QrackStabilizerHybridDevice — PennyLane devices
backed by QrackStabilizer and QrackStabilizerHybrid respectively.

Entry points:
    ``qrackbind.stabilizer``       → QrackStabilizerDevice
    ``qrackbind.stabilizer_hybrid`` → QrackStabilizerHybridDevice
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pennylane as qml
from pennylane.devices import Device, ExecutionConfig
from pennylane.tape import QuantumScript, QuantumScriptOrBatch
from pennylane.transforms import decompose
from pennylane.transforms.core.compile_pipeline import CompilePipeline

from qrackbind import QrackStabilizer, QrackStabilizerHybrid, Pauli
from qrackbind.pennylane._dispatch import (
    dispatch_gate,
    dispatch_clifford_gate,
    GATE_DISPATCH,
    CLIFFORD_GATE_DISPATCH,
)

_STABILIZER_TOML_PATH = pathlib.Path(__file__).parent / "qrack_stabilizer.toml"
_HYBRID_TOML_PATH = pathlib.Path(__file__).parent / "qrack.toml"


def _remap_wires(op, wire_map: dict):
    """Return a copy of an operation with wires remapped to integer indices."""
    old_wires = list(op.wires)
    new_wires = [wire_map[w] for w in old_wires]
    return op.map_wires(dict(zip(old_wires, new_wires)))


# ── QrackStabilizerDevice ─────────────────────────────────────────────────────


def _clifford_stopping_condition(op) -> bool:
    """True if an operation is natively supported by QrackStabilizer."""
    return op.name in CLIFFORD_GATE_DISPATCH


class QrackStabilizerDevice(Device):
    """PennyLane device backed by the pure Clifford stabilizer (QrackStabilizer).

    Uses polynomial memory in the qubit count (stabilizer tableau representation).
    Only Clifford gates are supported natively; non-Clifford operations (RX, RY,
    RZ, T, U, QubitUnitary, …) cause PennyLane to raise a decomposition error.

    Measurements are limited to expectation values and variance of Pauli
    observables, plus shot-based sampling. StateMP and ProbabilityMP are NOT
    supported — the stabilizer engine stores a tableau, not amplitudes.

    Usage::

        import pennylane as qml
        dev = qml.device("qrackbind.stabilizer", wires=2)

        @qml.qnode(dev)
        def bell_zz():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        print(bell_zz())  # ≈ 1.0
    """

    config_filepath = str(_STABILIZER_TOML_PATH)
    pennylane_requires = ">=0.43"

    def __init__(self, wires=None, shots=None):
        super().__init__(wires=wires, shots=shots)

    def _make_simulator(self, num_qubits: int) -> QrackStabilizer:
        return QrackStabilizer(qubitCount=num_qubits)

    # ── PennyLane Device API ──────────────────────────────────────────────

    def preprocess(
        self,
        execution_config: ExecutionConfig | None = None,
    ) -> tuple[CompilePipeline, ExecutionConfig]:
        config = execution_config or ExecutionConfig()
        pipeline = CompilePipeline()
        pipeline.add_transform(
            decompose,
            stopping_condition=_clifford_stopping_condition,
            max_expansion=10,
        )
        return pipeline, config

    def execute(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig | None = None,
    ) -> Any:
        if isinstance(circuits, QuantumScript):
            circuits = [circuits]
        return tuple(self._execute_one(circuit) for circuit in circuits)

    def _execute_one(self, circuit: QuantumScript):
        num_qubits = len(self.wires)
        wire_map = {wire: idx for idx, wire in enumerate(self.wires)}
        sim = self._make_simulator(num_qubits)

        for op in circuit.operations:
            remapped = _remap_wires(op, wire_map)
            dispatch_clifford_gate(sim, remapped)

        results = []
        for m in circuit.measurements:
            results.append(self._evaluate_measurement(sim, m, wire_map, circuit))

        return results[0] if len(results) == 1 else tuple(results)

    def _evaluate_measurement(
        self,
        sim: QrackStabilizer,
        m,
        wire_map: dict,
        circuit: QuantumScript,
    ) -> Any:
        import pennylane.measurements as meas

        if isinstance(m, meas.ExpectationMP):
            return self._expval(sim, m.obs, wire_map)
        elif isinstance(m, meas.VarianceMP):
            return self._variance(sim, m.obs, wire_map)
        elif isinstance(m, meas.SampleMP):
            shots_val = circuit.shots.total_shots if hasattr(circuit.shots, 'total_shots') else (
                circuit.shots.value if hasattr(circuit.shots, 'value') else circuit.shots
            )
            return self._sample(sim, m, wire_map, shots_val)
        elif isinstance(m, meas.CountsMP):
            shots_val = circuit.shots.total_shots if hasattr(circuit.shots, 'total_shots') else (
                circuit.shots.value if hasattr(circuit.shots, 'value') else circuit.shots
            )
            samples = self._sample(sim, m, wire_map, shots_val)
            unique, counts = np.unique(samples, axis=0, return_counts=True)
            return dict(zip(
                ["".join(str(b) for b in row) for row in unique],
                counts.tolist(),
            ))
        elif isinstance(m, meas.StateMP):
            raise NotImplementedError(
                "QrackStabilizerDevice does not support qml.state() — the pure "
                "stabilizer engine stores a tableau, not a state vector. "
                "Use qrackbind.stabilizer_hybrid or qrackbind.simulator instead.")
        elif isinstance(m, meas.ProbabilityMP):
            raise NotImplementedError(
                "QrackStabilizerDevice does not support qml.probs() — the pure "
                "stabilizer engine stores a tableau, not a state vector. "
                "Use qrackbind.stabilizer_hybrid or qrackbind.simulator instead.")

        raise NotImplementedError(
            f"QrackStabilizerDevice: measurement type {type(m).__name__} not supported.")

    def _expval(self, sim: QrackStabilizer, obs, wire_map: dict) -> float:
        """Expectation value — Pauli observables only."""
        paulis, qubits = self._observable_to_paulis(obs, wire_map)
        if paulis:
            return float(sim.exp_val_pauli(paulis, qubits))
        raise NotImplementedError(
            f"QrackStabilizerDevice: observable '{obs.name}' is not supported. "
            f"Only Pauli (X, Y, Z, I) and their tensor products are available "
            f"on the pure stabilizer backend.")

    def _variance(self, sim: QrackStabilizer, obs, wire_map: dict) -> float:
        """Variance — Pauli observables only."""
        paulis, qubits = self._observable_to_paulis(obs, wire_map)
        if paulis:
            return float(sim.variance_pauli(paulis, qubits))
        raise NotImplementedError(
            f"QrackStabilizerDevice: observable '{obs.name}' is not supported. "
            f"Only Pauli (X, Y, Z, I) and their tensor products are available "
            f"on the pure stabilizer backend.")

    def _observable_to_paulis(self, obs, wire_map: dict):
        """Convert a PennyLane observable to (list[Pauli], list[int]) if possible."""
        pauli_map = {
            "PauliX": Pauli.PauliX,
            "PauliY": Pauli.PauliY,
            "PauliZ": Pauli.PauliZ,
            "Identity": Pauli.PauliI,
        }
        if obs.name in pauli_map:
            return [pauli_map[obs.name]], [wire_map[obs.wires[0]]]

        if obs.name == "Prod":
            paulis, qubits = [], []
            for factor in obs.operands:
                if factor.name not in pauli_map:
                    return [], []
                paulis.append(pauli_map[factor.name])
                qubits.append(wire_map[factor.wires[0]])
            return paulis, qubits

        if obs.name == "SProd":
            base = obs.obs
            if base.name in pauli_map:
                return [pauli_map[base.name]], [wire_map[base.wires[0]]]
            return [], []

        return [], []

    def _sample(
        self, sim: QrackStabilizer, m, wire_map: dict, shots: int | None
    ) -> np.ndarray:
        """Generate shot samples via measure_shots.

        When ``m.obs`` is set (e.g. ``qml.sample(qml.PauliZ(0))``), returns a
        1-D array of shape ``(shots,)`` containing eigenvalues.  When ``m.obs``
        is None (raw ``qml.sample()``), returns a 2-D bit array of shape
        ``(shots, num_wires)``.
        """
        wires = [wire_map[w] for w in (m.wires if m.wires else self.wires)]
        if shots is None:
            shots = 1000

        results = sim.measure_shots(wires, shots)
        samples = []
        num_wires = len(wires)
        for outcome, count in sorted(results.items()):
            bits = [
                (outcome >> (num_wires - 1 - i)) & 1
                for i in range(num_wires)
            ]
            samples.extend([bits] * count)

        arr = np.array(samples, dtype=int)  # shape (shots, num_wires)

        # Observable-based sampling: map bit-string index → eigenvalue.
        if getattr(m, "obs", None) is not None:
            eigvals = np.array(m.obs.eigvals(), dtype=float)
            # Convert each row of bits to an integer index (big-endian).
            bit_indices = np.zeros(len(arr), dtype=int)
            for col in range(num_wires):
                bit_indices = bit_indices * 2 + arr[:, col]
            return eigvals[bit_indices]

        return arr

    def supports_derivatives(
        self,
        execution_config: ExecutionConfig | None = None,
        circuit: QuantumScript | None = None,
    ) -> bool:
        """No gradient support — no state vector on pure stabilizer."""
        return False


# ── QrackStabilizerHybridDevice ───────────────────────────────────────────────


def _hybrid_stopping_condition(op) -> bool:
    """True if an operation is natively supported by QrackStabilizerHybrid."""
    return op.name in GATE_DISPATCH


class QrackStabilizerHybridDevice(Device):
    """PennyLane device backed by the stabilizer-hybrid simulator.

    Stays in polynomial-memory Clifford (stabilizer tableau) mode for as long
    as the circuit contains only Clifford gates. Automatically falls back to
    full dense simulation on the first non-Clifford gate (T, RX, RY, RZ, U, …).

    Supports the same gate set and measurements as ``qrackbind.simulator``.

    Usage::

        import pennylane as qml
        dev = qml.device("qrackbind.stabilizer_hybrid", wires=2)

        @qml.qnode(dev)
        def circuit(theta):
            qml.Hadamard(wires=0)
            qml.RY(theta, wires=1)   # triggers dense fallback
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

    Constructor keyword arguments are forwarded to QrackStabilizerHybrid:
    ``isCpuGpuHybrid``, ``isOpenCL``, ``isHostPointer``, ``isSparse``.
    """

    config_filepath = str(_HYBRID_TOML_PATH)
    pennylane_requires = ">=0.43"

    def __init__(self, wires=None, shots=None, **simulator_kwargs):
        super().__init__(wires=wires, shots=shots)
        self._simulator_kwargs = simulator_kwargs

    def _make_simulator(self, num_qubits: int) -> QrackStabilizerHybrid:
        return QrackStabilizerHybrid(qubitCount=num_qubits, **self._simulator_kwargs)

    # ── PennyLane Device API ──────────────────────────────────────────────

    def preprocess(
        self,
        execution_config: ExecutionConfig | None = None,
    ) -> tuple[CompilePipeline, ExecutionConfig]:
        config = execution_config or ExecutionConfig()
        pipeline = CompilePipeline()
        pipeline.add_transform(
            decompose,
            stopping_condition=_hybrid_stopping_condition,
            max_expansion=10,
        )
        return pipeline, config

    def execute(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig | None = None,
    ) -> Any:
        if isinstance(circuits, QuantumScript):
            circuits = [circuits]
        return tuple(self._execute_one(circuit) for circuit in circuits)

    def _execute_one(self, circuit: QuantumScript):
        num_qubits = len(self.wires)
        wire_map = {wire: idx for idx, wire in enumerate(self.wires)}
        sim = self._make_simulator(num_qubits)

        for op in circuit.operations:
            remapped = _remap_wires(op, wire_map)
            dispatch_gate(sim, remapped)

        results = []
        for m in circuit.measurements:
            results.append(self._evaluate_measurement(sim, m, wire_map, circuit))

        return results[0] if len(results) == 1 else tuple(results)

    def _evaluate_measurement(
        self,
        sim: QrackStabilizerHybrid,
        m,
        wire_map: dict,
        circuit: QuantumScript,
    ) -> Any:
        import pennylane.measurements as meas

        if isinstance(m, meas.ExpectationMP):
            return self._expval(sim, m.obs, wire_map)
        elif isinstance(m, meas.VarianceMP):
            return self._variance(sim, m.obs, wire_map)
        elif isinstance(m, meas.ProbabilityMP):
            wires = (
                [wire_map[w] for w in m.wires]
                if m.wires
                else list(range(sim.num_qubits))
            )
            return self._probabilities(sim, wires)
        elif isinstance(m, meas.StateMP):
            return sim.state_vector.astype(np.complex128)
        elif isinstance(m, meas.SampleMP):
            shots_val = circuit.shots.total_shots if hasattr(circuit.shots, 'total_shots') else (
                circuit.shots.value if hasattr(circuit.shots, 'value') else circuit.shots
            )
            return self._sample(sim, m, wire_map, shots_val)
        elif isinstance(m, meas.CountsMP):
            shots_val = circuit.shots.total_shots if hasattr(circuit.shots, 'total_shots') else (
                circuit.shots.value if hasattr(circuit.shots, 'value') else circuit.shots
            )
            samples = self._sample(sim, m, wire_map, shots_val)
            unique, counts = np.unique(samples, axis=0, return_counts=True)
            return dict(zip(
                ["".join(str(b) for b in row) for row in unique],
                counts.tolist(),
            ))

        raise NotImplementedError(
            f"QrackStabilizerHybridDevice: measurement type {type(m).__name__} "
            f"not supported.")

    def _expval(self, sim: QrackStabilizerHybrid, obs, wire_map: dict) -> float:
        paulis, qubits = self._observable_to_paulis(obs, wire_map)
        if paulis:
            return float(sim.exp_val_pauli(paulis, qubits))
        return float(self._matrix_expval(sim, obs, wire_map))

    def _variance(self, sim: QrackStabilizerHybrid, obs, wire_map: dict) -> float:
        paulis, qubits = self._observable_to_paulis(obs, wire_map)
        if paulis:
            return float(sim.variance_pauli(paulis, qubits))
        ev = self._expval(sim, obs, wire_map)
        obs_sq = qml.prod(obs, obs)
        ev2 = self._matrix_expval(sim, obs_sq, wire_map)
        return float(ev2 - ev ** 2)

    def _observable_to_paulis(self, obs, wire_map: dict):
        pauli_map = {
            "PauliX": Pauli.PauliX,
            "PauliY": Pauli.PauliY,
            "PauliZ": Pauli.PauliZ,
            "Identity": Pauli.PauliI,
        }
        if obs.name in pauli_map:
            return [pauli_map[obs.name]], [wire_map[obs.wires[0]]]

        if obs.name == "Prod":
            paulis, qubits = [], []
            for factor in obs.operands:
                if factor.name not in pauli_map:
                    return [], []
                paulis.append(pauli_map[factor.name])
                qubits.append(wire_map[factor.wires[0]])
            return paulis, qubits

        if obs.name == "SProd":
            base = obs.obs
            if base.name in pauli_map:
                return [pauli_map[base.name]], [wire_map[base.wires[0]]]
            return [], []

        if obs.name in ("Sum", "Hamiltonian"):
            return [], []

        return [], []

    def _probabilities(
        self, sim: QrackStabilizerHybrid, qubits: list[int]
    ) -> np.ndarray:
        """Marginal probabilities for the given qubits."""
        full = list(sim.probabilities)
        n = sim.num_qubits
        k = len(qubits)
        out_size = 1 << k
        out = np.zeros(out_size, dtype=np.float64)
        for basis in range(1 << n):
            idx = 0
            for i, q in enumerate(reversed(qubits)):
                idx |= ((basis >> q) & 1) << i
            out[idx] += float(full[basis])
        return out

    def _sample(
        self, sim: QrackStabilizerHybrid, m, wire_map: dict, shots: int | None
    ) -> np.ndarray:
        """Generate shot samples via measure_shots.

        When ``m.obs`` is set (e.g. ``qml.sample(qml.PauliZ(0))``), returns a
        1-D array of shape ``(shots,)`` containing eigenvalues.  When ``m.obs``
        is None (raw ``qml.sample()``), returns a 2-D bit array of shape
        ``(shots, num_wires)``.
        """
        wires = [wire_map[w] for w in (m.wires if m.wires else self.wires)]
        if shots is None:
            shots = 1000

        results = sim.measure_shots(wires, shots)
        samples = []
        num_wires = len(wires)
        for outcome, count in sorted(results.items()):
            bits = [
                (outcome >> (num_wires - 1 - i)) & 1
                for i in range(num_wires)
            ]
            samples.extend([bits] * count)

        arr = np.array(samples, dtype=int)  # shape (shots, num_wires)

        # Observable-based sampling: map bit-string index → eigenvalue.
        if getattr(m, "obs", None) is not None:
            eigvals = np.array(m.obs.eigvals(), dtype=float)
            # Convert each row of bits to an integer index (big-endian).
            bit_indices = np.zeros(len(arr), dtype=int)
            for col in range(num_wires):
                bit_indices = bit_indices * 2 + arr[:, col]
            return eigvals[bit_indices]

        return arr

    def _matrix_expval(
        self, sim: QrackStabilizerHybrid, obs, wire_map: dict
    ) -> float:
        """Expectation value via state vector contraction for Hermitian observables."""
        sv = sim.state_vector.astype(np.complex128)
        qwire_list = list(wire_map.keys())
        matrix = qml.matrix(obs, wire_order=qwire_list)
        rho_psi = matrix @ sv
        return float(np.real(np.dot(sv.conj(), rho_psi)))

    def supports_derivatives(
        self,
        execution_config: ExecutionConfig | None = None,
        circuit: QuantumScript | None = None,
    ) -> bool:
        """Declare parameter-shift gradient support for analytic execution."""
        if execution_config is None:
            return True
        return (
            execution_config.gradient_method == "parameter-shift"
            and self.shots is None
        )
