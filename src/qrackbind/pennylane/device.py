"""QrackDevice — PennyLane device backed by QrackSimulator."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pennylane as qml
from pennylane.devices import Device, ExecutionConfig
from pennylane.tape import QuantumScript, QuantumScriptOrBatch
from pennylane.transforms import decompose
from pennylane.transforms.core.compile_pipeline import CompilePipeline

from qrackbind import QrackSimulator, Pauli
from qrackbind.pennylane._dispatch import dispatch_gate, GATE_DISPATCH

_TOML_PATH = pathlib.Path(__file__).parent / "qrack.toml"


def _remap_wires(op, wire_map: dict):
    """Return a copy of an operation with wires remapped to integer indices."""
    old_wires = list(op.wires)
    new_wires = [wire_map[w] for w in old_wires]
    return op.map_wires(dict(zip(old_wires, new_wires)))


def _stopping_condition(op) -> bool:
    """Return True if an operation is natively supported (no decomposition needed)."""
    return op.name in GATE_DISPATCH


class QrackDevice(Device):
    """PennyLane device backed by the Qrack quantum simulator via qrackbind.

    Usage::

        import pennylane as qml
        dev = qml.device("qrackbind.simulator", wires=4)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

    Constructor keyword arguments are forwarded to QrackSimulator.
    Common options: isTensorNetwork, isOpenCL, isStabilizerHybrid.
    See QrackSimulator.__init__ for the full list.
    """

    config_filepath = str(_TOML_PATH)
    pennylane_requires = ">=0.43"

    def __init__(self, wires=None, shots=None, **simulator_kwargs):
        super().__init__(wires=wires, shots=shots)
        self._simulator_kwargs = simulator_kwargs

    def _make_simulator(self, num_qubits: int) -> QrackSimulator:
        """Create a fresh QrackSimulator for a circuit execution."""
        return QrackSimulator(qubitCount=num_qubits, **self._simulator_kwargs)

    # ── PennyLane Device API ─────────────────────────────────────────────

    def preprocess(
        self,
        execution_config: ExecutionConfig | None = None,
    ) -> tuple[CompilePipeline, ExecutionConfig]:
        """Declare supported gate set and return a compile pipeline.

        PennyLane inserts decomposition transforms for any operation not
        listed in GATE_DISPATCH. By the time execute() is called, all
        operations in the circuit are guaranteed to be in GATE_DISPATCH.
        """
        config = execution_config or ExecutionConfig()
        pipeline = CompilePipeline()
        pipeline.add_transform(
            decompose,
            stopping_condition=_stopping_condition,
            max_expansion=10,
        )
        return pipeline, config

    def execute(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig | None = None,
    ) -> Any:
        """Execute one or more circuits and return measurement results.

        PennyLane calls this after preprocessing — all operations are
        guaranteed to be in GATE_DISPATCH.
        """
        if isinstance(circuits, QuantumScript):
            circuits = [circuits]

        return tuple(self._execute_one(circuit) for circuit in circuits)

    def _execute_one(self, circuit: QuantumScript):
        """Execute a single QuantumScript and return its measurement result."""
        # Use the actual number of qubits from the device's wire count.
        # For StateMP with no wires specified, PennyLane expects the full
        # 2^(num_device_qubits) state vector, not a reduced state.
        num_qubits = len(self.wires)
        wire_map = {wire: idx for idx, wire in enumerate(circuit.wires)}

        sim = self._make_simulator(num_qubits)

        # Apply all gate operations
        for op in circuit.operations:
            remapped = _remap_wires(op, wire_map)
            dispatch_gate(sim, remapped)

        # Evaluate measurements
        results = []
        for m in circuit.measurements:
            results.append(self._evaluate_measurement(sim, m, wire_map, circuit))

        return results[0] if len(results) == 1 else tuple(results)

    def _evaluate_measurement(
        self,
        sim: QrackSimulator,
        m,
        wire_map: dict,
        circuit: QuantumScript,
    ) -> Any:
        """Evaluate a single measurement process."""
        import pennylane.measurements as meas

        if isinstance(m, meas.ExpectationMP):
            return self._expval(sim, m.obs, wire_map)
        elif isinstance(m, meas.VarianceMP):
            return self._variance(sim, m.obs, wire_map)
        elif isinstance(m, meas.ProbabilityMP):
            wires = [wire_map[w] for w in m.wires]
            return self._probabilities(sim, wires)
        elif isinstance(m, meas.StateMP):
            return sim.state_vector.astype(np.complex128)
        elif isinstance(m, meas.SampleMP):
            shots_val = circuit.shots.value if hasattr(circuit.shots, 'value') else circuit.shots
            return self._sample(sim, m, wire_map, shots_val)
        elif isinstance(m, meas.CountsMP):
            shots_val = circuit.shots.value if hasattr(circuit.shots, 'value') else circuit.shots
            samples = self._sample(sim, m, wire_map, shots_val)
            unique, counts = np.unique(samples, axis=0, return_counts=True)
            return dict(zip(
                ["".join(str(b) for b in row) for row in unique],
                counts.tolist(),
            ))

        raise NotImplementedError(
            f"QrackDevice: measurement type {type(m).__name__} not supported.")

    def _expval(self, sim: QrackSimulator, obs, wire_map: dict) -> float:
        """Compute expectation value of an observable."""
        paulis, qubits = self._observable_to_paulis(obs, wire_map)
        if paulis:
            return float(sim.exp_val_pauli(paulis, qubits))
        # Fall back to matrix method for Hermitian observables
        return float(self._matrix_expval(sim, obs, wire_map))

    def _variance(self, sim: QrackSimulator, obs, wire_map: dict) -> float:
        """Compute variance of an observable."""
        paulis, qubits = self._observable_to_paulis(obs, wire_map)
        if paulis:
            var = sim.variance_pauli(paulis, qubits)
            return float(var)
        ev = self._expval(sim, obs, wire_map)
        obs_sq = qml.prod(obs, obs)
        ev2 = self._matrix_expval(sim, obs_sq, wire_map)
        return float(ev2 - ev ** 2)

    def _observable_to_paulis(self, obs, wire_map: dict):
        """Convert a PennyLane observable to (list[Pauli], list[int]) if possible."""
        # Map PennyLane observable names to qrackbind Pauli values
        # Qrack's Pauli enum is non-sequential: I=0, X=1, Z=2, Y=3
        pauli_map: dict[str, Pauli] = {
            "PauliX": Pauli.PauliX,
            "PauliY": Pauli.PauliY,
            "PauliZ": Pauli.PauliZ,
            "Identity": Pauli.PauliI,
        }

        if obs.name in pauli_map:
            return [pauli_map[obs.name]], [wire_map[obs.wires[0]]]

        if obs.name == "Prod":
            # Tensor product of Pauli operators
            paulis, qubits = [], []
            for factor in obs.operands:
                if factor.name not in pauli_map:
                    return [], []
                paulis.append(pauli_map[factor.name])
                qubits.append(wire_map[factor.wires[0]])
            return paulis, qubits

        if obs.name == "SProd":
            # Scalar-times-Pauli: just use the underlying observable
            base = obs.obs
            if base.name in pauli_map:
                return [pauli_map[base.name]], [wire_map[base.wires[0]]]
            return [], []

        if obs.name in ("Sum", "Hamiltonian"):
            # Mixed observable — fall back to matrix method
            return [], []

        return [], []

    def _probabilities(self, sim: QrackSimulator, qubits: list[int]) -> np.ndarray:
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
        self, sim, m, wire_map: dict, shots: int | None
    ) -> np.ndarray:
        """Generate shot samples via measure_shots."""
        wires = [wire_map[w] for w in (m.wires if m.wires else range(sim.num_qubits))]
        if shots is None:
            shots = 1000  # default shot count

        results = sim.measure_shots(wires, shots)
        samples = []
        num_wires = len(wires)
        for outcome, count in sorted(results.items()):
            bits = [
                (outcome >> (num_wires - 1 - i)) & 1
                for i in range(num_wires)
            ]
            samples.extend([bits] * count)
        return np.array(samples, dtype=int)

    def _matrix_expval(self, sim: QrackSimulator, obs, wire_map: dict) -> float:
        """Expectation value via state vector contraction for Hermitian observables."""
        sv = sim.state_vector.astype(np.complex128)
        qwire_list = list(wire_map.keys())
        matrix = qml.matrix(obs, wire_order=qwire_list)
        rho_psi = matrix @ sv
        return float(np.real(np.dot(sv.conj(), rho_psi)))

    # ── Gradient support ─────────────────────────────────────────────────

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
