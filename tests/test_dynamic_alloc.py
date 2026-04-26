"""Phase 2 — dynamic qubit allocation and disposal.

`allocate` / `dispose` / `allocate_qubits` are required for Bloqade's
``DynamicMemorySimulator``. They are incompatible with
``isTensorNetwork=True``; all tests construct simulators with
``isTensorNetwork=False``.
"""

import pytest

from qrackbind import QrackSimulator

ABS = 1e-5



# ── allocate_qubits (append at end) ──────────────────────────────────────────


def test_allocate_qubits_increases_count(dyn_sim):
    sim = dyn_sim(2)
    sim.allocate_qubits(3)
    assert sim.num_qubits == 5


def test_allocate_qubits_returns_first_new_index(dyn_sim):
    sim = dyn_sim(2)
    idx = sim.allocate_qubits(2)
    # Newly appended qubits start at index 2 (after the existing 0, 1)
    assert idx >= 2


def test_allocate_qubits_initializes_to_zero(dyn_sim):
    sim = dyn_sim(1)
    sim.allocate_qubits(1)
    # New qubit is the last index — should be in |0>
    assert sim.prob(sim.num_qubits - 1) == pytest.approx(0.0, abs=ABS)


def test_allocate_qubits_preserves_existing_state(dyn_sim):
    sim = dyn_sim(2)
    sim.x(0)
    sim.allocate_qubits(2)
    assert sim.num_qubits == 4
    assert sim.prob(0) == pytest.approx(1.0, abs=ABS)
    assert sim.prob(1) == pytest.approx(0.0, abs=ABS)


# ── allocate at explicit index ───────────────────────────────────────────────


def test_allocate_at_start(dyn_sim):
    sim = dyn_sim(2)
    sim.allocate(0, 1)
    assert sim.num_qubits == 3


def test_allocate_at_end(dyn_sim):
    sim = dyn_sim(2)
    sim.allocate(2, 1)
    assert sim.num_qubits == 3


# ── dispose ─────────────────────────────────────────────────────────────────


def test_dispose_decreases_count(dyn_sim):
    sim = dyn_sim(3)
    sim.dispose(2, 1)  # last qubit is |0>, safe to dispose
    assert sim.num_qubits == 2


def test_dispose_multiple_qubits(dyn_sim):
    sim = dyn_sim(5)
    sim.dispose(2, 3)  # remove last 3 (all |0>)
    assert sim.num_qubits == 2


def test_allocate_then_dispose_round_trip(dyn_sim):
    sim = dyn_sim(2)
    sim.allocate_qubits(2)
    assert sim.num_qubits == 4
    sim.dispose(2, 2)
    assert sim.num_qubits == 2


# ── interaction with tensor-network ──────────────────────────────────────────


def test_allocate_with_tensor_network_succeeds():
    # Qrack supports Allocate() on the tensor-network stack; only the
    # arithmetic operators are guarded against isTensorNetwork=True.
    sim = QrackSimulator(qubitCount=2, isTensorNetwork=True)
    sim.allocate_qubits(1)
    assert sim.num_qubits == 3
