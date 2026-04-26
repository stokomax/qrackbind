"""Shared pytest fixtures for the qrackbind test suite."""

import pytest

from qrackbind import QrackSimulator


@pytest.fixture
def arith_sim():
    """Factory fixture: build a simulator stack compatible with arithmetic gates.

    Arithmetic and shift gates are runtime-guarded against
    ``isTensorNetwork=True`` only. ``isSchmidtDecompose`` must stay enabled —
    QUnit is what implements QAlu (used by ``pown``/``mcpown``) and what
    propagates single-qubit gates correctly down to QPager.

    Usage::

        def test_something(arith_sim):
            sim = arith_sim(8)
            sim.add(3, 0, 4)
    """

    def _make(n: int) -> QrackSimulator:
        # isTensorNetwork=False is required for arithmetic and shift gates
        # (runtime-guarded). isSchmidtDecompose=True (QUnit) MUST stay on:
        # it implements QAlu (so pown/mcpown work), and removing it leaves
        # QPager directly atop QHybrid/CPU without the bookkeeping layer
        # that propagates single-qubit gates and probabilities correctly.
        return QrackSimulator(qubitCount=n, isTensorNetwork=False)

    return _make


@pytest.fixture
def dyn_sim():
    """Factory fixture: build a simulator that supports allocate / dispose.

    Dynamic qubit allocation is incompatible with ``isTensorNetwork=True``.
    """

    def _make(n: int) -> QrackSimulator:
        return QrackSimulator(qubitCount=n, isTensorNetwork=False)

    return _make
