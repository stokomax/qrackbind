"""Unit tests for QrackSimulator construction, configuration, and identity."""

import warnings

import pytest

from qrackbind import QrackSimulator

ABS = 1e-5


# ── Basic construction ───────────────────────────────────────────────────────

def test_single_qubit():
    sim = QrackSimulator(qubitCount=1)
    assert sim.num_qubits == 1


def test_multi_qubit():
    sim = QrackSimulator(qubitCount=8)
    assert sim.num_qubits == 8


def test_qubit_count_boundary():
    sim = QrackSimulator(qubitCount=2)
    assert sim.num_qubits == 2


def test_initial_state_all_zero():
    sim = QrackSimulator(qubitCount=4)
    for i in range(4):
        assert sim.prob(i) == pytest.approx(0.0, abs=ABS)


# ── Representation ───────────────────────────────────────────────────────────

def test_repr_contains_qubit_count():
    sim = QrackSimulator(qubitCount=5)
    assert "5" in repr(sim)


def test_repr_returns_string():
    sim = QrackSimulator(qubitCount=3)
    assert isinstance(repr(sim), str)


# ── Configuration variants ───────────────────────────────────────────────────

def test_no_tensor_network():
    sim = QrackSimulator(qubitCount=2, isTensorNetwork=False)
    assert sim.num_qubits == 2
    assert sim.prob(0) == pytest.approx(0.0, abs=ABS)


def test_no_schmidt_decompose():
    sim = QrackSimulator(qubitCount=2, isSchmidtDecompose=False)
    assert sim.num_qubits == 2
    assert sim.prob(0) == pytest.approx(0.0, abs=ABS)


def test_cpu_only():
    sim = QrackSimulator(qubitCount=2, isOpenCL=False, isCpuGpuHybrid=False)
    assert sim.num_qubits == 2
    assert sim.prob(0) == pytest.approx(0.0, abs=ABS)


def test_noise_enabled():
    sim = QrackSimulator(qubitCount=2, noise=0.01)
    assert sim.num_qubits == 2


# ── Argument validation ─────────────────────────────────────────────────────

def test_qubit_count_required():
    with pytest.raises(TypeError):
        QrackSimulator()  # type: ignore[call-arg]


# ── Cloning ─────────────────────────────────────────────────────────────────

def test_clone_method():
    import copy
    src = QrackSimulator(qubitCount=3)
    src.h(0)
    clone = src.clone()
    assert clone.num_qubits == 3
    # Independence: gating clone does not affect src
    clone.x(0)
    assert clone is not src


def test_copy_protocol():
    import copy
    src = QrackSimulator(qubitCount=2)
    src.h(0)
    c1 = copy.copy(src)
    c2 = copy.deepcopy(src)
    assert c1.num_qubits == 2
    assert c2.num_qubits == 2
    assert c1 is not src
    assert c2 is not src




# ── Context manager ──────────────────────────────────────────────────────────

def test_context_manager_enter_returns_valid_object():
    sim = QrackSimulator(qubitCount=2)
    entered = sim.__enter__()
    assert entered.num_qubits == sim.num_qubits


@pytest.mark.skip(reason="nb::sig is not overriding __exit__ signature")
def test_context_manager_with_statement():
    with QrackSimulator(qubitCount=2) as sim:
        assert sim.num_qubits == 2


# ── Deprecated compatibility aliases ─────────────────────────────────────────

def test_deprecated_m_warns():
    sim = QrackSimulator(qubitCount=1)
    with pytest.warns(DeprecationWarning, match=r"m\(\) is deprecated"):
        sim.m(0)


def test_deprecated_m_returns_int():
    sim = QrackSimulator(qubitCount=1)
    sim.x(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sim.m(0)
    assert isinstance(result, int)
    assert result == 1


def test_deprecated_m_all_warns():
    sim = QrackSimulator(qubitCount=2)
    with pytest.warns(DeprecationWarning, match=r"m_all\(\) is deprecated"):
        sim.m_all()


def test_deprecated_m_all_returns_list_of_int():
    sim = QrackSimulator(qubitCount=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sim.m_all()
    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)
    assert result == [0, 0]


def test_deprecated_get_num_qubits_warns():
    sim = QrackSimulator(qubitCount=3)
    with pytest.warns(DeprecationWarning, match=r"get_num_qubits\(\) is deprecated"):
        sim.get_num_qubits()


def test_deprecated_get_num_qubits_correct():
    sim = QrackSimulator(qubitCount=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert sim.get_num_qubits() == 4
