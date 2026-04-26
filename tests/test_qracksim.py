import math
import pytest
from qrackbind import QrackSimulator


# ── Construction ───────────────────────────────────────────────────────────────

def test_constructs():
    sim = QrackSimulator(qubitCount=4)
    assert sim.num_qubits == 4

def test_repr():
    sim = QrackSimulator(qubitCount=3)
    assert "3" in repr(sim)

@pytest.mark.skip(reason="nb::sig is not overriding __exit__ signature")
def test_context_manager():
    with QrackSimulator(qubitCount=2) as sim:
        assert sim.num_qubits == 2


# ── Basic gates ────────────────────────────────────────────────────────────────

def test_x_flips_zero_to_one():
    sim = QrackSimulator(qubitCount=1)
    sim.x(0)
    assert sim.prob(0) == pytest.approx(1.0, abs=1e-5)

def test_double_x_is_identity():
    sim = QrackSimulator(qubitCount=1)
    sim.x(0); sim.x(0)
    assert sim.prob(0) == pytest.approx(0.0, abs=1e-5)

def test_h_creates_superposition():
    results = []
    for _ in range(200):
        s = QrackSimulator(qubitCount=1)
        s.h(0)
        results.append(s.measure(0))
    ratio = sum(results) / len(results)
    assert 0.35 < ratio < 0.65, f"H gate ratio {ratio:.2f} not near 0.5"

def test_z_no_population_change():
    sim = QrackSimulator(qubitCount=1)
    sim.z(0)
    assert sim.prob(0) == pytest.approx(0.0, abs=1e-5)

def test_rx_pi_flips():
    sim = QrackSimulator(qubitCount=1)
    sim.rx(math.pi, 0)
    assert sim.prob(0) == pytest.approx(1.0, abs=1e-4)


# ── Measurement ────────────────────────────────────────────────────────────────

def test_measure_collapses_state():
    sim = QrackSimulator(qubitCount=1)
    sim.h(0)
    result = sim.measure(0)
    after = sim.prob(0)
    expected = 1.0 if result else 0.0
    assert after == pytest.approx(expected, abs=1e-5)

def test_measure_all_length_and_type():
    sim = QrackSimulator(qubitCount=5)
    results = sim.measure_all()
    assert len(results) == 5
    assert all(isinstance(r, bool) for r in results)

def test_reset_all():
    sim = QrackSimulator(qubitCount=3)
    sim.x(0); sim.x(1); sim.x(2)
    sim.reset_all()
    for i in range(3):
        assert sim.prob(i) == pytest.approx(0.0, abs=1e-5)


# ── Error handling ─────────────────────────────────────────────────────────────

def test_out_of_range_qubit_raises():
    sim = QrackSimulator(qubitCount=3)
    with pytest.raises(Exception):
        sim.h(99)
