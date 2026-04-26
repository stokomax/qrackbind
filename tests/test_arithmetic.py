"""Phase 2 — arithmetic, shift/rotate, modular, and QFT tests.

Arithmetic gates are runtime-guarded against ``isTensorNetwork=True``;
all tests here construct simulators with ``isTensorNetwork=False``.
"""

import pytest

from qrackbind import QrackSimulator

ABS = 1e-4



# ── add / sub ────────────────────────────────────────────────────────────────


def test_add_to_zero_register(arith_sim):
    sim = arith_sim(4)
    sim.add(3, 0, 4)  # |0> + 3 = |3>
    assert int(sim.m_reg(0, 4)) == 3


def test_add_then_sub_round_trip(arith_sim):
    sim = arith_sim(5)
    sim.add(7, 0, 5)
    sim.sub(7, 0, 5)
    assert int(sim.m_reg(0, 5)) == 0


def test_sub_decrements(arith_sim):
    sim = arith_sim(4)
    sim.add(10, 0, 4)
    sim.sub(3, 0, 4)
    assert int(sim.m_reg(0, 4)) == 7


def test_arithmetic_raises_with_tensor_network(arith_sim):
    sim = QrackSimulator(qubitCount=4, isTensorNetwork=True)
    with pytest.raises(Exception, match="isTensorNetwork"):
        sim.add(1, 0, 4)


def test_sub_raises_with_tensor_network(arith_sim):
    sim = QrackSimulator(qubitCount=4, isTensorNetwork=True)
    with pytest.raises(Exception, match="isTensorNetwork"):
        sim.sub(1, 0, 4)


# ── modular multiply / divide ────────────────────────────────────────────────


def test_mul_round_trip_with_div(arith_sim):
    # in * 3 mod 7  then  div by 3 mod 7  should restore the input register.
    sim = arith_sim(8)  # 4 in-bits, 4 out-bits
    sim.set_permutation(2)  # in = 2
    sim.mul(3, 7, 0, 4, 4)
    sim.div(3, 7, 0, 4, 4)
    # Output register should be cleared back to zero after exact inverse
    assert int(sim.m_reg(4, 4)) == 0


def test_mul_basic_smoke(arith_sim):
    sim = arith_sim(8)
    sim.set_permutation(3)
    sim.mul(2, 11, 0, 4, 4)  # out = 3*2 mod 11 = 6
    assert int(sim.m_reg(4, 4)) == 6


# ── pown (modular exponentiation, QAlu-routed) ──────────────────────────────


def test_pown_smoke_executes_on_qalu_capable_stack(arith_sim):
    # Just verify pown reaches POWModNOut without raising and produces
    # the correct value for a known classical input.
    sim = arith_sim(8)
    sim.set_permutation(2)  # in = 2
    sim.pown(3, 11, 0, 4, 4)  # out = 3^2 mod 11 = 9
    assert int(sim.m_reg(4, 4)) == 9


def test_pown_raises_with_tensor_network(arith_sim):
    sim = QrackSimulator(qubitCount=8, isTensorNetwork=True)
    with pytest.raises(Exception, match="isTensorNetwork"):
        sim.pown(3, 11, 0, 4, 4)


def test_mcpown_with_inactive_control_is_identity(arith_sim):
    sim = arith_sim(9)  # 1 control + 4 in + 4 out
    sim.set_permutation(2 << 1)  # control=0, in=2
    sim.mcpown(3, 11, 1, 5, 4, [0])
    # control is |0>, so out should remain 0
    assert int(sim.m_reg(5, 4)) == 0


@pytest.mark.xfail(
    reason="Qrack CPOWModNOut on the QUnit→QPager→QHybrid stack collapses "
    "the input register and produces 0. POWModNOut on the same stack works. "
    "Tracked as a Qrack-side issue; the inactive-control case passes "
    "(see test_mcpown_with_inactive_control_is_identity).",
    strict=True,
)
def test_mcpown_with_active_control_applies(arith_sim):
    sim = arith_sim(9)
    # control=1, in=2  -> permutation bits: bit0=1, bits1..4=0010
    sim.set_permutation((2 << 1) | 1)
    sim.mcpown(3, 11, 1, 5, 4, [0])
    # 3^2 mod 11 = 9
    assert int(sim.m_reg(5, 4)) == 9


# ── controlled mul / div ────────────────────────────────────────────────────


def test_mcmul_with_inactive_control(arith_sim):
    sim = arith_sim(9)
    sim.set_permutation(3 << 1)  # control=0, in=3
    sim.mcmul(2, 11, 1, 5, 4, [0])
    assert int(sim.m_reg(5, 4)) == 0


def test_mcmul_with_active_control(arith_sim):
    sim = arith_sim(9)
    sim.set_permutation((3 << 1) | 1)  # control=1, in=3
    sim.mcmul(2, 11, 1, 5, 4, [0])
    # out = 3 * 2 mod 11 = 6
    assert int(sim.m_reg(5, 4)) == 6


def test_mcdiv_inverts_mcmul(arith_sim):
    sim = arith_sim(9)
    sim.set_permutation((3 << 1) | 1)
    sim.mcmul(2, 11, 1, 5, 4, [0])
    sim.mcdiv(2, 11, 1, 5, 4, [0])
    assert int(sim.m_reg(5, 4)) == 0


# ── shift / rotate ──────────────────────────────────────────────────────────


def test_lsl_shifts_left(arith_sim):
    sim = arith_sim(4)
    sim.x(0)  # value 1
    sim.lsl(1, 0, 4)  # → 2
    assert int(sim.m_reg(0, 4)) == 2


def test_lsr_shifts_right(arith_sim):
    sim = arith_sim(4)
    sim.set_permutation(8)  # 0b1000
    sim.lsr(2, 0, 4)
    assert int(sim.m_reg(0, 4)) == 2


def test_rol_circular_left(arith_sim):
    sim = arith_sim(4)
    sim.set_permutation(8)  # 0b1000 — MSB
    sim.rol(1, 0, 4)  # MSB wraps to bit 0
    assert int(sim.m_reg(0, 4)) == 1


def test_ror_circular_right(arith_sim):
    sim = arith_sim(4)
    sim.x(0)  # 0b0001
    sim.ror(1, 0, 4)  # bit 0 wraps to MSB
    assert int(sim.m_reg(0, 4)) == 8


def test_rol_then_ror_round_trip(arith_sim):
    sim = arith_sim(4)
    sim.set_permutation(0b1011)
    sim.rol(2, 0, 4)
    sim.ror(2, 0, 4)
    assert int(sim.m_reg(0, 4)) == 0b1011


# ── QFT ────────────────────────────────────────────────────────────────────


def test_qft_iqft_round_trip_basis_state(arith_sim):
    sim = QrackSimulator(qubitCount=3, isTensorNetwork=False)
    sim.x(0)  # |001>
    sim.qft(0, 3)
    sim.iqft(0, 3)
    assert sim.prob(0) == pytest.approx(1.0, abs=ABS)
    assert sim.prob(1) == pytest.approx(0.0, abs=ABS)
    assert sim.prob(2) == pytest.approx(0.0, abs=ABS)


def test_qft_iqft_round_trip_arbitrary_state(arith_sim):
    sim = QrackSimulator(qubitCount=3, isTensorNetwork=False)
    sim.x(0)
    sim.x(2)  # |101>
    sim.qft(0, 3)
    sim.iqft(0, 3)
    assert sim.prob(0) == pytest.approx(1.0, abs=ABS)
    assert sim.prob(1) == pytest.approx(0.0, abs=ABS)
    assert sim.prob(2) == pytest.approx(1.0, abs=ABS)


def test_qftr_iqftr_round_trip(arith_sim):
    sim = QrackSimulator(qubitCount=3, isTensorNetwork=False)
    sim.x(1)
    sim.qftr([0, 1, 2])
    sim.iqftr([0, 1, 2])
    assert sim.prob(0) == pytest.approx(0.0, abs=ABS)
    assert sim.prob(1) == pytest.approx(1.0, abs=ABS)
    assert sim.prob(2) == pytest.approx(0.0, abs=ABS)


def test_qft_on_zero_state_creates_uniform_superposition(arith_sim):
    sim = QrackSimulator(qubitCount=2, isTensorNetwork=False)
    sim.qft(0, 2)
    # QFT of |00> is uniform superposition: each qubit in equal superposition
    assert sim.prob(0) == pytest.approx(0.5, abs=ABS)
    assert sim.prob(1) == pytest.approx(0.5, abs=ABS)
