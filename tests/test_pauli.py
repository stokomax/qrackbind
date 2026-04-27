"""Phase 4 tests — Pauli enum, measure_pauli, exp_val(_pauli), variance_pauli,
exp_val_all, exp_val_floats, variance_floats."""

import numpy as np
import pytest

from qrackbind import Pauli, QrackArgumentError, QrackSimulator


# ── Enum properties ──────────────────────────────────────────────────────────


class TestPauliEnum:
    def test_members_exist(self):
        assert hasattr(Pauli, "PauliI")
        assert hasattr(Pauli, "PauliX")
        assert hasattr(Pauli, "PauliY")
        assert hasattr(Pauli, "PauliZ")

    def test_integer_values(self):
        # Qrack's convention — non-sequential.
        assert int(Pauli.PauliI) == 0
        assert int(Pauli.PauliX) == 1
        assert int(Pauli.PauliZ) == 2
        assert int(Pauli.PauliY) == 3

    def test_is_arithmetic_integer_accepted(self):
        # nb::is_arithmetic() makes integer codes interchangeable
        # with Pauli members.
        sim = QrackSimulator(qubitCount=1)
        result_enum = sim.exp_val(Pauli.PauliZ, 0)
        result_int = sim.exp_val(2, 0)  # 2 == PauliZ
        assert result_enum == pytest.approx(result_int, abs=1e-5)


# ── measure_pauli ────────────────────────────────────────────────────────────


class TestMeasurePauli:
    """``measure_pauli`` returns the post-rotation computational-basis bit
    (True=|1>, False=|0>) — the same convention as :meth:`measure`.

    Mapping back to Pauli eigenvalues:

    * Pauli Z eigenstate |0> (eigenvalue +1) → False
    * Pauli Z eigenstate |1> (eigenvalue −1) → True
    * Pauli X eigenstate |+> (eigenvalue +1) → False  (H|+> = |0>)
    * Pauli X eigenstate |−> (eigenvalue −1) → True   (H|−> = |1>)
    """

    def test_z_basis_ground_state_always_zero(self):
        # |0> is Z eigenstate; measurement bit is 0 → False.
        for _ in range(20):
            sim = QrackSimulator(qubitCount=1)
            assert sim.measure_pauli(Pauli.PauliZ, 0) is False

    def test_z_basis_excited_state_always_one(self):
        for _ in range(20):
            sim = QrackSimulator(qubitCount=1)
            sim.x(0)
            assert sim.measure_pauli(Pauli.PauliZ, 0) is True

    def test_x_basis_plus_state_always_zero(self):
        # |+> rotates to |0> under H, measurement bit is 0 → False.
        for _ in range(20):
            sim = QrackSimulator(qubitCount=1)
            sim.h(0)
            assert sim.measure_pauli(Pauli.PauliX, 0) is False

    def test_x_basis_minus_state_always_one(self):
        # |-> rotates to |1> under H, measurement bit is 1 → True.
        for _ in range(20):
            sim = QrackSimulator(qubitCount=1)
            sim.x(0)
            sim.h(0)
            assert sim.measure_pauli(Pauli.PauliX, 0) is True

    def test_pauli_i_does_not_disturb_probability(self):
        sim = QrackSimulator(qubitCount=1)
        sim.h(0)
        prob_before = sim.prob(0)
        sim.measure_pauli(Pauli.PauliI, 0)
        prob_after = sim.prob(0)
        assert prob_before == pytest.approx(prob_after, abs=1e-5)

    def test_returns_bool(self):
        sim = QrackSimulator(qubitCount=1)
        result = sim.measure_pauli(Pauli.PauliZ, 0)
        assert isinstance(result, bool)

    def test_out_of_range_qubit_raises(self):
        sim = QrackSimulator(qubitCount=1)
        with pytest.raises(Exception):
            sim.measure_pauli(Pauli.PauliZ, 5)


# ── exp_val ──────────────────────────────────────────────────────────────────


class TestExpVal:
    def test_z_ground_state(self):
        sim = QrackSimulator(qubitCount=1)
        assert sim.exp_val(Pauli.PauliZ, 0) == pytest.approx(1.0, abs=1e-5)

    def test_z_excited_state(self):
        sim = QrackSimulator(qubitCount=1)
        sim.x(0)
        assert sim.exp_val(Pauli.PauliZ, 0) == pytest.approx(-1.0, abs=1e-5)

    def test_x_plus_state(self):
        sim = QrackSimulator(qubitCount=1)
        sim.h(0)
        assert sim.exp_val(Pauli.PauliX, 0) == pytest.approx(1.0, abs=1e-5)

    def test_x_minus_state(self):
        sim = QrackSimulator(qubitCount=1)
        sim.x(0)
        sim.h(0)
        assert sim.exp_val(Pauli.PauliX, 0) == pytest.approx(-1.0, abs=1e-5)

    def test_z_superposition_is_zero(self):
        sim = QrackSimulator(qubitCount=1)
        sim.h(0)
        assert sim.exp_val(Pauli.PauliZ, 0) == pytest.approx(0.0, abs=1e-4)

    def test_x_ground_state_is_zero(self):
        sim = QrackSimulator(qubitCount=1)
        assert sim.exp_val(Pauli.PauliX, 0) == pytest.approx(0.0, abs=1e-4)

    def test_identity_always_one(self):
        sim = QrackSimulator(qubitCount=1)
        sim.h(0)
        assert sim.exp_val(Pauli.PauliI, 0) == pytest.approx(1.0, abs=1e-5)

    def test_result_in_range(self):
        sim = QrackSimulator(qubitCount=1)
        sim.rx(0.7, 0)
        for basis in (Pauli.PauliX, Pauli.PauliY, Pauli.PauliZ):
            val = sim.exp_val(basis, 0)
            assert -1.0 - 1e-4 <= val <= 1.0 + 1e-4

    def test_does_not_collapse_state(self):
        sim = QrackSimulator(qubitCount=1)
        sim.h(0)
        prob_before = sim.prob(0)
        sim.exp_val(Pauli.PauliZ, 0)
        prob_after = sim.prob(0)
        assert prob_before == pytest.approx(prob_after, abs=1e-5)

    def test_out_of_range_qubit_raises(self):
        sim = QrackSimulator(qubitCount=1)
        with pytest.raises(Exception):
            sim.exp_val(Pauli.PauliZ, 5)


# ── exp_val_pauli ────────────────────────────────────────────────────────────


class TestExpValPauli:
    def test_zz_bell_state(self):
        # <Bell|ZZ|Bell> = +1 for (|00>+|11>)/√2.
        sim = QrackSimulator(qubitCount=2)
        sim.h(0)
        sim.cnot(0, 1)
        result = sim.exp_val_pauli([Pauli.PauliZ, Pauli.PauliZ], [0, 1])
        assert result == pytest.approx(1.0, abs=1e-4)

    def test_zi_bell_state_is_zero(self):
        sim = QrackSimulator(qubitCount=2)
        sim.h(0)
        sim.cnot(0, 1)
        result = sim.exp_val_pauli([Pauli.PauliZ, Pauli.PauliI], [0, 1])
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_xx_bell_state(self):
        sim = QrackSimulator(qubitCount=2)
        sim.h(0)
        sim.cnot(0, 1)
        result = sim.exp_val_pauli([Pauli.PauliX, Pauli.PauliX], [0, 1])
        assert result == pytest.approx(1.0, abs=1e-4)

    def test_mismatched_lengths_raises(self):
        sim = QrackSimulator(qubitCount=2)
        with pytest.raises(Exception):
            sim.exp_val_pauli([Pauli.PauliZ], [0, 1])

    def test_out_of_range_qubit_raises(self):
        sim = QrackSimulator(qubitCount=2)
        with pytest.raises(Exception):
            sim.exp_val_pauli([Pauli.PauliZ, Pauli.PauliZ], [0, 5])

    def test_single_qubit_matches_exp_val(self):
        sim = QrackSimulator(qubitCount=2)
        sim.rx(1.2, 0)
        direct = sim.exp_val(Pauli.PauliX, 0)
        via_list = sim.exp_val_pauli([Pauli.PauliX], [0])
        assert direct == pytest.approx(via_list, abs=1e-5)


# ── variance_pauli ───────────────────────────────────────────────────────────


class TestVariancePauli:
    def test_eigenstate_variance_is_zero(self):
        sim = QrackSimulator(qubitCount=1)
        v = sim.variance_pauli([Pauli.PauliZ], [0])
        assert v == pytest.approx(0.0, abs=1e-5)

    def test_superposition_variance_is_one(self):
        # <+|Z²|+> - <+|Z|+>² = 1 - 0 = 1.
        sim = QrackSimulator(qubitCount=1)
        sim.h(0)
        v = sim.variance_pauli([Pauli.PauliZ], [0])
        assert v == pytest.approx(1.0, abs=1e-4)

    def test_variance_in_range(self):
        sim = QrackSimulator(qubitCount=1)
        sim.rx(0.5, 0)
        v = sim.variance_pauli([Pauli.PauliZ], [0])
        assert 0.0 - 1e-4 <= v <= 1.0 + 1e-4

    def test_variance_equals_one_minus_expval_squared(self):
        # Pauli operators square to I, so Var(P) = 1 - <P>².
        sim = QrackSimulator(qubitCount=1)
        sim.rx(0.8, 0)
        sim.ry(0.3, 0)
        ev = sim.exp_val(Pauli.PauliZ, 0)
        var = sim.variance_pauli([Pauli.PauliZ], [0])
        assert var == pytest.approx(1.0 - ev * ev, abs=1e-4)

    def test_mismatched_lengths_raises(self):
        sim = QrackSimulator(qubitCount=2)
        with pytest.raises(Exception):
            sim.variance_pauli([Pauli.PauliZ], [0, 1])


# ── exp_val_all ──────────────────────────────────────────────────────────────


class TestExpValAll:
    def test_all_z_ground_state(self):
        sim = QrackSimulator(qubitCount=3)
        assert sim.exp_val_all(Pauli.PauliZ) == pytest.approx(1.0, abs=1e-5)

    def test_all_z_bell_pair_with_spectator(self):
        # |Bell> ⊗ |0>: <ZZZ> = <ZZ>·<Z> = 1·1 = 1.
        sim = QrackSimulator(qubitCount=3)
        sim.h(0)
        sim.cnot(0, 1)
        assert sim.exp_val_all(Pauli.PauliZ) == pytest.approx(1.0, abs=1e-4)


# ── exp_val_floats / variance_floats ─────────────────────────────────────────


class TestExpValFloats:
    """``weights`` is a flat list of two entries per qubit: ``weights[2*i]``
    is qubit ``i``'s eigenvalue for ``|0>`` and ``weights[2*i+1]`` for
    ``|1>``."""

    def test_ground_state_picks_zero_weight(self):
        # |0> with weights [w0=3, w1=7] → 3 (qubit is in |0>).
        sim = QrackSimulator(qubitCount=1)
        result = sim.exp_val_floats([0], [3.0, 7.0])
        assert result == pytest.approx(3.0, abs=1e-5)

    def test_excited_state_picks_one_weight(self):
        sim = QrackSimulator(qubitCount=1)
        sim.x(0)
        result = sim.exp_val_floats([0], [3.0, 7.0])
        assert result == pytest.approx(7.0, abs=1e-5)

    def test_superposition_averages_weights(self):
        # |+> has equal probability of |0> and |1>: returns (w0 + w1) / 2.
        sim = QrackSimulator(qubitCount=1)
        sim.h(0)
        result = sim.exp_val_floats([0], [4.0, 6.0])
        assert result == pytest.approx(5.0, abs=1e-4)

    def test_mismatched_lengths_raises(self):
        sim = QrackSimulator(qubitCount=2)
        # Need 2*1 = 2 weights for one qubit; supplying 1 must fail.
        with pytest.raises(Exception):
            sim.exp_val_floats([0], [1.0])

    def test_variance_floats_eigenstate_is_zero(self):
        # |0> is an eigenstate of the diagonal observable defined by the
        # weight pair, so its variance is exactly 0.
        sim = QrackSimulator(qubitCount=1)
        v = sim.variance_floats([0], [3.0, 7.0])
        assert v == pytest.approx(0.0, abs=1e-5)

    def test_variance_floats_superposition_nonzero(self):
        sim = QrackSimulator(qubitCount=1)
        sim.h(0)
        # For the diagonal observable diag(w0, w1) on |+>:
        # <O> = (w0+w1)/2,  <O²> = (w0² + w1²)/2,
        # Var = (w1 − w0)² / 4. With w0=4, w1=6: Var = 1.0.
        v = sim.variance_floats([0], [4.0, 6.0])
        assert v == pytest.approx(1.0, abs=1e-4)

    def test_variance_floats_mismatched_lengths_raises(self):
        sim = QrackSimulator(qubitCount=2)
        with pytest.raises(Exception):
            sim.variance_floats([0], [1.0])


# ── Integration with state_vector ────────────────────────────────────────────


class TestPhase4Integration:
    def test_exp_val_consistent_with_state_vector(self):
        sim = QrackSimulator(qubitCount=1)
        sim.ry(1.1, 0)
        # <ψ|Z|ψ> = |α₀|² - |α₁|².
        sv = sim.state_vector
        expected_z = float(abs(sv[0]) ** 2 - abs(sv[1]) ** 2)
        measured_z = sim.exp_val(Pauli.PauliZ, 0)
        assert measured_z == pytest.approx(expected_z, abs=1e-4)

    def test_exp_val_does_not_disturb_state_vector(self):
        sim = QrackSimulator(qubitCount=2)
        sim.h(0)
        sim.h(1)
        sv_before = sim.state_vector.copy()
        sim.exp_val_pauli([Pauli.PauliZ, Pauli.PauliX], [0, 1])
        sv_after = sim.state_vector
        assert np.allclose(np.abs(sv_before), np.abs(sv_after), atol=1e-5)


# ── Deferred Phase 4: exp_val_unitary, variance_unitary, exp_val_bits_factorized ──


class TestExpValUnitary:
    def test_z_matrix_matches_pauli_z(self):
        # Z matrix: [[1,0],[0,-1]] — row-major flat list
        z_matrix = [1 + 0j, 0 + 0j, 0 + 0j, -1 + 0j]
        sim = QrackSimulator(qubitCount=1)
        ev_unitary = sim.exp_val_unitary([0], z_matrix)
        ev_pauli = sim.exp_val(Pauli.PauliZ, 0)
        assert ev_unitary == pytest.approx(ev_pauli, abs=1e-4)

    def test_z_matrix_on_one_state(self):
        # Z matrix: [[1,0],[0,-1]] — diagonal elements are used as eigenvalues
        # Z on |1⟩: diag[0]*P(|0⟩) + diag[1]*P(|1⟩) = 1*0 + (-1)*1 = -1.0
        z_matrix = [1 + 0j, 0 + 0j, 0 + 0j, -1 + 0j]
        sim = QrackSimulator(qubitCount=1)
        sim.x(0)  # |1⟩ state
        ev = sim.exp_val_unitary([0], z_matrix)
        assert ev == pytest.approx(-1.0, abs=1e-4)

    def test_mismatched_ops_raises(self):
        sim = QrackSimulator(qubitCount=2)
        with pytest.raises(QrackArgumentError):
            # 2 qubits need 8 elements, only 4 given
            sim.exp_val_unitary([0, 1], [1 + 0j] * 4)

    def test_two_qubit_tensor_product(self):
        # Z⊗Z on |00> should give +1
        z_matrix = [1 + 0j, 0 + 0j, 0 + 0j, -1 + 0j]
        sim = QrackSimulator(qubitCount=2)
        # 2 qubits, each with 4 complex values = 8 total
        ev = sim.exp_val_unitary([0, 1], z_matrix * 2)
        assert ev == pytest.approx(1.0, abs=1e-4)


class TestVarianceUnitary:
    def test_z_variance_on_zero_is_zero(self):
        z_matrix = [1 + 0j, 0 + 0j, 0 + 0j, -1 + 0j]
        sim = QrackSimulator(qubitCount=1)
        var = sim.variance_unitary([0], z_matrix)
        # |0⟩ is an eigenstate of Z — variance should be 0
        assert var == pytest.approx(0.0, abs=1e-4)

    def test_z_variance_on_plus(self):
        # Z matrix: [[1,0],[0,-1]] — diagonal elements are used as eigenvalues
        # |+⟩ = equal superposition: <Z>=0, <Z²>=1 → Var(Z)=1
        z_matrix = [1 + 0j, 0 + 0j, 0 + 0j, -1 + 0j]
        sim = QrackSimulator(qubitCount=1)
        sim.h(0)  # |+⟩ state
        var = sim.variance_unitary([0], z_matrix)
        assert var == pytest.approx(1.0, abs=1e-4)

    def test_mismatched_ops_raises(self):
        sim = QrackSimulator(qubitCount=2)
        with pytest.raises(QrackArgumentError):
            sim.variance_unitary([0, 1], [1 + 0j] * 4)


class TestExpValBitsFactorized:
    def test_identity_permutation_on_zero(self):
        sim = QrackSimulator(qubitCount=1)
        # Identity permutation (0b0) means weight 0 for |0>, weight 0 for any
        # other basis. On |0⟩, the value should be 0.
        # Qrack requires 2 perms per qubit: [weight_for_|0⟩, weight_for_|1⟩]
        ev = sim.exp_val_bits_factorized([0], [0, 0])
        assert ev == pytest.approx(0.0, abs=1e-4)

    def test_weighted_permutation(self):
        sim = QrackSimulator(qubitCount=1)
        sim.h(0)  # equal superposition
        # perm weight 0 → contribution from |0⟩ is zero
        # perm weight 1 → contribution from |1⟩ is 1.0
        # On a uniform superposition, <P(qubit=1)> = 0.5
        # So exp_val ≈ 0.5 * 1.0 = 0.5
        # Qrack requires 2 perms per qubit: [weight_for_|0⟩, weight_for_|1⟩]
        # weight=0 for |0⟩, weight=1 for |1⟩ → <P(qubit=1)> = 0.5 * 1.0 = 0.5
        ev = sim.exp_val_bits_factorized([0], [0, 1])
        assert ev == pytest.approx(0.5, abs=1e-4)
