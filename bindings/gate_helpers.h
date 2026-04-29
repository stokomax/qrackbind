// bindings/gate_helpers.h
// Templated .def() helpers shared by QrackSimulator, QrackStabilizer, and
// QrackStabilizerHybrid.  Each helper takes an nb::class_<WrapperT>& and
// appends a group of related gate or measurement methods to it.
//
// Requirements on WrapperT:
//   - QInterfacePtr  sim        — the underlying Qrack simulator
//   - bitLenInt      numQubits  — qubit count (kept in sync with sim)
//   - void check_qubit(bitLenInt q, const char* method) const
//
// GIL note: Phase 10 does not attach nb::call_guard<nb::gil_scoped_release>
// to any of these helpers.  Phase 11 will retrofit that uniformly.
#pragma once

#include "qrackbind_core.h"
#include <cmath>
#include <numeric>   // std::iota — used by exp_val_all

// ── Precision type aliases (needed by add_state_access) ─────────────────────
// On the default Qrack build (FPPOW=5) real1 == float, so:
//   cf_t == std::complex<float>  → numpy.complex64
//   r_t  == float                → numpy.float32
// On double-precision builds both widen automatically.
using r_t  = Qrack::real1;
using cf_t = std::complex<r_t>;
static_assert(sizeof(cf_t) == sizeof(Qrack::complex),
    "cf_t and Qrack::complex must be layout-compatible for reinterpret_cast");

// ── Shared constants ─────────────────────────────────────────────────────────
namespace gh_detail {
    inline const Qrack::complex ONE_C   = Qrack::complex( 1.0f,  0.0f);
    inline const Qrack::complex NEG1_C  = Qrack::complex(-1.0f,  0.0f);
    inline const Qrack::complex I_C     = Qrack::complex( 0.0f,  1.0f);
    inline const Qrack::complex NEG_I_C = Qrack::complex( 0.0f, -1.0f);
    inline const Qrack::real1   SQRT1_2 = static_cast<Qrack::real1>(std::sqrt(0.5));
    // H gate matrix: [1/√2,  1/√2, 1/√2, -1/√2]
    inline const Qrack::complex H_MTRX[4] = {
        {SQRT1_2, 0}, {SQRT1_2, 0}, {SQRT1_2, 0}, {-SQRT1_2, 0}
    };
} // namespace gh_detail

// ── Shared Pauli-basis measurement helper ───────────────────────────────────
// Rotates a qubit into the computational basis, measures, then rotates back.
// Template-free — takes QInterfacePtr directly so all three wrapper types
// can call it without capturing the wrapper struct itself.
inline bool measure_in_basis(QInterfacePtr& sim, Qrack::Pauli basis, bitLenInt q)
{
    if (basis == Qrack::Pauli::PauliI)
        return false;   // identity: no observable, no collapse

    switch (basis) {
        case Qrack::Pauli::PauliX:
            sim->H(q);
            break;
        case Qrack::Pauli::PauliY:
            sim->IS(q);   // S†
            sim->H(q);
            break;
        case Qrack::Pauli::PauliZ:
        default:
            break;
    }

    const bool result = sim->M(q);

    switch (basis) {
        case Qrack::Pauli::PauliX:
            sim->H(q);
            break;
        case Qrack::Pauli::PauliY:
            sim->H(q);
            sim->S(q);
            break;
        case Qrack::Pauli::PauliZ:
        default:
            break;
    }

    return result;
}


// ════════════════════════════════════════════════════════════════════════════
// Gate helper templates
// ════════════════════════════════════════════════════════════════════════════

// ── Strict Clifford 1-qubit gates: H, X, Y, Z, S, S†, √X, √X† ─────────────
template <typename WrapperT>
void add_clifford_gates(nb::class_<WrapperT>& cls)
{
    cls
        .def("h",    [](WrapperT& w, bitLenInt q) { w.check_qubit(q, "h");    w.sim->H(q);      }, nb::arg("qubit"), "Hadamard gate.")
        .def("x",    [](WrapperT& w, bitLenInt q) { w.check_qubit(q, "x");    w.sim->X(q);      }, nb::arg("qubit"), "Pauli X (bit flip) gate.")
        .def("y",    [](WrapperT& w, bitLenInt q) { w.check_qubit(q, "y");    w.sim->Y(q);      }, nb::arg("qubit"), "Pauli Y gate.")
        .def("z",    [](WrapperT& w, bitLenInt q) { w.check_qubit(q, "z");    w.sim->Z(q);      }, nb::arg("qubit"), "Pauli Z (phase flip) gate.")
        .def("s",    [](WrapperT& w, bitLenInt q) { w.check_qubit(q, "s");    w.sim->S(q);      }, nb::arg("qubit"), "S gate — phase shift π/2.")
        .def("sdg",  [](WrapperT& w, bitLenInt q) { w.check_qubit(q, "sdg");  w.sim->IS(q);     }, nb::arg("qubit"), "S† (inverse S) gate.")
        .def("sx",   [](WrapperT& w, bitLenInt q) { w.check_qubit(q, "sx");   w.sim->SqrtX(q);  }, nb::arg("qubit"), "√X gate (half-X). Native Qiskit basis gate.")
        .def("sxdg", [](WrapperT& w, bitLenInt q) { w.check_qubit(q, "sxdg"); w.sim->ISqrtX(q); }, nb::arg("qubit"), "√X† gate (inverse √X).")
    ;
}


// ── Two-qubit Clifford + multi-controlled Clifford variants ─────────────────
// Includes: CNOT, CY, CZ, SWAP, iSWAP, CCNOT, MCX/MACX, MCY/MACY, MCZ/MACZ, MCH
template <typename WrapperT>
void add_clifford_two_qubit(nb::class_<WrapperT>& cls)
{
    using namespace gh_detail;

    cls
        .def("cnot",
            [](WrapperT& w, bitLenInt ctrl, bitLenInt tgt) {
                w.check_qubit(ctrl, "cnot"); w.check_qubit(tgt, "cnot");
                w.sim->CNOT(ctrl, tgt);
            },
            nb::arg("control"), nb::arg("target"),
            "Controlled-NOT (CNOT / CX) gate.")

        .def("cy",
            [](WrapperT& w, bitLenInt ctrl, bitLenInt tgt) {
                w.check_qubit(ctrl, "cy"); w.check_qubit(tgt, "cy");
                w.sim->CY(ctrl, tgt);
            },
            nb::arg("control"), nb::arg("target"), "Controlled-Y gate.")

        .def("cz",
            [](WrapperT& w, bitLenInt ctrl, bitLenInt tgt) {
                w.check_qubit(ctrl, "cz"); w.check_qubit(tgt, "cz");
                w.sim->CZ(ctrl, tgt);
            },
            nb::arg("control"), nb::arg("target"), "Controlled-Z gate.")

        .def("swap",
            [](WrapperT& w, bitLenInt q1, bitLenInt q2) {
                w.check_qubit(q1, "swap"); w.check_qubit(q2, "swap");
                w.sim->Swap(q1, q2);
            },
            nb::arg("qubit1"), nb::arg("qubit2"), "SWAP gate.")

        .def("iswap",
            [](WrapperT& w, bitLenInt q1, bitLenInt q2) {
                w.check_qubit(q1, "iswap"); w.check_qubit(q2, "iswap");
                w.sim->ISwap(q1, q2);
            },
            nb::arg("qubit1"), nb::arg("qubit2"), "iSWAP gate.")

        .def("ccnot",
            [](WrapperT& w, bitLenInt c1, bitLenInt c2, bitLenInt tgt) {
                w.sim->CCNOT(c1, c2, tgt);
            },
            nb::arg("control1"), nb::arg("control2"), nb::arg("target"),
            "Toffoli (CCX / CCNOT) gate.")

        // ── mcx / macx ───────────────────────────────────────────────────────
        .def("mcx",
            [](WrapperT& w, std::vector<bitLenInt> controls, bitLenInt tgt) {
                w.sim->MCInvert(controls, ONE_C, ONE_C, tgt);
            },
            nb::arg("controls"), nb::arg("target"),
            "Multiply-controlled X. Fires when all controls are |1>.")

        .def("macx",
            [](WrapperT& w, std::vector<bitLenInt> controls, bitLenInt tgt) {
                w.sim->MACInvert(controls, ONE_C, ONE_C, tgt);
            },
            nb::arg("controls"), nb::arg("target"),
            "Anti-controlled X. Fires when all controls are |0>.")

        // ── mcy / macy ───────────────────────────────────────────────────────
        .def("mcy",
            [](WrapperT& w, std::vector<bitLenInt> controls, bitLenInt tgt) {
                w.sim->MCInvert(controls, NEG_I_C, I_C, tgt);
            },
            nb::arg("controls"), nb::arg("target"), "Multiply-controlled Y.")

        .def("macy",
            [](WrapperT& w, std::vector<bitLenInt> controls, bitLenInt tgt) {
                w.sim->MACInvert(controls, NEG_I_C, I_C, tgt);
            },
            nb::arg("controls"), nb::arg("target"), "Anti-controlled Y.")

        // ── mcz / macz ───────────────────────────────────────────────────────
        .def("mcz",
            [](WrapperT& w, std::vector<bitLenInt> controls, bitLenInt tgt) {
                w.sim->MCPhase(controls, ONE_C, NEG1_C, tgt);
            },
            nb::arg("controls"), nb::arg("target"), "Multiply-controlled Z.")

        .def("macz",
            [](WrapperT& w, std::vector<bitLenInt> controls, bitLenInt tgt) {
                w.sim->MACPhase(controls, ONE_C, NEG1_C, tgt);
            },
            nb::arg("controls"), nb::arg("target"), "Anti-controlled Z.")

        // ── mch — multiply-controlled Hadamard ───────────────────────────────
        .def("mch",
            [](WrapperT& w, std::vector<bitLenInt> controls, bitLenInt tgt) {
                w.sim->MCMtrx(controls, H_MTRX, tgt);
            },
            nb::arg("controls"), nb::arg("target"), "Multiply-controlled H.")
    ;
}


// ── T, T† — non-Clifford phase gates ────────────────────────────────────────
template <typename WrapperT>
void add_t_gates(nb::class_<WrapperT>& cls)
{
    cls
        .def("t",   [](WrapperT& w, bitLenInt q) { w.check_qubit(q, "t");   w.sim->T(q);  }, nb::arg("qubit"), "T gate — phase shift π/4.")
        .def("tdg", [](WrapperT& w, bitLenInt q) { w.check_qubit(q, "tdg"); w.sim->IT(q); }, nb::arg("qubit"), "T† (inverse T) gate.")
    ;
}


// ── Rotations RX, RY, RZ, R1 — non-Clifford ─────────────────────────────────
template <typename WrapperT>
void add_rotation_gates(nb::class_<WrapperT>& cls)
{
    cls
        .def("rx", [](WrapperT& w, real1_f angle, bitLenInt q) { w.check_qubit(q, "rx"); w.sim->RX(angle, q); }, nb::arg("angle"), nb::arg("qubit"), "Rotate around X axis by angle radians. Equiv: exp(-i·angle/2·X).")
        .def("ry", [](WrapperT& w, real1_f angle, bitLenInt q) { w.check_qubit(q, "ry"); w.sim->RY(angle, q); }, nb::arg("angle"), nb::arg("qubit"), "Rotate around Y axis by angle radians.")
        .def("rz", [](WrapperT& w, real1_f angle, bitLenInt q) { w.check_qubit(q, "rz"); w.sim->RZ(angle, q); }, nb::arg("angle"), nb::arg("qubit"), "Rotate around Z axis by angle radians.")
        .def("r1", [](WrapperT& w, real1_f angle, bitLenInt q) { w.check_qubit(q, "r1"); w.sim->RT(angle, q); }, nb::arg("angle"), nb::arg("qubit"), "Phase rotation: apply e^(i·angle) to |1> state.")
    ;
}


// ── U, U2 — non-Clifford arbitrary single-qubit unitaries ───────────────────
template <typename WrapperT>
void add_u_gates(nb::class_<WrapperT>& cls)
{
    cls
        .def("u",
            [](WrapperT& w, real1_f theta, real1_f phi, real1_f lam, bitLenInt q) {
                w.check_qubit(q, "u");
                w.sim->U(q, theta, phi, lam);
            },
            nb::arg("theta"), nb::arg("phi"), nb::arg("lam"), nb::arg("qubit"),
            "General single-qubit unitary: U(θ,φ,λ). Decomposes to RZ·RY·RZ.")

        .def("u2",
            [](WrapperT& w, real1_f phi, real1_f lam, bitLenInt q) {
                w.check_qubit(q, "u2");
                w.sim->U(q, static_cast<real1_f>(M_PI / 2.0), phi, lam);
            },
            nb::arg("phi"), nb::arg("lam"), nb::arg("qubit"),
            "U2 gate: U(π/2, φ, λ).")
    ;
}


// ── Arbitrary matrix gates: mtrx, mcmtrx, macmtrx, multiplex1_mtrx, mcrz, mcu
template <typename WrapperT>
void add_matrix_gates(nb::class_<WrapperT>& cls)
{
    cls
        .def("mcrz",
            [](WrapperT& w, real1_f angle, std::vector<bitLenInt> controls, bitLenInt tgt) {
                const Qrack::real1 half = static_cast<Qrack::real1>(angle / 2.0f);
                const Qrack::complex phase0 = std::exp(Qrack::complex(0.0f, -half));
                const Qrack::complex phase1 = std::exp(Qrack::complex(0.0f,  half));
                w.sim->MCPhase(controls, phase0, phase1, tgt);
            },
            nb::arg("angle"), nb::arg("controls"), nb::arg("target"),
            "Multiply-controlled RZ.")

        .def("mcu",
            [](WrapperT& w, std::vector<bitLenInt> controls, bitLenInt tgt,
               real1_f theta, real1_f phi, real1_f lam) {
                w.sim->CU(controls, tgt, theta, phi, lam);
            },
            nb::arg("controls"), nb::arg("target"),
            nb::arg("theta"), nb::arg("phi"), nb::arg("lam"),
            "Multiply-controlled U(θ,φ,λ) gate.")

        .def("mtrx",
            [](WrapperT& w, std::vector<std::complex<float>> m, bitLenInt q) {
                if (m.size() < 4)
                    throw QrackError("mtrx: matrix must have 4 elements",
                                     QrackErrorKind::InvalidArgument);
                w.check_qubit(q, "mtrx");
                w.sim->Mtrx(m.data(), q);
            },
            nb::arg("matrix"), nb::arg("qubit"),
            "Apply arbitrary 2x2 unitary. matrix is [m00, m01, m10, m11] row-major.")

        .def("mcmtrx",
            [](WrapperT& w, std::vector<bitLenInt> controls,
               std::vector<std::complex<float>> m, bitLenInt q) {
                if (m.size() < 4)
                    throw QrackError("mcmtrx: matrix must have 4 elements",
                                     QrackErrorKind::InvalidArgument);
                w.sim->MCMtrx(controls, m.data(), q);
            },
            nb::arg("controls"), nb::arg("matrix"), nb::arg("qubit"),
            "Multiply-controlled arbitrary 2x2 unitary.")

        .def("macmtrx",
            [](WrapperT& w, std::vector<bitLenInt> controls,
               std::vector<std::complex<float>> m, bitLenInt q) {
                if (m.size() < 4)
                    throw QrackError("macmtrx: matrix must have 4 elements",
                                     QrackErrorKind::InvalidArgument);
                w.sim->MACMtrx(controls, m.data(), q);
            },
            nb::arg("controls"), nb::arg("matrix"), nb::arg("qubit"),
            "Anti-controlled arbitrary 2x2 unitary.")

        .def("multiplex1_mtrx",
            [](WrapperT& w, std::vector<bitLenInt> controls,
               std::vector<std::complex<float>> mtrxs, bitLenInt tgt) {
                const size_t expected = 4ULL << controls.size();
                if (mtrxs.size() < expected)
                    throw QrackError(
                        "multiplex1_mtrx: mtrxs must have at least 4 * 2^len(controls) = " +
                        std::to_string(expected) + " elements",
                        QrackErrorKind::InvalidArgument);
                w.check_qubit(tgt, "multiplex1_mtrx");
                w.sim->UniformlyControlledSingleBit(controls, tgt,
                    reinterpret_cast<const Qrack::complex*>(mtrxs.data()));
            },
            nb::arg("controls"), nb::arg("mtrxs"), nb::arg("target"),
            "Uniformly-controlled single-qubit gate. mtrxs is a flat list of "
            "4 * 2**len(controls) complex values — one 2x2 unitary per control "
            "permutation, in row-major order.")
    ;
}


// ── Measurement: measure, measure_all, force_measure, prob, prob_all ─────────
template <typename WrapperT>
void add_measurement(nb::class_<WrapperT>& cls)
{
    cls
        .def("measure",
            [](WrapperT& w, bitLenInt q) -> bool {
                w.check_qubit(q, "measure");
                return w.sim->M(q);
            },
            nb::arg("qubit"),
            "Measure qubit. Returns True=|1>, False=|0>. Collapses state.")

        .def("measure_all",
            [](WrapperT& w) -> std::vector<bool> {
                std::vector<bool> out;
                out.reserve(w.numQubits);
                for (bitLenInt i = 0; i < w.numQubits; i++)
                    out.push_back(w.sim->M(i));
                return out;
            },
            "Measure all qubits. Returns list[bool], LSB first.")

        .def("force_measure",
            [](WrapperT& w, bitLenInt q, bool result) -> bool {
                w.check_qubit(q, "force_measure");
                return w.sim->ForceM(q, result);
            },
            nb::arg("qubit"), nb::arg("result"),
            "Force measurement outcome. Projects state to result without random draw.")

        .def("prob",
            [](WrapperT& w, bitLenInt q) -> real1_f {
                w.check_qubit(q, "prob");
                return w.sim->Prob(q);
            },
            nb::arg("qubit"),
            "Probability of |1> for qubit. Does NOT collapse state.")

        .def("prob_all",
            [](WrapperT& w) -> std::vector<real1_f> {
                std::vector<real1_f> out(w.numQubits);
                for (bitLenInt i = 0; i < w.numQubits; i++)
                    out[i] = w.sim->Prob(i);
                return out;
            },
            "Per-qubit |1> probabilities for all qubits. Does NOT collapse state.")
    ;
}


// ── Pauli observables ────────────────────────────────────────────────────────
// measure_pauli, exp_val, exp_val_pauli, variance_pauli,
// exp_val_all, exp_val_floats, variance_floats
template <typename WrapperT>
void add_pauli_methods(nb::class_<WrapperT>& cls)
{
    cls
        .def("measure_pauli",
            [](WrapperT& w, Qrack::Pauli basis, bitLenInt q) -> bool {
                w.check_qubit(q, "measure_pauli");
                return measure_in_basis(w.sim, basis, q);
            },
            nb::arg("basis"), nb::arg("qubit"),
            "Measure a qubit in the specified Pauli basis.\n\n"
            "Rotates the qubit into the computational basis, measures, and\n"
            "rotates back. Returns True if the rotated qubit collapsed to |1>.\n"
            "The state is collapsed in the chosen basis.")

        .def("exp_val",
            [](WrapperT& w, Qrack::Pauli basis, bitLenInt q) -> real1_f {
                w.check_qubit(q, "exp_val");
                return w.sim->ExpectationPauliAll({q}, {basis});
            },
            nb::arg("basis"), nb::arg("qubit"),
            "Single-qubit Pauli expectation value. Result is in [-1.0, +1.0].\n"
            "Does not collapse the state.")

        .def("exp_val_pauli",
            [](WrapperT& w,
               std::vector<Qrack::Pauli> paulis,
               std::vector<bitLenInt>    qubits) -> real1_f
            {
                if (paulis.size() != qubits.size())
                    throw QrackError(
                        "exp_val_pauli: paulis and qubits must have the same length",
                        QrackErrorKind::InvalidArgument);
                for (auto q : qubits)
                    w.check_qubit(q, "exp_val_pauli");
                return w.sim->ExpectationPauliAll(qubits, paulis);
            },
            nb::arg("paulis"), nb::arg("qubits"),
            "Expectation value of a Pauli tensor product observable.\n\n"
            "Returns <ψ|P₀⊗P₁⊗…⊗Pₙ|ψ>. Result is in [-1.0, +1.0].\n"
            "Does not collapse the state.")

        .def("variance_pauli",
            [](WrapperT& w,
               std::vector<Qrack::Pauli> paulis,
               std::vector<bitLenInt>    qubits) -> real1_f
            {
                if (paulis.size() != qubits.size())
                    throw QrackError(
                        "variance_pauli: paulis and qubits must have the same length",
                        QrackErrorKind::InvalidArgument);
                for (auto q : qubits)
                    w.check_qubit(q, "variance_pauli");
                return w.sim->VariancePauliAll(qubits, paulis);
            },
            nb::arg("paulis"), nb::arg("qubits"),
            "Variance of a Pauli tensor product observable.\n\n"
            "For a Pauli operator P (P² = I), Var(P) = 1 − <P>².\n"
            "Result is in [0.0, 1.0]. Does not collapse the state.")

        .def("exp_val_all",
            [](WrapperT& w, Qrack::Pauli basis) -> real1_f {
                std::vector<bitLenInt> qubits(w.numQubits);
                std::iota(qubits.begin(), qubits.end(), bitLenInt(0));
                std::vector<Qrack::Pauli> paulis(w.numQubits, basis);
                return w.sim->ExpectationPauliAll(qubits, paulis);
            },
            nb::arg("basis"),
            "Expectation value of the same Pauli operator applied to every qubit.")

        .def("exp_val_floats",
            [](WrapperT& w,
               std::vector<bitLenInt> qubits,
               std::vector<float>     weights) -> real1_f
            {
                if (weights.size() != 2 * qubits.size())
                    throw QrackError(
                        "exp_val_floats: weights must contain exactly 2 entries "
                        "per qubit (weights[2*i] for |0>, weights[2*i+1] for |1>)",
                        QrackErrorKind::InvalidArgument);
                for (auto q : qubits)
                    w.check_qubit(q, "exp_val_floats");
                std::vector<Qrack::real1_f> wts(weights.begin(), weights.end());
                return w.sim->ExpectationFloatsFactorized(qubits, wts);
            },
            nb::arg("qubits"), nb::arg("weights"),
            "Expectation value of a weighted single-qubit observable.\n\n"
            "weights must have length 2 * len(qubits): "
            "[w_|0> for q0, w_|1> for q0, w_|0> for q1, ...]")

        .def("variance_floats",
            [](WrapperT& w,
               std::vector<bitLenInt> qubits,
               std::vector<float>     weights) -> real1_f
            {
                if (weights.size() != 2 * qubits.size())
                    throw QrackError(
                        "variance_floats: weights must contain exactly 2 entries "
                        "per qubit (weights[2*i] for |0>, weights[2*i+1] for |1>)",
                        QrackErrorKind::InvalidArgument);
                for (auto q : qubits)
                    w.check_qubit(q, "variance_floats");
                std::vector<Qrack::real1_f> wts(weights.begin(), weights.end());
                return w.sim->VarianceFloatsFactorized(qubits, wts);
            },
            nb::arg("qubits"), nb::arg("weights"),
            "Variance of a weighted single-qubit observable. "
            "See exp_val_floats for weight convention.")
    ;
}


// ── Multi-shot measurement: measure_shots ────────────────────────────────────
// Non-collapsing multi-shot sampler using MultiShotMeasureMask.
// Returns dict[int, int]: measurement result (as integer bit pattern) → count.
// Available on QrackSimulator, QrackStabilizer, and QrackStabilizerHybrid.
template <typename WrapperT>
void add_measure_shots(nb::class_<WrapperT>& cls)
{
    cls
        .def("measure_shots",
            [](WrapperT& w, std::vector<bitLenInt> qubits, unsigned shots)
                -> std::map<uint64_t, int>
            {
                std::vector<BigInteger> qpowers;
                qpowers.reserve(qubits.size());
                for (auto q : qubits) {
                    w.check_qubit(q, "measure_shots");
                    qpowers.push_back(BigInteger(1) << q);
                }
                auto raw = w.sim->MultiShotMeasureMask(qpowers, shots);
                std::map<uint64_t, int> out;
                for (const auto& kv : raw)
                    out.emplace(static_cast<uint64_t>(kv.first), kv.second);
                return out;
            },
            nb::arg("qubits"), nb::arg("shots"),
            "Sample 'shots' measurements of 'qubits' without collapsing state.\n"
            "Returns dict[int, int]: measurement result (integer bit pattern) → count.")
    ;
}


// ── State access: _state_vector_impl, _probabilities_impl, get/set_amplitude ─
// Cost-bearing for stabilizer engines (materialises amplitudes on demand).
// Not added to QrackStabilizer; only QrackSimulator and QrackStabilizerHybrid.
//
// state_vector and probabilities are exposed as _impl methods and wrapped as
// @property in Python (same pattern as QrackSimulator) because def_prop_ro is
// incompatible with capsule-owned ndarrays.
template <typename WrapperT>
void add_state_access(nb::class_<WrapperT>& cls)
{
    cls
        .def("_state_vector_impl",
            [](WrapperT& w) -> nb::ndarray<nb::numpy, cf_t, nb::shape<-1>>
            {
                const size_t n = size_t(1) << w.numQubits;
                cf_t* buf = new cf_t[n];
                for (size_t i = 0; i < n; i++) {
                    const Qrack::complex amp = w.sim->GetAmplitude(bitCapInt(i));
                    buf[i] = cf_t(amp.real(), amp.imag());
                }
                nb::capsule owner(buf, [](void* p) noexcept {
                    delete[] static_cast<cf_t*>(p);
                });
                return nb::ndarray<nb::numpy, cf_t, nb::shape<-1>>(buf, {n}, owner);
            },
            nb::rv_policy::reference,
            "Internal: backing function for the state_vector property.")

        .def("_probabilities_impl",
            [](WrapperT& w) -> nb::ndarray<nb::numpy, r_t, nb::shape<-1>>
            {
                const size_t n = size_t(1) << w.numQubits;
                r_t* buf = new r_t[n];
                for (size_t i = 0; i < n; i++)
                    buf[i] = static_cast<r_t>(w.sim->ProbAll(bitCapInt(i)));
                nb::capsule owner(buf, [](void* p) noexcept {
                    delete[] static_cast<r_t*>(p);
                });
                return nb::ndarray<nb::numpy, r_t, nb::shape<-1>>(buf, {n}, owner);
            },
            nb::rv_policy::reference,
            "Internal: backing function for the probabilities property.")

        .def("get_amplitude",
            [](WrapperT& w, bitCapInt perm) -> std::complex<float>
            {
                const Qrack::complex amp = w.sim->GetAmplitude(perm);
                return std::complex<float>(
                    static_cast<float>(amp.real()),
                    static_cast<float>(amp.imag()));
            },
            nb::arg("index"),
            "Get the complex amplitude of a specific basis state by integer index.\n"
            "Does not collapse the state.")

        .def("set_amplitude",
            [](WrapperT& w, bitCapInt perm, std::complex<float> amp)
            {
                w.sim->SetAmplitude(perm,
                    Qrack::complex(static_cast<r_t>(amp.real()),
                                   static_cast<r_t>(amp.imag())));
            },
            nb::arg("index"), nb::arg("amplitude"),
            "Set the complex amplitude of a specific basis state.\n"
            "Does NOT re-normalise — call update_running_norm() if needed.")
    ;
}
