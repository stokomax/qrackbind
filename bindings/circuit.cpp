// bindings/circuit.cpp
// ─────────────────────────────────────────────────────────────────────────────
// Phase 6 — QrackCircuit binding.
//
// Wraps Qrack::QCircuit — a replayable, optimisable, serialisable quantum
// circuit object independent of any specific simulator instance.
// ─────────────────────────────────────────────────────────────────────────────

#include "qrackbind_core.h"

#include <cmath>
#include <set>
#include <sstream>

// ── Gate matrices for 1-qubit Clifford gates ─────────────────────────────────
namespace {

using C = Qrack::complex;
using R = Qrack::real1;

// H = 1/√2 * [[1, 1], [1, -1]]
static const C H_MTRX[4] = {
    {M_SQRT1_2, 0}, {M_SQRT1_2, 0},
    {M_SQRT1_2, 0}, {-M_SQRT1_2, 0}};

static const C X_MTRX[4] = {{0,0},{1,0},{1,0},{0,0}};
static const C Y_MTRX[4] = {{0,0},{0,-1},{0,1},{0,0}};
static const C Z_MTRX[4] = {{1,0},{0,0},{0,0},{-1,0}};

// S = [[1, 0], [0, i]]
static const C S_MTRX[4] = {{1,0},{0,0},{0,0},{0,1}};
// IS = S† = [[1, 0], [0, -i]]
static const C IS_MTRX[4] = {{1,0},{0,0},{0,0},{0,-1}};

// SqrtX = 1/2 * [[1+i, 1-i], [1-i, 1+i]]
static const C SQRTX_MTRX[4] = {
    {0.5f, 0.5f}, {0.5f, -0.5f},
    {0.5f, -0.5f}, {0.5f, 0.5f}};
// ISqrtX = 1/2 * [[1-i, 1+i], [1+i, 1-i]]
static const C ISQRTX_MTRX[4] = {
    {0.5f, -0.5f}, {0.5f, 0.5f},
    {0.5f, 0.5f}, {0.5f, -0.5f}};

} // anonymous namespace

// ── GateType enum ────────────────────────────────────────────────────────────

enum class GateType {
    H, X, Y, Z, S, T, IS, IT,
    SqrtX, ISqrtX,
    RX, RY, RZ, R1,
    CNOT, CY, CZ, CH,
    MCX, MCY, MCZ,
    SWAP, ISWAP,
    U,
    Mtrx,
    MCMtrx,
};

// ── AppendGate helper: build a QCircuitGatePtr from type + operands ──────────

static QCircuitGatePtr make_circuit_gate(
    GateType gate,
    const std::vector<bitLenInt>& qubits,
    const std::vector<float>& params,
    bitLenInt numQubits)
{
    using C = Qrack::complex;
    using R = Qrack::real1;

    // Validate qubit indices
    for (auto q : qubits) {
        if (q >= numQubits)
            throw QrackError(
                "append_gate: qubit " + std::to_string(q) +
                " out of range [0, " + std::to_string(numQubits - 1) + "]",
                QrackErrorKind::QubitOutOfRange);
    }

    const bitLenInt q0 = qubits.empty() ? 0 : qubits[0];
    const bitLenInt q1 = qubits.size() > 1 ? qubits[1] : 0;

    switch (gate) {
        // ── Clifford 1-qubit gates ────────────────────────────────────────
        case GateType::H:
            return std::make_shared<Qrack::QCircuitGate>(q0, H_MTRX);
        case GateType::X:
            return std::make_shared<Qrack::QCircuitGate>(q0, X_MTRX);
        case GateType::Y:
            return std::make_shared<Qrack::QCircuitGate>(q0, Y_MTRX);
        case GateType::Z:
            return std::make_shared<Qrack::QCircuitGate>(q0, Z_MTRX);
        case GateType::S:
            return std::make_shared<Qrack::QCircuitGate>(q0, S_MTRX);
        case GateType::IS:
            return std::make_shared<Qrack::QCircuitGate>(q0, IS_MTRX);
        case GateType::SqrtX:
            return std::make_shared<Qrack::QCircuitGate>(q0, SQRTX_MTRX);
        case GateType::ISqrtX:
            return std::make_shared<Qrack::QCircuitGate>(q0, ISQRTX_MTRX);

        // ── T / IT — phase gates with π/4 angle ───────────────────────────
        case GateType::T: {
            const C phase = std::exp(C(0.0f, M_PI / 4.0f));
            const C mtrx[4] = {{1,0},{0,0},{0,0},phase};
            return std::make_shared<Qrack::QCircuitGate>(q0, mtrx);
        }
        case GateType::IT: {
            const C phase = std::exp(C(0.0f, -M_PI / 4.0f));
            const C mtrx[4] = {{1,0},{0,0},{0,0},phase};
            return std::make_shared<Qrack::QCircuitGate>(q0, mtrx);
        }

        // ── Rotation gates ────────────────────────────────────────────────
        case GateType::RX: {
            if (params.empty())
                throw QrackError("RX requires 1 angle param",
                                 QrackErrorKind::InvalidArgument);
            const R half = static_cast<R>(params[0] / 2.0f);
            const C mtrx[4] = {
                {std::cos(half), 0}, {0, -std::sin(half)},
                {0, -std::sin(half)}, {std::cos(half), 0}};
            return std::make_shared<Qrack::QCircuitGate>(q0, mtrx);
        }
        case GateType::RY: {
            if (params.empty())
                throw QrackError("RY requires 1 angle param",
                                 QrackErrorKind::InvalidArgument);
            const R half = static_cast<R>(params[0] / 2.0f);
            const C mtrx[4] = {
                {std::cos(half), 0}, {-std::sin(half), 0},
                {std::sin(half), 0}, {std::cos(half), 0}};
            return std::make_shared<Qrack::QCircuitGate>(q0, mtrx);
        }
        case GateType::RZ: {
            if (params.empty())
                throw QrackError("RZ requires 1 angle param",
                                 QrackErrorKind::InvalidArgument);
            const R half = static_cast<R>(params[0] / 2.0f);
            const C ph0 = std::exp(C(0.0f, -half));
            const C ph1 = std::exp(C(0.0f,  half));
            const C mtrx[4] = {ph0, {0,0}, {0,0}, ph1};
            return std::make_shared<Qrack::QCircuitGate>(q0, mtrx);
        }
        case GateType::R1: {
            if (params.empty())
                throw QrackError("R1 requires 1 angle param",
                                 QrackErrorKind::InvalidArgument);
            const C phase = std::exp(C(0.0f, static_cast<R>(params[0])));
            const C mtrx[4] = {{1,0},{0,0},{0,0},phase};
            return std::make_shared<Qrack::QCircuitGate>(q0, mtrx);
        }

        // ── 2-qubit controlled gates ──────────────────────────────────────
        case GateType::CNOT: {
            if (qubits.size() < 2)
                throw QrackError("CNOT requires 2 qubits",
                                 QrackErrorKind::InvalidArgument);
            const std::set<bitLenInt> controls{q0};
            return std::make_shared<Qrack::QCircuitGate>(
                q1, X_MTRX, controls, ONE_BCI);
        }
        case GateType::CY: {
            if (qubits.size() < 2)
                throw QrackError("CY requires 2 qubits",
                                 QrackErrorKind::InvalidArgument);
            const std::set<bitLenInt> controls{q0};
            return std::make_shared<Qrack::QCircuitGate>(
                q1, Y_MTRX, controls, ONE_BCI);
        }
        case GateType::CZ: {
            if (qubits.size() < 2)
                throw QrackError("CZ requires 2 qubits",
                                 QrackErrorKind::InvalidArgument);
            const std::set<bitLenInt> controls{q0};
            return std::make_shared<Qrack::QCircuitGate>(
                q1, Z_MTRX, controls, ONE_BCI);
        }
        case GateType::CH: {
            if (qubits.size() < 2)
                throw QrackError("CH requires 2 qubits",
                                 QrackErrorKind::InvalidArgument);
            const std::set<bitLenInt> controls{q0};
            return std::make_shared<Qrack::QCircuitGate>(
                q1, H_MTRX, controls, ONE_BCI);
        }

        // ── Multi-controlled gates (target is last qubit) ─────────────────
        case GateType::MCX: {
            if (qubits.size() < 2)
                throw QrackError("MCX requires at least 2 qubits (controls + target)",
                                 QrackErrorKind::InvalidArgument);
            const bitLenInt tgt = qubits.back();
            std::set<bitLenInt> controls(qubits.begin(), qubits.end() - 1);
            return std::make_shared<Qrack::QCircuitGate>(
                tgt, X_MTRX, controls, ONE_BCI);
        }
        case GateType::MCY: {
            if (qubits.size() < 2)
                throw QrackError("MCY requires at least 2 qubits (controls + target)",
                                 QrackErrorKind::InvalidArgument);
            const bitLenInt tgt = qubits.back();
            std::set<bitLenInt> controls(qubits.begin(), qubits.end() - 1);
            return std::make_shared<Qrack::QCircuitGate>(
                tgt, Y_MTRX, controls, ONE_BCI);
        }
        case GateType::MCZ: {
            if (qubits.size() < 2)
                throw QrackError("MCZ requires at least 2 qubits (controls + target)",
                                 QrackErrorKind::InvalidArgument);
            const bitLenInt tgt = qubits.back();
            std::set<bitLenInt> controls(qubits.begin(), qubits.end() - 1);
            return std::make_shared<Qrack::QCircuitGate>(
                tgt, Z_MTRX, controls, ONE_BCI);
        }

        // ── SWAP — uses QCircuit::Swap() which decomposes to 3 CNOTs ──────
        case GateType::SWAP:
            // SWAP is special: QCircuit has a native Swap() method that adds
            // 3 internal gates. Return nullptr to signal the caller to use
            // the native path.
            return nullptr;

        // ── Arbitrary unitary U(θ, φ, λ) ──────────────────────────────────
        case GateType::U: {
            if (params.size() < 3)
                throw QrackError("U requires 3 angle params (theta, phi, lam)",
                                 QrackErrorKind::InvalidArgument);
            const R theta = static_cast<R>(params[0]);
            const R phi   = static_cast<R>(params[1]);
            const R lam   = static_cast<R>(params[2]);
            const R cos_t2 = std::cos(theta / 2.0f);
            const R sin_t2 = std::sin(theta / 2.0f);
            const C mtrx[4] = {
                {cos_t2, 0},
                {-sin_t2 * std::cos(lam), -sin_t2 * std::sin(lam)},
                {sin_t2 * std::cos(phi), sin_t2 * std::sin(phi)},
                {cos_t2 * std::cos(phi + lam), cos_t2 * std::sin(phi + lam)}};
            return std::make_shared<Qrack::QCircuitGate>(q0, mtrx);
        }

        // ── Arbitrary 2x2 matrix (8 floats = 4 complex in row-major) ─────
        case GateType::Mtrx: {
            if (params.size() < 8)
                throw QrackError("Mtrx requires 8 floats (4 complex values)",
                                 QrackErrorKind::InvalidArgument);
            const C mtrx[4] = {
                C{params[0], params[1]}, C{params[2], params[3]},
                C{params[4], params[5]}, C{params[6], params[7]}};
            return std::make_shared<Qrack::QCircuitGate>(q0, mtrx);
        }

        // ── Multi-controlled arbitrary 2x2 matrix ─────────────────────────
        case GateType::MCMtrx: {
            if (qubits.size() < 2)
                throw QrackError("MCMtrx requires at least 2 qubits (controls + target)",
                                 QrackErrorKind::InvalidArgument);
            if (params.size() < 8)
                throw QrackError("MCMtrx requires 8 floats (4 complex values)",
                                 QrackErrorKind::InvalidArgument);
            const bitLenInt tgt = qubits.back();
            std::set<bitLenInt> controls(qubits.begin(), qubits.end() - 1);
            const C mtrx[4] = {
                C{params[0], params[1]}, C{params[2], params[3]},
                C{params[4], params[5]}, C{params[6], params[7]}};
            return std::make_shared<Qrack::QCircuitGate>(
                tgt, mtrx, controls, ONE_BCI);
        }

        // ── ISWAP — not yet implemented (requires decomposition) ──────────
        case GateType::ISWAP:
            throw QrackError(
                "ISWAP gate recording is not yet implemented. "
                "Use a decomposition (CNOT + single-qubit gates) instead.",
                QrackErrorKind::InvalidArgument);

        default:
            throw QrackError(
                "append_gate: unsupported gate type",
                QrackErrorKind::InvalidArgument);
    }
}

// ── Binding ──────────────────────────────────────────────────────────────────

void bind_circuit(nb::module_& m) {

    // ── GateType enum ────────────────────────────────────────────────────
    nb::enum_<GateType>(m, "GateType",
        "Gate type identifier for ``QrackCircuit.append_gate()``.\n\n"
        "Used to specify which gate to add to the circuit without\n"
        "immediately applying it to a simulator.")
        .value("H",      GateType::H,      "Hadamard gate")
        .value("X",      GateType::X,      "Pauli X (bit flip)")
        .value("Y",      GateType::Y,      "Pauli Y")
        .value("Z",      GateType::Z,      "Pauli Z (phase flip)")
        .value("S",      GateType::S,      "S gate (phase π/2)")
        .value("T",      GateType::T,      "T gate (phase π/4)")
        .value("IS",     GateType::IS,     "S† (inverse S)")
        .value("IT",     GateType::IT,     "T† (inverse T)")
        .value("SqrtX",  GateType::SqrtX,  "√X gate")
        .value("ISqrtX", GateType::ISqrtX, "√X† gate")
        .value("RX",     GateType::RX,     "X rotation — 1 angle param")
        .value("RY",     GateType::RY,     "Y rotation — 1 angle param")
        .value("RZ",     GateType::RZ,     "Z rotation — 1 angle param")
        .value("R1",     GateType::R1,     "Phase rotation — 1 angle param")
        .value("CNOT",   GateType::CNOT,   "Controlled NOT — 2 qubits")
        .value("CY",     GateType::CY,     "Controlled Y — 2 qubits")
        .value("CZ",     GateType::CZ,     "Controlled Z — 2 qubits")
        .value("CH",     GateType::CH,     "Controlled H — 2 qubits")
        .value("MCX",    GateType::MCX,    "Multi-controlled X — last qubit is target")
        .value("MCY",    GateType::MCY,    "Multi-controlled Y — last qubit is target")
        .value("MCZ",    GateType::MCZ,    "Multi-controlled Z — last qubit is target")
        .value("SWAP",   GateType::SWAP,   "SWAP gate — 2 qubits")
        .value("ISWAP",  GateType::ISWAP,  "iSWAP gate — not yet implemented")
        .value("U",      GateType::U,      "Arbitrary unitary U(θ, φ, λ) — 3 angle params")
        .value("Mtrx",   GateType::Mtrx,   "Arbitrary 2x2 unitary — 8 float params (4 complex)")
        .value("MCMtrx", GateType::MCMtrx, "Multi-controlled arbitrary 2x2 — 8 float params");

    // ── QrackCircuit class ───────────────────────────────────────────────
    nb::class_<QrackCircuit>(m, "QrackCircuit",
        "A replayable, optimisable quantum circuit.\n\n"
        "Records gate operations that can be executed on any\n"
        ":class:`QrackSimulator` via :meth:`run`. Circuits can be\n"
        "inverted, combined, and (in future) serialised to QASM.\n\n"
        "Example::\n\n"
        "    circ = QrackCircuit(2)\n"
        "    circ.append_gate(GateType.H, [0])\n"
        "    circ.append_gate(GateType.CNOT, [0, 1])\n"
        "    sim = QrackSimulator(qubitCount=2)\n"
        "    circ.run(sim)  # Bell state prepared")
        .def("__init__",
            [](QrackCircuit* self, bitLenInt qubitCount) {
                new (self) QrackCircuit(qubitCount);
            },
            nb::arg("qubitCount"),
            "Construct a circuit with the given number of qubits.")

        .def("__repr__",
            [](const QrackCircuit& c) {
                return "QrackCircuit(qubits=" + std::to_string(c.numQubits) +
                       ", gates=" + std::to_string(c.circuit->GetGateList().size()) + ")";
            })

        // ── append_gate ────────────────────────────────────────────────────
        .def("append_gate",
            [](QrackCircuit& c,
               GateType gate,
               std::vector<bitLenInt> qubits,
               std::vector<float> params)
            {
                // SWAP uses QCircuit's native Swap() method (decomposes
                // internally to 3 CNOT-like gates).
                if (gate == GateType::SWAP) {
                    if (qubits.size() < 2)
                        throw QrackError("SWAP requires 2 qubits",
                                         QrackErrorKind::InvalidArgument);
                    for (auto q : qubits) {
                        if (q >= c.numQubits)
                            throw QrackError(
                                "append_gate: qubit " + std::to_string(q) +
                                " out of range [0, " +
                                std::to_string(c.numQubits - 1) + "]",
                                QrackErrorKind::QubitOutOfRange);
                    }
                    c.circuit->Swap(qubits[0], qubits[1]);
                    return;
                }

                auto gatePtr = make_circuit_gate(gate, qubits, params, c.numQubits);
                c.circuit->AppendGate(gatePtr);
            },
            nb::arg("gate"), nb::arg("qubits"), nb::arg("params") = std::vector<float>{},
            "Append a gate to the circuit without executing it.\n\n"
            "Gates are accumulated and can be optimised before running.\n"
            "``params`` carries angle values for rotation gates, or complex\n"
            "components (real, imag pairs in row-major order) for matrix gates.\n\n"
            "Multi-controlled gates (MCX, MCY, MCZ, MCMtrx) treat the *last*\n"
            "qubit as the target and all others as controls.")

        // ── run ────────────────────────────────────────────────────────────
        .def("run",
            [](QrackCircuit& c, QrackSim& sim) {
                if (sim.numQubits < c.numQubits)
                    throw QrackError(
                        "run: circuit has " + std::to_string(c.numQubits) +
                        " qubits but simulator has only " +
                        std::to_string(sim.numQubits),
                        QrackErrorKind::InvalidArgument);
                c.circuit->Run(sim.sim);
            },
            nb::arg("simulator"),
            nb::keep_alive<1, 2>(),
            "Apply the circuit to the given simulator.\n\n"
            "The simulator's state is updated in place. The circuit itself\n"
            "is not consumed — it can be run on multiple simulators.\n"
            "The simulator must have at least as many qubits as the circuit.")

        // ── inverse ────────────────────────────────────────────────────────
        .def("inverse",
            [](const QrackCircuit& c) -> QrackCircuit {
                return QrackCircuit(c.circuit->Inverse(), c.numQubits);
            },
            "Return a new circuit that is the adjoint (inverse) of this circuit.\n\n"
            "Applies all gates in reverse order with conjugate-transposed matrices.\n"
            "Useful for uncomputation and ansatz construction.\n\n"
            "Example::\n\n"
            "    circ = QrackCircuit(2)\n"
            "    circ.append_gate(GateType.H, [0])\n"
            "    circ_inv = circ.inverse()   # applies H† = H\n"
            "    circ.run(sim)\n"
            "    circ_inv.run(sim)            # net effect: identity")

        // ── append (combine circuits) ──────────────────────────────────────
        .def("append",
            [](QrackCircuit& c, const QrackCircuit& other) {
                if (other.numQubits > c.numQubits)
                    throw QrackError(
                        "append: other circuit has more qubits (" +
                        std::to_string(other.numQubits) + ") than this circuit (" +
                        std::to_string(c.numQubits) + ")",
                        QrackErrorKind::InvalidArgument);
                c.circuit->Append(other.circuit);
            },
            nb::arg("other"),
            "Append all gates from another circuit to the end of this circuit.\n"
            "The other circuit's qubit count must be <= this circuit's qubit count.")

        // ── gate_count property ────────────────────────────────────────────
        .def_prop_ro("gate_count",
            [](const QrackCircuit& c) -> size_t {
                return c.circuit->GetGateList().size();
            },
            "Number of gates currently recorded in the circuit.")

        // ── num_qubits property ────────────────────────────────────────────
        .def_prop_ro("num_qubits",
            [](const QrackCircuit& c) { return c.numQubits; },
            "Number of qubits this circuit operates on.");
}
