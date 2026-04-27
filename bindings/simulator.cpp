#include "qrackbind_core.h"
#include "qalu.hpp"
#include <numeric>   // std::iota — used by exp_val_all
#if ENABLE_OPENCL
#include "common/oclengine.hpp"
#endif

// ── Constants needed for multi-control gates ─────────────────────────────────
static const Qrack::complex ONE_C  = Qrack::complex(1.0f, 0.0f);
static const Qrack::complex NEG1_C = Qrack::complex(-1.0f, 0.0f);
static const Qrack::complex I_C    = Qrack::complex(0.0f, 1.0f);
static const Qrack::complex NEG_I_C= Qrack::complex(0.0f, -1.0f);
static const Qrack::real1 SQRT1_2  = (Qrack::real1)std::sqrt(0.5f);

// H gate matrix: [1/√2, 1/√2, 1/√2, -1/√2]
static const Qrack::complex H_MTRX[4] = {
    {SQRT1_2, 0}, {SQRT1_2, 0}, {SQRT1_2, 0}, {-SQRT1_2, 0}
};

// ── Phase 3 type aliases ─────────────────────────────────────────────────────
// The ndarray element type follows Qrack's compile-time `real1` precision.
// On the default Qrack build (FPPOW=5) `real1 == float`, so cf_t is
// std::complex<float> → numpy.complex64 and r_t is float → numpy.float32.
// On a double-precision build, both widen automatically and Python sees
// complex128 / float64. Half / float128 builds are not supported here.
using r_t  = Qrack::real1;
using cf_t = std::complex<r_t>;
static_assert(sizeof(cf_t) == sizeof(Qrack::complex),
    "cf_t and Qrack::complex must be layout-compatible for reinterpret_cast");

// Detect whether the running process actually has at least one usable
// OpenCL device. Qrack will happily emit an "OCL stack" config even when
// the host has no OpenCL runtime, but the resulting simulator silently
// returns zero amplitudes for entangled states. Any caller that asks for
// `isOpenCL=True` on such a host is transparently downgraded to a CPU-only
// stack.
static bool runtime_has_opencl()
{
#if ENABLE_OPENCL
    try {
        return Qrack::OCLEngine::Instance().GetDeviceCount() > 0;
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

QInterfacePtr make_simulator(bitLenInt n, SimConfig c) {
    // Honour the user's request to enable OpenCL, but only if the host
    // actually has a runtime device. Without this clamp, the QPager /
    // QHybrid / QEngineOCL layers in the default stack produce silent
    // zero-amplitude bugs on entangled states.
    if (c.isOpenCL && !runtime_has_opencl()) {
        c.isOpenCL       = false;
        c.isCpuGpuHybrid = false;
        c.isPaged        = false;
    }

    // Qrack 10.6.2 QPager workaround: QPager in the stack — whether atop
    // QHybrid/OpenCL on a GPU host or QEngineCPU on a CPU-only host —
    // silently produces zero amplitudes for entangled states (Bell, GHZ,
    // …), breaks `CPOWModNOut`, and segfaults from `SetAmplitude` when a
    // direct amplitude write reaches QEngineCPU through the QPager
    // dispatch. Per-permutation queries (`GetAmplitude`, `ProbAll`) and
    // sampling (`MultiShotMeasureMask`) all read consistent values, but
    // anything that materialises the full state vector or mutates a
    // single amplitude collapses. Drop QPager unconditionally on this
    // Qrack release. Users who explicitly opt-in via `isPaged=True` are
    // accepting the risk — but for almost all qubit counts the QHybrid
    // (or pure QEngineCPU) layer alone is faster anyway.
    c.isPaged = false;

    std::vector<QInterfaceEngine> stack;

    if (c.isTensorNetwork)
        stack.push_back(QINTERFACE_TENSOR_NETWORK);

    if (c.isSchmidtDecompose)
        stack.push_back(QINTERFACE_QUNIT);

    if (c.isStabilizerHybrid)
        stack.push_back(QINTERFACE_STABILIZER_HYBRID);

    if (c.isBinaryDecisionTree) {
        stack.push_back(QINTERFACE_BDT);
    } else {
        if (c.isPaged && c.isOpenCL)
            stack.push_back(QINTERFACE_QPAGER);
        if (c.isCpuGpuHybrid && c.isOpenCL)
            stack.push_back(QINTERFACE_HYBRID);
        else if (c.isOpenCL)
            stack.push_back(QINTERFACE_OPENCL);
        else
            stack.push_back(QINTERFACE_CPU);
    }

    if (stack.empty())
        stack.push_back(QINTERFACE_OPTIMAL);

    return CreateQuantumInterface(
        stack,
        n,
        /*initState=*/  0,
        /*rgp=*/        nullptr,
        /*phaseFac=*/   CMPLX_DEFAULT_ARG,
        /*doNorm=*/     false,
        // randomGP=false: keep the ground-state amplitude exactly 1+0j
        // rather than e^{iφ} for some random φ. The randomized global
        // phase is unobservable physically but breaks naive comparisons
        // against a reference state vector (sv[0] == 1) and complicates
        // user code that expects |0…0⟩ to look like the textbook
        // computational-basis state. Users who need a randomized global
        // phase can call `update_running_norm()` after applying any
        // initial gates, or compose with their own phase rotation.
        /*randomGP=*/   false,
        /*useHostMem=*/ c.isHostPointer,
        /*deviceId=*/   -1,
        /*useHWRNG=*/   true,
        /*isSparse=*/   c.isSparse
    );
}

QrackSim::QrackSim(bitLenInt n, const SimConfig& cfg)
    : numQubits(n), config(cfg), sim(make_simulator(n, cfg))
{
    if (!sim) throw QrackError("QrackSimulator: factory returned null");
}

// Clone constructor — used by .clone() / __copy__ / __deepcopy__
QrackSim::QrackSim(const QrackSim& src)
    : numQubits(src.numQubits)
    , config(src.config)
    , sim(src.sim->Clone())
{
    if (!sim) throw QrackError("QrackSimulator: Clone() returned null");
}

void QrackSim::check_qubit(bitLenInt q, const char* method) const {
    if (q >= numQubits) {
        throw QrackError(
            std::string(method) + ": qubit index " + std::to_string(q) +
            " is out of range [0, " + std::to_string(numQubits) + ")"
            " (simulator has " + std::to_string(numQubits) + " qubits)",
            QrackErrorKind::QubitOutOfRange);
    }
}

std::string QrackSim::repr() const {
    return "QrackSimulator(qubits=" + std::to_string(numQubits) + ")";
}

static void check_arithmetic(const QrackSim& s, const char* method)
{
    if (s.config.isTensorNetwork)
        throw QrackError(
            std::string(method) + ": isTensorNetwork=True is incompatible with "
            "arithmetic gates. Construct with isTensorNetwork=False.");
}

// ── Pauli-basis measurement helper ───────────────────────────────────────────
// QInterface has no single "MeasurePauli" entry point — we manually rotate
// the qubit into the computational basis, measure, and rotate back. The
// rotation table mirrors pyqrack so behaviour is identical:
//
//   Basis   | rotate before M       | rotate back
//   --------|-----------------------|----------------
//   PauliI  | none                  | none
//   PauliX  | H                     | H
//   PauliY  | S†, H   (= SH adjoint)| H, S
//   PauliZ  | none                  | none
//
// Returns true for the +1 eigenvalue, false for -1.
static bool measure_in_basis(QrackSim& s, Qrack::Pauli basis, bitLenInt q)
{
    // The identity operator has trivial eigenvalue +1 — there is no
    // observable to measure, so we short-circuit before touching the
    // state. This matches pyqrack's behaviour and means
    // measure_pauli(PauliI, q) is a no-op that returns False.
    if (basis == Qrack::Pauli::PauliI)
        return false;

    switch (basis) {
        case Qrack::Pauli::PauliX:
            s.sim->H(q);
            break;
        case Qrack::Pauli::PauliY:
            s.sim->IS(q);  // S†
            s.sim->H(q);
            break;
        case Qrack::Pauli::PauliZ:
        default:
            break;
    }

    const bool result = s.sim->M(q);

    switch (basis) {
        case Qrack::Pauli::PauliX:
            s.sim->H(q);
            break;
        case Qrack::Pauli::PauliY:
            s.sim->H(q);
            s.sim->S(q);
            break;
        case Qrack::Pauli::PauliZ:
        default:
            break;
    }

    return result;
}


// Helper macro to reduce boilerplate
#define GATE1(pyname, cppfn, doc) \
    .def(pyname, [](QrackSim& s, bitLenInt q) { \
        s.check_qubit(q, pyname); s.sim->cppfn(q); }, \
        nb::arg("qubit"), doc)

#define RGATE(pyname, cppfn, doc) \
    .def(pyname, [](QrackSim& s, real1_f angle, bitLenInt q) { \
        s.check_qubit(q, pyname); s.sim->cppfn(angle, q); }, \
        nb::arg("angle"), nb::arg("qubit"), doc)

// ── Binding ───────────────────────────────────────────────────────────────────
void bind_simulator(nb::module_& m) {

nb::class_<QrackSim>(m, "QrackSimulator",
    "Qrack quantum simulator.\n\n"
    "Simulator Cloning\n"
    "-----------------\n"
    "A simulator can be deep-copied — producing an independent simulator with\n"
    "the same quantum state and configuration — using either the ``clone()``\n"
    "method or the standard ``copy.deepcopy`` / ``copy.copy`` protocol::\n\n"
    "    import copy\n"
    "    original = QrackSimulator(qubitCount=4)\n"
    "    original.h(0)\n"
    "    original.cnot(0, 1)              # Bell state on qubits 0 and 1\n\n"
    "    branch_a = original.clone()      # explicit\n"
    "    branch_b = copy.deepcopy(original)   # protocol-driven\n\n"
    "After construction, the clone is fully independent — gates applied to one\n"
    "have no effect on the other. The clone inherits the source simulator's\n"
    "qubit count and configuration.\n\n"
    "Typical use case is mid-circuit branching — capturing the state at a\n"
    "decision point so that multiple continuations can be explored without\n"
    "re-running the expensive state preparation."
)
    .def("__init__",
        [](QrackSim* self,
        bitLenInt qubitCount,
        bool isTensorNetwork,
        bool isSchmidtDecompose,
        bool isSchmidtDecomposeMulti,
        bool isStabilizerHybrid,
        bool isBinaryDecisionTree,
        bool isPaged,
        bool isCpuGpuHybrid,
        bool isOpenCL,
        bool isHostPointer,
        bool isSparse,
        real1_f noise)
        {
            SimConfig cfg{
                isTensorNetwork,
                isSchmidtDecompose,
                isSchmidtDecomposeMulti,
                isStabilizerHybrid,
                isBinaryDecisionTree,
                isPaged,
                isCpuGpuHybrid,
                isOpenCL,
                isHostPointer,
                isSparse,
                noise > 0.0f
            };
            new (self) QrackSim(static_cast<bitLenInt>(qubitCount), cfg);
        },
        nb::arg("qubitCount"),
        nb::arg("isTensorNetwork")         = true,
        nb::arg("isSchmidtDecompose")      = true,
        nb::arg("isSchmidtDecomposeMulti") = false,
        nb::arg("isStabilizerHybrid")      = false,
        nb::arg("isBinaryDecisionTree")    = false,
        nb::arg("isPaged")                 = true,
        nb::arg("isCpuGpuHybrid")          = true,
        nb::arg("isOpenCL")                = true,
        nb::arg("isHostPointer")           = false,
        nb::arg("isSparse")                = false,
        nb::arg("noise")                   = 0.0f
        )

        .def("__repr__", &QrackSim::repr)

        // ── Gates ─────────────────────────────────────────────────────────
        GATE1("h",        H,   "Hadamard gate.")
        GATE1("x",        X,   "Pauli X (bit flip) gate.")
        GATE1("y",        Y,   "Pauli Y gate.")
        GATE1("z",        Z,   "Pauli Z (phase flip) gate.")
        GATE1("s",        S,   "S gate — phase shift π/2.")
        GATE1("t",        T,   "T gate — phase shift π/4.")
        GATE1("sdg",      IS,  "S† (inverse S) gate.")
        GATE1("tdg",      IT,  "T† (inverse T) gate.")
        GATE1("sx",       SqrtX,  "√X gate (half-X). Native Qiskit basis gate.")
        GATE1("sxdg",     ISqrtX, "√X† gate (inverse √X).")
        RGATE("rx",  RX, "Rotate around X axis by angle radians. Equiv: exp(-i·angle/2·X).")
        RGATE("ry",  RY, "Rotate around Y axis by angle radians.")
        RGATE("rz",  RZ, "Rotate around Z axis by angle radians.")
        RGATE("r1",  RT, "Phase rotation: apply e^(i·angle) to |1> state.")

        .def("u",
            [](QrackSim& s, real1_f theta, real1_f phi, real1_f lam, bitLenInt q) {
                s.check_qubit(q, "u");
                s.sim->U(q, theta, phi, lam);
            },
            nb::arg("theta"), nb::arg("phi"), nb::arg("lam"), nb::arg("qubit"),
            "General single-qubit unitary: U(θ,φ,λ). Decomposes to RZ·RY·RZ.")

        .def("u2",
            [](QrackSim& s, real1_f phi, real1_f lam, bitLenInt q) {
                s.check_qubit(q, "u2");
                s.sim->U(q, M_PI / 2.0f, phi, lam);
            },
            nb::arg("phi"), nb::arg("lam"), nb::arg("qubit"),
            "U2 gate: U(π/2, φ, λ).")

        // Two-qubit convenience gates
        .def("cnot",
            [](QrackSim& s, bitLenInt ctrl, bitLenInt tgt) {
                s.check_qubit(ctrl, "cnot"); s.check_qubit(tgt, "cnot");
                s.sim->CNOT(ctrl, tgt);
            },
            nb::arg("control"), nb::arg("target"),
            "Controlled-NOT (CNOT / CX) gate.")

        .def("cy",
            [](QrackSim& s, bitLenInt ctrl, bitLenInt tgt) {
                s.check_qubit(ctrl, "cy"); s.check_qubit(tgt, "cy");
                s.sim->CY(ctrl, tgt);
            },
            nb::arg("control"), nb::arg("target"), "Controlled-Y gate.")

        .def("cz",
            [](QrackSim& s, bitLenInt ctrl, bitLenInt tgt) {
                s.check_qubit(ctrl, "cz"); s.check_qubit(tgt, "cz");
                s.sim->CZ(ctrl, tgt);
            },
            nb::arg("control"), nb::arg("target"), "Controlled-Z gate.")

        .def("swap",
            [](QrackSim& s, bitLenInt q1, bitLenInt q2) {
                s.check_qubit(q1, "swap"); s.check_qubit(q2, "swap");
                s.sim->Swap(q1, q2);
            },
            nb::arg("qubit1"), nb::arg("qubit2"), "SWAP gate.")

        .def("iswap",
            [](QrackSim& s, bitLenInt q1, bitLenInt q2) {
                s.check_qubit(q1, "iswap"); s.check_qubit(q2, "iswap");
                s.sim->ISwap(q1, q2);
            },
            nb::arg("qubit1"), nb::arg("qubit2"), "iSWAP gate.")

        .def("ccnot",
            [](QrackSim& s, bitLenInt c1, bitLenInt c2, bitLenInt tgt) {
                s.sim->CCNOT(c1, c2, tgt);
            },
            nb::arg("control1"), nb::arg("control2"), nb::arg("target"),
            "Toffoli (CCX / CCNOT) gate.") 

        // ── mcx / macx  (X = Invert with phase [1,1]) ────────────────────────────────
        .def("mcx",
            [](QrackSim& s, std::vector<bitLenInt> controls, bitLenInt tgt) {
                s.sim->MCInvert(controls, ONE_C, ONE_C, tgt);
            },
            nb::arg("controls"), nb::arg("target"),
            "Multiply-controlled X. Fires when all controls are |1>.")

        .def("macx",
            [](QrackSim& s, std::vector<bitLenInt> controls, bitLenInt tgt) {
                s.sim->MACInvert(controls, ONE_C, ONE_C, tgt);
            },
            nb::arg("controls"), nb::arg("target"),
            "Anti-controlled X. Fires when all controls are |0>.")

        // ── mcy / macy  (Y = Invert with phase [-i, i]) ──────────────────────────────
        .def("mcy",
            [](QrackSim& s, std::vector<bitLenInt> controls, bitLenInt tgt) {
                s.sim->MCInvert(controls, NEG_I_C, I_C, tgt);
            },
            nb::arg("controls"), nb::arg("target"), "Multiply-controlled Y.")

        .def("macy",
            [](QrackSim& s, std::vector<bitLenInt> controls, bitLenInt tgt) {
                s.sim->MACInvert(controls, NEG_I_C, I_C, tgt);
            },
            nb::arg("controls"), nb::arg("target"), "Anti-controlled Y.")

        // ── mcz / macz  (Z = Phase [1, -1]) ─────────────────────────────────────────
        .def("mcz",
            [](QrackSim& s, std::vector<bitLenInt> controls, bitLenInt tgt) {
                s.sim->MCPhase(controls, ONE_C, NEG1_C, tgt);
            },
            nb::arg("controls"), nb::arg("target"), "Multiply-controlled Z.")

        .def("macz",
            [](QrackSim& s, std::vector<bitLenInt> controls, bitLenInt tgt) {
                s.sim->MACPhase(controls, ONE_C, NEG1_C, tgt);
            },
            nb::arg("controls"), nb::arg("target"), "Anti-controlled Z.")

        // ── mch  (H = MCMtrx with Hadamard matrix) ───────────────────────────────────
        .def("mch",
            [](QrackSim& s, std::vector<bitLenInt> controls, bitLenInt tgt) {
                s.sim->MCMtrx(controls, H_MTRX, tgt);
            },
            nb::arg("controls"), nb::arg("target"), "Multiply-controlled H.")

        // ── mcrz  (RZ = Phase [exp(-iθ/2), exp(iθ/2)]) ──────────────────────────────
        .def("mcrz",
            [](QrackSim& s, real1_f angle, std::vector<bitLenInt> controls, bitLenInt tgt) {
                const Qrack::real1 half = (Qrack::real1)(angle / 2.0f);
                const Qrack::complex phase0 = std::exp(Qrack::complex(0.0f, -half));
                const Qrack::complex phase1 = std::exp(Qrack::complex(0.0f,  half));
                s.sim->MCPhase(controls, phase0, phase1, tgt);
            },
            nb::arg("angle"), nb::arg("controls"), nb::arg("target"),
            "Multiply-controlled RZ.")

        // ── mcu  (arbitrary multi-controlled U gate) ─────────────────────────────────
        .def("mcu",
            [](QrackSim& s, std::vector<bitLenInt> controls, bitLenInt tgt,
            real1_f theta, real1_f phi, real1_f lam) {
                s.sim->CU(controls, tgt, theta, phi, lam);
            },
            nb::arg("controls"), nb::arg("target"),
            nb::arg("theta"), nb::arg("phi"), nb::arg("lam"),
            "Multiply-controlled U(θ,φ,λ) gate.")

        .def("mtrx",
            [](QrackSim& s,
            std::vector<std::complex<float>> m,
            bitLenInt q)
            {
                if (m.size() < 4)
                    throw QrackError("mtrx: matrix must have 4 elements",
                                     QrackErrorKind::InvalidArgument);
                s.check_qubit(q, "mtrx");
                s.sim->Mtrx(m.data(), q);
            },
            nb::arg("matrix"), nb::arg("qubit"),
            "Apply arbitrary 2x2 unitary. matrix is [m00, m01, m10, m11] row-major.")

        .def("mcmtrx",
            [](QrackSim& s,
            std::vector<bitLenInt> controls,
            std::vector<std::complex<float>> m,
            bitLenInt q)
            {
                if (m.size() < 4)
                    throw QrackError("mcmtrx: matrix must have 4 elements",
                                     QrackErrorKind::InvalidArgument);
                // Signature: MCMtrx(const std::vector<bitLenInt>& controls,
                //                   const complex* mtrx, bitLenInt target)
                // Pass vector directly — do NOT use controls.size(), controls.data()
                s.sim->MCMtrx(controls, m.data(), q);
            },
            nb::arg("controls"), nb::arg("matrix"), nb::arg("qubit"),
            "Multiply-controlled arbitrary 2x2 unitary.")

        .def("macmtrx",
            [](QrackSim& s,
            std::vector<bitLenInt> controls,
            std::vector<std::complex<float>> m,
            bitLenInt q)
            {
                if (m.size() < 4)
                    throw QrackError("macmtrx: matrix must have 4 elements",
                                     QrackErrorKind::InvalidArgument);
                s.sim->MACMtrx(controls, m.data(), q);
            },
            nb::arg("controls"), nb::arg("matrix"), nb::arg("qubit"),
            "Anti-controlled arbitrary 2x2 unitary.")

        // ── multiplex1_mtrx  (uniformly-controlled arbitrary unitary) ────────────────
        // Maps to QInterface::UniformlyControlledSingleBit.
        // mtrxs is a flat array of 4 * 2^len(controls) complex values — one 2x2 matrix
        // per control permutation, in row-major order. Used by Bloqade's QASM2 interpreter.
        .def("multiplex1_mtrx",
            [](QrackSim& s,
            std::vector<bitLenInt> controls,
            std::vector<std::complex<float>> mtrxs,
            bitLenInt tgt)
            {
                const size_t expected = 4ULL << controls.size();
                if (mtrxs.size() < expected)
                    throw QrackError(
                        "multiplex1_mtrx: mtrxs must have at least 4 * 2^len(controls) = " +
                        std::to_string(expected) + " elements",
                        QrackErrorKind::InvalidArgument);
                s.check_qubit(tgt, "multiplex1_mtrx");
                s.sim->UniformlyControlledSingleBit(controls, tgt,
                    reinterpret_cast<const Qrack::complex*>(mtrxs.data()));
            },
            nb::arg("controls"), nb::arg("mtrxs"), nb::arg("target"),
            "Uniformly-controlled single-qubit gate. mtrxs is a flat list of "
            "4 * 2**len(controls) complex values — one 2x2 unitary per control permutation, "
            "in row-major order.")

        // ── Pauli observables (Phase 4) ──────────────────────────────────
        // The Pauli enum is registered in qrackbind_ext.cpp with
        // nb::is_arithmetic(), so callers may pass either Pauli members
        // or the underlying integer codes (Qrack's convention is
        // PauliI=0, PauliX=1, PauliZ=2, PauliY=3 — note non-sequential).
        .def("measure_pauli",
            [](QrackSim& s, Qrack::Pauli basis, bitLenInt q) -> bool {
                s.check_qubit(q, "measure_pauli");
                return measure_in_basis(s, basis, q);
            },
            nb::arg("basis"), nb::arg("qubit"),
            "Measure a qubit in the specified Pauli basis.\n\n"
            "Rotates the qubit into the computational basis, measures, and\n"
            "rotates back. Returns the same bit-valued bool as :meth:`measure`:\n"
            "``True`` if the rotated qubit collapsed to ``|1>``, ``False`` for\n"
            "``|0>``. For Pauli Z, this means ``True`` ↔ −1 eigenvalue and\n"
            "``False`` ↔ +1 eigenvalue. The state is collapsed in the chosen\n"
            "basis.\n\n"
            "Example::\n\n"
            "    sim.x(0)\n"
            "    sim.measure_pauli(Pauli.PauliZ, 0)  # → True (|1>)")

        .def("exp_val",
            [](QrackSim& s, Qrack::Pauli basis, bitLenInt q) -> real1_f {
                s.check_qubit(q, "exp_val");
                return s.sim->ExpectationPauliAll({q}, {basis});
            },
            nb::arg("basis"), nb::arg("qubit"),
            "Single-qubit Pauli expectation value.\n\n"
            "Equivalent to ``exp_val_pauli([basis], [qubit])``. Result is\n"
            "in [-1.0, +1.0]. Does not collapse the state.\n\n"
            "Example::\n\n"
            "    sim.h(0)\n"
            "    print(sim.exp_val(Pauli.PauliX, 0))  # → 1.0")

        .def("exp_val_pauli",
            [](QrackSim& s,
               std::vector<Qrack::Pauli> paulis,
               std::vector<bitLenInt>    qubits) -> real1_f
            {
                if (paulis.size() != qubits.size())
                    throw QrackError(
                        "exp_val_pauli: paulis and qubits must have the same length",
                        QrackErrorKind::InvalidArgument);
                for (auto q : qubits)
                    s.check_qubit(q, "exp_val_pauli");
                return s.sim->ExpectationPauliAll(qubits, paulis);
            },
            nb::arg("paulis"), nb::arg("qubits"),
            "Expectation value of a Pauli tensor product observable.\n\n"
            "Returns <ψ|P₀⊗P₁⊗…⊗Pₙ|ψ> where each Pᵢ is a Pauli operator\n"
            "acting on the corresponding qubit. Result is in [-1.0, +1.0].\n"
            "Does not collapse the state.\n\n"
            "``paulis`` and ``qubits`` must have equal length.\n\n"
            "Example::\n\n"
            "    # Measure <ZZ> on a Bell state — should be +1\n"
            "    sim.h(0); sim.cnot(0, 1)\n"
            "    sim.exp_val_pauli([Pauli.PauliZ, Pauli.PauliZ], [0, 1])")

        .def("variance_pauli",
            [](QrackSim& s,
               std::vector<Qrack::Pauli> paulis,
               std::vector<bitLenInt>    qubits) -> real1_f
            {
                if (paulis.size() != qubits.size())
                    throw QrackError(
                        "variance_pauli: paulis and qubits must have the same length",
                        QrackErrorKind::InvalidArgument);
                for (auto q : qubits)
                    s.check_qubit(q, "variance_pauli");
                return s.sim->VariancePauliAll(qubits, paulis);
            },
            nb::arg("paulis"), nb::arg("qubits"),
            "Variance of a Pauli tensor product observable.\n\n"
            "For a Pauli operator P (P² = I), Var(P) = 1 − <P>².\n"
            "Result is in [0.0, 1.0]. Does not collapse the state.\n\n"
            "Eigenstates have variance 0; maximally-mixed states have\n"
            "variance 1.")

        .def("exp_val_all",
            [](QrackSim& s, Qrack::Pauli basis) -> real1_f {
                std::vector<bitLenInt> qubits(s.numQubits);
                std::iota(qubits.begin(), qubits.end(), bitLenInt(0));
                std::vector<Qrack::Pauli> paulis(s.numQubits, basis);
                return s.sim->ExpectationPauliAll(qubits, paulis);
            },
            nb::arg("basis"),
            "Expectation value of the same Pauli operator applied to every\n"
            "qubit. Equivalent to::\n\n"
            "    sim.exp_val_pauli([basis] * sim.num_qubits,\n"
            "                      list(range(sim.num_qubits)))")

        .def("exp_val_floats",
            [](QrackSim& s,
               std::vector<bitLenInt> qubits,
               std::vector<float>     weights) -> real1_f
            {
                // Qrack requires exactly two weights per qubit: weights
                // [2*i] is qubit i's classical eigenvalue for |0>, and
                // weights[2*i+1] is its eigenvalue for |1>. The header
                // signature looks "bits + weights" but the runtime check
                // is `weights.size() >= 2 * bits.size()`.
                if (weights.size() != 2 * qubits.size())
                    throw QrackError(
                        "exp_val_floats: weights must contain exactly 2 entries "
                        "per qubit (weights[2*i] for |0>, weights[2*i+1] for |1>)",
                        QrackErrorKind::InvalidArgument);
                for (auto q : qubits)
                    s.check_qubit(q, "exp_val_floats");
                std::vector<Qrack::real1_f> w(weights.begin(), weights.end());
                return s.sim->ExpectationFloatsFactorized(qubits, w);
            },
            nb::arg("qubits"), nb::arg("weights"),
            "Expectation value of a weighted single-qubit observable.\n\n"
            "Each qubit gets two classical eigenvalues — one for ``|0>`` and\n"
            "one for ``|1>``. ``weights`` must have length ``2 * len(qubits)``::\n\n"
            "    weights = [w0_for_|0>, w0_for_|1>,\n"
            "               w1_for_|0>, w1_for_|1>,\n"
            "               ...]\n\n"
            "Returns ``Σᵢ (wᵢ⁰ · P(qᵢ=|0>) + wᵢ¹ · P(qᵢ=|1>))``. Used by\n"
            "PennyLane's Hamiltonian expectation-value path.")

        .def("variance_floats",
            [](QrackSim& s,
               std::vector<bitLenInt> qubits,
               std::vector<float>     weights) -> real1_f
            {
                if (weights.size() != 2 * qubits.size())
                    throw QrackError(
                        "variance_floats: weights must contain exactly 2 entries "
                        "per qubit (weights[2*i] for |0>, weights[2*i+1] for |1>)",
                        QrackErrorKind::InvalidArgument);
                for (auto q : qubits)
                    s.check_qubit(q, "variance_floats");
                std::vector<Qrack::real1_f> w(weights.begin(), weights.end());
                return s.sim->VarianceFloatsFactorized(qubits, w);
            },
            nb::arg("qubits"), nb::arg("weights"),
            "Variance of a weighted single-qubit observable. Symmetric\n"
            "counterpart to :meth:`exp_val_floats`. ``weights`` must have\n"
            "length ``2 * len(qubits)`` (see :meth:`exp_val_floats`).")

        // ── Deferred Phase 4: arbitrary unitary observables ─────────────
        .def("exp_val_unitary",
            [](QrackSim& s,
               std::vector<bitLenInt> qubits,
               std::vector<std::complex<float>> basisOps,
               std::vector<float> eigenVals) -> float
            {
                if (basisOps.size() != qubits.size() * 4)
                    throw QrackError(
                        "exp_val_unitary: basisOps must have 4 * len(qubits) elements",
                        QrackErrorKind::InvalidArgument);
                for (auto q : qubits)
                    s.check_qubit(q, "exp_val_unitary");

                std::vector<std::shared_ptr<Qrack::complex>> ops;
                ops.reserve(qubits.size());
                for (size_t i = 0; i < qubits.size(); ++i) {
                    std::shared_ptr<Qrack::complex> m(
                        new Qrack::complex[4], std::default_delete<Qrack::complex[]>());
                    const auto* src = reinterpret_cast<const Qrack::complex*>(&basisOps[i * 4]);
                    std::copy(src, src + 4, m.get());
                    ops.push_back(m);
                }

                std::vector<Qrack::real1_f> ev(eigenVals.begin(), eigenVals.end());
                return static_cast<float>(
                    s.sim->ExpectationUnitaryAll(qubits, ops, ev));
            },
            nb::arg("qubits"), nb::arg("basis_ops"),
            nb::arg("eigen_vals") = std::vector<float>{},
            "Expectation value of a tensor product of arbitrary 2x2 unitary\n"
            "observables. ``basis_ops`` is a flat list of ``4 * len(qubits)``\n"
            "complex values — one 2x2 matrix per qubit, in row-major order.")

        .def("variance_unitary",
            [](QrackSim& s,
               std::vector<bitLenInt> qubits,
               std::vector<std::complex<float>> basisOps,
               std::vector<float> eigenVals) -> float
            {
                if (basisOps.size() != qubits.size() * 4)
                    throw QrackError(
                        "variance_unitary: basisOps must have 4 * len(qubits) elements",
                        QrackErrorKind::InvalidArgument);
                for (auto q : qubits)
                    s.check_qubit(q, "variance_unitary");

                std::vector<std::shared_ptr<Qrack::complex>> ops;
                ops.reserve(qubits.size());
                for (size_t i = 0; i < qubits.size(); ++i) {
                    std::shared_ptr<Qrack::complex> m(
                        new Qrack::complex[4], std::default_delete<Qrack::complex[]>());
                    const auto* src = reinterpret_cast<const Qrack::complex*>(&basisOps[i * 4]);
                    std::copy(src, src + 4, m.get());
                    ops.push_back(m);
                }

                std::vector<Qrack::real1_f> ev(eigenVals.begin(), eigenVals.end());
                return static_cast<float>(
                    s.sim->VarianceUnitaryAll(qubits, ops, ev));
            },
            nb::arg("qubits"), nb::arg("basis_ops"),
            nb::arg("eigen_vals") = std::vector<float>{},
            "Variance of a tensor product of arbitrary 2x2 unitary observables.\n"
            "See :meth:`exp_val_unitary` for parameter conventions.")

        .def("exp_val_bits_factorized",
            [](QrackSim& s,
               std::vector<bitLenInt> qubits,
               std::vector<bitCapInt> perms) -> float
            {
                if (perms.size() < 2 * qubits.size())
                    throw QrackError(
                        "exp_val_bits_factorized: perms must contain at least 2 entries per qubit",
                        QrackErrorKind::InvalidArgument);
                for (auto q : qubits)
                    s.check_qubit(q, "exp_val_bits_factorized");
                // Qrack's ExpectationBitsFactorized takes
                // std::vector<BigInteger> (the compiled bitCapInt),
                // not our uint64_t typedef. Convert explicitly.
                std::vector<BigInteger> bPerms;
                bPerms.reserve(perms.size());
                for (auto p : perms)
                    bPerms.push_back(BigInteger(p));
                return static_cast<float>(
                    s.sim->ExpectationBitsFactorized(qubits, bPerms));
            },
            nb::arg("qubits"), nb::arg("perms"),
            "Per-qubit weighted expectation value using bitCapInt permutation\n"
            "weights. Low-level API used by Shor's and arithmetic expectation\n"
            "paths.")


        // ── Measurement ───────────────────────────────────────────────────
        .def("measure",
            [](QrackSim& s, bitLenInt q) -> bool {
                s.check_qubit(q, "measure");
                return s.sim->M(q);
            },
            nb::arg("qubit"),
            "Measure qubit. Returns True=|1>, False=|0>. Collapses state.")

        .def("measure_all",
            [](QrackSim& s) -> std::vector<bool> {
                std::vector<bool> out;
                out.reserve(s.numQubits);
                for (bitLenInt i = 0; i < s.numQubits; i++)
                    out.push_back(s.sim->M(i));
                return out;
            },
            "Measure all qubits. Returns list[bool], LSB first.")

        .def("force_measure",
            [](QrackSim& s, bitLenInt q, bool result) -> bool {
                s.check_qubit(q, "force_measure");
                return s.sim->ForceM(q, result);
            },
            nb::arg("qubit"), nb::arg("result"),
            "Force measurement outcome. Projects state to result without random draw.")

        .def("prob",
            [](QrackSim& s, bitLenInt q) -> real1_f {
                s.check_qubit(q, "prob");
                return s.sim->Prob(q);
            },
            nb::arg("qubit"),
            "Probability of |1> for qubit. Does NOT collapse state.")

        .def("prob_all",
            [](QrackSim& s) -> std::vector<real1_f> {
                std::vector<real1_f> out(s.numQubits);
                for (bitLenInt i = 0; i < s.numQubits; i++)
                    out[i] = s.sim->Prob(i);
                return out;
            },
            "Per-qubit |1> probabilities for all qubits. Does NOT collapse state.")


        // ── allocators ─────────────────────────────────────────────────
        .def("allocate",
            [](QrackSim& s, bitLenInt start, bitLenInt length) -> bitLenInt {
                bitLenInt offset = s.sim->Allocate(start, length);
                s.numQubits = s.sim->GetQubitCount();   // sync after allocation
                return offset;
            },
            nb::arg("start"), nb::arg("length"),
            "Allocate 'length' new |0> qubits at index 'start'. Returns the start offset.\n"
            "Existing qubits at >= start shift up. Updates num_qubits automatically.\n"
            "Incompatible with isTensorNetwork=True.")

        .def("dispose",
            [](QrackSim& s, bitLenInt start, bitLenInt length) {
                s.sim->Dispose(start, length);
                s.numQubits = s.sim->GetQubitCount();   // sync after disposal
            },
            nb::arg("start"), nb::arg("length"),
            "Remove 'length' qubits starting at 'start'. Qubits must be separably |0> or |1>.\n"
            "Updates num_qubits automatically.")

        // Convenience: allocate at end (most common case)
        .def("allocate_qubits",
            [](QrackSim& s, bitLenInt n) -> bitLenInt {
                bitLenInt offset = s.sim->Allocate(n);   // Allocate(length) appends at end
                s.numQubits = s.sim->GetQubitCount();
                return offset;
            },
            nb::arg("n"),
            "Allocate n new |0> qubits at the end. Returns the index of the first new qubit.")


        // ── Quantum Fourier Transform ─────────────────────
        .def("qft",
            [](QrackSim& s, bitLenInt start, bitLenInt length, bool trySeparate) {
                s.sim->QFT(start, length, trySeparate);
            },
            nb::arg("start"), nb::arg("length"), nb::arg("try_separate") = false,
            "Quantum Fourier Transform on a contiguous register [start, start+length).\n"
            "try_separate: optimization hint for QUnit — set True if you expect a permutation\n"
            "basis eigenstate result; otherwise leave False.")

        .def("iqft",
            [](QrackSim& s, bitLenInt start, bitLenInt length, bool trySeparate) {
                s.sim->IQFT(start, length, trySeparate);
            },
            nb::arg("start"), nb::arg("length"), nb::arg("try_separate") = false,
            "Inverse Quantum Fourier Transform on a contiguous register.")

        // Random-access variants (non-contiguous qubit lists)
        .def("qftr",
            [](QrackSim& s, std::vector<bitLenInt> qubits, bool trySeparate) {
                s.sim->QFTR(qubits, trySeparate);
            },
            nb::arg("qubits"), nb::arg("try_separate") = false,
            "Quantum Fourier Transform on an arbitrary list of qubit indices.")

        .def("iqftr",
            [](QrackSim& s, std::vector<bitLenInt> qubits, bool trySeparate) {
                s.sim->IQFTR(qubits, trySeparate);
            },
            nb::arg("qubits"), nb::arg("try_separate") = false,
            "Inverse Quantum Fourier Transform on an arbitrary list of qubit indices.")


        // ── State control ─────────────────────────────────────────────────
        .def("reset_all",
            [](QrackSim& s) { s.sim->SetPermutation(0); },
            "Reset all qubits to |0...0>.")

        .def("m_reg",
            [](QrackSim& s, bitLenInt start, bitLenInt length) -> uint64_t {
                return static_cast<uint64_t>(s.sim->MReg(start, length));
            },
            nb::arg("start"), nb::arg("length"),
            "Measure a contiguous register of 'length' qubits starting at 'start'. "
            "Collapses state. Returns result as a classical integer.")

        .def("set_permutation",
            [](QrackSim& s, bitCapInt value) { s.sim->SetPermutation(value); },
            nb::arg("value"),
            "Reset state to the computational basis state |value>. "
            "Bit i of value sets qubit i.")

        // ── Register measurement ──────────────────────────────────────────
        // NOTE: Qrack's `bitCapInt` is a BigInteger when the library is
        // compiled for >64 qubits. The qrackbind public surface uses
        // uint64_t. We bridge with explicit uint64_t casts on the way out
        // and BigInteger construction on the way in.
        .def("measure_shots",
            [](QrackSim& s, std::vector<bitLenInt> qubits, unsigned shots)
                -> std::map<uint64_t, int>
            {
                // BigInteger is Qrack's wide-integer type; the bitCapInt
                // macro was #undef'd in qrackbind_core.h to expose plain
                // uint64_t to Python, but Qrack's compiled API still wants
                // BigInteger here.
                std::vector<BigInteger> qpowers;
                qpowers.reserve(qubits.size());
                for (auto q : qubits) {
                    s.check_qubit(q, "measure_shots");
                    qpowers.push_back(BigInteger(1) << q);
                }
                auto raw = s.sim->MultiShotMeasureMask(qpowers, shots);
                std::map<uint64_t, int> out;
                for (const auto& kv : raw)
                    out.emplace(static_cast<uint64_t>(kv.first), kv.second);
                return out;
            },
            nb::arg("qubits"), nb::arg("shots"),
            "Sample 'shots' measurements of 'qubits' without collapsing state. "
            "Returns dict[int, int]: measurement result -> count.")

        // ── Arithmetic opertors ────────────────────────────────────────────
        .def("add",
            [](QrackSim& s, bitCapInt val, bitLenInt start, bitLenInt length) {
                check_arithmetic(s, "add");
                s.sim->INC(val, start, length);
            },
            nb::arg("value"), nb::arg("start"), nb::arg("length"),
            "Add classical integer 'value' to the quantum register [start, start+length).")

        .def("sub",
            [](QrackSim& s, bitCapInt val, bitLenInt start, bitLenInt length) {
                check_arithmetic(s, "sub");
                s.sim->DEC(val, start, length);
            },
            nb::arg("value"), nb::arg("start"), nb::arg("length"),
            "Subtract classical integer 'value' from the quantum register.")

        .def("mul",
            [](QrackSim& s, bitCapInt toMul, bitCapInt modN,
            bitLenInt inStart, bitLenInt outStart, bitLenInt length) {
                check_arithmetic(s, "mul");
                s.sim->MULModNOut(toMul, modN, inStart, outStart, length);
            },
            nb::arg("to_mul"), nb::arg("mod_n"),
            nb::arg("in_start"), nb::arg("out_start"), nb::arg("length"),
            "Modular multiplication: out = in * to_mul mod mod_n (out of place).")

        .def("div",
            [](QrackSim& s, bitCapInt toDiv, bitCapInt modN,
            bitLenInt inStart, bitLenInt outStart, bitLenInt length) {
                check_arithmetic(s, "div");
                s.sim->IMULModNOut(toDiv, modN, inStart, outStart, length);
            },
            nb::arg("to_div"), nb::arg("mod_n"),
            nb::arg("in_start"), nb::arg("out_start"), nb::arg("length"),
            "Inverse modular multiplication (modular division, out of place).")

        .def("pown",
            [](QrackSim& s, bitCapInt base, bitCapInt modN,
            bitLenInt inStart, bitLenInt outStart, bitLenInt length) {
                check_arithmetic(s, "pown");
                // POWModNOut lives on QAlu, not QInterface. Cast through QAlu;
                // QUnit, QPager, QHybrid, QStabilizerHybrid, QBdt etc. all
                // expose it via QAlu inheritance or composition.
                auto alu = std::dynamic_pointer_cast<Qrack::QAlu>(s.sim);
                if (!alu)
                    throw QrackError(
                        "pown: current simulator stack does not implement "
                        "POWModNOut. Construct with isSchmidtDecompose=False "
                        "to fall through to a QAlu-capable engine, or disable "
                        "isTensorNetwork.");
                alu->POWModNOut(base, modN, inStart, outStart, length);
            },
            nb::arg("base"), nb::arg("mod_n"),
            nb::arg("in_start"), nb::arg("out_start"), nb::arg("length"),
            "Modular exponentiation: out = base^in mod mod_n (out of place). "
            "Central operation of Shor's algorithm.")

        .def("mcpown",
            [](QrackSim& s, bitCapInt base, bitCapInt modN,
            bitLenInt inStart, bitLenInt outStart, bitLenInt length,
            std::vector<bitLenInt> controls) {
                check_arithmetic(s, "mcpown");
                auto alu = std::dynamic_pointer_cast<Qrack::QAlu>(s.sim);
                if (!alu)
                    throw QrackError(
                        "mcpown: current simulator stack does not implement "
                        "CPOWModNOut.");
                alu->CPOWModNOut(base, modN, inStart, outStart, length, controls);
            },
            nb::arg("base"), nb::arg("mod_n"),
            nb::arg("in_start"), nb::arg("out_start"), nb::arg("length"),
            nb::arg("controls"),
            "Controlled modular exponentiation.")

        // Controlled variants
        .def("mcmul",
            [](QrackSim& s, bitCapInt toMul, bitCapInt modN,
            bitLenInt inStart, bitLenInt outStart, bitLenInt length,
            std::vector<bitLenInt> controls) {
                check_arithmetic(s, "mcmul");
                s.sim->CMULModNOut(toMul, modN, inStart, outStart, length, controls);
            },
            nb::arg("to_mul"), nb::arg("mod_n"),
            nb::arg("in_start"), nb::arg("out_start"), nb::arg("length"),
            nb::arg("controls"),
            "Controlled modular multiplication.")

        .def("mcdiv",
            [](QrackSim& s, bitCapInt toDiv, bitCapInt modN,
            bitLenInt inStart, bitLenInt outStart, bitLenInt length,
            std::vector<bitLenInt> controls) {
                check_arithmetic(s, "mcdiv");
                s.sim->CIMULModNOut(toDiv, modN, inStart, outStart, length, controls);
            },
            nb::arg("to_div"), nb::arg("mod_n"),
            nb::arg("in_start"), nb::arg("out_start"), nb::arg("length"),
            nb::arg("controls"),
            "Controlled modular division (inverse modular multiplication).")

        // ── Shift and Rotate ──────────────────────────────────────────────
        .def("lsl",
            [](QrackSim& s, bitLenInt shift, bitLenInt start, bitLenInt length) {
                s.sim->LSL(shift, start, length); },
            nb::arg("shift"), nb::arg("start"), nb::arg("length"),
            "Logical shift left — fills vacated bits with |0>.")

        .def("lsr",
            [](QrackSim& s, bitLenInt shift, bitLenInt start, bitLenInt length) {
                s.sim->LSR(shift, start, length); },
            nb::arg("shift"), nb::arg("start"), nb::arg("length"),
            "Logical shift right — fills vacated bits with |0>.")

        .def("rol",
            [](QrackSim& s, bitLenInt shift, bitLenInt start, bitLenInt length) {
                s.sim->ROL(shift, start, length); },
            nb::arg("shift"), nb::arg("start"), nb::arg("length"),
            "Circular rotate left.")

        .def("ror",
            [](QrackSim& s, bitLenInt shift, bitLenInt start, bitLenInt length) {
                s.sim->ROR(shift, start, length); },
            nb::arg("shift"), nb::arg("start"), nb::arg("length"),
            "Circular rotate right.")

        // ── State vector access (Phase 3) ────────────────────────────────
        // These are exposed as plain methods (with a leading underscore) and
        // wrapped as @property by the Python __init__.py. The reason: nanobind's
        // def_prop_ro forces rv_policy::reference_internal on the getter, which
        // is incompatible with capsule-owned ndarrays. Plain .def() lets us pass
        // rv_policy::move so the capsule retains lifetime ownership.
        //
        // Snapshot semantics. Each call allocates a fresh buffer, copies the
        // state into it, and hands ownership to a Python capsule. The
        // returned ndarray reflects the state *at the moment of the call*;
        // subsequent gates do not mutate it.
        // NOTE on implementation strategy
        // ───────────────────────────────
        // Qrack 10.6.2's QInterface::GetQuantumState / GetProbs /
        // GetReducedDensityMatrix paths through the default
        // QTensorNetwork → QUnit → QPager → QHybrid stack are observed to
        // silently leave the destination buffer untouched (all zeros) when
        // the state has not been "realized" — even though
        // GetAmplitude(perm) and ProbAll(perm) return correct values for
        // any individual basis state.
        //
        // Workaround: build the snapshot by per-permutation queries. This
        // is O(2^n) calls instead of one bulk copy, but it is correct on
        // every Qrack backend and dwarfed by the Python overhead at the
        // sizes (≤ ~20 qubits) where dense state extraction makes sense
        // anyway. For larger circuits, users should use prob_perm /
        // get_amplitude / measure_shots directly instead of materialising
        // the full vector.
        .def("_state_vector_impl",
            [](QrackSim& s) -> nb::ndarray<nb::numpy, cf_t, nb::shape<-1>>
            {
                const size_t n = size_t(1) << s.numQubits;
                cf_t* buf = new cf_t[n];
                for (size_t i = 0; i < n; i++) {
                    const Qrack::complex amp = s.sim->GetAmplitude(bitCapInt(i));
                    buf[i] = cf_t(amp.real(), amp.imag());
                }
                nb::capsule owner(buf, [](void* p) noexcept {
                    delete[] static_cast<cf_t*>(p);
                });
                return nb::ndarray<nb::numpy, cf_t, nb::shape<-1>>(
                    buf, {n}, owner);
            },
            nb::rv_policy::reference,
            "Internal: backing function for the ``state_vector`` property.")

        .def("_probabilities_impl",
            [](QrackSim& s) -> nb::ndarray<nb::numpy, r_t, nb::shape<-1>>
            {
                const size_t n = size_t(1) << s.numQubits;
                r_t* buf = new r_t[n];
                for (size_t i = 0; i < n; i++)
                    buf[i] = static_cast<r_t>(s.sim->ProbAll(bitCapInt(i)));
                nb::capsule owner(buf, [](void* p) noexcept {
                    delete[] static_cast<r_t*>(p);
                });
                return nb::ndarray<nb::numpy, r_t, nb::shape<-1>>(
                    buf, {n}, owner);
            },
            nb::rv_policy::reference,
            "Internal: backing function for the ``probabilities`` property.")

        .def("set_state_vector",
            [](QrackSim& s,
               nb::ndarray<nb::numpy, const cf_t, nb::ndim<1>, nb::c_contig> arr)
            {
                const size_t expected = size_t(1) << s.numQubits;
                if (arr.shape(0) != expected)
                    throw QrackError(
                        "set_state_vector: array length " +
                        std::to_string(arr.shape(0)) +
                        " does not match state space size " +
                        std::to_string(expected) +
                        " (2^" + std::to_string(s.numQubits) + ")",
                        QrackErrorKind::InvalidArgument);
                s.sim->SetQuantumState(
                    reinterpret_cast<const Qrack::complex*>(arr.data()));
            },
            nb::arg("state"),
            "Set the simulator's quantum state from a 1-D complex NumPy array.\n"
            "Array must be C-contiguous, have length 2**num_qubits, and use the\n"
            "build's complex dtype (complex64 by default). The array is copied\n"
            "into the simulator. SetQuantumState does NOT renormalise — call\n"
            "update_running_norm() afterwards if the input may not be unit-norm.")

        .def("get_amplitude",
            [](QrackSim& s, bitCapInt perm) -> std::complex<float>
            {
                const Qrack::complex amp = s.sim->GetAmplitude(perm);
                return std::complex<float>(
                    static_cast<float>(amp.real()),
                    static_cast<float>(amp.imag()));
            },
            nb::arg("index"),
            "Get the complex amplitude of a specific basis state by integer index.\n"
            "index must be in [0, 2**num_qubits). Does not collapse the state.")

        .def("set_amplitude",
            [](QrackSim& s, bitCapInt perm, std::complex<float> amp)
            {
                s.sim->SetAmplitude(perm,
                    Qrack::complex(static_cast<r_t>(amp.real()),
                                   static_cast<r_t>(amp.imag())));
            },
            nb::arg("index"), nb::arg("amplitude"),
            "Set the complex amplitude of a specific basis state.\n"
            "Does NOT re-normalise — call update_running_norm() if the resulting\n"
            "state may not be unit-norm.")

        .def("get_reduced_density_matrix",
            [](QrackSim& s, std::vector<bitLenInt> qubits)
                -> nb::ndarray<nb::numpy, cf_t, nb::shape<-1, -1>>
            {
                // Compute the reduced density matrix from per-amplitude
                // queries rather than calling Qrack's bulk
                // GetReducedDensityMatrix, which (like GetQuantumState)
                // is not reliable through the default simulator stack in
                // this Qrack release. See the "implementation strategy"
                // comment on _state_vector_impl above for context.
                for (auto q : qubits)
                    s.check_qubit(q, "get_reduced_density_matrix");

                const size_t k       = qubits.size();
                const size_t dim     = size_t(1) << k;
                const size_t total   = size_t(1) << s.numQubits;
                const size_t traceN  = size_t(1) << (s.numQubits - k);

                // Build a list of qubits being traced out (complement of
                // the selected set within [0, numQubits)).
                std::vector<bool> selected(s.numQubits, false);
                for (auto q : qubits) selected[q] = true;
                std::vector<bitLenInt> traced;
                traced.reserve(s.numQubits - k);
                for (bitLenInt q = 0; q < s.numQubits; q++)
                    if (!selected[q]) traced.push_back(q);

                // Cache the full state vector once (O(2^n) GetAmplitude
                // calls) so the rho_{ij} = Σ_t ψ_{i,t} · conj(ψ_{j,t})
                // double loop is O(dim^2 · traceN) = O(2^(2k) · 2^(n-k))
                // = O(2^(n+k)) on the cache rather than O(2^(n+k)) calls
                // into Qrack.
                std::vector<cf_t> psi(total);
                for (size_t i = 0; i < total; i++) {
                    const Qrack::complex amp = s.sim->GetAmplitude(bitCapInt(i));
                    psi[i] = cf_t(amp.real(), amp.imag());
                }

                // For permutation index `i` over the selected qubits and
                // `t` over the traced qubits, the corresponding full-
                // register basis index is composed bit-wise: bit `qubits[b]`
                // takes (i >> b) & 1; bit `traced[b]` takes (t >> b) & 1.
                auto compose = [&](size_t i, size_t t) -> size_t {
                    size_t idx = 0;
                    for (size_t b = 0; b < k; b++)
                        if ((i >> b) & 1u) idx |= (size_t(1) << qubits[b]);
                    for (size_t b = 0; b < traced.size(); b++)
                        if ((t >> b) & 1u) idx |= (size_t(1) << traced[b]);
                    return idx;
                };

                cf_t* buf = new cf_t[dim * dim];
                for (size_t i = 0; i < dim; i++) {
                    for (size_t j = 0; j < dim; j++) {
                        cf_t sum(0.0f, 0.0f);
                        for (size_t t = 0; t < traceN; t++) {
                            const cf_t a = psi[compose(i, t)];
                            const cf_t b = psi[compose(j, t)];
                            sum += a * std::conj(b);
                        }
                        buf[i * dim + j] = sum;
                    }
                }

                nb::capsule owner(buf, [](void* p) noexcept {
                    delete[] static_cast<cf_t*>(p);
                });
                return nb::ndarray<nb::numpy, cf_t, nb::shape<-1, -1>>(
                    buf, {dim, dim}, owner);
            },
            nb::arg("qubits"),
            nb::rv_policy::reference,
            "Reduced density matrix of the specified qubits as a 2-D complex\n"
            "NumPy array of shape (2**k, 2**k), where k = len(qubits). All other\n"
            "qubits are traced out. The result is Hermitian, positive semi-definite,\n"
            "and has trace 1.")

        .def("prob_perm",
            [](QrackSim& s, bitCapInt perm) -> real1_f {
                return s.sim->ProbAll(perm);
            },
            nb::arg("index"),
            "Probability of a specific full-register basis state by integer index.\n"
            "More efficient than ``probabilities[index]`` for sparse queries.\n"
            "Does not collapse the state.\n\n"
            "Note: distinct from the ``prob_all`` property, which returns the\n"
            "per-qubit |1> probabilities (length num_qubits).")

        .def("prob_mask",
            [](QrackSim& s, bitCapInt mask, bitCapInt permutation) -> real1_f {
                return s.sim->ProbMask(mask, permutation);
            },
            nb::arg("mask"), nb::arg("permutation"),
            "Probability that the masked qubits match the given permutation.\n"
            "mask selects which qubits to check; permutation gives their expected\n"
            "values. Bits not in mask should be 0 in permutation.")

        .def("update_running_norm",
            [](QrackSim& s) { s.sim->UpdateRunningNorm(); },
            "Recompute and apply the state vector normalisation factor.\n"
            "Call after set_amplitude() or set_state_vector() if the injected\n"
            "state may not be exactly unit-norm.")

        .def("first_nonzero_phase",
            [](QrackSim& s) -> real1_f {
                return static_cast<real1_f>(s.sim->FirstNonzeroPhase());
            },
            "Return the phase angle of the lowest-index nonzero amplitude, in\n"
            "radians. Useful for global phase normalisation before state\n"
            "comparison.")

        // ── Properties ────────────────────────────────────────────────────
        .def_prop_ro("num_qubits",
            [](const QrackSim& s) { return s.numQubits; },
            "Number of qubits in this simulator.")

        // ── Cloning ───────────────────────────────────────────────────────
        .def("clone",
            [](const QrackSim& s) { return new QrackSim(s); },
            "Return an independent deep copy of this simulator. The clone\n"
            "starts with a full copy of this simulator's quantum state and\n"
            "configuration; subsequent gates applied to either simulator\n"
            "have no effect on the other.")

        .def("__copy__",
            [](const QrackSim& s) { return new QrackSim(s); },
            "Support for ``copy.copy(sim)``. Equivalent to ``sim.clone()``.")

        .def("__deepcopy__",
            [](const QrackSim& s, nb::object /*memo*/) { return new QrackSim(s); },
            nb::arg("memo"),
            "Support for ``copy.deepcopy(sim)``. Equivalent to ``sim.clone()``.")

        // ── Context manager ───────────────────────────────────────────────
        .def("__enter__",
            [](QrackSim& s) -> QrackSim& { return s; })

        .def("__exit__",
            [](QrackSim& s, const nb::object&, const nb::object&, const nb::object&) -> bool {
                s.sim.reset();
                return false;
            },
            nb::arg("exc_type").none(), nb::arg("exc_val").none(), nb::arg("exc_tb").none());
}
