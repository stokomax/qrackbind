#include "qrackbind_core.h"
#include "qalu.hpp"

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

struct SimConfig {
    bool isTensorNetwork     = true;
    bool isSchmidtDecompose  = true;
    bool isSchmidtDecomposeMulti = false;
    bool isStabilizerHybrid  = false;
    bool isBinaryDecisionTree = false;
    bool isPaged             = true;
    bool isCpuGpuHybrid      = true;
    bool isOpenCL            = true;
    bool isHostPointer       = false;
    bool isSparse            = false;
    bool isNoise             = false;
};

QInterfacePtr make_simulator(bitLenInt n, const SimConfig& c) {
    std::vector<QInterfaceEngine> stack;

    if (c.isTensorNetwork)
        stack.push_back(QINTERFACE_TENSOR_NETWORK);

    if (c.isSchmidtDecompose)
        stack.push_back(QINTERFACE_QUNIT);

    if (c.isStabilizerHybrid)
        stack.push_back(QINTERFACE_STABILIZER_HYBRID);

    if (c.isBinaryDecisionTree) {
        stack.push_back(QINTERFACE_BDT);
    } else if (c.isPaged) {
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
        /*randomGP=*/   true,
        /*useHostMem=*/ c.isHostPointer,
        /*deviceId=*/   -1,
        /*useHWRNG=*/   true,
        /*isSparse=*/   c.isSparse
    );
}

struct QrackSim {
    QInterfacePtr sim;
    bitLenInt     numQubits;
    SimConfig     config;

    QrackSim(bitLenInt n, const SimConfig& cfg)
        : numQubits(n), config(cfg), sim(make_simulator(n, cfg))
    {
        if (!sim) throw std::runtime_error("QrackSimulator: factory returned null");
    }

    // Clone constructor — used by .clone() / __copy__ / __deepcopy__
    explicit QrackSim(const QrackSim& src)
        : numQubits(src.numQubits)
        , config(src.config)
        , sim(src.sim->Clone())
    {
        if (!sim) throw std::runtime_error("QrackSimulator: Clone() returned null");
    }

    void check_qubit(bitLenInt q, const char* method) const {
        if (q >= numQubits) {
            throw std::out_of_range(
                std::string(method) + ": qubit index " + std::to_string(q) +
                " is out of range [0, " + std::to_string(numQubits) + ")");
        }
    }
    std::string repr() const {
        return "QrackSimulator(qubits=" + std::to_string(numQubits) + ")";
    }
};

static void check_arithmetic(const QrackSim& s, const char* method)
{
    if (s.config.isTensorNetwork)
        throw std::runtime_error(
            std::string(method) + ": isTensorNetwork=True is incompatible with "
            "arithmetic gates. Construct with isTensorNetwork=False.");
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
                    throw std::invalid_argument("mtrx: matrix must have 4 elements");
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
                    throw std::invalid_argument("mcmtrx: matrix must have 4 elements");
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
                    throw std::invalid_argument("macmtrx: matrix must have 4 elements");
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
                    throw std::invalid_argument(
                        "multiplex1_mtrx: mtrxs must have at least 4 * 2^len(controls) = " +
                        std::to_string(expected) + " elements");
                s.check_qubit(tgt, "multiplex1_mtrx");
                s.sim->UniformlyControlledSingleBit(controls, tgt,
                    reinterpret_cast<const Qrack::complex*>(mtrxs.data()));
            },
            nb::arg("controls"), nb::arg("mtrxs"), nb::arg("target"),
            "Uniformly-controlled single-qubit gate. mtrxs is a flat list of "
            "4 * 2**len(controls) complex values — one 2x2 unitary per control permutation, "
            "in row-major order.")

        // ── Pauli expectation value (deferred to pauli.cpp phase) ─────────
        // When added, use Qrack::Pauli from pauli.hpp directly — do NOT
        // redefine the enum. pauli.cpp will register it as:
        //   nb::enum_<Qrack::Pauli>(m, "Pauli", nb::is_arithmetic())
        //       .value("PauliI", Qrack::Pauli::PauliI)  ...
        // and exp_val here will accept it with no casting:
        //   .def("exp_val",
        //       [](QrackSim& s, Qrack::Pauli basis, bitLenInt q) -> real1_f {
        //           return s.sim->ExpectationBitsFactorized(...); },
        //       nb::arg("basis"), nb::arg("qubit"))


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
                    throw std::runtime_error(
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
                    throw std::runtime_error(
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
