#include "gate_helpers.h"   // includes qrackbind_core.h; defines r_t, cf_t, helper templates
#include "qalu.hpp"
#if ENABLE_OPENCL
#include "common/oclengine.hpp"
#endif

// Detect whether the running process actually has at least one usable
// OpenCL device. Qrack will happily emit an "OCL stack" config even when
// the host has no OpenCL runtime, but the resulting simulator silently
// returns zero amplitudes for entangled states. Any caller that asks for
// `isOpenCL=True` on such a host is transparently downgraded to a CPU-only
// stack.
bool runtime_has_opencl()
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
    // Qrack release.
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
        // computational-basis state.
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


// ── Binding ───────────────────────────────────────────────────────────────────
void bind_simulator(nb::module_& m) {

auto cls = nb::class_<QrackSim>(m, "QrackSimulator",
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
);

cls.def("__init__",
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

    .def("__repr__", &QrackSim::repr);

// ── Gate helpers (from gate_helpers.h) ────────────────────────────────────
add_clifford_gates(cls);
add_clifford_two_qubit(cls);
add_t_gates(cls);
add_rotation_gates(cls);
add_u_gates(cls);
add_matrix_gates(cls);
add_measurement(cls);
add_measure_shots(cls);
add_pauli_methods(cls);
add_state_access(cls);

// ── Deferred Phase 4: arbitrary unitary observables ─────────────────────
// These are QrackSimulator-specific (BigInteger-dependent); not templated.
cls
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

    // ── allocators ─────────────────────────────────────────────────────────
    .def("allocate",
        [](QrackSim& s, bitLenInt start, bitLenInt length) -> bitLenInt {
            bitLenInt offset = s.sim->Allocate(start, length);
            s.numQubits = s.sim->GetQubitCount();
            return offset;
        },
        nb::arg("start"), nb::arg("length"),
        "Allocate 'length' new |0> qubits at index 'start'. Returns the start offset.\n"
        "Existing qubits at >= start shift up. Updates num_qubits automatically.\n"
        "Incompatible with isTensorNetwork=True.")

    .def("dispose",
        [](QrackSim& s, bitLenInt start, bitLenInt length) {
            s.sim->Dispose(start, length);
            s.numQubits = s.sim->GetQubitCount();
        },
        nb::arg("start"), nb::arg("length"),
        "Remove 'length' qubits starting at 'start'. Qubits must be separably |0> or |1>.\n"
        "Updates num_qubits automatically.")

    .def("allocate_qubits",
        [](QrackSim& s, bitLenInt n) -> bitLenInt {
            bitLenInt offset = s.sim->Allocate(n);
            s.numQubits = s.sim->GetQubitCount();
            return offset;
        },
        nb::arg("n"),
        "Allocate n new |0> qubits at the end. Returns the index of the first new qubit.")

    // ── Quantum Fourier Transform ─────────────────────────────────────────
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

    // ── State control ─────────────────────────────────────────────────────
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

    // ── Arithmetic ────────────────────────────────────────────────────────
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

    // ── Shift and Rotate ──────────────────────────────────────────────────
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

    // ── Additional state vector methods (QrackSimulator-specific) ─────────
    // Note: _state_vector_impl, _probabilities_impl, get_amplitude,
    // set_amplitude are already added above by add_state_access(cls).

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

    .def("get_reduced_density_matrix",
        [](QrackSim& s, std::vector<bitLenInt> qubits)
            -> nb::ndarray<nb::numpy, cf_t, nb::shape<-1, -1>>
        {
            for (auto q : qubits)
                s.check_qubit(q, "get_reduced_density_matrix");

            const size_t k      = qubits.size();
            const size_t dim    = size_t(1) << k;
            const size_t total  = size_t(1) << s.numQubits;
            const size_t traceN = size_t(1) << (s.numQubits - k);

            std::vector<bool> selected(s.numQubits, false);
            for (auto q : qubits) selected[q] = true;
            std::vector<bitLenInt> traced;
            traced.reserve(s.numQubits - k);
            for (bitLenInt q = 0; q < s.numQubits; q++)
                if (!selected[q]) traced.push_back(q);

            std::vector<cf_t> psi(total);
            for (size_t i = 0; i < total; i++) {
                const Qrack::complex amp = s.sim->GetAmplitude(bitCapInt(i));
                psi[i] = cf_t(amp.real(), amp.imag());
            }

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
                        const cf_t b_val = psi[compose(j, t)];
                        sum += a * std::conj(b_val);
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

    // ── Properties ────────────────────────────────────────────────────────
    .def_prop_ro("num_qubits",
        [](const QrackSim& s) { return s.numQubits; },
        "Number of qubits in this simulator.")

    // ── Cloning ───────────────────────────────────────────────────────────
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

    // ── Context manager ───────────────────────────────────────────────────
    .def("__enter__",
        [](QrackSim& s) -> QrackSim& { return s; })

    .def("__exit__",
        [](QrackSim& s, const nb::object&, const nb::object&, const nb::object&) -> bool {
            s.sim.reset();
            return false;
        },
        nb::arg("exc_type").none(), nb::arg("exc_val").none(), nb::arg("exc_tb").none());
}
