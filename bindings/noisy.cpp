// bindings/noisy.cpp
// QrackNoisySimulator and QrackNoisyStabilizerHybrid
#include "gate_helpers.h"

bool runtime_has_opencl();

namespace {

enum class NoisyBase { SIMULATOR, STABILIZER_HYBRID };

struct NoisyConfig {
    NoisyBase base           = NoisyBase::SIMULATOR;
    real1_f   noise_param    = 0.01f;
    bool      isCpuGpuHybrid = true;
    bool      isOpenCL       = true;
    bool      isHostPointer  = false;
    bool      isSparse       = false;
};

QInterfacePtr make_noisy(bitLenInt n, const NoisyConfig& c)
{
    NoisyConfig cfg = c;
    if (cfg.isOpenCL && !runtime_has_opencl())
        { cfg.isOpenCL = false; cfg.isCpuGpuHybrid = false; }

    std::vector<QInterfaceEngine> stack{QINTERFACE_NOISY};
    switch (cfg.base) {
    case NoisyBase::SIMULATOR:
        stack.push_back(QINTERFACE_TENSOR_NETWORK);
        stack.push_back(QINTERFACE_QUNIT);
        stack.push_back(QINTERFACE_STABILIZER_HYBRID);
        if (cfg.isCpuGpuHybrid && cfg.isOpenCL) stack.push_back(QINTERFACE_HYBRID);
        else if (cfg.isOpenCL)                   stack.push_back(QINTERFACE_OPENCL);
        else                                     stack.push_back(QINTERFACE_CPU);
        break;
    case NoisyBase::STABILIZER_HYBRID:
        stack.push_back(QINTERFACE_STABILIZER_HYBRID);
        if (cfg.isCpuGpuHybrid && cfg.isOpenCL) stack.push_back(QINTERFACE_HYBRID);
        else if (cfg.isOpenCL)                   stack.push_back(QINTERFACE_OPENCL);
        else                                     stack.push_back(QINTERFACE_CPU);
        break;
    }

    auto sim = CreateQuantumInterface(
        stack, n, 0, nullptr, CMPLX_DEFAULT_ARG,
        false, true, cfg.isHostPointer, -1, true, cfg.isSparse);
    if (!sim)
        throw QrackError("noisy factory returned null", QrackErrorKind::InvalidArgument);
    // Always call SetNoiseParameter — even at 0.0f — so the engine
    // doesn't use its own internal default when the caller passes 0.
    sim->SetNoiseParameter(cfg.noise_param);
    return sim;
}

struct QrackNoisySim {
    QInterfacePtr sim;
    bitLenInt     numQubits;
    NoisyConfig   config;

    QrackNoisySim(bitLenInt n, const NoisyConfig& cfg)
        : numQubits(n), config(cfg), sim(make_noisy(n, cfg)) {}

    void check_qubit(bitLenInt q, const char* m) const {
        if (q >= numQubits)
            throw QrackError(
                std::string(m) + ": qubit index " + std::to_string(q) +
                " out of range [0, " + std::to_string(numQubits) + ")",
                QrackErrorKind::QubitOutOfRange);
    }

    std::string repr() const {
        return "QrackNoisySimulator(qubits=" + std::to_string(numQubits) +
               ", noise=" + std::to_string(sim->GetNoiseParameter()) +
               ", fidelity=" + std::to_string(sim->GetUnitaryFidelity()) + ")";
    }
};

struct QrackNoisyStabHybrid {
    QInterfacePtr sim;
    bitLenInt     numQubits;
    NoisyConfig   config;

    QrackNoisyStabHybrid(bitLenInt n, const NoisyConfig& cfg)
        : numQubits(n), config(cfg), sim(make_noisy(n, cfg)) {}

    void check_qubit(bitLenInt q, const char* m) const {
        if (q >= numQubits)
            throw QrackError(
                std::string(m) + ": qubit index " + std::to_string(q) +
                " out of range [0, " + std::to_string(numQubits) + ")",
                QrackErrorKind::QubitOutOfRange);
    }

    std::string repr() const {
        return "QrackNoisyStabilizerHybrid(qubits=" + std::to_string(numQubits) +
               ", noise=" + std::to_string(sim->GetNoiseParameter()) +
               ", fidelity=" + std::to_string(sim->GetUnitaryFidelity()) + ")";
    }
};

template <typename WrapperT>
std::map<uint64_t, int> do_sample_trajectories(WrapperT& s, unsigned shots)
{
    std::vector<BigInteger> qp;
    qp.reserve(s.numQubits);
    for (bitLenInt q = 0; q < s.numQubits; ++q) qp.push_back(BigInteger(1) << q);
    auto raw = s.sim->MultiShotMeasureMask(qp, shots);
    std::map<uint64_t, int> out;
    for (const auto& kv : raw) out.emplace((uint64_t)kv.first, kv.second);
    return out;
}

void bind_noisy_simulator_class(nb::module_& m)
{
    auto cls = nb::class_<QrackNoisySim>(m, "QrackNoisySimulator",
        "Quantum simulator with depolarizing noise injected around every gate.\n\n"
        "IMPORTANT: state_vector returns one trajectory sample, not the ensemble.\n"
        "Use sample_trajectories(shots) for ensemble statistics.\n"
        "noise_param defaults to 0.01. At 0.0 the wrapper is a pass-through.")
        .def("__init__",
            [](QrackNoisySim* self, bitLenInt qubitCount, NoisyBase base,
               real1_f noise_param, bool isCpuGpuHybrid, bool isOpenCL,
               bool isHostPointer, bool isSparse) {
                new (self) QrackNoisySim(qubitCount,
                    NoisyConfig{base, noise_param, isCpuGpuHybrid,
                                isOpenCL, isHostPointer, isSparse});
            },
            nb::arg("qubitCount")     = 0,
            nb::arg("base")           = NoisyBase::SIMULATOR,
            nb::arg("noise_param")    = 0.01f,
            nb::arg("isCpuGpuHybrid") = true,
            nb::arg("isOpenCL")       = true,
            nb::arg("isHostPointer")  = false,
            nb::arg("isSparse")       = false)
        .def("__repr__", &QrackNoisySim::repr);

    add_clifford_gates(cls);    add_clifford_two_qubit(cls);
    add_t_gates(cls);           add_rotation_gates(cls);
    add_u_gates(cls);           add_matrix_gates(cls);
    add_measurement(cls);       add_measure_shots(cls);
    add_pauli_methods(cls);     add_state_access(cls);
    add_noise_methods(cls);

    cls
        .def_prop_ro("num_qubits",
            [](const QrackNoisySim& s) { return s.numQubits; })
        .def_prop_ro("base",
            [](const QrackNoisySim& s) { return s.config.base; },
            "Underlying engine base this noisy layer wraps.")
        .def("sample_trajectories",
            [](QrackNoisySim& s, unsigned shots) {
                return do_sample_trajectories(s, shots); },
            nb::arg("shots"),
            nb::call_guard<nb::gil_scoped_release>(),
            "Run shots noise trajectories; return dict[int,int] histogram.")
        .def("reset_all",
            [](QrackNoisySim& s) {
                s.sim->SetPermutation(0); s.sim->ResetUnitaryFidelity(); },
            "Reset qubits to |0> and unitary fidelity to 1.0.")
        .def("set_permutation",
            [](QrackNoisySim& s, bitCapInt p) { s.sim->SetPermutation(p); },
            nb::arg("permutation"))
        .def("__enter__",
            [](QrackNoisySim& s) -> QrackNoisySim& { return s; })
        .def("__exit__",
            [](QrackNoisySim& s, nb::object, nb::object, nb::object) {
                s.sim.reset(); },
            nb::arg("exc_type").none(),
            nb::arg("exc_val").none(),
            nb::arg("exc_tb").none());
}

void bind_noisy_stabilizer_hybrid_class(nb::module_& m)
{
    auto cls = nb::class_<QrackNoisyStabHybrid>(m, "QrackNoisyStabilizerHybrid",
        "Stabilizer-hybrid simulator with depolarizing noise injected around\n"
        "every gate. Stays in stabilizer mode while gates are Clifford AND\n"
        "noise has not pushed the state out of the stabilizer manifold.\n\n"
        "Same density-matrix semantics caveat as QrackNoisySimulator.")
        .def("__init__",
            [](QrackNoisyStabHybrid* self, bitLenInt qubitCount,
               real1_f noise_param, bool isCpuGpuHybrid, bool isOpenCL,
               bool isHostPointer, bool isSparse) {
                new (self) QrackNoisyStabHybrid(qubitCount,
                    NoisyConfig{NoisyBase::STABILIZER_HYBRID, noise_param,
                                isCpuGpuHybrid, isOpenCL, isHostPointer, isSparse});
            },
            nb::arg("qubitCount")     = 0,
            nb::arg("noise_param")    = 0.01f,
            nb::arg("isCpuGpuHybrid") = true,
            nb::arg("isOpenCL")       = true,
            nb::arg("isHostPointer")  = false,
            nb::arg("isSparse")       = false)
        .def("__repr__", &QrackNoisyStabHybrid::repr);

    add_clifford_gates(cls);    add_clifford_two_qubit(cls);
    add_t_gates(cls);           add_rotation_gates(cls);
    add_u_gates(cls);           add_matrix_gates(cls);
    add_measurement(cls);       add_measure_shots(cls);
    add_pauli_methods(cls);     add_state_access(cls);
    add_noise_methods(cls);

    cls
        .def_prop_ro("num_qubits",
            [](const QrackNoisyStabHybrid& s) { return s.numQubits; })
        .def_prop_ro("is_clifford",
            [](const QrackNoisyStabHybrid& s) { return s.sim->isClifford(); },
            "True if the engine is currently in stabilizer mode.")
        .def("sample_trajectories",
            [](QrackNoisyStabHybrid& s, unsigned shots) {
                return do_sample_trajectories(s, shots); },
            nb::arg("shots"),
            nb::call_guard<nb::gil_scoped_release>(),
            "Run shots noise trajectories; return dict[int,int] histogram.")
        .def("reset_all",
            [](QrackNoisyStabHybrid& s) {
                s.sim->SetPermutation(0); s.sim->ResetUnitaryFidelity(); },
            "Reset qubits to |0> and unitary fidelity to 1.0.")
        .def("set_permutation",
            [](QrackNoisyStabHybrid& s, bitCapInt p) { s.sim->SetPermutation(p); },
            nb::arg("permutation"))
        .def("__enter__",
            [](QrackNoisyStabHybrid& s) -> QrackNoisyStabHybrid& { return s; })
        .def("__exit__",
            [](QrackNoisyStabHybrid& s, nb::object, nb::object, nb::object) {
                s.sim.reset(); },
            nb::arg("exc_type").none(),
            nb::arg("exc_val").none(),
            nb::arg("exc_tb").none());
}

} // namespace


void bind_noisy(nb::module_& m)
{
    nb::enum_<NoisyBase>(m, "NoisyBase",
        "Underlying engine base for QrackNoisySimulator.\n"
        "SIMULATOR         - full standard stack\n"
        "STABILIZER_HYBRID - stabilizer-hybrid with dense fallback")
        .value("SIMULATOR",         NoisyBase::SIMULATOR)
        .value("STABILIZER_HYBRID", NoisyBase::STABILIZER_HYBRID);

    bind_noisy_simulator_class(m);
    bind_noisy_stabilizer_hybrid_class(m);
}
