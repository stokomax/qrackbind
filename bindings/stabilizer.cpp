// bindings/stabilizer.cpp
// Standalone QrackStabilizer and QrackStabilizerHybrid classes.
//
// QrackStabilizer   → [QINTERFACE_STABILIZER]
//   Pure Clifford engine. Polynomial memory. No non-Clifford gates, no
//   state_vector / probabilities (would force expensive dense materialisation).
//
// QrackStabilizerHybrid → [QINTERFACE_STABILIZER_HYBRID, QINTERFACE_HYBRID/OPENCL/CPU]
//   Clifford engine that falls back to dense simulation on the first
//   non-Clifford gate. Full gate surface + state access.
//
// Both classes share gate helpers with QrackSimulator via gate_helpers.h.

#include "gate_helpers.h"   // includes qrackbind_core.h + r_t, cf_t, all helpers

// runtime_has_opencl() is defined in simulator.cpp (linked into the same
// extension module) — forward-declare so we can reuse the same detection logic.
bool runtime_has_opencl();

namespace {

// ── QrackStabilizer ──────────────────────────────────────────────────────────

struct QrackStab {
    QInterfacePtr sim;
    bitLenInt     numQubits;

    explicit QrackStab(bitLenInt n)
        : numQubits(n)
        , sim(CreateQuantumInterface(
              std::vector<QInterfaceEngine>{QINTERFACE_STABILIZER},
              n,
              /*initState=*/   0,
              /*rgp=*/         nullptr,
              /*phaseFac=*/    CMPLX_DEFAULT_ARG,
              /*doNorm=*/      false,
              /*randomGP=*/    true,
              /*useHostMem=*/  false,
              /*deviceId=*/    -1,
              /*useHWRNG=*/    true,
              /*isSparse=*/    false))
    {
        if (!sim)
            throw QrackError("QrackStabilizer: factory returned null",
                             QrackErrorKind::InvalidArgument);
    }

    void check_qubit(bitLenInt q, const char* method) const {
        if (q >= numQubits)
            throw QrackError(
                std::string(method) + ": qubit index " + std::to_string(q) +
                " is out of range [0, " + std::to_string(numQubits) + ")"
                " (stabilizer has " + std::to_string(numQubits) + " qubits)",
                QrackErrorKind::QubitOutOfRange);
    }

    std::string repr() const {
        return "QrackStabilizer(qubits=" + std::to_string(numQubits) + ")";
    }
};


// ── QrackStabilizerHybrid ────────────────────────────────────────────────────

struct StabHybridConfig {
    bool isCpuGpuHybrid = true;
    bool isOpenCL       = true;
    bool isHostPointer  = false;
    bool isSparse       = false;
};

struct QrackStabHybrid {
    QInterfacePtr    sim;
    bitLenInt        numQubits;
    StabHybridConfig config;

    QrackStabHybrid(bitLenInt n, const StabHybridConfig& cfg)
        : numQubits(n), config(cfg)
    {
        // Apply the same OpenCL guard as make_simulator() so that a system
        // without an OpenCL runtime silently falls through to CPU.
        StabHybridConfig c = cfg;
        if (c.isOpenCL && !runtime_has_opencl()) {
            c.isOpenCL       = false;
            c.isCpuGpuHybrid = false;
        }

        std::vector<QInterfaceEngine> stack{QINTERFACE_STABILIZER_HYBRID};
        if (c.isCpuGpuHybrid && c.isOpenCL)
            stack.push_back(QINTERFACE_HYBRID);
        else if (c.isOpenCL)
            stack.push_back(QINTERFACE_OPENCL);
        else
            stack.push_back(QINTERFACE_CPU);

        sim = CreateQuantumInterface(
            stack,
            n,
            /*initState=*/   0,
            /*rgp=*/         nullptr,
            /*phaseFac=*/    CMPLX_DEFAULT_ARG,
            /*doNorm=*/      false,
            /*randomGP=*/    false,
            /*useHostMem=*/  c.isHostPointer,
            /*deviceId=*/    -1,
            /*useHWRNG=*/    true,
            /*isSparse=*/    c.isSparse);

        if (!sim)
            throw QrackError("QrackStabilizerHybrid: factory returned null",
                             QrackErrorKind::InvalidArgument);
    }

    void check_qubit(bitLenInt q, const char* method) const {
        if (q >= numQubits)
            throw QrackError(
                std::string(method) + ": qubit index " + std::to_string(q) +
                " is out of range [0, " + std::to_string(numQubits) + ")"
                " (stabilizer-hybrid has " + std::to_string(numQubits) + " qubits)",
                QrackErrorKind::QubitOutOfRange);
    }

    std::string repr() const {
        return "QrackStabilizerHybrid(qubits=" + std::to_string(numQubits) +
               ", clifford=" + (sim->isClifford() ? "true" : "false") + ")";
    }
};


// ── Binding helpers ──────────────────────────────────────────────────────────

void bind_stabilizer_class(nb::module_& m)
{
    auto cls = nb::class_<QrackStab>(m, "QrackStabilizer",
        "Pure Clifford-only quantum simulator. Polynomial memory in qubit count.\n\n"
        "Supports H, X, Y, Z, S, S†, √X, √X† single-qubit gates; CNOT, CY, CZ,\n"
        "SWAP, iSWAP two-qubit gates; and their multiply-controlled forms.\n\n"
        "Non-Clifford gates (RX, RY, RZ, U, T, T†, arbitrary matrices) are NOT\n"
        "exposed — use QrackStabilizerHybrid or QrackSimulator for those.\n\n"
        "State vector and probabilities are also intentionally omitted — the\n"
        "stabilizer engine stores a tableau, not amplitudes. Use\n"
        "QrackStabilizerHybrid if you need state access.")
        .def(nb::init<bitLenInt>(),
             nb::arg("qubitCount") = 0,
             "Create a stabilizer simulator on n qubits, initialised to |0...0>.")
        .def("__repr__", &QrackStab::repr);

    add_clifford_gates(cls);
    add_clifford_two_qubit(cls);
    add_measurement(cls);
    add_measure_shots(cls);
    add_pauli_methods(cls);

    cls
        .def_prop_ro("num_qubits",
            [](const QrackStab& s) { return s.numQubits; },
            "Number of qubits in this stabilizer simulator.")

        .def("reset_all",
            [](QrackStab& s) { s.sim->SetPermutation(0); },
            "Reset all qubits to |0...0>.")

        .def("set_permutation",
            [](QrackStab& s, bitCapInt p) { s.sim->SetPermutation(p); },
            nb::arg("permutation"),
            "Reset state to the computational basis state |permutation>.")

        .def("__enter__",
            [](QrackStab& s) -> QrackStab& { return s; })

        .def("__exit__",
            [](QrackStab& s, nb::object, nb::object, nb::object) {
                s.sim.reset();
            },
            nb::arg("exc_type").none(),
            nb::arg("exc_val").none(),
            nb::arg("exc_tb").none());
}


void bind_stabilizer_hybrid_class(nb::module_& m)
{
    auto cls = nb::class_<QrackStabHybrid>(m, "QrackStabilizerHybrid",
        "Stabilizer simulator with automatic fallback to dense simulation\n"
        "when non-Clifford gates are applied. Stays in polynomial-memory\n"
        "stabilizer mode for as long as the circuit is Clifford; switches to\n"
        "a QHybrid (CPU+GPU) dense backend on the first non-Clifford gate.\n\n"
        "The is_clifford property lets callers check which mode is active.\n"
        "set_t_injection enables the near-Clifford T-injection gadget for\n"
        "circuits with few T gates (Clifford+T / RZ workloads).")
        .def("__init__",
            [](QrackStabHybrid* self,
               bitLenInt qubitCount,
               bool isCpuGpuHybrid,
               bool isOpenCL,
               bool isHostPointer,
               bool isSparse)
            {
                StabHybridConfig cfg{isCpuGpuHybrid, isOpenCL,
                                     isHostPointer, isSparse};
                new (self) QrackStabHybrid(qubitCount, cfg);
            },
            nb::arg("qubitCount")     = 0,
            nb::arg("isCpuGpuHybrid") = true,
            nb::arg("isOpenCL")       = true,
            nb::arg("isHostPointer")  = false,
            nb::arg("isSparse")       = false,
            "Create a stabilizer-hybrid simulator. Flags select the dense\n"
            "fallback engine — same semantics as the matching flags on\n"
            "QrackSimulator.")
        .def("__repr__", &QrackStabHybrid::repr);

    add_clifford_gates(cls);
    add_clifford_two_qubit(cls);
    add_t_gates(cls);
    add_rotation_gates(cls);
    add_u_gates(cls);
    add_matrix_gates(cls);
    add_measurement(cls);
    add_measure_shots(cls);
    add_pauli_methods(cls);
    add_state_access(cls);   // _state_vector_impl, _probabilities_impl, get/set_amplitude

    cls
        .def_prop_ro("num_qubits",
            [](const QrackStabHybrid& s) { return s.numQubits; },
            "Number of qubits in this stabilizer-hybrid simulator.")

        .def_prop_ro("is_clifford",
            [](const QrackStabHybrid& s) { return s.sim->isClifford(); },
            "True if the engine is currently in stabilizer (Clifford) mode.\n"
            "Becomes False after the first non-Clifford gate forces the\n"
            "fallback to dense simulation.")

        .def("set_t_injection",
            [](QrackStabHybrid& s, bool useGadget) {
                s.sim->SetTInjection(useGadget);
            },
            nb::arg("use_gadget"),
            "Enable or disable the T-injection gadget for near-Clifford circuits.\n"
            "With the gadget enabled, T gates are deferred as long as possible\n"
            "using a Clifford+T simulation approach before triggering the\n"
            "dense fallback.")

        .def("set_use_exact_near_clifford",
            [](QrackStabHybrid& s, bool exact) {
                s.sim->SetUseExactNearClifford(exact);
            },
            nb::arg("exact"),
            "Toggle exact near-Clifford simulation. When True, Qrack uses an\n"
            "exact (slower) path for near-Clifford circuits instead of the\n"
            "approximate gadget approach.")

        .def("reset_all",
            [](QrackStabHybrid& s) { s.sim->SetPermutation(0); },
            "Reset all qubits to |0...0>.")

        .def("set_permutation",
            [](QrackStabHybrid& s, bitCapInt p) { s.sim->SetPermutation(p); },
            nb::arg("permutation"),
            "Reset state to the computational basis state |permutation>.")

        .def("__enter__",
            [](QrackStabHybrid& s) -> QrackStabHybrid& { return s; })

        .def("__exit__",
            [](QrackStabHybrid& s, nb::object, nb::object, nb::object) {
                s.sim.reset();
            },
            nb::arg("exc_type").none(),
            nb::arg("exc_val").none(),
            nb::arg("exc_tb").none());
}

} // anonymous namespace


// ── Public entry point ───────────────────────────────────────────────────────
void bind_stabilizer(nb::module_& m)
{
    bind_stabilizer_class(m);
    bind_stabilizer_hybrid_class(m);
}
