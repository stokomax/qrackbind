#include "qrackbind_core.h"

// Forward declarations — one per binding .cpp file
void bind_simulator(nb::module_& m);
void bind_circuit  (nb::module_& m);
void bind_stabilizer(nb::module_& m);   // Phase 10: QrackStabilizer + QrackStabilizerHybrid

// Module name must exactly match the CMakeLists.txt target name: _core
NB_MODULE(_core, m) {
    m.doc() = "qrackbind — nanobind bindings for the Qrack quantum simulator";
    m.attr("__version__") = "0.1.0";

    // bind_exceptions MUST be called first. nanobind processes
    // exception translators in LIFO order — the most recently registered
    // translator runs first. Registering exceptions first means our
    // QrackError translator runs before nanobind's default
    // ``std::exception → RuntimeError`` translator, so qrackbind errors
    // surface as the typed QrackException hierarchy instead of generic
    // RuntimeError. See bindings/exceptions.cpp for the full design.
    bind_exceptions(m);

    // Pauli operator basis for single-qubit observables.
    //
    // Used by measure_pauli(), exp_val(), exp_val_pauli(), and
    // variance_pauli(). nb::is_arithmetic() makes this an IntEnum-
    // compatible type — integer values are accepted wherever Pauli is
    // expected, which preserves compatibility with frameworks that pass
    // Pauli bases as raw integers (PennyLane, Bloqade QASM interpreter).
    //
    // NOTE: Qrack's enum values are NOT sequential — PauliI=0, PauliX=1,
    // PauliZ=2, PauliY=3. We expose them verbatim. Stubs render this as
    // ``class Pauli(enum.IntEnum)``.
    nb::enum_<Qrack::Pauli>(m, "Pauli",
        nb::is_arithmetic(),
        "Pauli operator basis for single-qubit observables.\n\n"
        "Used by measure_pauli(), exp_val(), exp_val_pauli(), and\n"
        "variance_pauli(). Integer codes are accepted wherever a Pauli\n"
        "is expected (IntEnum semantics).\n\n"
        "Qrack's underlying values are non-sequential:\n"
        "    PauliI = 0, PauliX = 1, PauliZ = 2, PauliY = 3.")
        .value("PauliI", Qrack::Pauli::PauliI,
               "Identity operator — no rotation applied.")
        .value("PauliX", Qrack::Pauli::PauliX,
               "Pauli X basis — measures in the X (Hadamard) basis.")
        .value("PauliY", Qrack::Pauli::PauliY,
               "Pauli Y basis — measures in the Y basis (S†H rotation).")
        .value("PauliZ", Qrack::Pauli::PauliZ,
               "Pauli Z basis — computational basis, no rotation needed.");

    bind_simulator(m);
    bind_circuit(m);
    bind_stabilizer(m);   // after simulator and circuit
}
