#include "qrackbind_core.h"

PyObject* QrackExceptionType = nullptr;

// Forward declarations — one per binding .cpp file
void bind_simulator(nb::module_& m);

// Module name must exactly match the CMakeLists.txt target name: _qrackbind_core
NB_MODULE(_core, m) {
    m.doc() = "qrackbind — nanobind bindings for the Qrack quantum simulator";
    m.attr("__version__") = "0.1.0";

    // Register a custom Python exception class.
    // NOTE: PyErr_NewExceptionWithDoc returns a raw `PyObject*`. Assigning a
    // raw PyObject* directly to nb::module_::attr triggers `std::bad_cast`
    // inside nanobind's metadata machinery during NB_MODULE init, which
    // surfaces only at module-import time as an opaque ImportError. Always
    // wrap with nb::borrow() (or nb::handle) before assigning to attr.
    QrackExceptionType = PyErr_NewExceptionWithDoc(
        "qrackbind.QrackException",
        "Exception raised by the Qrack C++ library.",
        PyExc_RuntimeError,
        nullptr);
    if (QrackExceptionType)
        m.attr("QrackException") = nb::borrow(QrackExceptionType);

    nb::register_exception_translator([](const std::exception_ptr& p, void*) {
        try { std::rethrow_exception(p); }
        catch (const std::exception& e) {
            PyErr_SetString(QrackExceptionType, e.what());
        }
    });

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

    // Future bind_circuit(m);  bind_exceptions(m);
}
