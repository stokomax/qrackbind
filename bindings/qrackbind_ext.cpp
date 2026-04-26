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

    nb::enum_<Qrack::Pauli>(m, "Pauli", nb::is_arithmetic())
        .value("PauliI", Qrack::Pauli::PauliI)
        .value("PauliX", Qrack::Pauli::PauliX)
        .value("PauliY", Qrack::Pauli::PauliY)
        .value("PauliZ", Qrack::Pauli::PauliZ);

    bind_simulator(m);

    // Future bind_circuit(m);  bind_exceptions(m);
}
