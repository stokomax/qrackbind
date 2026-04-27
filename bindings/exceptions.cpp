// bindings/exceptions.cpp
// ─────────────────────────────────────────────────────────────────────────────
// Phase 5 — Python exception hierarchy and C++ → Python translation.
//
// Owns:
//   1. The three Python exception type objects (QrackException,
//      QrackQubitError, QrackArgumentError) and their inheritance wiring.
//   2. The static PyObject* pointers for those types — populated once at
//      module init so translators can use them without re-importing the
//      qrackbind package on every error.
//   3. Two nb::register_exception_translator calls:
//        • catch-all   std::exception → QrackException (safety net)
//        • typed       QrackError      → QrackQubitError /
//                                       QrackArgumentError /
//                                       QrackException (by kind)
//
// Translators are processed in LIFO order, so the typed QrackError
// translator (registered second) runs first; if a thrown exception is not
// a QrackError the catch-all translator (registered first) runs and routes
// it to QrackException. This lets us subsume every std::exception escape
// from the bindings (including Qrack's own std::runtime_error) without
// any message-prefix filtering.
// ─────────────────────────────────────────────────────────────────────────────

#include "qrackbind_core.h"

PyObject* QrackExceptionType     = nullptr;
PyObject* QrackQubitErrorType    = nullptr;
PyObject* QrackArgumentErrorType = nullptr;

// Set the active Python error using one of the three captured type
// pointers. ``type_obj`` may be null if module init failed; we fall back
// to PyExc_RuntimeError so the user still sees something rather than a
// silent abort.
static void set_python_error(PyObject* type_obj, const char* what) noexcept
{
    PyErr_SetString(type_obj ? type_obj : PyExc_RuntimeError, what);
}

void bind_exceptions(nb::module_& m)
{
    // ── QrackException (base) ──────────────────────────────────────────────
    // Inherits from RuntimeError so a plain ``except RuntimeError`` clause
    // still catches qrackbind errors — preserving the historical behaviour
    // (Phase 1 errors were all generic RuntimeError) while letting users
    // narrow with ``except QrackException`` or its subclasses.
    QrackExceptionType = PyErr_NewExceptionWithDoc(
        "qrackbind.QrackException",
        "Base class for all qrackbind exceptions.\n\n"
        "Catch this to handle any error raised by QrackSimulator. Use\n"
        "the subclasses for more specific handling:\n\n"
        "  QrackQubitError    — qubit index out of valid range\n"
        "  QrackArgumentError — invalid method arguments (length\n"
        "                       mismatch, wrong array size, etc.)",
        PyExc_RuntimeError,   // base
        nullptr);             // dict

    // ── QrackQubitError ────────────────────────────────────────────────────
    // Subclass of QrackException. Raised whenever a qubit index falls
    // outside [0, num_qubits).
    if (QrackExceptionType) {
        QrackQubitErrorType = PyErr_NewExceptionWithDoc(
            "qrackbind.QrackQubitError",
            "Raised when a qubit index is out of the valid range\n"
            "[0, num_qubits).\n\n"
            "The error message includes the offending index, the valid\n"
            "range, and the simulator's qubit count, e.g.::\n\n"
            "    sim = QrackSimulator(qubitCount=2)\n"
            "    sim.h(5)\n"
            "    # QrackQubitError: h: qubit index 5 is out of range\n"
            "    #                  [0, 2) (simulator has 2 qubits)\n",
            QrackExceptionType,   // base
            nullptr);
    }

    // ── QrackArgumentError ────────────────────────────────────────────────
    // Subclass of QrackException. Raised for everything that's a *value*
    // problem rather than a *qubit-index* problem: length mismatches in
    // paulis/qubits or qubits/weights pairs, NumPy arrays of the wrong
    // size, weight arrays of the wrong shape (Qrack's
    // ExpectationFloatsFactorized expects 2 entries per qubit), and so on.
    if (QrackExceptionType) {
        QrackArgumentErrorType = PyErr_NewExceptionWithDoc(
            "qrackbind.QrackArgumentError",
            "Raised when method arguments are invalid — mismatched\n"
            "list lengths, wrong array sizes, or otherwise incompatible\n"
            "values.\n\n"
            "Example::\n\n"
            "    sim.exp_val_pauli([Pauli.PauliZ], [0, 1])\n"
            "    # QrackArgumentError: paulis and qubits must have the\n"
            "    #                     same length\n",
            QrackExceptionType,   // base
            nullptr);
    }

    // Expose all three on the module. nb::borrow takes a borrowed
    // reference and converts it into the nb::object representation
    // nanobind expects for ``m.attr(...) = ...``. Assigning a raw
    // PyObject* directly triggers ``std::bad_cast`` inside nanobind's
    // metadata machinery (see git history of qrackbind_ext.cpp for the
    // diagnostic that led to this).
    if (QrackExceptionType)
        m.attr("QrackException") = nb::borrow(QrackExceptionType);
    if (QrackQubitErrorType)
        m.attr("QrackQubitError") = nb::borrow(QrackQubitErrorType);
    if (QrackArgumentErrorType)
        m.attr("QrackArgumentError") = nb::borrow(QrackArgumentErrorType);

    // ── Translator 1: catch-all std::exception → QrackException ────────────
    // Registered FIRST so it runs LAST in the LIFO chain. This is the
    // safety net for any std::exception that escapes the bindings without
    // being a typed QrackError — most importantly Qrack's own
    // std::runtime_error throws (factory failures, isTensorNetwork
    // incompatibility, etc.) — guaranteeing they surface as QrackException
    // rather than nanobind's default RuntimeError.
    nb::register_exception_translator(
        [](const std::exception_ptr& p, void*) {
            try {
                if (p) std::rethrow_exception(p);
            } catch (const std::exception& e) {
                set_python_error(QrackExceptionType, e.what());
            }
        });

    // ── Translator 2: typed QrackError → kind-specific subclass ────────────
    // Registered SECOND so it runs FIRST. Switches on QrackError::kind()
    // to pick the right Python subclass.
    nb::register_exception_translator(
        [](const std::exception_ptr& p, void*) {
            try {
                if (p) std::rethrow_exception(p);
            } catch (const QrackError& e) {
                switch (e.kind()) {
                    case QrackErrorKind::QubitOutOfRange:
                        set_python_error(QrackQubitErrorType, e.what());
                        break;
                    case QrackErrorKind::InvalidArgument:
                        set_python_error(QrackArgumentErrorType, e.what());
                        break;
                    case QrackErrorKind::Generic:
                    default:
                        set_python_error(QrackExceptionType, e.what());
                        break;
                }
            }
        });
}
