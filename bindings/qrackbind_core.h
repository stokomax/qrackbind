// bindings/qrackbind_core.h
// Include this at the top of every binding .cpp file.
// Add new nanobind headers here; never include them piecemeal in .cpp files.
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/complex.h>   // required for std::complex<float> casters used by mtrx/mcmtrx/multiplex1_mtrx
#include <nanobind/stl/map.h>       // required for std::map return from measure_shots
#include <nanobind/ndarray.h>       // state_vector, probabilities, RDM (Phase 3)
// Add as needed:
// #include <nanobind/stl/optional.h>

// Qrack types used across all binding files
#include "qfactory.hpp"
#include "common/pauli.hpp" 

// Qrack defines these as preprocessor macros, we have to undefine them first
#undef bitLenInt
#undef bitCapInt
using bitLenInt = uint16_t;
using bitCapInt = uint64_t;
using real1_f   = float;

namespace nb = nanobind;
using namespace Qrack;

// ── Phase 5: typed exceptions ───────────────────────────────────────────────
// Every error path in the C++ bindings throws QrackError with a kind tag.
// bindings/exceptions.cpp registers a translator that maps each kind to one
// of three Python exception classes:
//
//   QrackError(...)                                    → qrackbind.QrackException
//   QrackError(..., QrackErrorKind::QubitOutOfRange)   → qrackbind.QrackQubitError
//   QrackError(..., QrackErrorKind::InvalidArgument)   → qrackbind.QrackArgumentError
//
// The Python type pointers below are populated by bind_exceptions(m).

enum class QrackErrorKind {
    Generic,
    QubitOutOfRange,
    InvalidArgument,
};

class QrackError : public std::exception {
public:
    explicit QrackError(std::string msg,
                        QrackErrorKind kind = QrackErrorKind::Generic)
        : msg_(std::move(msg)), kind_(kind) {}

    const char* what() const noexcept override { return msg_.c_str(); }
    QrackErrorKind kind() const noexcept { return kind_; }

private:
    std::string    msg_;
    QrackErrorKind kind_;
};

// Created by bind_exceptions(m). Borrowed strong references owned by the
// module — never decremented.
extern PyObject* QrackExceptionType;
extern PyObject* QrackQubitErrorType;
extern PyObject* QrackArgumentErrorType;

void bind_exceptions(nb::module_& m);
