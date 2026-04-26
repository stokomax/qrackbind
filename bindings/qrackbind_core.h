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

extern PyObject* QrackExceptionType;
