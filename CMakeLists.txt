cmake_minimum_required(VERSION 3.18...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

# Optional compilation options
option(BUILD_CPP_TESTS "Build C++ unit tests" OFF)

if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()

# Per nanobind docs — Development.Module and Development.SABIModule
# both require CMake >= 3.18. SABIModule provides the Python::SABIModule
# target that nanobind's STABLE_ABI guard checks for; without it, the flag
# is silently ignored and the extension compiles against full (non-abi3)
# headers, producing a version-tagged .so that only works on the build
# host's Python version.
find_package(Python 3.12
  COMPONENTS Interpreter Development.Module Development.SABIModule
  REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Per nanobind docs.
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into 
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# --- Qrack ---
find_library(QRACK_LIB qrack
    HINTS $ENV{QRACK_LIB_DIR}
          /usr/local/lib/qrack /usr/lib/qrack
          /usr/local/lib /usr/local/lib64
          /usr/lib /usr/lib64)
find_path(QRACK_INCLUDE qfactory.hpp
    HINTS $ENV{QRACK_INCLUDE_DIR}
          /usr/local/include/qrack
          /usr/include/qrack)

if(NOT QRACK_LIB OR NOT QRACK_INCLUDE)
    message(STATUS "Qrack not found on system — fetching pre-built binary...")
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FetchQrack)
endif()

message(STATUS "Qrack lib:     ${QRACK_LIB}")
message(STATUS "Qrack include: ${QRACK_INCLUDE}")

# --- OpenCL ---
find_package(OpenCL QUIET)

# --- Extension module ---
nanobind_add_module(
  # Name of the extension
  _core

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI

  # Build libnanobind statically and merge it into the
  # extension (which itself remains a shared library)
  #
  # If your project builds multiple extensions, you can
  # replace this flag by NB_SHARED to conserve space by
  # reusing a shared libnanobind across libraries
  NB_STATIC

  # NB_DOMAIN isolates type visibility to this project (prevents conflicts with
  # other nanobind extensions that might bind overlapping types).
  NB_DOMAIN qrackbind

  # Source code goes here
  bindings/qrackbind_ext.cpp
  bindings/exceptions.cpp
  bindings/simulator.cpp
  bindings/circuit.cpp
  bindings/stabilizer.cpp
)

target_compile_definitions(_core PRIVATE
    VERSION_INFO="${SKBUILD_PROJECT_VERSION}"
)

target_include_directories(_core PRIVATE ${QRACK_INCLUDE})
target_link_libraries(_core PRIVATE ${QRACK_LIB})
if(OpenCL_FOUND)
    target_link_libraries(_core PRIVATE OpenCL::OpenCL)
endif()

# Qrack configuration defines are already baked into the installed library
# We cannot override them - Qrack will use whatever it was built with

# Install directive for scikit-build-core
install(TARGETS _core LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
# NOTE: nanobind stubgen currently aborts while introspecting this module
# after adding cross-referenced QrackCircuit/QrackSimulator bindings. Until
# that is isolated, install the maintained source-tree stubs so wheel builds
# are not blocked by stub generation.
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/qrackbind/_core.pyi DESTINATION ${SKBUILD_PROJECT_NAME})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/qrackbind/py.typed DESTINATION ${SKBUILD_PROJECT_NAME})

# C++ Tests configuration
if(BUILD_CPP_TESTS)
    enable_testing()
    # Test target placeholder - will be implemented when tests are added
    add_custom_target(${SKBUILD_PROJECT_NAME}_tests
        COMMENT "C++ test target placeholder"
    )
endif()
