# cmake/FetchQrack.cmake
#
# Downloads a pre-built Qrack release archive from GitHub when Qrack is not
# found on the system. Called from CMakeLists.txt after find_library /
# find_path fail.
#
# After this module runs, QRACK_LIB and QRACK_INCLUDE are set to the
# extracted library and header paths and the caller can proceed with
# target_include_directories / target_link_libraries as normal.
#
# To pin a different release, update QRACK_VERSION and QRACK_LINUX_ARCHIVE
# below.

# ── Version pin ──────────────────────────────────────────────────────────────
set(QRACK_VERSION "vm6502q.v10.7.0")
set(QRACK_BASE_URL "https://github.com/unitaryfoundation/qrack/releases/download/${QRACK_VERSION}")

# ── Platform detection ───────────────────────────────────────────────────────
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(QRACK_ARCHIVE "libqrack-manylinux_2_39_x86_64.zip")
    else()
        message(FATAL_ERROR
            "FetchQrack: unsupported Linux architecture '${CMAKE_SYSTEM_PROCESSOR}'. "
            "Please install Qrack manually and set QRACK_LIB_DIR / QRACK_INCLUDE_DIR.")
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        set(QRACK_ARCHIVE "libqrack-macosx_14_0_arm64.zip")
    else()
        message(FATAL_ERROR
            "FetchQrack: unsupported macOS architecture '${CMAKE_SYSTEM_PROCESSOR}'. "
            "Please install Qrack manually and set QRACK_LIB_DIR / QRACK_INCLUDE_DIR.")
    endif()
else()
    message(FATAL_ERROR
        "FetchQrack: unsupported platform '${CMAKE_SYSTEM_NAME}'. "
        "Supported platforms: Linux x86_64, macOS ARM64. "
        "Please install Qrack manually and set QRACK_LIB_DIR / QRACK_INCLUDE_DIR.")
endif()

# ── Download ─────────────────────────────────────────────────────────────────
set(QRACK_DOWNLOAD_URL "${QRACK_BASE_URL}/${QRACK_ARCHIVE}")
set(QRACK_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/_qrack_fetch")
set(QRACK_ARCHIVE_PATH "${QRACK_DOWNLOAD_DIR}/${QRACK_ARCHIVE}")
set(QRACK_EXTRACT_DIR  "${QRACK_DOWNLOAD_DIR}/qrack")

if(NOT EXISTS "${QRACK_ARCHIVE_PATH}")
    message(STATUS "FetchQrack: downloading ${QRACK_DOWNLOAD_URL}")
    file(DOWNLOAD
        "${QRACK_DOWNLOAD_URL}"
        "${QRACK_ARCHIVE_PATH}"
        SHOW_PROGRESS
        STATUS _fetch_status
        TLS_VERIFY ON)
    list(GET _fetch_status 0 _fetch_code)
    if(NOT _fetch_code EQUAL 0)
        list(GET _fetch_status 1 _fetch_msg)
        message(FATAL_ERROR "FetchQrack: download failed — ${_fetch_msg}")
    endif()
    message(STATUS "FetchQrack: download complete")
else()
    message(STATUS "FetchQrack: using cached archive ${QRACK_ARCHIVE_PATH}")
endif()

# ── Extract ───────────────────────────────────────────────────────────────────
if(NOT EXISTS "${QRACK_EXTRACT_DIR}")
    message(STATUS "FetchQrack: extracting ${QRACK_ARCHIVE}")
    file(ARCHIVE_EXTRACT
        INPUT   "${QRACK_ARCHIVE_PATH}"
        DESTINATION "${QRACK_EXTRACT_DIR}")
endif()

# ── Locate library and headers inside the extracted tree ─────────────────────
# The Qrack release archive has the layout produced by cmake --install:
#   qrack/
#     include/qrack/qfactory.hpp   (and other headers)
#     lib/libqrack.a               (or libqrack.so)
file(GLOB_RECURSE _qrack_libs
    "${QRACK_EXTRACT_DIR}/lib/libqrack.*"
    "${QRACK_EXTRACT_DIR}/lib64/libqrack.*"
    "${QRACK_EXTRACT_DIR}/libqrack.*")

if(NOT _qrack_libs)
    message(FATAL_ERROR
        "FetchQrack: could not find libqrack.* inside the extracted archive at "
        "'${QRACK_EXTRACT_DIR}'. The archive layout may have changed — please "
        "file an issue or install Qrack manually.")
endif()
list(GET _qrack_libs 0 QRACK_LIB)

file(GLOB_RECURSE _qrack_header
    "${QRACK_EXTRACT_DIR}/include/qrack/qfactory.hpp"
    "${QRACK_EXTRACT_DIR}/qfactory.hpp")

if(NOT _qrack_header)
    message(FATAL_ERROR
        "FetchQrack: could not find qfactory.hpp inside the extracted archive at "
        "'${QRACK_EXTRACT_DIR}'.")
endif()
get_filename_component(_qrack_include_dir "${_qrack_header}" DIRECTORY)
# QRACK_INCLUDE should be the parent of qrack/ (so #include <qrack/...> works)
get_filename_component(QRACK_INCLUDE "${_qrack_include_dir}" DIRECTORY)

# If the headers are directly in include/ without a qrack/ sub-dir, adjust:
if(NOT EXISTS "${QRACK_INCLUDE}/qrack/qfactory.hpp")
    set(QRACK_INCLUDE "${_qrack_include_dir}")
endif()

message(STATUS "FetchQrack: lib     = ${QRACK_LIB}")
message(STATUS "FetchQrack: include = ${QRACK_INCLUDE}")
