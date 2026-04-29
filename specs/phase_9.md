---
tags:
  - qrack
  - nanobind
  - packaging
  - distribution
  - pypi
  - cicd
  - qrackbind
  - phase9
---
## qrackbind Phase 9 — Packaging and Distribution

Covers the full packaging and distribution story for qrackbind: publishing a pre-built wheel to PyPI so that end users can install with a single `pip install` command, and providing a clean developer workflow for contributors building from source.

**Prerequisite:** All Phase 7 checklist items passing. `.pyi` stubs generated and `pyright` passing.

---

## The Two Audiences

qrackbind has two distinct audiences with different installation requirements. Understanding this separation is the key to Phase 9.

**End users** install a pre-built binary wheel from PyPI. They run `pip install qrackbind` and receive a compiled `.whl` file containing the nanobind extension with the Qrack library already bundled. They do not interact with CMake, justfile, uv, or the source repository. Nothing compiles on their machine.

**Developers** clone the repository and build from source. They use uv as the package manager and the justfile as a task runner. They install the Qrack C++ library as a separate step before building the extension.

These two workflows are completely independent and do not interfere with each other.

---

## End User Installation

```bash
pip install qrackbind
```

That is the complete installation for an end user. The wheel on PyPI is a pre-built binary that includes the Qrack library. No system Qrack installation is required.

The wheel is built by the GitHub Actions CI pipeline using cibuildwheel inside a `manylinux_2_34` container. During the wheel build, the `before-all` script installs OpenCL build dependencies and compiles Qrack from source with `-DENABLE_OPENCL=ON`. The nanobind extension is compiled against this freshly-built Qrack, then `auditwheel repair` bundles `libqrack.so` into the wheel while explicitly excluding `libOpenCL.so.1` (the ICD loader is provided by the user's distro at runtime — see the OpenCL section below). By the time the `.whl` file is published to PyPI, the Qrack library is inside it; only the OpenCL ICD loader is a runtime system dependency.

---

## Developer Workflow

Developers work from a clone of the repository.

**Prerequisites:**
- `uv` — Python package manager and virtual environment tool
- `just` — task runner (optional; `uv run` scripts cover the same cases)
- A C++17 compiler and CMake
- The Qrack C++ library (installed separately — see below)

**One-time setup:**

```bash
git clone https://github.com/yourusername/qrackbind
cd qrackbind
uv sync --dev                   # create venv and install Python dependencies
bash scripts/install_qrack.sh   # install the Qrack C++ library
```

**Daily workflow:**

```bash
just build    # build the extension (or: uv run build)
just test     # run the test suite (or: uv run test)
just stubs    # regenerate .pyi stubs after binding changes
```

---

## Why Developers Need a Separate Qrack Install

The end-user wheel bundles Qrack at **wheel-build time** on the CI runner. When a developer runs `just build` locally, they are performing a fresh compile from source — not unpacking a pre-built wheel. CMake must find the Qrack headers and library on the developer's machine in order to compile the extension. The `FetchContent` auto-download in `cmake/FetchQrack.cmake` handles this automatically if Qrack is not found on the system, but developers who need a specific build (CUDA, CPU-only, double precision) install Qrack manually first.

The `scripts/install_qrack.sh` script handles the most common cases:

```bash
bash scripts/install_qrack.sh           # auto-detect (PPA, OpenCL, CPU-only)
bash scripts/install_qrack.sh --ppa     # Ubuntu PPA (simplest on Ubuntu)
bash scripts/install_qrack.sh --cpu     # CPU-only, no OpenCL
bash scripts/install_qrack.sh --cuda    # CUDA build from source
```

---

## scikit-build-core and uv — Key Constraints

### scikit-build-core has no pre-build hook

scikit-build-core's only extension points are CMake itself, `cmake.define` config settings, and `SKBUILD_*` environment variables. There is no mechanism to run a script before CMake fires. This means the `scripts/install_qrack.sh` script cannot be wired into `pip install` or `uv pip install` automatically. It must be run manually, or Qrack must be made available to CMake through `FetchContent`.

### Passing CMake flags through scikit-build-core

The `cmake.define` config-settings mechanism is the canonical way to pass configuration to CMake at install time:

```bash
# Specify a custom Qrack install path
uv pip install . --config-settings "cmake.define.QRACK_LIB_DIR=/opt/qrack/lib"

# CPU-only build
uv pip install . --config-settings "cmake.define.ENABLE_OPENCL=OFF"

# CUDA build
uv pip install . --config-settings "cmake.define.ENABLE_CUDA=ON"
```

Defaults for these defines can be set in `pyproject.toml`:

```toml
[tool.scikit-build.cmake.define]
ENABLE_OPENCL = "ON"
ENABLE_CUDA   = "OFF"
```

### uv run scripts

Since uv is already a required developer tool, scripts defined in `[tool.uv.scripts]` are available without installing anything extra. These provide the same convenience as the justfile without requiring `just`:

```toml
[tool.uv.scripts]
install-qrack     = "python scripts/install_qrack.py"
install-qrack-cpu = "python scripts/install_qrack.py --cpu"
install-qrack-cuda= "python scripts/install_qrack.py --cuda"
build             = "uv pip install -e . --no-build-isolation"
test              = "pytest tests/ -v"
stubs             = "python -m nanobind.stubgen -i build/ -m _qrackbind_core -o src/qrackbind/"
```

```bash
uv run install-qrack   # install Qrack (auto-detect)
uv run build           # build qrackbind
uv run test            # run tests
```

The justfile remains available for developers who prefer it and have `just` installed.

---

## GitHub Actions — Wheel Build Pipeline

The wheel build is triggered by pushing a version tag. cibuildwheel runs inside GitHub Actions, builds the wheel for each supported platform inside a manylinux container, and publishes to PyPI via trusted publishing (no API key required).

**`.github/workflows/wheels.yml`:**

```yaml
name: Build and publish wheels

on:
  push:
    tags:
      - "v*"
  workflow_dispatch:

jobs:
  build_wheels:
    name: Build wheels
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pypa/cibuildwheel@v2.21.3
      - uses: actions/upload-artifact@v4
        with:
          name: wheels
          path: ./wheelhouse/*.whl

  publish:
    needs: build_wheels
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v')
    environment: pypi
    permissions:
      id-token: write
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: wheels
          path: dist/
      - uses: pypa/gh-action-pypi-publish@release/v1
```

**`pyproject.toml` — cibuildwheel section:**

```toml
[tool.cibuildwheel]
build = "cp39-* cp310-* cp311-* cp312-* cp313-*"
skip  = "*-win32 *-manylinux_i686 *-musllinux*"
build-frontend = "build"

[tool.cibuildwheel.linux]
# manylinux_2_34 (AlmaLinux 9, glibc 2.34) is the highest standard PyPA image.
# Qrack's pre-built archives on the releases page (manylinux_2_35 / manylinux_2_39)
# are too new to use here without a custom base image — so Qrack is built from
# source inside the container, producing a libqrack.so that matches the
# container's glibc and lets the wheel claim manylinux_2_34 honestly.
# Resulting wheel runs on Ubuntu 22.04+, RHEL 9+, Fedora 36+, Debian 12+.
manylinux-x86_64-image = "manylinux_2_34"

# Build-time deps + Qrack source build with OpenCL enabled.
# ocl-icd-devel provides libOpenCL.so.1 as a link-time stub. The actual
# ICD loader is provided by the user's distro at runtime — we do NOT
# bundle it (see repair-wheel-command below).
before-all = """
    dnf install -y cmake gcc-c++ git ocl-icd-devel opencl-headers
    git clone --depth=1 --branch vm6502q.v10.7.0 \\
        https://github.com/unitaryfoundation/qrack.git /tmp/qrack
    cmake -S /tmp/qrack -B /tmp/qrack/build \\
        -DCMAKE_BUILD_TYPE=Release \\
        -DCMAKE_INSTALL_PREFIX=/usr/local \\
        -DENABLE_OPENCL=ON \\
        -DBUILD_SHARED_LIBS=ON
    cmake --build /tmp/qrack/build --parallel
    cmake --install /tmp/qrack/build
    ldconfig
"""

# Exclude the OpenCL ICD loader from auditwheel bundling.
# Bundling it would pin the wheel to whatever vendor was present in the
# build container and break on every other user's machine. Users supply
# libOpenCL.so.1 via ocl-icd-libopencl1 (Debian/Ubuntu) or ocl-icd (Fedora/RHEL).
repair-wheel-command = "auditwheel repair -w {dest_dir} {wheel} --exclude libOpenCL.so.1"

environment = { QRACK_INCLUDE_DIR="/usr/local/include", QRACK_LIB_DIR="/usr/local/lib64" }
```

**Release workflow:**

```bash
# Bump version in pyproject.toml, commit, then:
git tag v0.1.0
git push origin v0.1.0
# GitHub Actions builds and publishes automatically
```

---

## OpenCL — Build-Time vs Runtime

OpenCL on Linux uses a two-layer architecture that the wheel has to respect to be portable.

**The ICD loader (`libOpenCL.so.1`)** is a thin shim shipped by the user's distribution (`ocl-icd-libopencl1` on Debian/Ubuntu, `ocl-icd` on Fedora/RHEL). It does no GPU work itself — it dlopens vendor driver libraries listed under `/etc/OpenCL/vendors/*.icd` at runtime.

**The vendor driver** is the actual OpenCL implementation: NVIDIA's proprietary driver, AMD's `rocm-opencl-runtime`, Intel's `intel-opencl-icd`, `pocl` for CPU-only, or the WSL2 d3d12 stub at `/usr/lib/wsl/lib/libOpenCL.so.1`.

The wheel must link **against** the loader at build time but must **not bundle** it. Bundling pins the wheel to whichever ICD loader was present in the build container, which then conflicts with the user's real loader and breaks vendor driver discovery. This is the same pattern `pyqrack` uses today and the same pattern `torch` uses for CUDA.

The cibuildwheel config above implements this:

- `dnf install ocl-icd-devel opencl-headers` provides `libOpenCL.so.1` as a link-time stub inside the container.
- Qrack is built with `-DENABLE_OPENCL=ON` so the OpenCL code paths are compiled in.
- `auditwheel repair --exclude libOpenCL.so.1` keeps the loader as an unresolved external in the final wheel — the linker records a dependency on `libOpenCL.so.1` but the file is not copied into the wheel.

**End-user runtime requirement.** The README installation section needs a note like:

> qrackbind requires an OpenCL ICD loader at runtime. On Debian/Ubuntu: `apt install ocl-icd-libopencl1`. On Fedora/RHEL: `dnf install ocl-icd`. For GPU acceleration also install a vendor driver: NVIDIA's proprietary driver (CUDA toolkit installs the OpenCL ICD), `rocm-opencl-runtime` (AMD), `intel-opencl-icd` (Intel), or `pocl-opencl-icd` (CPU fallback). On WSL2 with an NVIDIA GPU the Windows host driver provides the ICD via `/usr/lib/wsl/lib/libOpenCL.so.1` automatically — see [[OpenCL on WSL2 with NVIDIA GPU]].
>
> If `import qrackbind` fails with `libOpenCL.so.1: cannot open shared object file`, the loader is missing.

**Build-runner GPU not required.** GitHub Actions runners have no GPU, but the build only needs the link-time stub. Real driver discovery happens at the user's runtime. CI does not need to be a GPU runner.

**Optional fail-fast.** Qrack's `qrack_cl_precompile` binary, which ships in the source tree's bin output, can be invoked from a `qrackbind-precompile` console-script entry point to force OpenCL kernel compilation up front. This converts a confusing JIT-time failure on first simulator construction into a clear early error if no ICD or vendor driver is configured. Cheap to add later if support burden warrants it.

---

## CMake FetchContent — How the Auto-Download Works

`cmake/FetchQrack.cmake` is included by `CMakeLists.txt` when `find_library` does not locate Qrack on the system. This is the developer-machine fallback that makes `just build` work on a clean clone without a prior manual Qrack install.

> [!note] Not used in the CI wheel build
> The CI pipeline builds Qrack from source inside `before-all` (see cibuildwheel section above) rather than using FetchContent, because the pre-built archives on the Qrack releases page require glibc 2.35+ which is newer than the `manylinux_2_34` build container. FetchContent is purely for developer convenience.

> [!warning] Developer-side glibc compatibility
> FetchContent downloads `libqrack-manylinux_2_35_x86_64.zip` by default, which requires glibc 2.35. Developers on systems with glibc < 2.35 will hit a runtime error and need to fall back to `bash scripts/install_qrack.sh --cpu` or `--cuda` to build Qrack from source. `cmake/FetchQrack.cmake` should detect glibc and emit an actionable error message when the prebuilt archive would not run.

```cmake
# CMakeLists.txt — Qrack discovery with auto-download fallback
find_library(QRACK_LIB qrack
    HINTS $ENV{QRACK_LIB_DIR} /usr/local/lib /usr/lib/qrack /usr/lib)

find_path(QRACK_INCLUDE qfactory.hpp
    HINTS $ENV{QRACK_INCLUDE_DIR}
          /usr/local/include/qrack
          /usr/include/qrack)

if(NOT QRACK_LIB OR NOT QRACK_INCLUDE)
    message(STATUS "Qrack not found on system — fetching pre-built binary...")
    include(cmake/FetchQrack)
endif()
```

`cmake/FetchQrack.cmake` detects the platform, downloads the correct archive from the Qrack GitHub releases page, and sets `QRACK_LIB` and `QRACK_INCLUDE` for the rest of the build.

---

## Inspecting Installed Qrack Build Features

There is no official CLI utility in Qrack that prints a summary of which features were compiled in. One binary does ship with the installation — `qrack_cl_precompile` — which precompiles OpenCL kernels for all available devices and saves them to `~/.qrack/`. Its presence confirms OpenCL support is compiled in, but it reports nothing about other feature flags.

The following approaches cover the full picture:

### `qrack_cl_precompile` — OpenCL kernel precompilation

```bash
# Confirm it exists in the install
find /usr/local/bin /usr/bin -name "qrack_cl_precompile" 2>/dev/null

# Run it to precompile kernels for all detected OpenCL devices
# (avoids JIT compilation overhead on first simulator use)
qrack_cl_precompile

# Precompiled kernels are saved to ~/.qrack/ by default.
# Override with QRACK_OCL_PATH environment variable.
# Delete the directory to force JIT recompilation.
```

### Library symbol inspection

Works on any installed `libqrack.a` or `libqrack.so` regardless of how it was built:

```bash
# OpenCL support (ENABLE_OPENCL=ON at build time)
nm /usr/lib/qrack/libqrack.a 2>/dev/null | grep -c "clBuildProgram"

# CUDA support (ENABLE_CUDA=ON)
nm /usr/lib/qrack/libqrack.a 2>/dev/null | grep -c "cudaLaunchKernel"

# __int128 bitCapInt (ENABLE_UINT128=ON — requires nb::sig() overrides in qrackbind)
nm /usr/lib/qrack/libqrack.a 2>/dev/null | grep -c "__int128"
```

### Header inspection

`qrack_types.hpp` reflects the precision and integer width the library was compiled with:

```bash
grep -E "FPPOW|UINTPOW|typedef.*bitCapInt|using.*bitCapInt|real1\b" \
    /usr/include/qrack/qrack_types.hpp | head -10
```

Key values to check:

| Finding | Meaning for qrackbind |
|---|---|
| `bitCapInt` = `uint64_t` | Standard — no special handling needed |
| `bitCapInt` = `unsigned __int128` | `nb::sig()` overrides required on all `bitCapInt` params/returns |
| `real1` = `float` | Standard — `real1_f` maps to Python `float` |
| `real1` = `double` | Double precision — state vector buffers use `complex128` internally |

### CMake cache (source builds only)

```bash
grep -E "ENABLE_OPENCL|ENABLE_CUDA|FPPOW|UINTPOW|ENABLE_UINT128" \
    /path/to/qrack/build/CMakeCache.txt
```

### `just info` — combined diagnostic report

The Phase 9 justfile includes an `info` target that runs all of the above in one command:

```makefile
info:
    #!/usr/bin/env bash
    echo "=== Qrack library ==="
    LIB=$(find /usr/lib/qrack /usr/local/lib /usr/lib -name "libqrack.*" 2>/dev/null | head -1)
    [ -n "${LIB}" ] && echo "Location: ${LIB}" || echo "libqrack: not found"
    if [ -n "${LIB}" ]; then
        nm "${LIB}" 2>/dev/null | grep -q "clBuildProgram"  \
            && echo "OpenCL:   compiled in" || echo "OpenCL:   not compiled in"
        nm "${LIB}" 2>/dev/null | grep -q "cudaLaunchKernel" \
            && echo "CUDA:     compiled in" || echo "CUDA:     not compiled in"
        nm "${LIB}" 2>/dev/null | grep -q "__int128" \
            && echo "uint128:  YES — nb::sig() overrides required in qrackbind" \
            || echo "uint128:  NO — standard uint64_t bitCapInt"
    fi
    echo ""
    echo "=== Qrack headers ==="
    HFILE="/usr/include/qrack/qrack_types.hpp"
    [ -f "${HFILE}" ] \
        && grep -E "FPPOW|UINTPOW|bitCapInt|real1\b" "${HFILE}" | head -6 \
        || echo "qrack_types.hpp not found"
    echo ""
    echo "=== qrack_cl_precompile ==="
    command -v qrack_cl_precompile &>/dev/null \
        && echo "Found: $(which qrack_cl_precompile)" \
        || echo "Not found (normal for static-library installs)"
    echo ""
    echo "=== OpenCL runtime ==="
    command -v clinfo &>/dev/null \
        && clinfo 2>/dev/null | grep "Number of platforms" \
        || echo "clinfo not installed"
    echo ""
    echo "=== CUDA ==="
    command -v nvcc &>/dev/null \
        && nvcc --version | head -1 \
        || echo "nvcc not found"
    echo ""
    echo "=== WSL2 ==="
    ls /usr/lib/wsl/lib/libOpenCL.so.1 2>/dev/null \
        && echo "WSL2 stub: present" \
        || echo "WSL2 stub: not found"
    echo ""
    echo "=== ldconfig path ==="
    ldconfig -p 2>/dev/null | grep qrack || echo "libqrack: not on ldconfig path"
```

Running `just info` before filing any build issue provides the full picture of what is installed and which qrackbind code paths are relevant.

---

## Phase 9 Completion Checklist

```
□ cmake/FetchQrack.cmake downloads correct archive for Linux x86_64
□ cmake/FetchQrack.cmake detects glibc < 2.35 and emits actionable error
□ cmake/FetchQrack.cmake downloads correct archive for macOS ARM64
□ cmake/FetchQrack.cmake gives actionable error for unsupported platforms
□ pip install qrackbind succeeds on a clean Linux x86_64 machine (Ubuntu 22.04+, RHEL 9+, Fedora 36+)
□ pip install qrackbind succeeds on a clean macOS ARM64 machine
□ System Qrack install (PPA) takes precedence over FetchContent download
□ QRACK_LIB_DIR env var overrides both system and FetchContent paths
□ scripts/install_qrack.sh --ppa works on Ubuntu 22.04+
□ scripts/install_qrack.sh --cpu builds and installs CPU-only Qrack
□ scripts/install_qrack.sh --cuda builds and installs CUDA Qrack
□ scripts/install_qrack.sh auto-detect selects PPA on Ubuntu
□ uv run install-qrack works from a clean clone
□ uv run build produces a working extension after install-qrack
□ uv run test passes after uv run build
□ justfile build target works for developers who have just installed
□ GitHub Actions wheels.yml triggers on version tags
□ Wheel builds succeed in cibuildwheel manylinux_2_34 container
□ Qrack source build in before-all completes with ENABLE_OPENCL=ON
□ auditwheel repair --exclude libOpenCL.so.1 produces a manylinux_2_34 wheel
□ Built wheel imports successfully on a system with ocl-icd-libopencl1 installed
□ Built wheel emits clear error on a system without libOpenCL.so.1
□ Built wheel discovers NVIDIA OpenCL on a host with the proprietary driver
□ Built wheel discovers WSL2 d3d12 OpenCL stub on Windows + WSL2 + NVIDIA
□ PyPI trusted publishing configured (no API key stored in repo)
□ pip install qrackbind from PyPI — python -c "from qrackbind import QrackSimulator; print(sim)" works
□ Wheel is self-contained — no system Qrack required at runtime
□ README documents OpenCL ICD runtime requirement per distro
□ pyproject.toml cmake.define defaults documented
□ README installation section matches actual user experience
```

---

## Related

- [[qrackbind GitHub Wheel Publishing]]
- [[qrackbind Streamlined Installation]]
- [[qrackbind Installing Qrack]]
- [[qrackbind Project Phase Breakdown]]
- [[qrackbind Starter — Scaffold and First Binding]]
- [[OpenCL on WSL2 with NVIDIA GPU]]
