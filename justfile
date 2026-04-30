
# Read project name from pyproject.toml at load time
# Hyphens are normalised to underscores to match SKBUILD_PROJECT_NAME behaviour
package := `python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); print(d['project']['name'].replace('-','_'))"`

default:
    @just --list

# Install build dependencies into an existing virtual environment
setup:
    uv pip install scikit-build-core nanobind cmake ninja

# Run uv sync
sync: setup
    uv sync

dev:
    uv pip install -e . --no-build-isolation

# Display Qrack library features.
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

install:
    rm -rf build/
    uv pip install . --no-build-isolation

# Compile C++, create Python wheel and run nanobind stubgen
build:
    uv build --no-build-isolation
    just retag

# Append cp313-abi3 and cp314-abi3 tags so that resolvers which do not
# fully honour the PEP 425 abi3 compatibility chain (e.g. uv on Python
# 3.14) still recognise the wheel as compatible across 3.12–3.14.
retag *args="":
    @for w in dist/{{package}}*.whl wheelhouse/{{package}}*.whl; do \
        [ -f "$w" ] || continue; \
        echo "retagging $w"; \
        uv run python -m wheel tags --python-tag '+cp313.cp314' --remove "$w"; \
    done

# NOTE: build-debug pins its own build-dir so switching between
# Release and Debug does not reuse stale CMake cache fragments.

build-debug:
    uv pip install . --no-build-isolation \
        --config-settings "build-dir=build/{wheel_tag}-debug" \
        --config-settings "cmake.build-type=Debug"

build-verbose:
    uv pip install . --no-build-isolation -v \
        --config-settings "build-dir=build/{wheel_tag}-verbose" \
        --config-settings "build.verbose=true" \
        --config-settings "logging.level=DEBUG"

build-define define:
    uv pip install . --no-build-isolation \
        --config-settings "cmake.define.{{define}}"

test: install
    uv run pytest

test-fast *args="":
    uv run pytest {{args}}

# Manually create stub files. 
stubs *args="": install
    uv run python -m nanobind.stubgen \
        -m {{package}}._core \
        -M src/{{package}}/py.typed \
        -O src/{{package}} \
        {{args}}


# Verify stubs match the installed extension (type-check against them)
check-stubs: stubs
    uv run pyright src/{{package}}

clean:
    rm -rf dist/ build/ _skbuild/ *.egg-info
    find . -name "*.so" -delete
    find . -name "*.pyd" -delete

clean-all: clean
    rm -rf .venv

fmt:
    uv run ruff format .

lint:
    uv run ruff check .

typecheck:
    uv run pyright .

# Build wheel locally using cibuildwheel (manylinux_2_34, OpenCL enabled).
# Requires Docker. Output lands in ./wheelhouse/.
# This is the spec-compliant build — matches what CI publishes to PyPI.
cibuild:
    python -m cibuildwheel --platform linux
    just retag

# CPU-only local wheel (manylinux_2_34, no OpenCL dependency).
# Faster than 'cibuild' — skips ocl-icd-devel and sets ENABLE_OPENCL=OFF.
# Useful for smoke-testing the packaging pipeline without GPU infrastructure.
cibuild-cpu:
    CIBW_BEFORE_ALL="dnf install -y cmake gcc-c++ git make && \
      git clone --depth=1 --branch vm6502q.v10.7.0 \
        https://github.com/unitaryfoundation/qrack.git /tmp/qrack_src && \
      cmake -S /tmp/qrack_src -B /tmp/qrack_build \
        -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENCL=OFF \
        -DCMAKE_INSTALL_PREFIX=/usr/local && \
      cmake --build /tmp/qrack_build --parallel \$(nproc) && \
      cmake --install /tmp/qrack_build" \
    python -m cibuildwheel --platform linux
    just retag

# Publish to TestPyPI (uses ~/.pypirc [testpypi] credentials).
# Uses 'uvx' so twine runs in an isolated env without triggering a project build.
# To publish to production PyPI when ready, run:
#   uvx twine upload --repository pypi wheelhouse/*.whl
publish:
    uvx twine upload --repository testpypi wheelhouse/*.whl

wheel:
    uv build --wheel --no-build-isolation
