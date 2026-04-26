
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

install:
    rm -rf build/
    uv pip install . --no-build-isolation

# Compile C++, create Python wheel and run nanobind stubgen
build:
    uv build --no-build-isolation

# NOTE: each variant pins its own build-dir under build/ so that switching
# between configurations (default ↔ cpu ↔ cuda ↔ debug ↔ ...) does not
# reuse stale CMake cache fragments from a prior configuration. Without
# this, CMake's compiler-check try_compile gets wired to the wrong
# generator and fails with "ninja: error: Makefile:5: expected '='".

build-debug:
    uv pip install . --no-build-isolation \
        --config-settings "build-dir=build/{wheel_tag}-debug" \
        --config-settings "cmake.build-type=Debug"

build-verbose:
    uv pip install . --no-build-isolation -v \
        --config-settings "build-dir=build/{wheel_tag}-verbose" \
        --config-settings "build.verbose=true" \
        --config-settings "logging.level=DEBUG"

# CPU only build (no OpenCL / GPU acceleration)
build-cpu:
    uv pip install . --no-build-isolation \
        --config-settings "build-dir=build/{wheel_tag}-cpu" \
        --config-settings "cmake.define.ENABLE_OPENCL=OFF"

# CUDA GPU build (NVIDIA)
build-cuda:
    uv pip install . --no-build-isolation \
        --config-settings "build-dir=build/{wheel_tag}-cuda" \
        --config-settings "cmake.define.ENABLE_CUDA=ON"

# Double precision floating point mode
build-double:
    uv pip install . --no-build-isolation \
        --config-settings "build-dir=build/{wheel_tag}-double" \
        --config-settings "cmake.define.FPPOW=6"

# Disable CPU SIMD instructions for compatibility
build-no-simd:
    uv pip install . --no-build-isolation \
        --config-settings "build-dir=build/{wheel_tag}-nosimd" \
        --config-settings "cmake.define.ENABLE_SSE3=OFF" \
        --config-settings "cmake.define.ENABLE_AVX=OFF"

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
    uv run mypy src/{{package}} --ignore-missing-imports

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
    uv run mypy .

publish: wheel
    uv publish

wheel:
    uv build --wheel --no-build-isolation
