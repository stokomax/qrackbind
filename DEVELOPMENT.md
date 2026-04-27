# Development Guide

This document walks through setting up a local development environment for
`qrackbind` on Ubuntu Linux: installing the Qrack C++ runtime from the
official PPA, installing the Python build toolchain (uv + scikit-build-core
+ CMake), and using the `justfile` recipes to compile the nanobind extension.

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Ubuntu 18.04 / 20.04 / 22.04 / 24.04 LTS | Required by the Qrack PPA |
| Python 3.12+ | Required by nanobind's stable ABI |
| C++17 compiler | `g++` ≥ 9 or `clang++` ≥ 10 |
| `cmake` ≥ 3.15 and `ninja` | Drives the C++ build |
| `git`, `curl` | For cloning + installing `uv` |
| OpenCL ICD (optional) | For GPU acceleration; `sudo apt install ocl-icd-opencl-dev` |
| CUDA Toolkit (optional) | Only if you want `just build-cuda` |

```sh
sudo apt update
sudo apt install -y build-essential cmake ninja-build git curl \
                    software-properties-common
```

## 2. Install Qrack from the PPA

`qrackbind` does **not** vendor Qrack — it links against a system install.
The Unitary Foundation publishes prebuilt `.deb` packages on Launchpad:

```sh
sudo add-apt-repository ppa:wrathfulspatula/vm6502q
sudo apt update
sudo apt install libqrack-dev
```

This installs:

- Headers under `/usr/include/qrack/` (notably `qfactory.hpp`)
- The shared library `libqrack.so` under `/usr/lib/`

`CMakeLists.txt` discovers them automatically via:

```cmake
find_library(QRACK_LIB qrack
    HINTS $ENV{QRACK_LIB_DIR} /usr/local/lib/qrack /usr/lib/qrack)
find_path(QRACK_INCLUDE qfactory.hpp
    HINTS $ENV{QRACK_INCLUDE_DIR}
          /usr/local/include/qrack
          /usr/include/qrack)
```

If you have a custom (e.g. source-built) Qrack install in a non-standard
location, point CMake at it via environment variables before building:

```sh
export QRACK_LIB_DIR=/opt/qrack/lib
export QRACK_INCLUDE_DIR=/opt/qrack/include/qrack
```

> **Note on Qrack build options.** GPU / CUDA / float-precision flags are
> baked into `libqrack.so` at the time it was compiled — `qrackbind` cannot
> override them. The `just build-cuda` / `just build-double` recipes only
> control how the *binding layer* is compiled. To use CUDA at runtime, you
> need a CUDA-enabled `libqrack.so` (the PPA build is OpenCL-enabled by
> default).

## 3. Build toolchain: uv, scikit-build-core, CMake

Three tools cooperate to turn the C++ sources under `bindings/` into an
importable Python module. Understanding their roles makes the `justfile`
recipes much easier to read.

### uv

[`uv`](https://docs.astral.sh/uv/) is a fast Python package + virtual-env
manager from Astral. In this project it is responsible for:

- Creating the project venv (`uv venv`)
- Resolving and locking dependencies declared in `pyproject.toml` against
  `uv.lock` (`uv sync`)
- Driving editable and wheel installs (`uv pip install -e .`, `uv build`)
- Running tools inside the venv (`uv run pytest`, `uv run mypy`, …)

Install it once, system-wide:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
# or:  pipx install uv
```

You also need [`just`](https://github.com/casey/just) for the recipes:

```sh
sudo apt install just        # 22.04+ has it; otherwise: cargo install just
```

### scikit-build-core

[`scikit-build-core`](https://scikit-build-core.readthedocs.io/) is the
PEP 517 build backend declared in `pyproject.toml`. It is the bridge
between Python packaging and CMake:

1. When you run `uv pip install .` (or `uv build`), pip hands control to
   scikit-build-core.
2. scikit-build-core invokes CMake to **configure** the project, then
   **builds** the `_core` nanobind extension with Ninja.
3. It collects the resulting shared library, the generated `_core.pyi`
   stubs, and the `py.typed` marker into the wheel under
   `src/qrackbind/`.

The `--config-settings` flags you see in `justfile` recipes are
scikit-build-core options forwarded to CMake. For example:

```sh
uv pip install . --no-build-isolation \
    --config-settings "build-dir=build/{wheel_tag}-cuda" \
    --config-settings "cmake.define.ENABLE_CUDA=ON"
```

is exactly equivalent to

```sh
cmake -S . -B build/... -G Ninja -DENABLE_CUDA=ON
ninja -C build/...
```

…except the resulting artefacts are bundled into a wheel and installed
into the active Python environment.

### CMake (with Ninja)

`CMakeLists.txt` is the source of truth for the C++ build:

- `find_package(Python 3.12 …)` locates the interpreter and dev headers.
- `find_package(nanobind CONFIG REQUIRED)` pulls in nanobind's CMake
  helpers from the venv.
- `find_library(QRACK_LIB qrack ...)` / `find_path(QRACK_INCLUDE ...)`
  resolve the system Qrack install (see §2).
- `nanobind_add_module(_core … bindings/*.cpp)` defines the extension.
- `nanobind_add_stub(...)` regenerates `_core.pyi` after each build.

The CMake options that `qrackbind` exposes (and which the `build-*`
recipes toggle via `cmake.define.<KEY>=<VAL>`):

| Option | Default | Purpose |
|---|---|---|
| `ENABLE_OPENCL` | `ON` | Link the OpenCL ICD when present |
| `ENABLE_CUDA` | `OFF` | Compile against CUDA runtime |
| `ENABLE_SSE3` | `ON` | Allow SSE3 intrinsics in the binding layer |
| `ENABLE_AVX` | `ON` | Allow AVX intrinsics in the binding layer |
| `FPPOW` | `5` | Floating-point precision exponent (5 = single, 6 = double) |
| `BUILD_CPP_TESTS` | `OFF` | Build optional C++ unit tests |

### How it fits together

```
$ just build-cuda
   └── uv pip install . --no-build-isolation \
         --config-settings build-dir=build/<tag>-cuda \
         --config-settings cmake.define.ENABLE_CUDA=ON
         └── scikit-build-core   (PEP 517 backend)
              └── cmake -G Ninja -DENABLE_CUDA=ON …
                   ├── find_package(Python 3.12)
                   ├── find_package(nanobind)
                   ├── find_library(QRACK_LIB qrack)   ← from libqrack-dev
                   └── nanobind_add_module(_core …)
                        └── ninja → src/qrackbind/_core.<abi>.so
```

Each `build-*` recipe pins its own `build/<wheel_tag>-<variant>` directory
so that switching variants (cpu ↔ cuda ↔ debug …) does not reuse a stale
CMake cache from a previous configuration.

## 4. Clone and bootstrap the Python env

```sh
git clone https://github.com/stokomax/qrackbind.git
cd qrackbind

uv venv                       # create .venv/ using the pinned Python
source .venv/bin/activate

just sync                     # install build deps + uv sync against uv.lock
```

`just sync` runs `just setup` (which installs `scikit-build-core`,
`nanobind`, `cmake`, and `ninja` into the venv) and then `uv sync` to
materialise the locked development dependencies (`pytest`, `mypy`,
`ruff`, …).

## 5. Build qrackbind via the justfile

Run `just` (or `just --list`) to see every available recipe. The
build-related ones:

| Command | Description |
|---|---|
| `just dev` | Editable install — fastest iteration loop |
| `just install` | Clean release install (`rm -rf build/` first) |
| `just build` | Build a wheel into `dist/` (default config) |
| `just build-debug` | `cmake.build-type=Debug` |
| `just build-verbose` | Verbose ninja + DEBUG-level build logs |
| `just build-cpu` | `ENABLE_OPENCL=OFF` (CPU only) |
| `just build-cuda` | `ENABLE_CUDA=ON` (NVIDIA GPU) |
| `just build-double` | `FPPOW=6` (double precision) |
| `just build-no-simd` | Disable SSE3 + AVX for compatibility |
| `just build-define KEY=VAL` | Pass an arbitrary `-DKEY=VAL` to CMake |
| `just stubs` | Regenerate `.pyi` via `nanobind.stubgen` |
| `just check-stubs` | `stubs` + `mypy` against the generated stubs |
| `just test` | `just install` + `uv run pytest` |
| `just test-fast [args]` | `pytest [args]` without reinstalling |
| `just fmt` / `just lint` / `just typecheck` | Ruff format / lint / mypy |
| `just clean` | Remove `dist/`, `build/`, `*.egg-info`, stray `.so` files |
| `just clean-all` | `clean` + remove `.venv` |
| `just wheel` | `uv build --wheel` |
| `just publish` | Build a wheel and `uv publish` it |

### Recommended first build

```sh
just dev          # editable install of qrackbind into the active venv
just test         # run the full pytest suite
```

If everything is wired up correctly the smoke test below should print a
3-element list of booleans:

```sh
python -c "
from qrackbind import QrackSimulator
sim = QrackSimulator(qubitCount=3)
sim.h(0); sim.cnot(0, 1); sim.x(2)
print(sim.measure_all())
"
```

## 6. Troubleshooting

- **`Qrack library not found. Set QRACK_LIB_DIR env var.`**
  `libqrack-dev` is not installed, or you installed Qrack to a custom
  prefix. Either run the PPA install in §2 or export `QRACK_LIB_DIR` /
  `QRACK_INCLUDE_DIR`.

- **`Qrack headers not found.`** Same cause — the package
  `libqrack-dev` ships the headers under `/usr/include/qrack/`. The
  runtime-only `libqrack` package is *not* enough.

- **`ninja: error: Makefile:5: expected '='`** when switching variants.
  A previous `build/` directory configured a different generator. Run
  `just clean` (or rely on the per-variant build dirs the `build-*`
  recipes already use).

- **`isOpenCL=True` silently returns CPU results.** The Qrack runtime
  downgrades to CPU on hosts without an OpenCL ICD. Install
  `ocl-icd-opencl-dev` plus a vendor ICD (`mesa-opencl-icd`,
  `intel-opencl-icd`, `nvidia-opencl-icd`, …).

- **Stubs out of date.** Run `just stubs` after editing any binding
  signature; `just check-stubs` then verifies them with `mypy`.

- **Stale CMake cache after switching Python versions.** `just clean-all`
  removes `.venv` too — recreate it with `uv venv && just sync`.

## 7. Project documentation: `memory-bank/` and `specs/`

Two top-level documentation folders capture context that lives outside
the code itself. Contributors — and AI coding assistants such as Cline —
are expected to read them before making non-trivial changes.

### `memory-bank/` — Cline memory bank

`qrackbind` follows the [Cline](https://cline.bot) memory-bank convention
described in the project-root `.clinerules`. Because Cline starts each
session with no recollection of prior work, the memory bank is the
single source of truth for project context. The core files are:

| File | Role |
|---|---|
| `projectbrief.md` | Foundational scope and requirements |
| `productContext.md` | Why the project exists, UX / API goals |
| `systemPatterns.md` | Architecture, key technical decisions, design patterns |
| `techContext.md` | Tech stack, build/dev setup, constraints, dependencies |
| `activeContext.md` | Current focus, recent changes, next steps |
| `progress.md` | What works, what's left, status, known issues |

When you make a meaningful change — new patterns, new constraints,
shifted priorities — update the relevant memory-bank file in the same
commit. If a contributor (human or agent) tells Cline to **"update
memory bank"**, it will review *all* files and refresh them.

At the start of every Cline session, the agent re-reads the entire
`memory-bank/` directory before touching code; keeping these files
accurate is therefore the most direct way to steer future automated
work.

### `specs/` — phase specs and design notes

`specs/` holds the phased implementation plan and longer-form design
documents that don't fit in the memory bank:

- `phase_1.md` … `phase_6.md` — per-phase scope, acceptance criteria,
  and implementation notes. The README's "Project Phases and Status"
  table tracks completion.
- `compatibility_review.md` — pyqrack ↔ qrackbind API drift analysis.
- `bloqade_pyrack_dependency.md` — integration notes for downstream
  Bloqade users.
- `type_casting.md` — nanobind type-caster reference; consult before
  adding any new STL or custom-type parameter to a binding.

When starting work on a new phase, read the matching `specs/phase_N.md`
together with the current `memory-bank/activeContext.md` and
`memory-bank/progress.md` to align on scope before writing code.

## 8. Reference

- Project README: [`README.md`](./README.md)
- Build configuration: [`CMakeLists.txt`](./CMakeLists.txt),
  [`pyproject.toml`](./pyproject.toml), [`justfile`](./justfile)
- Qrack upstream: <https://github.com/unitaryfoundation/qrack>
- uv docs: <https://docs.astral.sh/uv/>
- scikit-build-core docs: <https://scikit-build-core.readthedocs.io/>
- nanobind docs: <https://nanobind.readthedocs.io/>
