#!/usr/bin/env bash
# scripts/install_qrack.sh
#
# Install the Qrack C++ library for qrackbind development.
# Not needed for end-user pip install — the wheel bundles Qrack automatically.
#
# Usage:
#   bash scripts/install_qrack.sh              # auto-detect best method
#   bash scripts/install_qrack.sh --ppa        # Ubuntu PPA (simplest on Ubuntu 22.04+)
#   bash scripts/install_qrack.sh --cpu        # build from source, CPU-only (no OpenCL)
#   bash scripts/install_qrack.sh --cuda       # build from source with CUDA support
#   bash scripts/install_qrack.sh --version X  # override Qrack version tag (default vm6502q.v10.7.0)
#
# After installation, run:
#   just build      # (or: uv pip install -e . --no-build-isolation)
#   just test

set -euo pipefail

QRACK_VERSION="${QRACK_VERSION:-vm6502q.v10.7.0}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"
MODE=""

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ppa)    MODE="ppa";  shift ;;
        --cpu)    MODE="cpu";  shift ;;
        --cuda)   MODE="cuda"; shift ;;
        --version)
            QRACK_VERSION="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--ppa|--cpu|--cuda] [--version TAG]"
            exit 1 ;;
    esac
done

# ── Auto-detect ───────────────────────────────────────────────────────────────
if [[ -z "$MODE" ]]; then
    if command -v lsb_release &>/dev/null && lsb_release -i 2>/dev/null | grep -qi ubuntu; then
        echo "install_qrack: detected Ubuntu — using PPA"
        MODE="ppa"
    else
        echo "install_qrack: non-Ubuntu system — building from source (CPU-only)"
        MODE="cpu"
    fi
fi

# ── PPA install (Ubuntu 22.04+) ───────────────────────────────────────────────
ppa_install() {
    echo "install_qrack: installing Qrack via Ubuntu PPA..."
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends software-properties-common
    sudo add-apt-repository -y ppa:wrathfulspatula/vm6502q
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends libqrack-dev
    echo "install_qrack: PPA install complete"
    echo "  Library: /usr/lib/qrack/libqrack.a (or .so)"
    echo "  Headers: /usr/include/qrack/"
}

# ── Source build (CPU-only) ───────────────────────────────────────────────────
cpu_install() {
    echo "install_qrack: building Qrack from source (CPU-only)..."
    local tmpdir
    tmpdir=$(mktemp -d)
    trap "rm -rf '$tmpdir'" EXIT

    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends \
        build-essential cmake git libboost-dev

    git clone --depth=1 --branch "${QRACK_VERSION}" \
        https://github.com/unitaryfoundation/qrack.git "${tmpdir}/qrack"

    cmake -S "${tmpdir}/qrack" -B "${tmpdir}/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
        -DENABLE_OPENCL=OFF \
        -DENABLE_CUDA=OFF
    cmake --build  "${tmpdir}/build" --parallel "$(nproc)"
    sudo cmake --install "${tmpdir}/build"

    echo "install_qrack: CPU source build complete"
    echo "  Library: ${INSTALL_PREFIX}/lib/libqrack.a"
    echo "  Headers: ${INSTALL_PREFIX}/include/qrack/"
}

# ── Source build (CUDA) ──────────────────────────────────────────────────────
cuda_install() {
    echo "install_qrack: building Qrack from source (CUDA)..."
    if ! command -v nvcc &>/dev/null; then
        echo "ERROR: nvcc not found. Install CUDA toolkit first."
        exit 1
    fi

    local tmpdir
    tmpdir=$(mktemp -d)
    trap "rm -rf '$tmpdir'" EXIT

    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends \
        build-essential cmake git libboost-dev

    git clone --depth=1 --branch "${QRACK_VERSION}" \
        https://github.com/unitaryfoundation/qrack.git "${tmpdir}/qrack"

    cmake -S "${tmpdir}/qrack" -B "${tmpdir}/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
        -DENABLE_CUDA=ON \
        -DENABLE_OPENCL=OFF
    cmake --build  "${tmpdir}/build" --parallel "$(nproc)"
    sudo cmake --install "${tmpdir}/build"

    echo "install_qrack: CUDA source build complete"
    echo "  Library: ${INSTALL_PREFIX}/lib/libqrack.a"
    echo "  Headers: ${INSTALL_PREFIX}/include/qrack/"
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
case "$MODE" in
    ppa)  ppa_install  ;;
    cpu)  cpu_install  ;;
    cuda) cuda_install ;;
esac

echo ""
echo "install_qrack: done. Run 'just build' or 'uv run build' to compile qrackbind."
