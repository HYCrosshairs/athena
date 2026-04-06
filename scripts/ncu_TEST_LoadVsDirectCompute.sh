#!/usr/bin/env bash
# Run Nsight Compute on TEST_LoadVsDirectCompute with the full metric preset (--set full).
#
# Usage:
#   NCU_USE_SUDO=1 ./scripts/ncu_TEST_LoadVsDirectCompute.sh [CMAKE_BINARY_DIR]
#   Avoid plain `sudo ./scripts/...`: root’s PATH often omits CUDA/Nsight Compute (then: NCU=/path/to/ncu).
#
# Default binary dir is <repo>/build. Build first:
#   cmake --build <dir> --target TEST_LoadVsDirectCompute
#
# Open the report:
#   ncu-ui ncu_TEST_LoadVsDirectCompute.ncu-rep
#
# Optional / required environment:
#   NCU_USE_SUDO=1 (recommended)  Run ncu under sudo — required on most Linux installs unless you
#                                 changed the driver policy (see below). Without this you still get
#                                 ERR_NVGPUCTRPERM and no real counter data.
#   NCU_ALLOW_NONADMIN=1          Skip the sudo check (use after NVreg_RestrictProfilingToAdminUsers=0
#                                 or Windows “all users” GPU counters; see link below).
#   NCU=/path/to/ncu             Use if ncu is not on PATH (common when using sudo: secure_path)
#   NCU_REPORT=/path/to/prefix   Base name for -o (default: <repo>/ncu_TEST_LoadVsDirectCompute)
#   NCU_IMPORT_SOURCE=no         Omit --import-source yes if your ncu is too old
#
# ERR_NVGPUCTRPERM fixes:
#   Linux (quick):  NCU_USE_SUDO=1 ./scripts/ncu_TEST_LoadVsDirectCompute.sh
#   Linux (persist): https://developer.nvidia.com/ERR_NVGPUCTRPERM modprobe.d + reboot
#   WSL2: On Windows host open NVIDIA Control Panel → Desktop → Enable Developer Settings;
#         Developer → Manage GPU Performance Counters → allow all users. Then reconnect WSL.
#
# This test runs many kernels (warmup + timed sweep). For a shorter replay, use the small test,
# replacing BUILD with your CMake binary dir (e.g. build):
#   ncu --set full -f --import-source yes -o ncu_small \\
#       BUILD/tests/concepts/TEST_LoadVsDirectCompute -t LoadVsDirectCompute.ResultsMatchBetweenPaths

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY_DIR="${1:-${REPO_ROOT}/build}"
EXE="${BINARY_DIR}/tests/concepts/TEST_LoadVsDirectCompute"

if [[ ! -x "$EXE" ]]; then
    echo "Missing executable: ${EXE}" >&2
    echo "Build: cmake --build \"${BINARY_DIR}\" --target TEST_LoadVsDirectCompute" >&2
    exit 1
fi

if [[ "$(id -u)" -ne 0 ]] && [[ "${NCU_USE_SUDO:-}" != "1" ]] && [[ "${NCU_ALLOW_NONADMIN:-}" != "1" ]]; then
    echo "" >&2
    echo "ncu: refusing to run as a normal user — you will hit ERR_NVGPUCTRPERM (no counter data)." >&2
    echo "  Fix:  NCU_USE_SUDO=1 $0 \"${BINARY_DIR}\"" >&2
    echo "  Or:   NCU_ALLOW_NONADMIN=1 ...  (only if the driver already allows your user; see script header)" >&2
    echo "" >&2
    exit 1
fi

OUT="${NCU_REPORT:-${REPO_ROOT}/ncu_TEST_LoadVsDirectCompute}"
IMPORT=(--import-source yes)
if [[ "${NCU_IMPORT_SOURCE:-}" == "no" ]]; then
    IMPORT=()
fi

# sudo uses a minimal PATH — resolve ncu to a full path (or honor $NCU).
resolve_ncu_bin() {
    if [[ -n "${NCU:-}" ]]; then
        if [[ -x "${NCU}" ]]; then
            printf '%s\n' "${NCU}"
            return 0
        fi
        echo "NCU=${NCU} is not an executable file." >&2
        return 1
    fi
    local found
    found="$(command -v ncu 2>/dev/null || true)"
    if [[ -n "${found}" ]]; then
        printf '%s\n' "${found}"
        return 0
    fi
    local d
    for d in "/usr/local/cuda/bin" "/usr/lib/nsight-compute"; do
        if [[ -x "${d}/ncu" ]]; then
            printf '%s\n' "${d}/ncu"
            return 0
        fi
    done
    shopt -s nullglob
    for d in /opt/nvidia/nsight-compute/*/target/linux-desktop-*-x64; do
        if [[ -x "${d}/ncu" ]]; then
            printf '%s\n' "${d}/ncu"
            shopt -u nullglob
            return 0
        fi
    done
    for d in "${HOME}"/NVIDIA-Nsight-Compute-*/target/linux-desktop-*-x64; do
        if [[ -x "${d}/ncu" ]]; then
            printf '%s\n' "${d}/ncu"
            shopt -u nullglob
            return 0
        fi
    done
    shopt -u nullglob
    return 1
}

NCU_BIN="$(resolve_ncu_bin)" || {
    echo "ncu not found. Install Nsight Compute, add its directory to PATH, or set NCU=/path/to/ncu." >&2
    echo "If you use sudo, either:  NCU_USE_SUDO=1 $0   or   sudo env \"PATH=\$PATH\" $0" >&2
    exit 1
}

# --set full        : all built-in sections / metrics (replay may take several minutes)
# -f                : overwrite existing report
# --import-source   : tie metrics to CUDA source when built with -lineinfo
if [[ "${NCU_USE_SUDO:-}" == "1" ]]; then
    SUDO_ENV=(env "PATH=${PATH}")
    if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
        SUDO_ENV+=("LD_LIBRARY_PATH=${LD_LIBRARY_PATH}")
    fi
    exec sudo "${SUDO_ENV[@]}" "${NCU_BIN}" --set full -f "${IMPORT[@]}" -o "${OUT}" \
        "${EXE}" -t LoadVsDirectCompute.TippingPointSweep
fi
exec "${NCU_BIN}" --set full -f "${IMPORT[@]}" -o "${OUT}" \
    "${EXE}" -t LoadVsDirectCompute.TippingPointSweep
