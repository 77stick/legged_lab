#!/usr/bin/env bash
# Package the legged_lab project for porting to another machine.
#
# What gets INCLUDED by default:
#   - All source code, configs, scripts, docs
#   - Local rsl_rl fork (./rsl_rl)
#   - Motion data + robot assets under source/legged_lab/legged_lab/data/
#   - .codex/env.local.sh.example (template)
#   - Top-level config files (pyproject.toml, AGENTS.md, README.md, ...)
#   - .git/ (including .git/lfs/) unless --no-git is given
#
# What gets EXCLUDED (always):
#   - logs/, outputs/, temp/, wandb/, runs/, videos/, recordings/
#   - **/__pycache__/, **/*.pyc, **/*.egg-info/
#   - .pytest_cache, .mypy_cache
#   - .codex/env.local.sh (machine-specific; must be recreated on target)
#   - Untitled (scratch file)
#
# Optional:
#   --with-checkpoint PATH   Include the given checkpoint file (relative to project root).
#                            Can be repeated. Example:
#                              --with-checkpoint logs/rsl_rl/v1_amp/2026-04-20_14-37-03/model_3000.pt
#
# Usage:
#   bash scripts/tools/pack_project.sh                  # default: .tar.zst, include .git
#   bash scripts/tools/pack_project.sh --format gz      # use gzip instead of zstd
#   bash scripts/tools/pack_project.sh --no-git         # smaller archive, no git history
#   bash scripts/tools/pack_project.sh --output /tmp/ll.tar.zst
#   bash scripts/tools/pack_project.sh --dry-run        # just list what would be packed
#
set -euo pipefail

# ------------------------------------------------------------------
# Resolve project root (script lives in <root>/scripts/tools/)
# ------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_NAME="$(basename "${PROJECT_ROOT}")"

if [[ ! -f "${PROJECT_ROOT}/pyproject.toml" || ! -d "${PROJECT_ROOT}/.codex" ]]; then
    echo "ERROR: ${PROJECT_ROOT} does not look like the legged_lab project root." >&2
    exit 1
fi

# ------------------------------------------------------------------
# Defaults and argument parsing
# ------------------------------------------------------------------
FORMAT=""
OUTPUT=""
INCLUDE_GIT=1
DRY_RUN=0
EXTRA_CHECKPOINTS=()

usage() {
    sed -n '1,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//' | sed -n '2,$p'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --format)           FORMAT="$2"; shift 2 ;;
        --format=*)         FORMAT="${1#*=}"; shift ;;
        --output|-o)        OUTPUT="$2"; shift 2 ;;
        --output=*)         OUTPUT="${1#*=}"; shift ;;
        --no-git)           INCLUDE_GIT=0; shift ;;
        --with-git)         INCLUDE_GIT=1; shift ;;
        --with-checkpoint)  EXTRA_CHECKPOINTS+=("$2"); shift 2 ;;
        --dry-run)          DRY_RUN=1; shift ;;
        -h|--help)          usage ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

# Auto-detect format: prefer zstd, fall back to gzip.
if [[ -z "${FORMAT}" ]]; then
    if command -v zstd >/dev/null 2>&1; then
        FORMAT="zst"
    else
        FORMAT="gz"
    fi
fi

case "${FORMAT}" in
    zst)
        command -v zstd >/dev/null 2>&1 || { echo "zstd not found on PATH"; exit 3; }
        TAR_COMPRESS=(--use-compress-program "zstd -T0 -19")
        EXT="tar.zst"
        ;;
    gz)
        TAR_COMPRESS=(--gzip)
        EXT="tar.gz"
        ;;
    *) echo "Unsupported format: ${FORMAT} (expected zst|gz)"; exit 4 ;;
esac

# Default output path: one level above project root, dated.
if [[ -z "${OUTPUT}" ]]; then
    STAMP="$(date +%Y%m%d-%H%M%S)"
    OUTPUT="$(dirname "${PROJECT_ROOT}")/${PROJECT_NAME}-${STAMP}.${EXT}"
fi

# ------------------------------------------------------------------
# Build exclude list (paths are relative to the tar transform, i.e.
# relative to <project_name>/ inside the archive).
# ------------------------------------------------------------------
EXCLUDE_FILE="$(mktemp)"
trap 'rm -f "${EXCLUDE_FILE}"' EXIT

cat > "${EXCLUDE_FILE}" <<'EOF'
logs
outputs
temp
tmp
wandb
runs
videos
recordings
docker/artifacts
docker/.cache
Untitled
.codex/env.local.sh
.pytest_cache
.mypy_cache
.ruff_cache
.ipynb_checkpoints
*.pyc
*.pyo
*.pyd
*.dmp
*.tmp
__pycache__
*.egg-info
EOF

# Optionally exclude .git (including LFS cache).
if [[ "${INCLUDE_GIT}" -eq 0 ]]; then
    echo ".git" >> "${EXCLUDE_FILE}"
fi

# ------------------------------------------------------------------
# Sanity-check extra checkpoints (they are inside logs/ which is excluded;
# we add them back via explicit --add-file).
# ------------------------------------------------------------------
ADD_FILES=()
for ck in "${EXTRA_CHECKPOINTS[@]+"${EXTRA_CHECKPOINTS[@]}"}"; do
    if [[ "${ck}" = /* ]]; then
        echo "ERROR: --with-checkpoint expects a path relative to project root, got: ${ck}" >&2
        exit 5
    fi
    if [[ ! -f "${PROJECT_ROOT}/${ck}" ]]; then
        echo "ERROR: checkpoint not found: ${ck}" >&2
        exit 6
    fi
    ADD_FILES+=(--add-file "${ck}")
done

# ------------------------------------------------------------------
# Dry-run: just list what would be included (filtered by excludes).
# ------------------------------------------------------------------
echo "========================================================"
echo "  legged_lab packaging"
echo "========================================================"
echo "  Project root : ${PROJECT_ROOT}"
echo "  Output       : ${OUTPUT}"
echo "  Format       : ${EXT}"
echo "  Include .git : $([[ ${INCLUDE_GIT} -eq 1 ]] && echo yes || echo no)"
if [[ ${#EXTRA_CHECKPOINTS[@]} -gt 0 ]]; then
    echo "  Extra files  :"
    for ck in "${EXTRA_CHECKPOINTS[@]}"; do echo "                 ${ck}"; done
fi
echo "========================================================"

cd "$(dirname "${PROJECT_ROOT}")"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[DRY RUN] File list that would be packed (top 40 lines):"
    tar -cvf /dev/null \
        --exclude-from="${EXCLUDE_FILE}" \
        "${ADD_FILES[@]+"${ADD_FILES[@]}"}" \
        "${PROJECT_NAME}" 2>&1 | head -40
    echo "..."
    # Total uncompressed size estimate (directory size minus excluded dirs).
    EST="$(du -sb --exclude=logs --exclude=outputs --exclude=temp --exclude=wandb \
        --exclude=__pycache__ --exclude=.pytest_cache \
        "${PROJECT_NAME}" 2>/dev/null | awk '{print $1}')"
    if [[ -n "${EST:-}" ]]; then
        echo "Estimated uncompressed size: $(numfmt --to=iec "${EST}")"
    fi
    exit 0
fi

# ------------------------------------------------------------------
# Real pack.
# ------------------------------------------------------------------
START_TS=$(date +%s)
tar -cf "${OUTPUT}" \
    "${TAR_COMPRESS[@]}" \
    --exclude-from="${EXCLUDE_FILE}" \
    --warning=no-file-changed \
    "${ADD_FILES[@]+"${ADD_FILES[@]}"}" \
    "${PROJECT_NAME}"
END_TS=$(date +%s)

# ------------------------------------------------------------------
# Checksum and summary.
# ------------------------------------------------------------------
SHA="${OUTPUT}.sha256"
(cd "$(dirname "${OUTPUT}")" && sha256sum "$(basename "${OUTPUT}")") > "${SHA}"

SIZE_BYTES="$(stat -c%s "${OUTPUT}")"
SIZE_HUMAN="$(numfmt --to=iec --suffix=B "${SIZE_BYTES}")"

echo "========================================================"
echo "  DONE in $((END_TS - START_TS))s"
echo "========================================================"
echo "  Archive : ${OUTPUT}"
echo "  Size    : ${SIZE_HUMAN}"
echo "  SHA256  : $(awk '{print $1}' "${SHA}")"
echo
echo "Copy to the new machine, then on that machine:"
echo "  tar -xf $(basename "${OUTPUT}")"
echo "  cd ${PROJECT_NAME}"
echo "  cat docs/PORTING.md      # follow the bootstrap steps"
echo "========================================================"
