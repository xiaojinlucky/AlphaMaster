#!/usr/bin/env bash
set -euo pipefail
umask 077

readonly ROOT="/hwdata/home/jinqc/Quant/AlphaMaster"
readonly PYTHON="${ROOT}/.venv/bin/python"
readonly WORKER="${ROOT}/scripts/train_slurm_worker.py"
readonly RUN_ID="${1:-}"

if [[ ! "${RUN_ID}" =~ ^run_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}$ ]]; then
    echo "invalid run_id" >&2
    exit 2
fi

readonly RUN_DIR="${ROOT}/runs/${RUN_ID}"
if [[ ! -d "${RUN_DIR}" || -L "${RUN_DIR}" ]]; then
    echo "run directory is missing or unsafe" >&2
    exit 2
fi
if [[ ! -x "${PYTHON}" || ! -f "${WORKER}" || -L "${WORKER}" ]]; then
    echo "fixed worker runtime is unavailable" >&2
    exit 2
fi

: "${SLURM_JOB_ID:?missing SLURM_JOB_ID}"
: "${SLURM_JOB_NAME:?missing SLURM_JOB_NAME}"
: "${SLURM_CPUS_PER_TASK:?missing SLURM_CPUS_PER_TASK}"
if [[ ! "${SLURM_JOB_ID}" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid SLURM_JOB_ID" >&2
    exit 2
fi
if [[ ! "${SLURM_CPUS_PER_TASK}" =~ ^[0-9]+$ ]] \
    || (( SLURM_CPUS_PER_TASK < 1 || SLURM_CPUS_PER_TASK > 64 )); then
    echo "invalid SLURM_CPUS_PER_TASK" >&2
    exit 2
fi

cd -- "${RUN_DIR}"

exec /usr/bin/env -i \
    HOME="${RUN_DIR}" \
    PATH="${ROOT}/.venv/bin:/usr/bin:/bin" \
    LANG="C.UTF-8" \
    LC_ALL="C.UTF-8" \
    PYTHONNOUSERSITE="1" \
    PYTHONDONTWRITEBYTECODE="1" \
    OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" \
    MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}" \
    OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" \
    NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}" \
    SLURM_JOB_ID="${SLURM_JOB_ID}" \
    SLURM_JOB_NAME="${SLURM_JOB_NAME}" \
    SLURM_JOB_PARTITION="${SLURM_JOB_PARTITION:-}" \
    SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}" \
    SLURMD_NODENAME="${SLURMD_NODENAME:-}" \
    SLURM_JOB_NODELIST="${SLURM_JOB_NODELIST:-}" \
    "${PYTHON}" "${WORKER}" "${RUN_ID}"
