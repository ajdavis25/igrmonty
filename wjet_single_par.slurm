#!/bin/bash

#SBATCH --job-name=grmonty_wjet
#SBATCH --partition=compute2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --time=1-00:00:00
#SBATCH --output=/work/vmo703/igrmonty_logs/%x_%j.out
#SBATCH --error=/work/vmo703/igrmonty_logs/%x_%j.err

set -euo pipefail

# -------------------------------
# INPUT PAR FILE
# -------------------------------
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
DEFAULT_PAR="$SUBMIT_DIR/testjet.par"
PAR_FILE="${PAR_FILE:-$DEFAULT_PAR}"

if [[ "$PAR_FILE" != /* ]]; then
  PAR_FILE="$SUBMIT_DIR/$PAR_FILE"
fi

PAR_DIR="$(cd "$(dirname "$PAR_FILE")" && pwd)"

# -------------------------------
# HARD-SET WORKDIR
# -------------------------------
WORKDIR="/work/vmo703/igrmonty"

JOB_LABEL="${JOB_LABEL:-$(basename "${PAR_FILE%.*}")}"
MODULEFILE="${MODULEFILE:-}"

# -------------------------------
# VALIDATION
# -------------------------------
if [[ ! -f "$PAR_FILE" ]]; then
  echo "[ERROR] PAR_FILE '$PAR_FILE' not found" >&2
  exit 1
fi

if [[ ! -x "$WORKDIR/grmonty" ]]; then
  echo "[ERROR] GRMONTY NOT FOUND: $WORKDIR/grmonty" >&2
  exit 1
fi

# -------------------------------
# MODULES & ENV
# -------------------------------
module purge
module load gcc/11.3.0
module load hdf5/1.12.0

if [[ -n "$MODULEFILE" ]]; then
  source "$MODULEFILE"
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export LD_LIBRARY_PATH="/work/vmo703/aricarte/gsl-1.16/lib:/work/vmo703/aricarte/gsl-1.16/lib64:${LD_LIBRARY_PATH:-}"

LOGROOT="/work/vmo703/igrmonty_logs"
mkdir -p "$LOGROOT"

echo "[INFO] $(date) starting GRMONTY job '$JOB_LABEL' on $(hostname)"
echo "[INFO] PAR_FILE = $PAR_FILE"
echo "[INFO] WORKDIR  = $WORKDIR"

# -------------------------------
# RUN GRMONTY (DIRECT EXECUTION)
# -------------------------------
cd "$WORKDIR"
./grmonty -par "$PAR_FILE"

echo "[INFO] $(date) finished GRMONTY job '$JOB_LABEL'"
