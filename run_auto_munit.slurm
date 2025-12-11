#!/bin/bash

#SBATCH --job-name=auto_munit
#SBATCH --partition=compute2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --output=/work/vmo703/igrmonty_logs/auto_munit_%x_%A_%a.out
#SBATCH --error=/work/vmo703/igrmonty_logs/auto_munit_%x_%A_%a.err

set -euo pipefail

# ---------------------------
# ENVIRONMENT (GRMONTY)
# ---------------------------
module purge
module load gcc/11.3.0
module load hdf5/1.12.0

export LD_LIBRARY_PATH="/work/vmo703/aricarte/gsl-1.16/lib:/work/vmo703/aricarte/gsl-1.16/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---------------------------
# PYTHON VENV
# ---------------------------
source /work/vmo703/ipole_venv/bin/activate

# ---------------------------
# PATHS
# ---------------------------
SCRIPT_DIR="/work/vmo703/igrmonty"
BRACKET="${SCRIPT_DIR}/auto_munit_bracket.py"

# ROW index (array safe)
ROW="${ROW:-$SLURM_ARRAY_TASK_ID}"

echo "[INFO] Running auto_munit_bracket for row $ROW"
cd "$SCRIPT_DIR"

# ---------------------------
# CALL PYTHON WITH ALL FLAGS
# ---------------------------
python "$BRACKET" \
    --row "$ROW" \
    "$@"

echo "[INFO] Finished row $ROW"
