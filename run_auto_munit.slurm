#!/bin/bash
#SBATCH --job-name=auto_munit_c2
#SBATCH --partition=compute2
#SBATCH --nodes=1
#SBATCH --ntasks=80
#SBATCH --cpus-per-task=1
#SBATCH --time=10-00:01:00
#SBATCH --output=/work/vmo703/igrmonty_logs/%x_%j.out
#SBATCH --error=/work/vmo703/igrmonty_logs/%x_%j.err

set -euo pipefail

module purge
module load gcc/11.3.0
module load hdf5/1.12.0

export LD_LIBRARY_PATH="/work/vmo703/aricarte/gsl-1.16/lib:/work/vmo703/aricarte/gsl-1.16/lib64:${LD_LIBRARY_PATH:-}"
source /work/vmo703/ipole_venv/bin/activate

SCRIPT_DIR="/work/vmo703/igrmonty"
BRACKET="${SCRIPT_DIR}/auto_munit_bracket.py"
cd "$SCRIPT_DIR"

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=true
export OMP_PLACES=cores

MAX_CONCURRENT=80

for ROW in $(seq 0 336); do
  srun --exclusive -N1 -n1 -c1 \
    python "$BRACKET" --row "$ROW" \
    > /work/vmo703/igrmonty_logs/row_${SLURM_JOB_ID}_${ROW}.out \
    2> /work/vmo703/igrmonty_logs/row_${SLURM_JOB_ID}_${ROW}.err &
  while (( $(jobs -pr | wc -l) >= MAX_CONCURRENT )); do
    sleep 1
  done
done

wait
echo "[INFO] all rows finished"
