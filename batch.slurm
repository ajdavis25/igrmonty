#!/bin/bash
#SBATCH --job-name=grmonty_m87
#SBATCH --output=/work/vmo703/igrmonty_logs/%x_%A_%a.out
#SBATCH --error=/work/vmo703/igrmonty_logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --time=2-00:00:00
#SBATCH --partition=compute2
#SBATCH --array=10-17%10

# optional overrides when submitting:
#   sbatch --export=RUN_STATE=SANE,RUN_DUMPS="4000 4200" igrmonty/batch.slurm
# RUN_STATE defaults to ALL and maps directly to run_spectra.sh.
# ROWS_PER_JOB controls how many CSV rows a single array element processes.

set -euo pipefail

module purge
module load gcc/11.3.0
module load hdf5/1.12.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export LD_LIBRARY_PATH=/work/vmo703/aricarte/gsl-1.16/lib:/work/vmo703/aricarte/gsl-1.16/lib64:$LD_LIBRARY_PATH

CSV=/work/vmo703/data/munits_results.csv
NROWS=$(($(wc -l < "$CSV") - 1))
RUN_STATE="${RUN_STATE:-ALL}"
RUN_DUMPS="${RUN_DUMPS:-}"
ROWS_PER_JOB=${ROWS_PER_JOB:-1}

CHUNK_INDEX=${SLURM_ARRAY_TASK_ID}
CHUNK_START=$((CHUNK_INDEX * ROWS_PER_JOB))
if (( CHUNK_START >= NROWS )); then
  echo "Chunk $CHUNK_INDEX beyond available rows ($NROWS); skipping."
  exit 0
fi
CHUNK_END=$((CHUNK_START + ROWS_PER_JOB - 1))
if (( CHUNK_END >= NROWS )); then
  CHUNK_END=$((NROWS - 1))
fi

read -r -a EXTRA_DUMPS <<<"$RUN_DUMPS"

LOGROOT="/work/vmo703/igrmonty_logs"
mkdir -p "$LOGROOT"

cd /work/vmo703/igrmonty

for ((ROW = CHUNK_START; ROW <= CHUNK_END; ROW++)); do
  CMD=(./run_spectra.sh --row "$ROW" "$RUN_STATE")
  if ((${#EXTRA_DUMPS[@]})); then
    CMD+=("${EXTRA_DUMPS[@]}")
  fi
  echo "[INFO] $(date) launching row $ROW / $NROWS with: ${CMD[*]}"
  srun --exclusive "${CMD[@]}"
  echo "[INFO] $(date) completed row $ROW"
done
