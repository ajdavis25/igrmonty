#!/bin/bash
#SBATCH --job-name=auto_munit_bhg
#SBATCH --partition=anantuabhg
#SBATCH --account=anantuabhg
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=20-00:01:00
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

CSV_PATH="/work/vmo703/data/paper_output_dedupe.csv"
TOTAL_ROWS=$(( $(wc -l < "$CSV_PATH") - 1 ))
LAST_ROW=$(( TOTAL_ROWS - 1 ))
MAX_CONCURRENT="${MAX_CONCURRENT:-${SLURM_NTASKS}}"
SKIP_CRIT_RHIGH_DUPES="${SKIP_CRIT_RHIGH_DUPES:-1}"

ROW_LIST_FILE="$(mktemp)"

if (( TOTAL_ROWS <= 0 )); then
  echo "[ERROR] no data rows found in ${CSV_PATH}" >&2
  exit 1
fi

if [[ "${SKIP_CRIT_RHIGH_DUPES}" == "1" ]]; then
  python3 - <<'PY' > "${ROW_LIST_FILE}"
import csv
import math
from pathlib import Path

csv_path = Path("/work/vmo703/data/paper_output_dedupe.csv")
seen = set()
rows = []


def _norm_key(key: str) -> str:
    return "".join(ch.lower() for ch in key if ch.isalnum())


def find_row_value(row: dict, *candidate_keys: str):
    key_map = {_norm_key(str(k)): k for k in row.keys()}
    for cand in candidate_keys:
        match_key = key_map.get(_norm_key(cand))
        if match_key is None:
            continue
        raw = row.get(match_key)
        if raw is None:
            continue
        val = str(raw).strip()
        if val == "":
            continue
        return val
    return None


def parse_positive(raw: str, *, default: float, label: str) -> float:
    if raw is None:
        return default
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} must be finite and > 0 (got {raw!r})")
    return value

with csv_path.open("r", newline="") as fh:
    reader = csv.DictReader(fh)
    for idx, row in enumerate(reader):
        state = str(row.get("state", "")).strip().upper()
        model = str(row.get("model", "")).strip().upper()
        spin = str(row.get("spin", "")).strip()
        dump_idx = str(row.get("dump_index", "")).strip()
        rhigh = str(row.get("Rhigh", "")).strip()

        if model.startswith("CRITBETA"):
            beta_crit = parse_positive(
                find_row_value(row, "beta_crit", "betaCrit", "betacrit"),
                default=1.0,
                label="beta_crit",
            )
            beta_crit_coeff = parse_positive(
                find_row_value(
                    row,
                    "beta_crit_coefficient",
                    "beta_crit_coeff",
                    "betaCritCoefficient",
                    "f",
                ),
                default=0.5,
                label="beta_crit_coefficient/f",
            )
            key = (
                state,
                model,
                spin,
                dump_idx,
                f"{beta_crit:.12g}",
                f"{beta_crit_coeff:.12g}",
            )
        else:
            key = (state, model, spin, dump_idx, rhigh)

        if key in seen:
            continue
        seen.add(key)
        rows.append(idx)

for idx in rows:
    print(idx)
PY
else
  seq 0 "${LAST_ROW}" > "${ROW_LIST_FILE}"
fi

mapfile -t ROW_LIST < "${ROW_LIST_FILE}"
rm -f "${ROW_LIST_FILE}"

echo "[INFO] job=${SLURM_JOB_ID} selected_rows=${#ROW_LIST[@]} max_concurrent=${MAX_CONCURRENT} skip_crit_rhigh_dupes=${SKIP_CRIT_RHIGH_DUPES}"

for ROW in "${ROW_LIST[@]}"; do
  while (( $(jobs -pr | wc -l) >= MAX_CONCURRENT )); do
    sleep 1
  done

  srun --exclusive -N1 -n1 -c1 \
    python "$BRACKET" --row "$ROW" \
    > /work/vmo703/igrmonty_logs/row_${SLURM_JOB_ID}_${ROW}.out \
    2> /work/vmo703/igrmonty_logs/row_${SLURM_JOB_ID}_${ROW}.err &
done

wait
echo "[INFO] all rows finished"
