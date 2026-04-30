#!/bin/bash
#SBATCH --job-name=auto_munit_bhg
#SBATCH --partition=anantuabhg,compute3,compute2,compute1
#SBATCH --account=anantuabhg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=3-00:00:00
#SBATCH --array=0-47%5
#SBATCH --output=/work/vmo703/igrmonty_logs/%x_%A_%a.out
#SBATCH --error=/work/vmo703/igrmonty_logs/%x_%A_%a.err

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

CSV_PATH="${CSV_PATH:-/work/vmo703/data/final_grmonty_paper.csv}"
TOTAL_ROWS=$(( $(wc -l < "$CSV_PATH") - 1 ))
SKIP_CRIT_RHIGH_DUPES="${SKIP_CRIT_RHIGH_DUPES:-1}"
EXPAND_POSITRON_BRANCHES="${EXPAND_POSITRON_BRANCHES:-1}"
AUTO_MUNIT_RESUME="${AUTO_MUNIT_RESUME:-1}"

TASK_LIST_FILE="$(mktemp)"
trap 'rm -f "${TASK_LIST_FILE}"' EXIT
export CSV_PATH
export SKIP_CRIT_RHIGH_DUPES
export EXPAND_POSITRON_BRANCHES

if (( TOTAL_ROWS <= 0 )); then
  echo "[ERROR] no data rows found in ${CSV_PATH}" >&2
  exit 1
fi

python3 - <<'PY' > "${TASK_LIST_FILE}"
import csv
import math
import os
import re
from pathlib import Path

csv_path = Path(os.environ["CSV_PATH"])
skip_crit_rhigh_dupes = os.environ.get("SKIP_CRIT_RHIGH_DUPES", "1") == "1"
expand_positron_branches = os.environ.get("EXPAND_POSITRON_BRANCHES", "1") == "1"
selected_rows = []
seen = set()


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


def parse_nonnegative(raw: str, label: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be finite and >= 0 (got {raw!r})")
    return value


def format_float_tag(value: float) -> str:
    nearest = round(value)
    if abs(value - nearest) < 1e-12:
        return str(int(nearest))
    return f"{value:.6g}"


def discover_positron_runs(row: dict):
    raw = find_row_value(
        row,
        "positron_ratio",
        "positronRatio",
        "positron frac",
        "positron_frac",
        "positron fraction",
        "pair_ratio",
        "pair ratio",
        "pair_fraction",
        "pair frac",
        "pos",
    )
    if raw is not None:
        return [(parse_nonnegative(raw, "positron_ratio"), "explicit")]

    found = []
    seen_ratios = set()
    for key, raw_value in row.items():
        if raw_value is None:
            continue
        value_str = str(raw_value).strip()
        if value_str == "":
            continue

        key_compact = re.sub(r"\s+", "", str(key))
        match = re.match(
            r"^munitused_?pos(?P<tag>\d+(?:\.\d+)?)$",
            key_compact,
            flags=re.IGNORECASE,
        )
        if match is None:
            continue

        ratio = parse_nonnegative(match.group("tag"), f"CSV column '{key}' positron tag")
        ratio_key = f"{ratio:.12g}"
        if ratio_key in seen_ratios:
            continue
        seen_ratios.add(ratio_key)
        found.append((ratio, f"csv:{key}"))

    if found:
        found.sort(key=lambda item: item[0])
        return found

    return [(0.0, "default")]


with csv_path.open("r", newline="") as fh:
    reader = csv.DictReader(fh)
    rows = list(reader)

for idx, row in enumerate(rows):
    if skip_crit_rhigh_dupes:
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
            dedupe_key = (
                state,
                model,
                spin,
                dump_idx,
                f"{beta_crit:.12g}",
                f"{beta_crit_coeff:.12g}",
            )
        else:
            dedupe_key = (state, model, spin, dump_idx, rhigh)

        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

    selected_rows.append(idx)

for row_idx in selected_rows:
    row = rows[row_idx]
    if expand_positron_branches:
        for ratio, _source in discover_positron_runs(row):
            ratio_tag = format_float_tag(ratio)
            print(f"{row_idx}\t{ratio:.12g}\t_pos{ratio_tag}")
    else:
        print(f"{row_idx}\t\t")
PY

mapfile -t TASK_LIST < "${TASK_LIST_FILE}"

if (( ${#TASK_LIST[@]} == 0 )); then
  echo "[INFO] job=${SLURM_JOB_ID:-manual} selected_tasks=0; nothing to do"
  exit 0
fi

TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if ! [[ "${TASK_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] SLURM_ARRAY_TASK_ID must be a nonnegative integer (got ${TASK_INDEX})" >&2
  exit 1
fi

if (( TASK_INDEX >= ${#TASK_LIST[@]} )); then
  echo "[INFO] task_index=${TASK_INDEX} selected_tasks=${#TASK_LIST[@]}; nothing to do"
  exit 0
fi

if [[ -n "${SLURM_ARRAY_TASK_MAX:-}" ]] && (( SLURM_ARRAY_TASK_MAX + 1 < ${#TASK_LIST[@]} )); then
  echo "[WARN] array range ends at ${SLURM_ARRAY_TASK_MAX}, but selected_tasks=${#TASK_LIST[@]}; later tasks will not run" >&2
fi

IFS=$'\t' read -r ROW POSITRON_RATIO TASK_SUFFIX <<< "${TASK_LIST[TASK_INDEX]}"

CMD=(python "$BRACKET" --csv "$CSV_PATH" --row "$ROW")
if [[ -n "${POSITRON_RATIO}" ]]; then
  CMD+=(--positron-ratio "$POSITRON_RATIO")
fi
if [[ "${AUTO_MUNIT_RESUME}" == "1" ]]; then
  CMD+=(--resume)
fi

JOB_TAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
LOG_STEM="row_${JOB_TAG}_${TASK_INDEX}_${ROW}${TASK_SUFFIX}"

echo "[INFO] job=${SLURM_JOB_ID:-manual} array_task=${TASK_INDEX} csv=${CSV_PATH} total_rows=${TOTAL_ROWS} selected_tasks=${#TASK_LIST[@]} row=${ROW} positron_ratio=${POSITRON_RATIO:-auto} partitions=${SLURM_JOB_PARTITION:-unknown}"

srun --exclusive -N1 -n1 -c "${SLURM_CPUS_PER_TASK:-1}" "${CMD[@]}" \
  > "/work/vmo703/igrmonty_logs/${LOG_STEM}.out" \
  2> "/work/vmo703/igrmonty_logs/${LOG_STEM}.err"

echo "[INFO] task finished row=${ROW} positron_ratio=${POSITRON_RATIO:-auto}"
