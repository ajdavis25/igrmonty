#!/usr/bin/env bash
set -euo pipefail

show_usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [ALL|SANE|MAD] [dump_idx ...]

Options:
  --row N       Only process the Nth data row of munit CSV (0-based index).
  --help        Show this message.
EOF
}

ROW_FILTER=""

while (($# > 0)); do
  case "$1" in
    --row)
      ROW_FILTER="${2:-}"
      shift 2
      ;;
    --row=*)
      ROW_FILTER="${1#*=}"
      shift
      ;;
    --help|-h)
      show_usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      show_usage
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

if [[ -n "$ROW_FILTER" && ! "$ROW_FILTER" =~ ^[0-9]+$ ]]; then
  echo "Invalid value for --row: $ROW_FILTER" >&2
  exit 1
fi

STATE_FILTER="ALL"
if (($# > 0)); then
  STATE_FILTER="${1^^}"
  shift
fi

case "$STATE_FILTER" in
  ALL|SANE|MAD)
    ;;
  *)
    echo "Invalid state filter: $STATE_FILTER" >&2
    show_usage
    exit 1
    ;;
esac

REQUESTED_DUMPS=("$@")

should_run_dump() {
  local idx="$1"
  if ((${#REQUESTED_DUMPS[@]} == 0)); then
    return 0
  fi
  for req in "${REQUESTED_DUMPS[@]}"; do
    if [[ "$req" == "$idx" ]]; then
      return 0
    fi
  done
  return 1
}

model_to_electron_mode() {
  local raw="${1^^}"
  case "$raw" in
    RBETA|RBETAWJET)
      echo 2
      ;;
    CRITBETA|CRITBETAWJET)
      echo 3
      ;;
    *)
      return 1
      ;;
  esac
}

format_spin_tag() {
  local spin="$1"
  spin="${spin// /}"
  spin="${spin#+}"
  if [[ "$spin" == -* ]]; then
    echo "$spin"
  else
    echo "+$spin"
  fi
}

state_to_prefix() {
  case "$1" in
    SANE)
      echo "S"
      ;;
    MAD)
      echo "M"
      ;;
    *)
      return 1
      ;;
  esac
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

export LD_LIBRARY_PATH="/work/vmo703/aricarte/gsl-1.16/lib:/work/vmo703/aricarte/gsl-1.16/lib64:${LD_LIBRARY_PATH:-}"

CSV="$REPO_ROOT/data/munits_results.csv"
DUMP_DIR="$REPO_ROOT/grmhd_dump_samples"
OUT_DIR="$REPO_ROOT/igrmonty_outputs/m87"
LOG_DIR="$SCRIPT_DIR/logs"

NS=2e5
MBH=6.5e9
TP_OVER_TE_VALUE=3.0
FIT_BIAS_NS=50000
BIAS=0.05
TARGET_RATIO=1.4142135623730951

mkdir -p "$OUT_DIR" "$LOG_DIR"

[[ -s "$CSV" ]] || { echo "Missing CSV: $CSV" >&2; exit 1; }
[[ -d "$DUMP_DIR" ]] || { echo "Missing dump directory: $DUMP_DIR" >&2; exit 1; }
[[ -x "$SCRIPT_DIR/grmonty" ]] || { echo "grmonty binary not found in $SCRIPT_DIR" >&2; exit 1; }

row_index=-1
rows_ran=0
while IFS=, read -r dump_idx timestep sim_state model spin pos MunitOffset MunitSlope Munit MunitUsed notes; do
  ((row_index++))
  if [[ -n "$ROW_FILTER" && "$row_index" -ne "$ROW_FILTER" ]]; then
    continue
  fi

  dump_idx="${dump_idx// /}"
  dump_idx="${dump_idx//$'\r'/}"
  [[ -n "$dump_idx" ]] || continue
  should_run_dump "$dump_idx" || continue

  sim_state="${sim_state//$'\r'/}"
  sim_state="${sim_state^^}"
  if [[ "$STATE_FILTER" != "ALL" && "$sim_state" != "$STATE_FILTER" ]]; then
    continue
  fi

  state_prefix="$(state_to_prefix "$sim_state")" || { echo "Skipping row with unknown state '$sim_state' (dump $dump_idx)" >&2; continue; }

  spin="${spin// /}"
  spin="${spin//$'\r'/}"
  spin_tag="$(format_spin_tag "$spin")"

  dump_file="$DUMP_DIR/${state_prefix}a${spin_tag}_${dump_idx}.h5"
  if [[ ! -f "$dump_file" ]]; then
    echo "Skipping missing dump $dump_file" >&2
    continue
  fi

  model="${model//$'\r'/}"
  electron_mode="$(model_to_electron_mode "$model")" || { echo "Skipping unknown model '$model' for dump $dump_idx" >&2; continue; }

  csv_munit_used="${MunitUsed// /}"
  csv_munit_used="${csv_munit_used//$'\r'/}"
  if [[ -n "$csv_munit_used" ]]; then
    MunitUsed="$csv_munit_used"
  elif [[ "$sim_state" == "SANE" ]]; then
    MunitUsed="1.83e27"
  elif [[ "$sim_state" == "MAD" ]]; then
    MunitUsed="7.49e24"
  else
    echo "Unknown state '$sim_state' for dump $dump_idx; cannot assign M_unit." >&2
    continue
  fi

  pos="${pos// /}"
  pos="${pos//$'\r'/}"
  formatted_munit="$(printf '%.3e' "$MunitUsed")"

  tag="${sim_state}_${model}_a${spin_tag}_t${dump_idx}_pos${pos}_M${formatted_munit}"
  par_file="$LOG_DIR/${tag}.par"
  log_file="$LOG_DIR/${tag}.log"
  spectrum_path="$OUT_DIR/spectrum_${state_prefix}a${spin_tag}_${dump_idx}_${model}_pos${pos}.h5"

  cat > "$par_file" <<EOF_PAR
seed -1
Ns $NS
MBH $MBH
M_unit $MunitUsed
dump $dump_file
spectrum $spectrum_path

fit_bias 1
fit_bias_ns $FIT_BIAS_NS
bias $BIAS
ratio $TARGET_RATIO

TP_OVER_TE $TP_OVER_TE_VALUE
beta_crit 1.0
beta_crit_coefficient 0.5
with_electrons $electron_mode
trat_small 1
trat_large 20
Thetae_max 1e100
EOF_PAR

  echo "[$sim_state] dump $dump_idx spin $spin_tag model $model pos $pos -> $(basename "$spectrum_path")"
  ./grmonty -par "$par_file" > "$log_file" 2>&1

  ((rows_ran++))
done < <(tail -n +2 "$CSV") || true

if [[ -n "$ROW_FILTER" && "$rows_ran" -eq 0 ]]; then
  echo "Requested row $ROW_FILTER not found or was skipped by filters." >&2
  exit 2
fi
