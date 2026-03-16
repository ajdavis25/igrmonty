#!/usr/bin/env python3
"""automatic M_unit tuning for GRMONTY using gentle power-law updates.

this script:
  * reads a model row from a CSV (MAD/SANE, model, spin, dump index, Rhigh, pos),
  * runs GRMONTY with a given M_unit,
  * measures the flux at a target frequency (default 230 GHz),
  * updates M_unit via  M_new = M_old * (F_target/F_old)^(1/P),
  * repeats until the flux is within tolerance or max iterations hit,
  * logs every trial to a CSV history file,
  * optionally resumes from existing spectra/logs/parfiles.

pair convention:
  if GRMONTY `positron_ratio` is used, keep `M_unit` baryonic and do not
  externally pre-scale density/M_unit by `(1 + 2*positron_ratio)`.
"""

import argparse
import csv
import math
import re
import subprocess
import sys
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# input table of models (row-based)
DEFAULT_CSV = REPO_ROOT / "data" / "paper_output_dedupe.csv"

# GRMHD dump and output locations
DUMP_DIR = REPO_ROOT / "grmhd_dump_samples"
OUT_DIR = REPO_ROOT / "igrmonty_outputs" / "m87"
LOG_DIR = SCRIPT_DIR / "logs"

# default GRMONTY binary (adjust if needed)
GRMONTY_BIN = SCRIPT_DIR / "grmonty"

# where to store tuning history
DEFAULT_HISTORY_CSV = REPO_ROOT / "data" / "munits_tuning_history.csv"

# baseline M_unit guesses per state
STATE_DEFAULT_MUNIT: Dict[str, float] = {
    "SANE": 1.83e27,
    "MAD": 7.49e24,
}

# mapping from model name to GRMONTY with_electrons mode
MODEL_TO_ELECTRON_MODE: Dict[str, int] = {
    "RBETA": 2,
    "RBETAWJET": 4,
    "CRITBETA": 3,
    "CRITBETAWJET": 5,
}

# geometry / units for flux conversion
D_MPC = 16.8
FREQ_TARGET_HZ = 230.0e9
F_TARGET_JY = 0.5
DEFAULT_BETA_CRIT = 1.0
DEFAULT_BETA_CRIT_COEFF = 0.5

PC_TO_CM = 3.085677581e18
D_CM = D_MPC * 1e6 * PC_TO_CM
LSUN_CGS = 3.827e33
FOUR_PI = 4.0 * math.pi

TrialResult = namedtuple(
    "TrialResult",
    (
        "index",
        "munit",
        "flux",
        "spec_path",
        "par_path",
        "log_path",
        "invalid_bias_count",
    ),
)


def cap_jump(prev_munit: float, new_munit: float, max_factor: float = 30.0) -> float:
    """prevent catastrophic GRMONTY jumps (e.g. x700)."""
    if prev_munit <= 0:
        return new_munit
    if new_munit > prev_munit * max_factor:
        return prev_munit * max_factor
    if new_munit < prev_munit / max_factor:
        return prev_munit / max_factor
    return new_munit


# helpers for CSV & context
def format_spin_tag(spin: str) -> str:
    """format spin tag like '+0.5' / '-0.5' / '+0'."""
    spin = spin.strip().replace(" ", "")
    if not spin:
        return "+0"
    spin = spin.lstrip("+")
    if spin.startswith("-"):
        return spin
    return f"+{spin}"


def state_to_prefix(state: str) -> str:
    if state == "SANE":
        return "S"
    if state == "MAD":
        return "M"
    raise ValueError(f"Unknown state: {state}")


def _norm_key(key: str) -> str:
    return "".join(ch.lower() for ch in key if ch.isalnum())


def find_row_value(row: dict, *candidate_keys: str) -> Tuple[Optional[str], Optional[str]]:
    """case-insensitive/format-insensitive row lookup for CSV columns."""
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
        return val, str(match_key)
    return None, None


def require_row_value(row: dict, *candidate_keys: str) -> str:
    val, key = find_row_value(row, *candidate_keys)
    if val is None:
        raise KeyError(
            f"Required CSV column missing: one of {candidate_keys}. "
            f"Available columns: {list(row.keys())}"
        )
    _ = key  # key is useful for debugging but not needed by the caller.
    return val


def parse_nonnegative_float(raw: object, label: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be finite and >= 0 (got {raw!r})")
    return value


def format_float_tag(value: float) -> str:
    """compact stable tag used in output filenames/log tags."""
    nearest = round(value)
    if abs(value - nearest) < 1e-12:
        return str(int(nearest))
    return f"{value:.6g}"


def format_pos_tag(positron_ratio: float) -> str:
    """compact stable tag used in output filenames/log tags (e.g. 0, 0.5, 1)."""
    return format_float_tag(positron_ratio)


def parse_positive_float(raw: object, label: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} must be finite and > 0 (got {raw!r})")
    return value


def resolve_critbeta_params(row: dict, *, model: str) -> Tuple[float, float]:
    """resolve critical-beta parameters from CSV with defaults."""
    beta_crit = DEFAULT_BETA_CRIT
    beta_crit_coeff = DEFAULT_BETA_CRIT_COEFF

    if not model.upper().startswith("CRITBETA"):
        return beta_crit, beta_crit_coeff

    beta_raw, beta_key = find_row_value(
        row,
        "beta_crit",
        "betaCrit",
        "betacrit",
    )
    if beta_raw is not None:
        beta_crit = parse_positive_float(beta_raw, f"CSV column '{beta_key}'")

    coeff_raw, coeff_key = find_row_value(
        row,
        "beta_crit_coefficient",
        "beta_crit_coeff",
        "betaCritCoefficient",
        "f",
    )
    if coeff_raw is not None:
        beta_crit_coeff = parse_positive_float(
            coeff_raw,
            f"CSV column '{coeff_key}'",
        )

    return beta_crit, beta_crit_coeff


def resolve_positron_ratio(
    row: dict, *, override: Optional[float]
) -> Tuple[float, str]:
    """resolve positron_ratio from CLI override or CSV columns."""
    if override is not None:
        return parse_nonnegative_float(override, "--positron-ratio"), "cli"

    raw, key = find_row_value(
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
    if raw is None:
        return 0.0, "default"

    return parse_nonnegative_float(raw, f"CSV column '{key}'"), f"csv:{key}"


def candidate_munit_keys(positron_ratio: float, pos_tag: str) -> List[str]:
    keys = [f"MunitUsed_pos{pos_tag}"]
    if abs(positron_ratio) < 1e-12:
        keys.append("MunitUsed_pos0")
    if abs(positron_ratio - 1.0) < 1e-12:
        keys.append("MunitUsed_pos1")
    keys.extend(["MunitUsed", "M_unit"])

    # de-duplicate while preserving order
    out: List[str] = []
    for key in keys:
        if key not in out:
            out.append(key)
    return out


def discover_positron_ratios_from_munit_columns(row: dict) -> List[Tuple[float, str]]:
    """infer available positron branches from CSV columns like MunitUsed_pos0/pos1."""
    found: List[Tuple[float, str]] = []
    seen: set = set()

    for key, raw in row.items():
        if raw is None:
            continue
        raw_val = str(raw).strip()
        if raw_val == "":
            continue

        key_compact = re.sub(r"\s+", "", str(key))
        match = re.match(
            r"^munitused_?pos(?P<tag>\d+(?:\.\d+)?)$",
            key_compact,
            flags=re.IGNORECASE,
        )
        if match is None:
            continue

        ratio = parse_nonnegative_float(
            match.group("tag"),
            f"CSV column '{key}' positron tag",
        )
        ratio_key = f"{ratio:.12g}"
        if ratio_key in seen:
            continue
        seen.add(ratio_key)
        found.append((ratio, f"csv:{key}"))

    found.sort(key=lambda item: item[0])
    return found


def resolve_positron_runs(
    row: dict, *, override: Optional[float]
) -> List[Tuple[float, str]]:
    """resolve one or more positron branches to execute for this row."""
    if override is not None:
        return [(parse_nonnegative_float(override, "--positron-ratio"), "cli")]

    ratio, source = resolve_positron_ratio(row, override=None)
    if source != "default":
        return [(ratio, source)]

    inferred = discover_positron_ratios_from_munit_columns(row)
    if inferred:
        return inferred

    return [(0.0, "default")]


def load_row(csv_path: Path, row_index: int) -> dict:
    """load a single row (0-based) from the CSV as a dict with stripped fields."""
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            if idx == row_index:
                return {
                    k.strip(): (v.strip() if isinstance(v, str) else v)
                    for k, v in row.items()
                }
    raise IndexError(f"Row {row_index} not found in {csv_path}")


def infer_electron_mode(model: str) -> int:
    try:
        return MODEL_TO_ELECTRON_MODE[model.upper()]
    except KeyError as exc:
        raise ValueError(f"Unknown model '{model}' for electron mode mapping") from exc


def build_context(row: dict, *, positron_ratio_override: Optional[float] = None) -> dict:
    """build a context dict (metadata) for this tuning job from the CSV row."""
    state = require_row_value(row, "state", "MAD/SANE").upper()
    if state not in STATE_DEFAULT_MUNIT:
        raise ValueError(f"Unexpected MAD/SANE value '{state}'")

    model = require_row_value(row, "model")
    electron_mode = infer_electron_mode(model)

    spin = require_row_value(row, "spin")
    spin_tag = format_spin_tag(spin)
    dump_idx = require_row_value(row, "dump_index")
    rhigh_raw, rhigh_key = find_row_value(row, "Rhigh", "rhigh", "r_high", "trat_large")
    if rhigh_raw is None:
        raise KeyError("Required CSV column missing: one of ('Rhigh', 'rhigh', 'r_high', 'trat_large').")
    rhigh = parse_nonnegative_float(rhigh_raw, f"CSV column '{rhigh_key}'")
    if rhigh <= 0.0:
        raise ValueError(f"Rhigh/trat_large must be > 0 (got {rhigh_raw!r})")
    rhigh_tag = format_float_tag(rhigh)
    beta_crit, beta_crit_coeff = resolve_critbeta_params(row, model=model)
    beta_crit_tag = format_float_tag(beta_crit)
    beta_crit_coeff_tag = format_float_tag(beta_crit_coeff)
    state_prefix = state_to_prefix(state)

    dump_file = DUMP_DIR / f"{state_prefix}a{spin_tag}_{dump_idx}.h5"
    if not dump_file.exists():
        raise FileNotFoundError(f"Dump not found: {dump_file}")

    positron_ratio, positron_source = resolve_positron_ratio(
        row, override=positron_ratio_override
    )
    pos = format_pos_tag(positron_ratio)
    crit_suffix = ""
    if model.upper().startswith("CRITBETA"):
        crit_suffix = f"_bc{beta_crit_tag}_f{beta_crit_coeff_tag}"

    spectrum_base = (
        OUT_DIR
        / f"spectrum_{state_prefix}a{spin_tag}_{dump_idx}_{model}_rh{rhigh_tag}{crit_suffix}_pos{pos}"
    )
    log_tag = f"{state}_{model}_a{spin_tag}_t{dump_idx}_rh{rhigh_tag}{crit_suffix}_pos{pos}"

    return {
        "state": state,
        "model": model,
        "spin": spin,
        "dump_index": dump_idx,
        "rhigh": rhigh,
        "beta_crit": beta_crit,
        "beta_crit_coeff": beta_crit_coeff,
        "pos": pos,
        "positron_ratio": positron_ratio,
        "positron_source": positron_source,
        "dump_file": dump_file,
        "electron_mode": electron_mode,
        "spectrum_base": spectrum_base,
        "log_tag": log_tag,
    }


# flux measurement
def measure_flux(
    spec_path: Path, freq_target: float, *, return_freq: bool = False
) -> Tuple[float, Optional[float]]:
    """measure F_nu at freq_target from a GRMONTY spectrum HDF5 file.

    returns:
        flux_Jy (and optionally nearest frequency used).
    """
    with h5py.File(spec_path, "r") as f:
        params = f["/params"]
        nuLnu = np.array(f["/output/nuLnu"], dtype=np.float64)
        dOmega = np.array(f["/output/dOmega"], dtype=np.float64)
        numin = float(params["NUMIN"][()])
        numax = float(params["NUMAX"][()])
        nfreq = None
        try:
            nfreq = int(params["N_EBINS"][()])
        except Exception:  # noqa: BLE001
            pass
        if nfreq is None:
            try:
                nfreq = int(f["/output/lnu"].shape[0])
            except Exception:  # noqa: BLE001
                nfreq = nuLnu.shape[-1]

    if nfreq <= 0:
        raise RuntimeError(
            f"Unable to determine frequency axis length for spectrum {spec_path}"
        )

    nu = np.logspace(np.log10(numin), np.log10(numax), nfreq)

    # bring the frequency axis to the end to simplify angular reductions
    freq_axis = None
    freq_matches = [axis for axis, size in enumerate(nuLnu.shape) if size == nfreq]
    if freq_matches:
        if 1 in freq_matches:
            freq_axis = 1
        else:
            freq_axis = freq_matches[0]
    else:
        freq_axis = nuLnu.ndim - 1
    if freq_axis != nuLnu.ndim - 1:
        nuLnu = np.moveaxis(nuLnu, freq_axis, -1)

    # convert from L_sun to cgs before integrating over angles
    nuLnu_cgs = nuLnu * LSUN_CGS

    spatial_shape = nuLnu_cgs.shape[:-1]
    if not spatial_shape:
        raise RuntimeError(
            f"Spectrum {spec_path} does not contain angular information (shape={nuLnu_cgs.shape})"
        )

    d_shape = dOmega.shape
    if not d_shape:
        raise RuntimeError(
            f"dOmega from {spec_path} must have at least one dimension (shape={dOmega.shape})"
        )
    if len(d_shape) > len(spatial_shape):
        raise RuntimeError(
            "dOmega has more dimensions than nuLnu's spatial axes: "
            f"nuLnu spatial shape {spatial_shape}, dOmega {d_shape}"
        )

    def find_block(container_shape: Tuple[int, ...], target_shape: Tuple[int, ...]) -> Optional[int]:
        if len(target_shape) > len(container_shape):
            return None
        max_start = len(container_shape) - len(target_shape)
        for start in range(max_start + 1):
            if container_shape[start : start + len(target_shape)] == target_shape:
                return start
        return None

    block_start = find_block(spatial_shape, d_shape)
    if block_start is None:
        raise RuntimeError(
            "dOmega shape does not match nuLnu spatial axes: "
            f"nuLnu spatial shape {spatial_shape}, dOmega {d_shape}"
        )

    # sum over any axes before the angular block (e.g., scattering components)
    for _ in range(block_start):
        nuLnu_cgs = np.sum(nuLnu_cgs, axis=0)

    # sum over axes after the angular block (e.g., polarization states)
    spatial_ndim = nuLnu_cgs.ndim - 1
    trailing_axes = spatial_ndim - len(d_shape)
    for _ in range(trailing_axes):
        nuLnu_cgs = np.sum(nuLnu_cgs, axis=len(d_shape))

    spatial_shape = nuLnu_cgs.shape[:-1]
    if spatial_shape != d_shape:
        raise RuntimeError(
            "Failed to align nuLnu and dOmega shapes after reductions: "
            f"nuLnu spatial shape {spatial_shape}, dOmega {d_shape}"
        )

    if dOmega.ndim == 2:
        # reshape dOmega -> (N_phi, N_theta, 1) to avoid accidental broadcasting
        dOmega_weight = dOmega.reshape(dOmega.shape + (1,))
        nuLnu_4pi = np.sum(nuLnu_cgs * dOmega_weight, axis=(0, 1)) / FOUR_PI
    elif dOmega.ndim == 1:
        dOmega_weight = dOmega.reshape(dOmega.shape + (1,))
        nuLnu_4pi = np.sum(nuLnu_cgs * dOmega_weight, axis=0) / FOUR_PI
    else:
        raise RuntimeError(
            "Unsupported dOmega rank. Expected 1D or 2D weights but got "
            f"shape {dOmega.shape} from {spec_path}."
        )

    idx = int(np.argmin(np.abs(nu - freq_target)))
    freq = float(nu[idx])

    Lnu = nuLnu_4pi[idx] / freq
    Fnu_cgs = Lnu / (FOUR_PI * D_CM**2)
    Fnu_Jy = float(Fnu_cgs * 1e23)

    if not np.isfinite(Fnu_Jy) or Fnu_Jy <= 0:
        raise RuntimeError(
            f"{spec_path} produced invalid flux at {freq_target/1e9:.0f} GHz (value={Fnu_Jy})"
        )

    if return_freq:
        return Fnu_Jy, freq
    return Fnu_Jy, None


# parfile writer
def write_par_file(
    path: Path,
    *,
    ns: float,
    mbh: float,
    m_unit: float,
    dump_file: Path,
    spectrum_path: Path,
    electron_mode: int,
    tp_over_te: float,
    trat_large: float,
    beta_crit: float,
    beta_crit_coefficient: float,
    positron_ratio: float,
    bias_ns: int,
    bias_start: float,
    target_ratio: float,
) -> None:
    """write a GRMONTY parameter (.par) file for a single trial."""
    path.write_text(
        "\n".join(
            [
                "seed -1",
                f"Ns {ns}",
                f"MBH {mbh}",
                f"M_unit {m_unit:.8e}",
                f"dump {dump_file}",
                f"spectrum {spectrum_path}",
                "",
                "fit_bias 1",  # 0 off or 1 on
                f"fit_bias_ns {bias_ns}",
                f"bias {bias_start}",
                f"ratio {target_ratio}",
                "",
                f"TP_OVER_TE {tp_over_te}",
                f"beta_crit {beta_crit:.12g}",
                f"beta_crit_coefficient {beta_crit_coefficient:.12g}",
                f"with_electrons {electron_mode}",
                "trat_small 1",
                f"trat_large {trat_large:.12g}",
                f"positron_ratio {positron_ratio:.12g}",
                "Thetae_max 1e100",
                "",
            ]
        )
    )


def parse_munit_from_par(par_path: Path) -> float:
    """extract M_unit from an existing .par file."""
    text = par_path.read_text().splitlines()
    for line in text:
        if line.strip().startswith("M_unit"):
            parts = line.split()
            if len(parts) >= 2:
                return float(parts[1])
    raise RuntimeError(f"Could not find M_unit in {par_path}")


# tolerance helper
class MunitBracketer:
    """simple wrapper for flux tolerance checks."""

    def __init__(self, *, target_flux: float, rel_tol: float, abs_tol: Optional[float]):
        self.target_flux = target_flux
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol

    def within_tolerance(self, flux: float) -> bool:
        if self.abs_tol is not None:
            return abs(flux - self.target_flux) <= self.abs_tol
        return abs(flux - self.target_flux) / self.target_flux <= self.rel_tol


# tuner
class MunitTuner:
    """manage M_unit tuning and optional resume."""

    def __init__(
        self,
        *,
        context: dict,
        args: argparse.Namespace,
        bracketer: MunitBracketer,
        row: dict,
    ) -> None:
        self.context = context
        self.args = args
        self.bracketer = bracketer
        self.row = row

        self.trials: List[TrialResult] = []
        self.next_index: int = 1
        self.history_csv: Path = args.history_csv

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        if self.args.resume:
            self._load_existing_trials()

    # resume support
    def _load_existing_trials(self) -> None:
        """load existing spectra/par/logs for this model, if any."""
        base_spec = self.context["spectrum_base"]
        log_tag = self.context["log_tag"]

        existing: List[TrialResult] = []

        # 1) look for a final spectrum (no _trialXX)
        # avoid Path.with_suffix here because the spin tag contains a '.' (e.g. -0.5),
        # which Path treats as an existing suffix. appending preserves the full name.
        final_spec = base_spec.with_name(f"{base_spec.name}.h5")
        final_par = LOG_DIR / f"{log_tag}.par"
        final_log = LOG_DIR / f"{log_tag}.log"

        if final_spec.exists() and final_par.exists():
            try:
                munit = parse_munit_from_par(final_par)
                flux, _ = measure_flux(final_spec, self.args.freq_target)
                existing.append(
                    TrialResult(
                        index=0,  # final result from a previous run
                        munit=munit,
                        flux=flux,
                        spec_path=final_spec,
                        par_path=final_par,
                        log_path=final_log,
                        invalid_bias_count=0,
                    )
                )
                print(
                    f"[resume] found final spectrum {final_spec.name}: "
                    f"M_unit={munit:.4e}, F={flux:.4e} Jy",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[resume] error reading final spectrum: {exc}", file=sys.stderr)

        # 2) look for any trial*.h5 spectra
        pattern = f"{base_spec.name}_trial*.h5"
        for spec in OUT_DIR.glob(pattern):
            stem = spec.stem  # e.g. spectrum_Sa+0.5_4000_RBETA_pos0_trial03
            try:
                trial_str = stem.rsplit("trial", 1)[-1]
                idx = int(trial_str)
            except (ValueError, IndexError):
                continue

            par_path = LOG_DIR / f"{log_tag}_trial{idx:02d}.par"
            log_path = LOG_DIR / f"{log_tag}_trial{idx:02d}.log"
            if not par_path.exists():
                continue

            try:
                munit = parse_munit_from_par(par_path)
                flux, _ = measure_flux(spec, self.args.freq_target)
                existing.append(
                    TrialResult(
                        index=idx,
                        munit=munit,
                        flux=flux,
                        spec_path=spec,
                        par_path=par_path,
                        log_path=log_path,
                        invalid_bias_count=0,
                    )
                )
                print(
                    f"[resume] found trial #{idx:02d}: "
                    f"M_unit={munit:.4e}, F={flux:.4e} Jy ({spec.name})",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[resume] error reading {spec.name}: {exc}",
                    file=sys.stderr,
                )

        if not existing:
            return

        # sort by trial index
        existing.sort(key=lambda t: t.index)
        self.trials.extend(existing)

        # next trial index should follow the max used index >= 1
        used_indices = [t.index for t in existing if t.index >= 1]
        if used_indices:
            self.next_index = max(used_indices) + 1
        else:
            self.next_index = 1

        # summarize best-so-far
        best = min(self.trials, key=lambda t: abs(t.flux - self.bracketer.target_flux))
        print(
            f"[resume] best-so-far: trial #{best.index:02d} "
            f"M_unit={best.munit:.4e}, F={best.flux:.4e} Jy",
            flush=True,
        )

    # core tuning
    def _compute_next_munit(self, trial: TrialResult) -> float:
        """compute next M_unit using SAFE power-law scaling + jump cap."""
        target = self.bracketer.target_flux
        if trial.flux <= 0 or not math.isfinite(trial.flux):
            raise RuntimeError(
                f"Non-positive or non-finite flux {trial.flux} for trial {trial.index}"
            )

        scale = target / trial.flux
        p = self.args.p_factor
        if p <= 0:
            raise ValueError(f"Invalid p-factor {p}")

        # make P gentler when scale is huge
        if scale > 1e3:
            p = max(p, 4.0)
        if scale > 1e4:
            p = max(p, 5.0)
        if scale > 1e5:
            p = max(p, 6.0)

        raw_new = trial.munit * (scale ** (1.0 / p))
        m_new = cap_jump(trial.munit, raw_new, max_factor=30.0)

        # clamp to [min_munit, max_munit]
        if m_new < self.args.min_munit:
            print(
                f"[warn] proposed M_unit={m_new:.4e} < min_munit; "
                f"clamping to {self.args.min_munit:.4e}",
                flush=True,
            )
            m_new = self.args.min_munit
        if m_new > self.args.max_munit:
            print(
                f"[warn] proposed M_unit={m_new:.4e} > max_munit; "
                f"clamping to {self.args.max_munit:.4e}",
                flush=True,
            )
            m_new = self.args.max_munit

        print(
            f"[update] trial={trial.index:02d} scale={scale:.3e} P={p:.1f} "
            f"raw_M={raw_new:.4e} capped_M={m_new:.4e}",
            flush=True,
        )

        return m_new

    def _append_history(
        self, trial: TrialResult, *, converged: bool, is_resumed: bool
    ) -> None:
        """append a row describing this trial to the tuning history CSV."""
        self.history_csv.parent.mkdir(parents=True, exist_ok=True)

        file_exists = self.history_csv.exists()
        fieldnames = [
            "timestamp_utc",
            "row_index",
            "state",
            "model",
            "spin",
            "dump_index",
            "pos",
            "iteration",
            "M_unit",
            "flux_Jy",
            "target_flux_Jy",
            "freq_target_Hz",
            "p_factor",
            "Ns",
            "converged",
            "resumed_trial",
            "spec_path",
            "par_path",
            "log_path",
        ]

        with self.history_csv.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            writer.writerow(
                {
                    "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
                    "row_index": self.args.row,
                    "state": self.context["state"],
                    "model": self.context["model"],
                    "spin": self.context["spin"],
                    "dump_index": self.context["dump_index"],
                    "pos": self.context["pos"],
                    "iteration": trial.index,
                    "M_unit": f"{trial.munit:.8e}",
                    "flux_Jy": f"{trial.flux:.8e}",
                    "target_flux_Jy": f"{self.bracketer.target_flux:.8e}",
                    "freq_target_Hz": f"{self.args.freq_target:.8e}",
                    "p_factor": f"{self.args.p_factor:.3f}",
                    "Ns": f"{self.args.ns:.3e}",
                    "converged": int(bool(converged)),
                    "resumed_trial": int(bool(is_resumed)),
                    "spec_path": str(trial.spec_path),
                    "par_path": str(trial.par_path),
                    "log_path": str(trial.log_path),
                }
            )

    def run_trial(self, munit: float) -> TrialResult:
        """run a single GRMONTY trial with the given M_unit."""
        if not (self.args.min_munit <= munit <= self.args.max_munit):
            raise ValueError(
                f"M_unit={munit:.3e} outside allowed range "
                f"[{self.args.min_munit:.3e}, {self.args.max_munit:.3e}]"
            )

        idx = self.next_index
        self.next_index += 1

        spec_path = self.context["spectrum_base"].with_name(
            f"{self.context['spectrum_base'].name}_trial{idx:02d}.h5"
        )
        par_tag = f"{self.context['log_tag']}_trial{idx:02d}"
        par_path = LOG_DIR / f"{par_tag}.par"
        log_path = LOG_DIR / f"{par_tag}.log"

        write_par_file(
            par_path,
            ns=self.args.ns,
            mbh=self.args.mbh,
            m_unit=munit,
            dump_file=self.context["dump_file"],
            spectrum_path=spec_path,
            electron_mode=self.context["electron_mode"],
            tp_over_te=self.args.tp_over_te,
            trat_large=self.context["rhigh"],
            beta_crit=self.context["beta_crit"],
            beta_crit_coefficient=self.context["beta_crit_coeff"],
            positron_ratio=self.context["positron_ratio"],
            bias_ns=self.args.fit_bias_ns,
            bias_start=self.args.bias,
            target_ratio=self.args.target_ratio,
        )

        print(
            f"[trial {idx:02d}] running GRMONTY with M_unit={munit:.4e}...",
            flush=True,
        )

        with log_path.open("w") as log_fh:
            # let CalledProcessError propagate to the caller.
            subprocess.run(
                [str(self.args.grmonty_bin), "-par", str(par_path)],
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                check=True,
            )

        flux, _ = measure_flux(spec_path, self.args.freq_target)

        # inspect log for invalid bias warnings
        invalid_bias_count = 0
        try:
            with log_path.open("r") as fh:
                for line in fh:
                    if "invalid bias" in line or "bias < 1" in line:
                        invalid_bias_count += 1
        except FileNotFoundError:
            pass

        if invalid_bias_count > 0:
            print(
                f"[trial {idx:02d}] detected {invalid_bias_count} invalid-bias events",
                flush=True,
            )

        trial = TrialResult(
            index=idx,
            munit=munit,
            flux=flux,
            spec_path=spec_path,
            par_path=par_path,
            log_path=log_path,
            invalid_bias_count=invalid_bias_count,
        )
        self.trials.append(trial)

        print(
            f"[trial {trial.index:02d}] M_unit={trial.munit:.4e} -> "
            f"F={trial.flux:.6e} Jy",
            flush=True,
        )

        # log immediately for cluster safety
        self._append_history(trial, converged=False, is_resumed=False)

        return trial

    def solve(self, initial_munit: float) -> TrialResult:
        """main tuning loop.

        uses power-law updates M_new = M_old*(F_target/F_old)^(1/P) only.
        no geometric bracketing/bisection to avoid catastrophic GRMONTY behavior.
        """
        # 1) if we have existing trials (resume), check if any already converged
        if self.trials:
            # best so far (closest to target)
            best_so_far = min(
                self.trials,
                key=lambda t: abs(t.flux - self.bracketer.target_flux),
            )

            if self.bracketer.within_tolerance(best_so_far.flux):
                print(
                    "[solve] resume: existing trial already within tolerance; "
                    f"returning trial #{best_so_far.index:02d}",
                    flush=True,
                )
                # log as converged so it's recorded in history as such
                self._append_history(best_so_far, converged=True, is_resumed=True)
                return best_so_far

            # otherwise, start from a gentle update of the best existing trial
            print(
                "[solve] resume: no existing trial within tolerance; "
                "continuing tuning from best-so-far.",
                flush=True,
            )
            current_munit = self._compute_next_munit(best_so_far)
        else:
            # fresh run
            current_munit = initial_munit

        # 2) main iteration loop
        for _ in range(self.args.max_iters):
            trial = self.run_trial(current_munit)

            # if invalid bias events too large, treat as unstable and back off
            if trial.invalid_bias_count > self.args.invalid_bias_max:
                print(
                    f"[solve] trial {trial.index:02d} unstable "
                    f"({trial.invalid_bias_count} invalid-bias events); "
                    f"backing off by factor {self.args.bias_backoff_factor:.1f}.",
                    flush=True,
                )
                # decrease M_unit and try again
                current_munit = max(
                    self.args.min_munit,
                    trial.munit / self.args.bias_backoff_factor,
                )
                continue

            if self.bracketer.within_tolerance(trial.flux):
                print(
                    f"[solve] converged: M_unit={trial.munit:.4e}, "
                    f"flux={trial.flux:.4f} Jy",
                    flush=True,
                )
                self._append_history(trial, converged=True, is_resumed=False)
                return trial

            current_munit = self._compute_next_munit(trial)

        # 3) if we reach here, no exact convergence: pick closest trial
        best = min(
            self.trials,
            key=lambda t: abs(t.flux - self.bracketer.target_flux),
        )
        print(
            "[solve] max iterations reached; using closest flux encountered:\n"
            f"        trial #{best.index:02d} M_unit={best.munit:.4e}, "
            f"flux={best.flux:.4f} Jy",
            flush=True,
        )
        # mark best as "converged" in the history file (best-so-far)
        self._append_history(best, converged=True, is_resumed=False)
        return best

    # cleanup / finalization
    def cleanup(self, best: TrialResult) -> None:
        """optionally remove non-winning trials and rename best trial to final."""
        base_spec = self.context["spectrum_base"]
        final_spec = base_spec.with_name(f"{base_spec.name}.h5")
        final_log = LOG_DIR / f"{self.context['log_tag']}.log"
        final_par = LOG_DIR / f"{self.context['log_tag']}.par"

        # remove or keep intermediate trials
        for trial in self.trials:
            if trial is best or self.args.keep_trials:
                continue
            for path in (trial.spec_path, trial.log_path, trial.par_path):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

        # move best trial results into a clean final name
        if best.spec_path != final_spec:
            try:
                final_spec.unlink()
            except FileNotFoundError:
                pass
            best.spec_path.rename(final_spec)
        if best.log_path != final_log:
            try:
                final_log.unlink()
            except FileNotFoundError:
                pass
            best.log_path.rename(final_log)
        if best.par_path != final_par:
            try:
                final_par.unlink()
            except FileNotFoundError:
                pass
            best.par_path.rename(final_par)


# CLI
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatic M_unit tuning for GRMONTY spectra (power-law updates)."
    )

    parser.add_argument("--row", type=int, required=True, help="0-based row index in the CSV.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to model table CSV.")

    parser.add_argument(
        "--target-flux",
        type=float,
        default=F_TARGET_JY,
        help="Desired flux in Jy (default: 0.5 Jy).",
    )
    parser.add_argument(
        "--freq-target",
        type=float,
        default=FREQ_TARGET_HZ,
        help="Frequency (Hz) used to compare fluxes (default: 230 GHz).",
    )
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=0.05,
        help="Relative tolerance (fraction) if abs_tol not set.",
    )
    parser.add_argument(
        "--abs-tol",
        type=float,
        default=None,
        help="Absolute tolerance in Jy (overrides relative tolerance when set).",
    )

    parser.add_argument("--ns", type=float, default=1e6, help="Photon count for tuning runs.")
    parser.add_argument("--mbh", type=float, default=6.5e9, help="Black hole mass in Msun.")
    parser.add_argument("--tp-over-te", type=float, default=3.0, help="TP_OVER_TE value.")
    parser.add_argument(
        "--positron-ratio",
        type=float,
        default=None,
        help=(
            "Override positron_ratio passed to GRMONTY. "
            "If unset, inferred from CSV (pos/positron columns) or defaults to 0."
        ),
    )

    parser.add_argument(
        "--fit-bias-ns",
        type=int,
        default=50000,
        help="fit_bias_ns parameter passed to GRMONTY.",
    )
    parser.add_argument(
        "--bias",
        type=float,
        default=0.05,
        help="Initial bias value passed to GRMONTY.",
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=math.sqrt(2.0),
        help="Target bias effectiveness ratio.",
    )

    parser.add_argument(
        "--min-munit",
        type=float,
        default=1e20,
        help="Hard lower bound for M_unit trials.",
    )
    parser.add_argument(
        "--max-munit",
        type=float,
        default=1e32,
        help="Hard upper bound for M_unit trials.",
    )

    parser.add_argument(
        "--p-factor",
        type=float,
        default=2.0,
        help=(
            "Exponent P in M_new = M_old * (F_target/F_old)^(1/P). "
            "Values ~2 are usually safe for GRMONTY."
        ),
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=8,
        help="Maximum number of new GRMONTY trials to run.",
    )

    parser.add_argument(
        "--grmonty-bin",
        type=Path,
        default=GRMONTY_BIN,
        help="Path to the GRMONTY executable.",
    )
    parser.add_argument(
        "--keep-trials",
        action="store_true",
        help="Keep intermediate spectra/logs/parfiles instead of cleaning up.",
    )

    parser.add_argument(
        "--history-csv",
        type=Path,
        default=DEFAULT_HISTORY_CSV,
        help="Path to CSV file where tuning history is appended.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing spectra/logs/parfiles if present.",
    )

    parser.add_argument(
        "--invalid-bias-max",
        type=int,
        default=200,
        help="Max allowed invalid-bias events before treating trial as unstable.",
    )
    parser.add_argument(
        "--bias-backoff-factor",
        type=float,
        default=5.0,
        help="Factor by which to reduce M_unit after an unstable trial.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    row = load_row(args.csv, args.row)
    positron_runs = resolve_positron_runs(row, override=args.positron_ratio)
    if len(positron_runs) > 1:
        tags = ", ".join(
            f"pos{format_pos_tag(ratio)}"
            for ratio, _ in positron_runs
        )
        print(
            f"[ctx] no explicit positron field; auto-running branches from CSV columns: {tags}",
            flush=True,
        )

    for run_idx, (branch_ratio, branch_source) in enumerate(positron_runs, start=1):
        if len(positron_runs) > 1:
            print(
                f"[ctx] ----- branch {run_idx}/{len(positron_runs)} "
                f"(positron_ratio={branch_ratio:.6g}, source={branch_source}) -----",
                flush=True,
            )

        context = build_context(row, positron_ratio_override=branch_ratio)
        context["positron_source"] = branch_source

        pos = str(context["pos"]).strip()
        positron_ratio = float(context["positron_ratio"])

        print(
            f"[ctx] positron_ratio={positron_ratio:.6g} (source={context['positron_source']})",
            flush=True,
        )
        if context["positron_source"] == "default":
            has_pos_munit_cols = any(
                _norm_key(str(key)).startswith(_norm_key("MunitUsed_pos"))
                for key in row.keys()
            )
            if has_pos_munit_cols:
                print(
                    "[note] CSV has MunitUsed_pos* columns but no explicit positron field; "
                    "defaulted to positron_ratio=0. Use --positron-ratio to select nonzero pairs.",
                    flush=True,
                )
        if positron_ratio != 0.0:
            print(
                "[note] nonzero positron_ratio: GRMONTY applies pair scaling internally; "
                "do not pre-scale M_unit by (1+2*positron_ratio).",
                flush=True,
            )

        munit_key = f"MunitUsed_pos{pos}"

        munit_used, matched_key = find_row_value(
            row, *candidate_munit_keys(positron_ratio, pos)
        )
        default_munit: float

        if munit_used:
            try:
                default_munit = float(munit_used)
                print(
                    f"[init] using CSV {matched_key}={default_munit:.4e}",
                    flush=True,
                )
            except ValueError:
                print(
                    f"[warn] invalid {matched_key}='{munit_used}', "
                    "falling back to state default",
                    flush=True,
                )
                default_munit = STATE_DEFAULT_MUNIT[context["state"]]
        else:
            fallback_raw, fallback_key = find_row_value(row, "Munit", "M_unit")
            if fallback_raw:
                try:
                    default_munit = float(fallback_raw)
                    print(
                        f"[init] no {munit_key} in CSV; using {fallback_key}={default_munit:.4e}",
                        flush=True,
                    )
                except ValueError:
                    print(
                        f"[warn] invalid {fallback_key}='{fallback_raw}', "
                        "falling back to state default",
                        flush=True,
                    )
                    default_munit = STATE_DEFAULT_MUNIT[context["state"]]
            else:
                default_munit = STATE_DEFAULT_MUNIT[context["state"]]

        # optional override if you ever add --init-munit
        init_munit = (
            args.init_munit
            if hasattr(args, "init_munit") and args.init_munit
            else default_munit
        )

        bracketer = MunitBracketer(
            target_flux=args.target_flux,
            rel_tol=args.rel_tol,
            abs_tol=args.abs_tol,
        )

        tuner = MunitTuner(context=context, args=args, bracketer=bracketer, row=row)

        try:
            best_trial = tuner.solve(init_munit)
        except subprocess.CalledProcessError as exc:
            print(
                f"[error] GRMONTY failed (see log). Command: {exc.cmd}",
                file=sys.stderr,
            )
            raise

        tuner.cleanup(best_trial)
        print(
            f"[done] best trial: #{best_trial.index:02d} "
            f"M_unit={best_trial.munit:.6e} flux={best_trial.flux:.4f} Jy "
            f"(target {args.target_flux:.3f} Jy)",
            flush=True,
        )


if __name__ == "__main__":
    main()
