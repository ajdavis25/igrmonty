#!/usr/bin/env python3
"""
postprocess GRMONTY theta-binned spectra into IPOLE-like cone-restricted products

this utility is no-recompile and works on existing GRMONTY HDF5 outputs in this
pipeline (`/output/nuLnu`, `/output/dOmega`, `/params/NUMIN`, `/params/NUMAX`)

key behavior:
- reconstructs baseline 4pi flux exactly as `igrmonty/auto_munit_bracket.py`
  (dOmega-weighted angular integration + distance conversion)
- applies cone restriction in theta-bin space around `thetacam` with half-angle
  `alpha` by selecting bins whose centers are in
  [`thetacam_folded - alpha`, `thetacam_folded + alpha`] clipped to [0, 90] deg
- produces BOTH cone interpretations:
  1) `cone_physical`: no renormalization (power from selected cone only)
  2) `cone_renorm_4pi_equiv`: multiply by (4pi / Omega_selected) to get a
     4pi-equivalent average of the selected cone

notes:
- in this GRMONTY branch, theta is folded about the equator during binning
  therefore a cone cut is an azimuth-averaged theta cut, not a true camera/FOV
- for IPOLE-like comparison (single line-of-sight camera), `cone_renorm_4pi_equiv`
  is usually the better proxy than baseline 4pi, but still cannot reproduce finite
  image-plane/FOV ray tracing

examples:
1) single file, IPOLE-like orientation near 17 deg, 10 deg cone:
   python3 viewing_cone_postprocess.py \
     /work/vmo703/igrmonty_outputs/m87/spectrum_Sa-0.5_4000_RBETA_pos0_trial04.h5 \
     --thetacam-deg 17 --cone-half-angle-deg 10

2) directory (recursive), same cone, explicit 228 GHz anchor:
   python3 viewing_cone_postprocess.py \
     /work/vmo703/igrmonty_outputs/m87 \
     --thetacam-deg 17 --cone-half-angle-deg 10 --anchor-ghz 228

3) use 163 deg camera convention (auto-folds to 17 deg):
   python3 viewing_cone_postprocess.py \
     /work/vmo703/igrmonty_outputs/m87 \
     --thetacam-deg 163 --cone-half-angle-deg 10
"""

from __future__ import print_function

import argparse
import csv
import math
import os
import sys

import h5py
import numpy as np


LSUN_CGS = 3.827e33
PC_TO_CM = 3.085677581e18
FOUR_PI = 4.0 * math.pi
DEFAULT_DISTANCE_MPC = 16.8
DEFAULT_TARGETS_GHZ = [86.0, 230.0, 345.0]
DEFAULT_ANCHOR_GHZ = 228.0
DEFAULT_BANDS = [
    ("mm86", 70.0e9, 110.0e9),
    ("mm230", 200.0e9, 260.0e9),
    ("mm345", 300.0e9, 390.0e9),
    ("high", 1.0e17, 1.0e20),
]


def _float_tag(val):
    s = "{:.1f}".format(float(val))
    s = s.replace("-", "m").replace(".", "p")
    return s


def _safe_name(text):
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in ["_", "-"]:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def _find_block(container_shape, target_shape):
    if len(target_shape) > len(container_shape):
        return None
    max_start = len(container_shape) - len(target_shape)
    for start in range(max_start + 1):
        if tuple(container_shape[start : start + len(target_shape)]) == tuple(target_shape):
            return start
    return None


def _align_nulnu_to_domega(nuLnu_cgs, dOmega_shape):
    """
    match `auto_munit_bracket.measure_flux` axis reductions

    returns an array with shape dOmega_shape + (nfreq,)
    """
    spatial_shape = nuLnu_cgs.shape[:-1]
    if len(spatial_shape) == 0:
        raise RuntimeError("nuLnu has no angular/spatial axes after frequency alignment")

    block_start = _find_block(spatial_shape, dOmega_shape)
    if block_start is None:
        raise RuntimeError(
            "dOmega shape does not match nuLnu spatial axes: "
            "nuLnu spatial shape {}, dOmega {}".format(spatial_shape, dOmega_shape)
        )

    # sum leading non-angular axes (e.g., scattering components)
    for _ in range(block_start):
        nuLnu_cgs = np.sum(nuLnu_cgs, axis=0)

    # sum trailing non-angular axes
    spatial_ndim = nuLnu_cgs.ndim - 1
    trailing_axes = spatial_ndim - len(dOmega_shape)
    for _ in range(trailing_axes):
        nuLnu_cgs = np.sum(nuLnu_cgs, axis=len(dOmega_shape))

    if tuple(nuLnu_cgs.shape[:-1]) != tuple(dOmega_shape):
        raise RuntimeError(
            "Failed to align nuLnu and dOmega after reductions: "
            "nuLnu spatial shape {}, dOmega {}".format(nuLnu_cgs.shape[:-1], dOmega_shape)
        )

    return nuLnu_cgs


def load_spectrum(spec_path):
    with h5py.File(spec_path, "r") as f:
        if "/output/nuLnu" not in f:
            raise RuntimeError("Missing /output/nuLnu in {}".format(spec_path))
        if "/output/dOmega" not in f:
            raise RuntimeError("Missing /output/dOmega in {}".format(spec_path))
        if "/params/NUMIN" not in f or "/params/NUMAX" not in f:
            raise RuntimeError("Missing /params/NUMIN or /params/NUMAX in {}".format(spec_path))

        nuLnu = np.array(f["/output/nuLnu"], dtype=np.float64)
        dOmega = np.array(f["/output/dOmega"], dtype=np.float64)
        numin = float(f["/params/NUMIN"][()])
        numax = float(f["/params/NUMAX"][()])
        nfreq = None
        if "/params/N_EBINS" in f:
            try:
                nfreq = int(f["/params/N_EBINS"][()])
            except Exception:
                nfreq = None
        if nfreq is None:
            if "/output/lnu" in f:
                nfreq = int(f["/output/lnu"].shape[0])
            else:
                nfreq = int(nuLnu.shape[-1])

    if nfreq <= 0:
        raise RuntimeError("Invalid N_EBINS ({}) for {}".format(nfreq, spec_path))

    # match tuner frequency axis exactly
    nu_hz = np.logspace(np.log10(numin), np.log10(numax), nfreq)

    # put frequency axis last
    freq_matches = [axis for axis, size in enumerate(nuLnu.shape) if size == nfreq]
    if len(freq_matches) > 0:
        if 1 in freq_matches:
            freq_axis = 1
        else:
            freq_axis = freq_matches[0]
    else:
        freq_axis = nuLnu.ndim - 1
    if freq_axis != nuLnu.ndim - 1:
        nuLnu = np.moveaxis(nuLnu, freq_axis, -1)

    # L_sun -> cgs
    nuLnu_cgs = nuLnu * LSUN_CGS

    if dOmega.ndim != 1:
        raise RuntimeError(
            "This postprocess tool currently supports 1D theta dOmega only; got shape {} in {}".format(
                dOmega.shape, spec_path
            )
        )

    nuLnu_theta_freq = _align_nulnu_to_domega(nuLnu_cgs, dOmega.shape)

    return {
        "nu_hz": nu_hz,
        "nuLnu_theta_freq_cgs": nuLnu_theta_freq,
        "dOmega_sr": dOmega,
        "nfreq": int(nfreq),
        "numin": numin,
        "numax": numax,
    }


def fold_theta_deg(theta_deg):
    theta = float(theta_deg) % 360.0
    if theta > 180.0:
        theta = 360.0 - theta
    if theta > 90.0:
        theta = 180.0 - theta
    return theta


def build_theta_bins(n_thbins):
    edges_deg = np.linspace(0.0, 90.0, int(n_thbins) + 1)
    centers_deg = 0.5 * (edges_deg[:-1] + edges_deg[1:])
    width_deg = 90.0 / float(n_thbins)
    return edges_deg, centers_deg, width_deg


def select_theta_bins(centers_deg, thetacam_deg, cone_half_angle_deg):
    tc_fold = fold_theta_deg(thetacam_deg)
    alpha = max(0.0, float(cone_half_angle_deg))
    lo = max(0.0, tc_fold - alpha)
    hi = min(90.0, tc_fold + alpha)

    mask = (centers_deg >= lo) & (centers_deg <= hi)
    if not np.any(mask):
        idx = int(np.argmin(np.abs(centers_deg - tc_fold)))
        mask[idx] = True

    return {
        "thetacam_input_deg": float(thetacam_deg),
        "thetacam_folded_deg": float(tc_fold),
        "cone_half_angle_deg": float(alpha),
        "theta_select_lo_deg": float(lo),
        "theta_select_hi_deg": float(hi),
        "theta_mask": mask,
        "theta_indices": np.where(mask)[0],
    }


def nuLnu_to_fnu_jy(nuLnu_cgs, nu_hz, distance_mpc):
    d_cm = float(distance_mpc) * 1.0e6 * PC_TO_CM
    Lnu_cgs = nuLnu_cgs / nu_hz
    Fnu_cgs = Lnu_cgs / (FOUR_PI * d_cm * d_cm)
    Fnu_jy = Fnu_cgs * 1.0e23
    return Fnu_jy


def compute_spectra_modes(nu_hz, nuLnu_theta_freq_cgs, dOmega_sr, theta_mask, distance_mpc):
    dOmega = dOmega_sr.reshape((-1, 1))

    # baseline: exact tuner-style 4pi integration
    nuLnu_4pi_cgs = np.sum(nuLnu_theta_freq_cgs * dOmega, axis=0) / FOUR_PI

    # cone physical: keep only selected theta bins, no renormalization
    sel = theta_mask.astype(bool)
    omega_sel = float(np.sum(dOmega_sr[sel]))
    nuLnu_cone_phys_cgs = np.sum(nuLnu_theta_freq_cgs[sel, :] * dOmega_sr[sel].reshape((-1, 1)), axis=0) / FOUR_PI

    # cone renormalized to 4pi-equivalent average over selected solid angle
    if omega_sel > 0.0:
        renorm_factor = FOUR_PI / omega_sel
    else:
        renorm_factor = float("nan")
    nuLnu_cone_renorm_cgs = nuLnu_cone_phys_cgs * renorm_factor

    fnu_4pi_jy = nuLnu_to_fnu_jy(nuLnu_4pi_cgs, nu_hz, distance_mpc)
    fnu_cone_phys_jy = nuLnu_to_fnu_jy(nuLnu_cone_phys_cgs, nu_hz, distance_mpc)
    fnu_cone_renorm_jy = nuLnu_to_fnu_jy(nuLnu_cone_renorm_cgs, nu_hz, distance_mpc)

    # per-theta isotropic-equivalent flux proxy (useful diagnostics only)
    fnu_theta_iso_jy = nuLnu_to_fnu_jy(nuLnu_theta_freq_cgs, nu_hz.reshape((1, -1)), distance_mpc)

    return {
        "omega_selected_sr": omega_sel,
        "renorm_factor": renorm_factor,
        "nuLnu_4pi_cgs": nuLnu_4pi_cgs,
        "nuLnu_cone_phys_cgs": nuLnu_cone_phys_cgs,
        "nuLnu_cone_renorm_cgs": nuLnu_cone_renorm_cgs,
        "fnu_4pi_jy": fnu_4pi_jy,
        "fnu_cone_phys_jy": fnu_cone_phys_jy,
        "fnu_cone_renorm_jy": fnu_cone_renorm_jy,
        "fnu_theta_iso_jy": fnu_theta_iso_jy,
    }


def interpolate_fnu(nu_hz, fnu_jy, target_hz, mode):
    x = np.log10(nu_hz)
    xt = math.log10(float(target_hz))
    if xt < x[0] or xt > x[-1]:
        return float("nan")

    if mode == "none":
        return float("nan")

    if mode == "loglog":
        if np.any(fnu_jy <= 0.0):
            # fallback to linear-in-log(nu)
            return float(np.interp(xt, x, fnu_jy))
        y = np.log10(fnu_jy)
        return float(10.0 ** np.interp(xt, x, y))

    # mode == linlog: linear in fnu, linear in log10(nu)
    return float(np.interp(xt, x, fnu_jy))


def anchor_metrics(nu_hz, fnu_jy, anchor_hz, interp_mode):
    idx = int(np.argmin(np.abs(nu_hz - anchor_hz)))
    nearest_hz = float(nu_hz[idx])
    nearest_jy = float(fnu_jy[idx])
    offset_hz = nearest_hz - float(anchor_hz)
    offset_pct = (offset_hz / float(anchor_hz)) * 100.0
    interp_jy = interpolate_fnu(nu_hz, fnu_jy, anchor_hz, interp_mode)
    return {
        "anchor_hz": float(anchor_hz),
        "nearest_hz": nearest_hz,
        "offset_hz": offset_hz,
        "offset_pct": offset_pct,
        "nearest_jy": nearest_jy,
        "interp_jy": interp_jy,
    }


def band_integral_jy_hz(nu_hz, fnu_jy, nu_lo_hz, nu_hi_hz):
    lo = float(min(nu_lo_hz, nu_hi_hz))
    hi = float(max(nu_lo_hz, nu_hi_hz))
    mask = (nu_hz >= lo) & (nu_hz <= hi)
    if int(np.count_nonzero(mask)) < 2:
        return float("nan")
    return float(np.trapz(fnu_jy[mask], nu_hz[mask]))


def parse_bands(band_args):
    if not band_args:
        return list(DEFAULT_BANDS)
    out = []
    for entry in band_args:
        parts = entry.split(":")
        if len(parts) != 3:
            raise ValueError(
                "Invalid --band '{}'. Expected format name:min_hz:max_hz".format(entry)
            )
        name = _safe_name(parts[0])
        nu_lo = float(parts[1])
        nu_hi = float(parts[2])
        out.append((name, nu_lo, nu_hi))
    return out


def parse_targets_ghz(text):
    vals = []
    for part in str(text).split(","):
        s = part.strip()
        if not s:
            continue
        vals.append(float(s))
    if len(vals) == 0:
        vals = list(DEFAULT_TARGETS_GHZ)
    return vals


def collect_input_files(inputs, recursive=True, glob_pattern="*.h5"):
    files = []
    for item in inputs:
        p = os.path.abspath(item)
        if os.path.isfile(p):
            if p.lower().endswith(".h5") or p.lower().endswith(".hdf5"):
                files.append(p)
            continue
        if not os.path.isdir(p):
            continue

        if recursive:
            for root, _dirs, names in os.walk(p):
                for name in names:
                    if name.lower().endswith(".h5") or name.lower().endswith(".hdf5"):
                        if glob_pattern == "*.h5" or glob_pattern == "*.hdf5":
                            files.append(os.path.join(root, name))
                        else:
                            # simple suffix fallback for custom glob; keep deterministic
                            if name.endswith(glob_pattern.replace("*", "")):
                                files.append(os.path.join(root, name))
        else:
            for name in sorted(os.listdir(p)):
                q = os.path.join(p, name)
                if os.path.isfile(q) and (name.lower().endswith(".h5") or name.lower().endswith(".hdf5")):
                    files.append(q)

    # exclude already-generated cone sidecars by default
    clean = []
    for fpath in sorted(set(files)):
        base = os.path.basename(fpath)
        if "_cone_" in base:
            continue
        clean.append(fpath)
    return clean


def build_summary_rows(
    spec_path,
    npz_path,
    nu_hz,
    spectra_modes,
    theta_meta,
    dOmega_sr,
    distance_mpc,
    anchor_hz,
    interp_mode,
    target_freqs_hz,
    bands,
):
    rows = []

    mode_map = [
        ("baseline_4pi", spectra_modes["fnu_4pi_jy"]),
        ("cone_physical", spectra_modes["fnu_cone_phys_jy"]),
        ("cone_renorm_4pi_equiv", spectra_modes["fnu_cone_renorm_jy"]),
    ]

    sel_idx = theta_meta["theta_indices"]
    sel_centers = []
    for i in sel_idx:
        sel_centers.append(theta_meta["theta_centers_deg"][int(i)])

    omega_sel = float(spectra_modes["omega_selected_sr"])
    omega_frac = omega_sel / FOUR_PI if omega_sel > 0.0 else float("nan")

    for mode_name, fnu_jy in mode_map:
        am = anchor_metrics(nu_hz, fnu_jy, anchor_hz, interp_mode)
        row = {
            "input_h5": spec_path,
            "output_npz": npz_path,
            "mode": mode_name,
            "distance_mpc": float(distance_mpc),
            "thetacam_input_deg": float(theta_meta["thetacam_input_deg"]),
            "thetacam_folded_deg": float(theta_meta["thetacam_folded_deg"]),
            "cone_half_angle_deg": float(theta_meta["cone_half_angle_deg"]),
            "theta_select_lo_deg": float(theta_meta["theta_select_lo_deg"]),
            "theta_select_hi_deg": float(theta_meta["theta_select_hi_deg"]),
            "selected_theta_indices": ";".join([str(int(i)) for i in sel_idx]),
            "selected_theta_centers_deg": ";".join(["{:.3f}".format(float(v)) for v in sel_centers]),
            "selected_bin_count": int(len(sel_idx)),
            "selected_omega_sr": omega_sel,
            "selected_omega_frac_4pi": omega_frac,
            "renorm_factor_used": float(spectra_modes["renorm_factor"]),
            "anchor_hz": am["anchor_hz"],
            "anchor_nearest_hz": am["nearest_hz"],
            "anchor_offset_hz": am["offset_hz"],
            "anchor_offset_pct": am["offset_pct"],
            "fnu_anchor_nearest_jy": am["nearest_jy"],
            "fnu_anchor_interp_jy": am["interp_jy"],
        }

        for target_hz in target_freqs_hz:
            ghz = float(target_hz) / 1.0e9
            key = "{:g}".format(ghz).replace(".", "p")
            tm = anchor_metrics(nu_hz, fnu_jy, target_hz, interp_mode)
            row["fnu_{}_nearest_hz".format(key)] = tm["nearest_hz"]
            row["fnu_{}_nearest_jy".format(key)] = tm["nearest_jy"]
            row["fnu_{}_interp_jy".format(key)] = tm["interp_jy"]

        for name, nu_lo, nu_hi in bands:
            val = band_integral_jy_hz(nu_hz, fnu_jy, nu_lo, nu_hi)
            row["band_{}_jy_hz".format(name)] = val
            if np.isfinite(val):
                row["band_{}_erg_s_cm2".format(name)] = val * 1.0e-23
            else:
                row["band_{}_erg_s_cm2".format(name)] = float("nan")

        rows.append(row)

    return rows


def write_csv(rows, csv_path):
    if len(rows) == 0:
        return

    fieldnames = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def output_paths(spec_path, thetacam_deg, cone_half_angle_deg):
    root, _ext = os.path.splitext(spec_path)
    tag = "tc{}_a{}".format(_float_tag(thetacam_deg), _float_tag(cone_half_angle_deg))
    npz_path = "{}_cone_{}.npz".format(root, tag)
    csv_path = "{}_cone_summary_{}.csv".format(root, tag)
    return npz_path, csv_path


def process_one_file(args, spec_path, bands, target_freqs_hz, anchor_hz):
    data = load_spectrum(spec_path)
    nu_hz = data["nu_hz"]
    nuLnu_theta_freq_cgs = data["nuLnu_theta_freq_cgs"]
    dOmega_sr = data["dOmega_sr"]

    n_thbins = int(dOmega_sr.shape[0])
    theta_edges_deg, theta_centers_deg, theta_width_deg = build_theta_bins(n_thbins)

    theta_meta = select_theta_bins(theta_centers_deg, args.thetacam_deg, args.cone_half_angle_deg)
    theta_meta["theta_edges_deg"] = theta_edges_deg
    theta_meta["theta_centers_deg"] = theta_centers_deg
    theta_meta["theta_bin_width_deg"] = theta_width_deg

    spectra_modes = compute_spectra_modes(
        nu_hz,
        nuLnu_theta_freq_cgs,
        dOmega_sr,
        theta_meta["theta_mask"],
        args.distance_mpc,
    )

    tc_fold = theta_meta["thetacam_folded_deg"]
    npz_path, csv_path = output_paths(spec_path, tc_fold, args.cone_half_angle_deg)

    summary_rows = build_summary_rows(
        spec_path=spec_path,
        npz_path=npz_path,
        nu_hz=nu_hz,
        spectra_modes=spectra_modes,
        theta_meta=theta_meta,
        dOmega_sr=dOmega_sr,
        distance_mpc=args.distance_mpc,
        anchor_hz=anchor_hz,
        interp_mode=args.interp_mode,
        target_freqs_hz=target_freqs_hz,
        bands=bands,
    )

    if not args.dry_run:
        if (not args.overwrite) and (os.path.exists(npz_path) or os.path.exists(csv_path)):
            raise RuntimeError(
                "Output exists and --overwrite not set: {} or {}".format(npz_path, csv_path)
            )

        np.savez_compressed(
            npz_path,
            input_h5=np.string_(spec_path),
            distance_mpc=float(args.distance_mpc),
            nu_hz=nu_hz,
            theta_edges_deg=theta_edges_deg,
            theta_centers_deg=theta_centers_deg,
            theta_bin_width_deg=float(theta_width_deg),
            dOmega_sr=dOmega_sr,
            thetacam_input_deg=float(theta_meta["thetacam_input_deg"]),
            thetacam_folded_deg=float(theta_meta["thetacam_folded_deg"]),
            cone_half_angle_deg=float(theta_meta["cone_half_angle_deg"]),
            theta_select_lo_deg=float(theta_meta["theta_select_lo_deg"]),
            theta_select_hi_deg=float(theta_meta["theta_select_hi_deg"]),
            theta_mask=theta_meta["theta_mask"].astype(np.uint8),
            theta_indices=theta_meta["theta_indices"].astype(np.int64),
            omega_selected_sr=float(spectra_modes["omega_selected_sr"]),
            renorm_factor=float(spectra_modes["renorm_factor"]),
            nuLnu_theta_freq_cgs=nuLnu_theta_freq_cgs,
            nuLnu_4pi_cgs=spectra_modes["nuLnu_4pi_cgs"],
            nuLnu_cone_phys_cgs=spectra_modes["nuLnu_cone_phys_cgs"],
            nuLnu_cone_renorm_cgs=spectra_modes["nuLnu_cone_renorm_cgs"],
            fnu_4pi_jy=spectra_modes["fnu_4pi_jy"],
            fnu_cone_phys_jy=spectra_modes["fnu_cone_phys_jy"],
            fnu_cone_renorm_jy=spectra_modes["fnu_cone_renorm_jy"],
            fnu_theta_iso_jy=spectra_modes["fnu_theta_iso_jy"],
            anchor_hz=float(anchor_hz),
            interp_mode=np.string_(args.interp_mode),
        )

        write_csv(summary_rows, csv_path)

    return {
        "input_h5": spec_path,
        "output_npz": npz_path,
        "output_csv": csv_path,
        "rows": summary_rows,
    }


def build_parser():
    p = argparse.ArgumentParser(
        description="Cone-restrict GRMONTY theta-binned spectra and emit baseline+cone sidecars."
    )
    p.add_argument(
        "inputs",
        nargs="+",
        help="One or more HDF5 files and/or directories containing GRMONTY outputs.",
    )
    p.add_argument("--thetacam-deg", type=float, required=True, help="Observer theta/camera angle in degrees.")
    p.add_argument(
        "--cone-half-angle-deg",
        type=float,
        required=True,
        help="Cone half-angle in degrees for theta-bin selection.",
    )
    p.add_argument(
        "--distance-mpc",
        type=float,
        default=DEFAULT_DISTANCE_MPC,
        help="Distance in Mpc for F_nu conversion (default: {:.3f}).".format(DEFAULT_DISTANCE_MPC),
    )
    p.add_argument(
        "--anchor-ghz",
        type=float,
        default=DEFAULT_ANCHOR_GHZ,
        help="Anchor frequency in GHz for nearest-bin + interpolation summaries (default: {:.1f}).".format(
            DEFAULT_ANCHOR_GHZ
        ),
    )
    p.add_argument(
        "--target-ghz",
        default=",".join(["{:g}".format(v) for v in DEFAULT_TARGETS_GHZ]),
        help="Comma-separated summary frequencies in GHz (default: 86,230,345).",
    )
    p.add_argument(
        "--interp-mode",
        choices=["linlog", "loglog", "none"],
        default="linlog",
        help="Interpolation mode for anchor/target values. 'linlog' = linear in F_nu vs log10(nu).",
    )
    p.add_argument(
        "--band",
        action="append",
        default=[],
        help="Band integral definition as name:min_hz:max_hz (repeatable).",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively scan directories for HDF5 files (default: true).",
    )
    p.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Disable recursive directory scan.",
    )
    p.add_argument(
        "--glob",
        default="*.h5",
        help="Optional suffix-like glob filter when scanning dirs (default: *.h5).",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing sidecar outputs.")
    p.add_argument("--dry-run", action="store_true", help="Inspect/process in memory without writing files.")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    anchor_hz = float(args.anchor_ghz) * 1.0e9
    target_ghz = parse_targets_ghz(args.target_ghz)
    target_freqs_hz = [v * 1.0e9 for v in target_ghz]
    bands = parse_bands(args.band)

    files = collect_input_files(args.inputs, recursive=args.recursive, glob_pattern=args.glob)
    if len(files) == 0:
        raise RuntimeError("No input HDF5 files found from: {}".format(args.inputs))

    print("[info] files_to_process={}".format(len(files)))
    print("[info] thetacam_deg={} (folded in [0,90] during selection)".format(args.thetacam_deg))
    print("[info] cone_half_angle_deg={}".format(args.cone_half_angle_deg))
    print("[info] anchor={} GHz".format(args.anchor_ghz))

    total_rows = 0
    failures = 0
    for spec_path in files:
        try:
            result = process_one_file(args, spec_path, bands, target_freqs_hz, anchor_hz)
            total_rows += len(result["rows"])
            if len(result["rows"]) > 0:
                sample = result["rows"][0]
                print(
                    "[ok] {} -> {} | anchor_nearest={:.6e} Hz offset={:+.3f}%".format(
                        spec_path,
                        result["output_npz"],
                        float(sample["anchor_nearest_hz"]),
                        float(sample["anchor_offset_pct"]),
                    )
                )
        except Exception as exc:
            failures += 1
            print("[fail] {} :: {}".format(spec_path, exc), file=sys.stderr)

    print("[done] rows_written={} failures={}".format(total_rows, failures))
    if failures > 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
