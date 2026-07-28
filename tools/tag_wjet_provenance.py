#!/usr/bin/env python3
"""Read-only provenance tagging for existing wJET GRMONTY outputs.

For every *wJET*.h5 spectrum found under igrmonty_outputs/, reads the
/params/electrons/* metadata embedded at write time (model/iharm/model.c,
h5io_add_data_* calls) and classifies the run as:

  - hard_override_reachable = YES  if jet_sigma_cut>0 or jet_beta_cut>0
                                    (the jet_thetae/jet_ne_mult hard-override
                                    path was actually exercisable)
  - hard_override_reachable = no   otherwise (additive constant_beta_e0
                                    supplement only -- "pre-fix era" behavior)

and separately flags constant_beta_e0_exponent == 0 (see audit finding H1).

This script does not modify any existing file. It writes one new summary CSV
under igrmonty_outputs/m87/_qa/.

See docs/audits/2026-07-23_jet_electron_temperature_audit.md, findings H1 and M3,
and docs/2026-07-23_jet_paper_readiness_plan.md, Phase 1.
"""
import csv
import glob
import sys
from pathlib import Path

import h5py

REPO_ROOT = Path(__file__).resolve().parent.parent
SEARCH_ROOT = REPO_ROOT.parent / "igrmonty_outputs"
OUT_CSV = REPO_ROOT.parent / "igrmonty_outputs" / "m87" / "_qa" / "wjet_provenance_report.csv"

ELECTRON_KEYS = (
    "sigma_transition",
    "constant_beta_e0",
    "constant_beta_e0_exponent",
    "jet_sigma_cut",
    "jet_beta_cut",
    "jet_thetae",
    "jet_ne_mult",
)

FIELDS = [
    "path",
    "with_electrons_from_name",
    "with_electrons_from_meta",
    *ELECTRON_KEYS,
    "hard_override_reachable",
    "exponent_is_zero",
    "name_meta_mismatch",
    "error",
]


def read_scalar(group, key):
    if key not in group:
        return None
    val = group[key][()]
    try:
        return float(val)
    except (TypeError, ValueError):
        return val


def classify(path: Path):
    row = {k: "" for k in FIELDS}
    row["path"] = str(path)
    name = path.name
    if "CRITBETAwJET" in name:
        row["with_electrons_from_name"] = "5 (CRITBETAwJET)"
        expect_we = 5
    elif "RBETAwJET" in name:
        row["with_electrons_from_name"] = "4 (RBETAwJET)"
        expect_we = 4
    else:
        row["with_electrons_from_name"] = "?"
        expect_we = None

    try:
        with h5py.File(path, "r") as f:
            grp = f.get("params/electrons")
            if grp is None:
                row["error"] = "no /params/electrons group in file"
                return row

            we = read_scalar(grp, "type")
            row["with_electrons_from_meta"] = we

            for key in ELECTRON_KEYS:
                row[key] = read_scalar(grp, key)

            jsc = row["jet_sigma_cut"]
            jbc = row["jet_beta_cut"]
            hard_active = (isinstance(jsc, float) and jsc > 0.0) or (
                isinstance(jbc, float) and jbc > 0.0
            )
            row["hard_override_reachable"] = "YES" if hard_active else "no (inert defaults)"

            exp = row["constant_beta_e0_exponent"]
            row["exponent_is_zero"] = "YES" if (isinstance(exp, float) and exp == 0.0) else "no"

            if expect_we is not None and we is not None:
                row["name_meta_mismatch"] = (
                    "no" if int(we) == expect_we else f"YES (metadata says {we})"
                )
    except Exception as exc:  # noqa: BLE001 -- report-and-continue tool, not production code
        row["error"] = repr(exc)

    return row


def main():
    paths = sorted(Path(p) for p in glob.glob(str(SEARCH_ROOT / "**" / "*wJET*.h5"), recursive=True))
    rows = [classify(p) for p in paths]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    n_hard = sum(1 for r in rows if r["hard_override_reachable"] == "YES")
    n_err = sum(1 for r in rows if r["error"])
    n_exp0 = sum(1 for r in rows if r["exponent_is_zero"] == "YES")
    n_mismatch = sum(1 for r in rows if r["name_meta_mismatch"] not in ("", "no"))

    print(f"scanned {n} wJET outputs under {SEARCH_ROOT}")
    print(f"  hard-override reachable (jet_sigma_cut/jet_beta_cut > 0): {n_hard}")
    print(f"  additive-only / pre-fix era:                              {n - n_hard - n_err}")
    print(f"  constant_beta_e0_exponent == 0:                           {n_exp0}")
    print(f"  filename/metadata with_electrons mismatch:                {n_mismatch}")
    print(f"  read errors:                                              {n_err}")
    print(f"full report written to: {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
