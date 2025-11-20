#!/usr/bin/env python3
from pathlib import Path
from typing import Dict, Optional

from auto_munit_bracket import FREQ_TARGET_HZ, F_TARGET_JY, measure_flux

OUTPUT_DIR = Path("/work/vmo703/igrmonty_outputs/m87")
STATE_PREFIX_TO_NAME = {"S": "SANE", "M": "MAD"}
STATE_TO_MUNIT = {"SANE": 1.83e27, "MAD": 7.49e24}
P = 2.0


def infer_state(spec_path: Path) -> Optional[str]:
    """return 'SANE' or 'MAD' by looking at the first character after 'spectrum_'."""
    stem = spec_path.stem
    suffix = stem.split("spectrum", 1)[-1].lstrip("_")
    if not suffix:
        return None
    return STATE_PREFIX_TO_NAME.get(suffix[0].upper())


def process_spectrum(spec_path: Path, munit_old: float) -> Dict[str, float]:
    flux, freq_used = measure_flux(spec_path, FREQ_TARGET_HZ, return_freq=True)
    scale = F_TARGET_JY / flux
    munit_new = munit_old * (scale ** (1.0 / P))
    return {"nearest_freq": freq_used, "Fnu_raw_Jy": flux, "scale": scale, "Munit_new": munit_new}


def main() -> None:
    spectra = sorted(p for p in OUTPUT_DIR.glob("spectrum*.h5") if p.is_file())
    if not spectra:
        raise SystemExit(f"no spectrum*.h5 files found in {OUTPUT_DIR}")

    for spec in spectra:
        state = infer_state(spec)
        if state is None:
            print(f"skipping {spec.name}: could not infer MAD/SANE from filename.")
            continue

        munit_old = STATE_TO_MUNIT[state]
        print(f"\n[{state}] {spec}")
        try:
            result = process_spectrum(spec, munit_old)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[error] processing {spec.name}: {exc}")
            continue

        print(f"nearest freq: {result['nearest_freq']/1e9:.3f} GHz")
        print(f"raw F_nu({FREQ_TARGET_HZ/1e9:.0f} GHz) = {result['Fnu_raw_Jy']:.3e} Jy")
        print(f"scale factor (target/raw) = {result['scale']:.3e}")
        print(f"M_unit old = {munit_old:.6g}")
        print(f"M_unit new (p={P:g}) = {result['Munit_new']:.6g}")


if __name__ == "__main__":
    main()
