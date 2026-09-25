# Crit-β Θe floor: 3×10⁻² is part of the model (decision 2026-09-25)

**Decision (ashton, 2026-09-25): `crit_floor = 3.e-2` is the production
convention for Crit-β electron models (with_electrons 3 and 5), documented as
part of the model definition.** Finding M1's change to 1×10⁻³ is reverted.

## What happened

- The July 2026 production grid (42/48 tuned branches, all published-comparison
  numbers) ran with the historical `crit_floor = 3.e-2` in
  `thetae_func` (model/iharm/model.c).
- Commit a6e1b40 (2026-07-28, "paper-readiness fixes", Finding M1) lowered it
  to 1×10⁻³ to match IPOLE's flat "secret floor". Crit-β modes were never
  spectrum-tested again until the Phase-4 campaign (2026-09-15).
- All 24 Crit-β Phase-4 tasks then failed in ~3 minutes with
  `fitbias_zero_ratio_stalled` (exit 41): **zero superphotons generated**.

## Root cause (bisect, 2026-09-25)

| build | crit_floor | photons (tiny Ns=2e3, SANE CRITBETAwJET a-0.5 d4000, July M_unit) |
|---|---|---|
| af16b3a (July freeze) | 3e-2 | 73,932 |
| a6e1b40 (M1) | 1e-3 | **0** |
| 65835d9 (P4 freeze) | 1e-3 | **0** |
| 65835d9 + floor reverted | 3e-2 | 71,829 |

Mechanism: ~45% of the SANE volume sits AT the floor, and that floored dense
disk dominates Crit-β photon generation. At Θe = 1e-3 its emissivity collapses
(K/KMAX synchrotron cutoffs), the frequency-bin sums fall below the hardcoded
`WEIGHT_MIN = 1e28` addend in `init_weight_table`, `n2gens → 0`, and grmonty
generates nothing — silently at fixed bias (status "ok", L = 0), loudly (exit
41) under the fitter.

## Consequences

1. A floor this influential is a **model parameter**: state it in the paper's
   methods next to β_crit and f (Θe ≥ 3×10⁻² in Crit-β models).
2. **IPOLE harmonization goes the other way**: before any grmonty↔ipole
   Crit-β comparison (P4.3), raise IPOLE's floor to 3e-2 for these models —
   same one-line-gate treatment as `constant_beta_paper_literal`.
3. M1's other siblings in a6e1b40 (D5 NaN guards, deep-KN sampling, H1
   exponent fix) are correct and retained; only the floor value reverts.
4. July M_units remain valid warm starts for Crit-β branches.
