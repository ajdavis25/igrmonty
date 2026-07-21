# GRMONTY M87 Output Validation Report

- **Audit date:** 2026-07-10
- **Auditor:** automated read-only audit (Claude Code); no files were modified, moved, deleted, or rerun
- **Campaign audited:** M87 M_unit tuning driven by `/work/vmo703/igrmonty/auto_munit_bracket.py` from `/work/vmo703/data/final_grmonty_paper.csv` (24 rows × pos0/pos1 = 48 branch tasks), Slurm array jobs `732850` (May 2026) and `755312` (June 2026)

---

## 1. Executive conclusion

**18 of 48 model branches are complete, genuinely converged, and scientifically clean; 30 branches are incomplete or failed and must not be used.**

| Verdict | Count | Which |
|---|---:|---|
| `PASS — ready for paper use` | **10** | All 10 converged **pos0** branches |
| `PASS WITH CAVEATS` | **8** | All 8 converged **pos1** branches (mixed pair-bremsstrahlung physics across the set — see §8, A2) |
| `NEEDS VERIFICATION` | 0 | — |
| `FAIL — rerun or repair required` | **30** | 16 timed-out partial searches, 12 MAD_CRITBETA hard init failures, 2 SANE_CRITBETAwJET a+0.94 t6000 branches with no measurable trial |
| `UNKNOWN` | 0 | — |

**Most important issue:** the entire **MAD_RBETA block except a=+0.94 t=5000** (10 branches) and the **entire MAD_CRITBETA block** (12 branches) have no final products. If the paper's model grid requires MAD models at all three dump times, the dataset is not yet paper-complete. Among the models that *did* finish, the only substantive concern is that the GRMONTY binary was rebuilt mid-campaign with a changed pair-bremsstrahlung treatment, splitting the 8 positron (pos1) finals across two physics versions (6 old / 2 new).

Every one of the 18 finals was independently re-validated in this audit: GRMONTY exited with `run status: code=2 label=ok`, the final `.par` M_unit matches the accepted trial in the tuning history byte-for-byte, and the 230 GHz flux re-measured from the final HDF5 agrees with the logged value and lies within the 5 % tolerance of the 0.5 Jy target (max |error| 4.96 %). No NaN/Inf bins exist in any final spectrum.

---

## 2. Workflow reconstruction

### 2.1 What `auto_munit_bracket.py` actually does

Despite its filename, the script does **no bracketing/bisection**. It is a damped fixed-point iterator:

1. Loads one row (0-based `--row`) from the model CSV (default `/work/vmo703/data/final_grmonty_paper.csv`): state (MAD/SANE), model (RBETA / RBETAWJET / CRITBETA / CRITBETAWJET → `with_electrons` 2/4/3/5), spin, dump_index, Rhigh, beta_crit/f, and per-positron-branch `MunitUsed_pos0/pos1` seeds.
2. Resolves the GRMHD dump as `/work/vmo703/grmhd_dump_samples/{M|S}a{spin}_{dump}.h5` (all referenced dumps exist, 151 MB each, dated Sep 2025).
3. Writes a trial `.par` to `/work/vmo703/igrmonty/logs/<TAG>_trialNN.par` (fresh file per trial; there is no template — the writer at `auto_munit_bracket.py:624` emits: `seed -1`, `Ns`, `MBH 6.5e9`, `M_unit`, dump, spectrum path, fit-bias controls, electron-model parameters, wJET parameters when applicable, `positron_ratio`, `Thetae_max 1e100`).
4. Runs `subprocess.run([grmonty_bin, "-par", par])` with stdout+stderr redirected to `logs/<TAG>_trialNN.log`; the binary is `/work/vmo703/igrmonty/grmonty`.
5. Writes the trial spectrum to `/work/vmo703/igrmonty_outputs/m87/spectrum_<tag>_trialNN.h5`.
6. Measures **the 4π-angle-averaged flux density at the bin nearest 230 GHz** from `/output/nuLnu` (units L_sun, converted via L_sun=3.827e33, D=16.8 Mpc): `measure_flux()` at `auto_munit_bracket.py:495`. Target **0.5 Jy**, relative tolerance **5 %** (`--rel-tol 0.05`). Note this is the angle-averaged flux, not the flux at M87's 17° inclination; inclination-specific extraction is a separate post-processing step (`tools/viewing_cone_postprocess.py`, validated in `/work/vmo703/_reports/munit_ipole_vs_grmonty_20260218/`). The known-good reference campaign used the identical convention, so this is consistent, but it should be stated in the paper's methods.
7. Updates `M_new = M_old × (F_target/F_old)^(1/P)` with P=2 (gentler P for huge ratios), a ×30 jump cap, and clamps to [1e20, 1e32]. **No lower/upper brackets are ever maintained** — the Phase-3 questions about bracket validity are moot by construction.
8. Repeats up to `--max-iters 8` new trials. Declares convergence at the **first** trial within tolerance.
9. On success, `cleanup()` deletes non-winning trial files and **renames** the winning trial's spectrum/log/par to suffix-free final names: `logs/<TAG>.{par,log}` and `igrmonty_outputs/m87/spectrum_<tag>.h5`.
10. Appends every measured trial to `/work/vmo703/data/munits_tuning_history.csv` (once with `converged=0` when measured, and again with `converged=1` when accepted).
11. `--resume` re-ingests existing final/trial spectra+pars and continues from the best one.

Failure handling: per-trial wallclock guard (`TrialTimeoutError` → exit 75), Slurm-budget guard (`JobRuntimeBudgetExceeded` → exit 75), GRMONTY `abort_bias` status → M_unit backoff ÷5 and retry, other GRMONTY failure → exit with GRMONTY's code.

### 2.2 Code path that can falsely mark non-convergence as "converged" — present but did not fire

At `auto_munit_bracket.py:1265-1281`, when `max_iters` is exhausted the script picks the **closest** trial and logs it to the history CSV with `converged=1` even if it is outside tolerance, and `cleanup()` still promotes it to final names. **This audit verified none of the 18 finals took that path**: every accepted trial's re-measured flux is within the 5 % tolerance. The trap remains latent for future runs.

A second latent trap: on resume, if a prior *out-of-tolerance* final exists it is re-ingested as trial index 0 and tuning continues — correct behavior, but combined with the max-iters path a stale final can be re-blessed. Again, not observed here.

### 2.3 Orchestration

`/work/vmo703/igrmonty/run_auto_munit.slurm` (array `0-47%5`, 3-day walltime, partitions anantuabhg/compute1-3) builds the 48-task list (24 rows × auto-discovered pos0/pos1 branches from `MunitUsed_pos*` columns), then runs one tuner per task with `--resume`. Task stdout/err: `/work/vmo703/igrmonty_logs/row_<jobid>_<task>_<row>_pos<tag>.{out,err}`. Environment: gcc/11.3.0, hdf5/1.12.0 modules, GSL from `/work/vmo703/aricarte/gsl-1.16`, Python from `/work/vmo703/ipole_venv`.

**Settings changed between the two campaigns** (current slurm file = June version):

| Setting | May job 732850 | June job 755312 |
|---|---|---|
| GRMONTY binary | `e6ede55-dirty` | `0bfdb56-dirty` (built Jun 11; source snapshot in `igrmonty/build_archive/`) |
| `fit_bias_ns` | 50000 | 14000 |
| bias `ratio` target | √2 ≈ 1.414 | 1.05 |
| `bias_abort_ratio` | 10 (binary default; not in May `.par`s) | 5 (explicit) |
| trial/job time guards | none visible in May `.par`s/logs | 60 h trial guard, 68 h job budget |

---

## 3. Located directories and files

Everything relevant was found in the expected trees plus a few side locations. No GRMONTY spectra exist outside `/work/vmo703/igrmonty_outputs/`.

| Path | Contents | Role |
|---|---|---|
| `/work/vmo703/igrmonty/auto_munit_bracket.py` | tuner (modified Jun 11 15:45) | driver |
| `/work/vmo703/igrmonty/run_auto_munit.slurm` | array submitter (June version; uncommitted edits) | orchestration |
| `/work/vmo703/igrmonty/grmonty` | binary, Jun 11 15:36 = `0bfdb56-dirty` build | executable (June) |
| `/work/vmo703/igrmonty/build_archive/` | full `src/` snapshot of the June build + binary copy; verified identical to current dirty tree (`main.c`, `model.c` match) | June-build provenance |
| `/work/vmo703/igrmonty/logs/` | 100 trial/final `.par`+`.log` for this campaign (May 1 – Jun 27) | trial + final params/logs |
| `/work/vmo703/igrmonty_outputs/m87/` | 18 final + 17 leftover trial spectra + 1 stale `*_TEST.h5` | scientific output |
| `/work/vmo703/data/final_grmonty_paper.csv` | 24-row model table (Apr 30) with `MunitUsed_pos0/pos1` seeds | input table |
| `/work/vmo703/data/munits_tuning_history.csv` | 706 trial rows, Nov 2025 – Jun 24 2026 (stale 19-col header; see §8 A3) | convergence history |
| `/work/vmo703/igrmonty_logs/row_755312_*`, `row_732850_*`, `auto_munit_bhg_*` | Slurm task stdout/err | exit-status evidence |
| `/work/vmo703/igrmonty/notebooks/grmonty_status_update_2026-06-11.md` | human status memo after May job: 14 converged / 31 walltime / 3 failed — independently confirmed by this audit | prior analysis |
| `/work/vmo703/igrmonty_outputs/m87/5e4_test/` + `/work/vmo703/igrmonty_outputs/m87/_qa/` | Feb 2026 5×10⁴-photon QA campaign (25 spectra) + machine QA report (0 failures) | **known-good reference (selected)** |
| `/work/vmo703/igrmonty_outputs/m87/pair_sweep_20260224/` + `/work/vmo703/_reports/positrons_20260224/` | 3-point positron sweep + implementation report | secondary reference |
| `/work/vmo703/igrmonty/logs/{5e4_test,pair_sweep_20260224}/`, `igrmonty/scratch_logs/` | par/logs of the reference campaigns and older scraps | reference provenance |
| `/work/vmo703/igrmonty_outputs/m87/test_scrap/`, `.../images/` (empty) | scrap/test artifacts | stale, ignore |
| `/work/vmo703/igrmonty_outputs/sgra/` | **empty** | unused |

---

## 4. Known-good comparison selection

Candidates considered:

1. **`/work/vmo703/igrmonty_outputs/m87/5e4_test` (SELECTED).** 25 completed spectra (Feb 16 2026) from the same pipeline/naming, with a machine-generated QA report at `/work/vmo703/igrmonty_outputs/m87/_qa/QA_REPORT.md` ("Pass sanity checks: 25; Total non-pass files: 0"), plus par/logs preserved under `igrmonty/logs/5e4_test/`. Commit `0900329` documents it: "ran QA on 5e4 runs and ready to rock and roll with 1e6 for paper quality data" — i.e., it is the intended pre-flight reference for exactly the audited campaign.
2. `/work/vmo703/igrmonty_outputs/m87/pair_sweep_20260224` (secondary). Only 3 spectra, but the finished positron-scaling validation with report `/work/vmo703/_reports/positrons_20260224/implementation_report.md`; used here to sanity-check pair-scaling direction.
3. `/work/vmo703/ipole_outputs` — rejected: ipole (imaging code), not GRMONTY.
4. `igrmonty_outputs/sgra`, `/work/vmo703/sgrA` — rejected: empty / scripts only.

Uncertainty: low. If you intended a different reference, it does not exist in this workspace.

---

## 5. Per-model convergence history (May–June 2026 campaign)

Reconstructed from `munits_tuning_history.csv` (schema-corrected; see §8 A3), trial `.par`/`.log` files, and Slurm task logs. `err%` is vs the 0.5 Jy target. The algorithm keeps no brackets; "iterations" are successive power-law updates. No duplicated *trial executions* were found (duplicate CSV *rows* exist — the by-design conv=0/conv=1 double-logging plus a few resume re-logs). The observable behaved monotonically with M_unit in every measured search.

### 5.1 Converged branches (18) — all stopped on genuine tolerance, none on the max-iters fallback

| Model branch | Trials (M_unit → F[Jy], err%) | Accepted | Binary |
|---|---|---|---|
| MAD_RBETA a+0.94 t5000 pos0 | 1.92e25→0.744 (+49); 1.55e25→0.556 (+11); 1.45e25→0.4998 (−0.04) | #03, M=1.44588445e+25 | e6ede55 |
| MAD_RBETA a+0.94 t5000 pos1 | 9.74e24→0.598 (+20); 8.90e24→0.523 (+4.7*); 8.66e24→0.5065 (+1.3) | #03, M=8.65851127e+24 | e6ede55 |
| SANE_RBETAwJET a−0.5 t4000 pos0 | 2.44e29→36.2 (+7137); 2.87e28→0.254 (−49); 4.03e28→0.584 (+17); 3.73e28→0.4875 (−2.5) | #04, M=3.72628145e+28 | e6ede55 |
| SANE_RBETAwJET a−0.5 t4000 pos1 | 1.36e29→31.4 (+6174); 1.72e28→0.216 (−57); 2.61e28→0.627 (+25); 2.33e28→0.4752 (−4.96) | #04, M=2.33153833e+28 | e6ede55 |
| SANE_RBETAwJET a−0.5 t5000 pos0 | 1.29e29→7.25 (+1350); 3.39e28→0.274 (−45); 4.59e28→0.594 (+19); 4.21e28→0.471 (−5.7); 4.33e28→0.508 (+1.7) | #05, M=4.33350115e+28 | e6ede55 |
| SANE_RBETAwJET a−0.5 t6000 pos0 | 2.05e29→3.63 (+626); 7.62e28→0.352 (−30); 9.09e28→0.533 (+6.6); 8.80e28→0.512 (+2.4) | #04, M=8.79848295e+28 | e6ede55 |
| SANE_RBETAwJET a−0.5 t6000 pos1 | 1.10e29→2.65 (+429); 4.77e28→0.344 (−31); 5.75e28→0.542 (+8.5); 5.52e28→0.501 (+0.2) | #04, M=5.52174851e+28 | e6ede55 |
| SANE_RBETAwJET a+0.94 t4000 pos0 | 7.23e27→3.62 (+624); 2.69e27→0.427 (−15); 2.91e27→0.501 (+0.2) | #03, M=2.90512620e+27 | e6ede55 |
| SANE_RBETAwJET a+0.94 t4000 pos1 | 3.89e27→2.94 (+489); 1.60e27→0.406 (−19); 1.78e27→0.523 (+4.6) | #03, M=1.78118234e+27 | e6ede55 |
| SANE_RBETAwJET a+0.94 t6000 pos0 | 6.30e27→6.10 (+1120, May); 1.80e27→0.397 (−21, May); 2.02e27→0.520 (+4.0, **Jun 12**) | #03, M=2.02283355e+27 | **0bfdb56** |
| SANE_CRITBETAwJET a−0.5 t4000 pos0 | 5.54e28→? … 3.81e28→0.507 (+1.5) [4 measured trials + 2 resume re-logs] | #best, M=3.81220332e+28 | e6ede55 |
| SANE_CRITBETAwJET a−0.5 t4000 pos1 | 2.98e28→? … 2.42e28→0.517 (+3.4) | M=2.42464766e+28 | e6ede55 |
| SANE_CRITBETAwJET a−0.5 t5000 pos0 | 4.41e28→? … 4.24e28→0.4989 (−0.2) | M=4.23715777e+28 | e6ede55 |
| SANE_CRITBETAwJET a−0.5 t6000 pos0 | 5.16e28→0.157 (−69); 9.21e28→0.631 (+26); 8.20e28→0.4984 (−0.3) | #03, M=8.20030883e+28 | e6ede55 |
| SANE_CRITBETAwJET a−0.5 t6000 pos1 | 2.72e28→0.099 (−80); 6.11e28→0.774 (+55); 4.91e28→0.428 (−14); 5.31e28→0.513 (+2.6) | #04, M=5.31216886e+28 | e6ede55 |
| SANE_CRITBETAwJET a+0.94 t4000 pos0 | May trials +332 %… ; converged **Jun 20**: 1.08e27→0.4992 (−0.16) | M=1.08359878e+27 | **0bfdb56** |
| SANE_CRITBETAwJET a+0.94 t4000 pos1 | … converged **Jun 21**: 6.85e26→0.477 (−4.6) | M=6.85123657e+26 | **0bfdb56** |
| SANE_CRITBETAwJET a+0.94 t5000 pos1 | 2.21 Jy start (May); Jun resume: 8.03e26→0.355; 9.53e26→0.556; 9.03e26→0.4827 (−3.5), **Jun 24** | #04, M=9.03279238e+26 | **0bfdb56** |

\* interpolated from log context; full per-trial rows are in `munits_tuning_history.csv`.

The initial seeds (`MunitUsed_pos*` from the Feb-era campaigns) were off by +49 % to +7000 %, so the first trial was never within tolerance, but the damped update converged in 3–5 measured trials wherever GRMONTY itself ran to completion.

### 5.2 Unconverged branches (30) — final dispositions

| Block | Branches | Best measured F (Jy) | Terminal evidence |
|---|---|---|---|
| MAD_RBETA a−0.5 t4000 | pos0, pos1 | 0.261 / 0.229 (1 trial each) | June retrial(s) killed by trial wallclock guard; `row_755312_*_.err`: exit 75 |
| MAD_RBETA a−0.5 t5000 | pos0, pos1 | 0.424 / 0.268 | trial03 guard-killed after 35 h / 20 h (`MAD_RBETA_a-0.5_t5000_rh20_pos*_trial03.log`) |
| MAD_RBETA a−0.5 t6000 | pos0, pos1 | 0.547 / 0.555 | trial04 guard-killed (scatter ratio ran to ~34 at limit 5) |
| MAD_RBETA a+0.94 t4000 | pos0, pos1 | 0.833 / 0.714 | Jun: trial02 `abort_bias` (ratio 5.00 @ limit 5), trial03 guard-killed, exit 75 |
| MAD_RBETA a+0.94 t6000 | pos0, pos1 | 0.472 / 0.474 (−5.5 %, just outside tol.) | trial03 `abort_bias`-adjacent, trial04 guard-killed |
| MAD_CRITBETA (bc0.01 f0.5), all 6 rows | 12 branches | none — no measurable trial ever | May: walltime before first measurement or `init_error fitbias_zero_ratio_single`; Jun 22 retry: **all 12 die in minutes** with `run status: code=5 label=init_error detail=fitbias_zero_ratio_stalled` |
| SANE_RBETAwJET a−0.5 t5000 pos1 | 1 | 0.625 (+25 %) | Jun trials 04/05 `abort_bias` (ratio ≈5 @ limit 5, biasTuning 740–1250), trial06 guard-killed |
| SANE_RBETAwJET a+0.94 t5000 | pos0, pos1 | 0.407 / 0.376 | Jun trial03 both: `init_error fitbias_zero_ratio_stalled` |
| SANE_RBETAwJET a+0.94 t6000 pos1 | 1 | 4.58 (+815 %, 1 trial) | trial02 guard-killed |
| SANE_CRITBETAwJET a−0.5 t5000 pos1 | 1 | 0.454 (−9 %) | May trials only; June job did not reach it before budget |
| SANE_CRITBETAwJET a+0.94 t5000 pos0 | 1 | 2.159 (+332 %) | trial03 `abort_bias`, trial04 `init_error`; task exit 41 |
| SANE_CRITBETAwJET a+0.94 t6000 | pos0, pos1 | none measured | pos0: trial01 `abort_bias`, trial02 guard-killed; pos1: trial01 60 h guard-killed |

All 30 have **no suffix-free final `.par`/`.log`/`.h5`** — the pipeline's own completion marker — so they cannot be mistaken for finished products if you select strictly on suffix-free names.

---

## 6. Per-model validation table (the 18 finals)

Flux re-measured in this audit directly from the final HDF5 with the same estimator the tuner uses. "Bins" = 200 log-spaced 10⁹–10²⁴ Hz; NaN/Inf = 0 for every file; "zero-bin %" (27–30 %) is the empty SED tails, matching the known-good reference. Jaggedness = median |Δlog₁₀ F| per bin within ±10 bins of 230 GHz.

| Model branch | Final .par / .log (in `igrmonty/logs/`) | Final spectrum (in `igrmonty_outputs/m87/`) | M_unit | F₂₃₀ (Jy) | err % | GRMONTY status | Warnings | Jag. | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| MAD_RBETA a+0.94 t5000 pos0 | `MAD_RBETA_a+0.94_t5000_rh20_pos0.{par,log}` | `spectrum_Ma+0.94_5000_RBETA_rh20_pos0.h5` | 1.4459e25 | 0.49979 | −0.04 | ok (code=2) | none | 0.026 | **PASS** |
| MAD_RBETA a+0.94 t5000 pos1 | `..._pos1.{par,log}` | `..._pos1.h5` | 8.6585e24 | 0.50647 | +1.29 | ok | 3× `isnan nu: track_super_photon` (3 / 3.6e7 photons — negligible) | 0.034 | **PASS w/ caveats** (pos1 physics mix) |
| SANE_RBETAwJET a−0.5 t4000 pos0 | `SANE_RBETAwJET_a-0.5_t4000_rh160_pos0.*` | `spectrum_Sa-0.5_4000_RBETAwJET_rh160_pos0.h5` | 3.7263e28 | 0.48753 | −2.49 | ok | none | 0.076 | **PASS** |
| SANE_RBETAwJET a−0.5 t4000 pos1 | 〃 pos1 | 〃 pos1 | 2.3315e28 | 0.47519 | −4.96 | ok | none | 0.095 | **PASS w/ caveats** |
| SANE_RBETAwJET a−0.5 t5000 pos0 | 〃 t5000 pos0 | 〃 | 4.3335e28 | 0.50839 | +1.68 | ok | none | 0.100 | **PASS** |
| SANE_RBETAwJET a−0.5 t6000 pos0 | 〃 t6000 pos0 | 〃 | 8.7985e28 | 0.51181 | +2.36 | ok | none | 0.082 | **PASS** |
| SANE_RBETAwJET a−0.5 t6000 pos1 | 〃 t6000 pos1 | 〃 | 5.5217e28 | 0.50121 | +0.24 | ok | none | 0.098 | **PASS w/ caveats** |
| SANE_RBETAwJET a+0.94 t4000 pos0 | 〃 a+0.94 t4000 pos0 | 〃 | 2.9051e27 | 0.50110 | +0.22 | ok | none | 0.045 | **PASS** |
| SANE_RBETAwJET a+0.94 t4000 pos1 | 〃 pos1 | 〃 | 1.7812e27 | 0.52293 | +4.59 | ok | none | 0.048 | **PASS w/ caveats** |
| SANE_RBETAwJET a+0.94 t6000 pos0 | 〃 t6000 pos0 | 〃 | 2.0228e27 | 0.52007 | +4.01 | ok | none | 0.047 | **PASS** (June binary — pos0 physics identical) |
| SANE_CRITBETAwJET a−0.5 t4000 pos0 | `SANE_CRITBETAwJET_a-0.5_t4000_rh20_bc1_f0.5_pos0.*` | `spectrum_Sa-0.5_4000_CRITBETAwJET_rh20_bc1_f0.5_pos0.h5` | 3.8122e28 | 0.50734 | +1.47 | ok | none | 0.078 | **PASS** |
| SANE_CRITBETAwJET a−0.5 t4000 pos1 | 〃 pos1 | 〃 | 2.4246e28 | 0.51714 | +3.43 | ok | none | 0.058 | **PASS w/ caveats** |
| SANE_CRITBETAwJET a−0.5 t5000 pos0 | 〃 t5000 pos0 | 〃 | 4.2372e28 | 0.49888 | −0.22 | ok | none | 0.106 | **PASS** |
| SANE_CRITBETAwJET a−0.5 t6000 pos0 | 〃 t6000 pos0 | 〃 | 8.2003e28 | 0.49845 | −0.31 | ok | none | 0.059 | **PASS** |
| SANE_CRITBETAwJET a−0.5 t6000 pos1 | 〃 t6000 pos1 | 〃 | 5.3122e28 | 0.51286 | +2.57 | ok | none | 0.059 | **PASS w/ caveats** |
| SANE_CRITBETAwJET a+0.94 t4000 pos0 | `..._a+0.94_t4000_...pos0.*` | 〃 | 1.0836e27 | 0.49921 | −0.16 | ok | none | 0.082 | **PASS** (June binary) |
| SANE_CRITBETAwJET a+0.94 t4000 pos1 | 〃 pos1 | 〃 | 6.8512e26 | 0.47727 | −4.55 | ok | none | 0.085 | **PASS w/ caveats** (June pair physics) |
| SANE_CRITBETAwJET a+0.94 t5000 pos1 | 〃 t5000 pos1 | 〃 | 9.0328e26 | 0.48265 | −3.47 | ok | none | 0.099 | **PASS w/ caveats** (June pair physics) |

All 18: dump paths in the `.par` resolve to existing files; `Ns 1e6` target photons with 24–36 M superphotons actually made (bias); `MBH 6.5e9`, `TP_OVER_TE 3`, `trat_small 1`, correct `trat_large` (20/160), correct `with_electrons` mode, `Thetae_max 1e100`, `positron_ratio` 0/1 as tagged, wJET parameters at repo defaults (σ_trans=2, β_e0=0.1, exponent 0, jet σ_cut=10, jet β_cut=0.1, jet Θe=50, jet ne_mult=1). CRITBETAwJET finals carry `beta_crit 1`, `beta_crit_coefficient 0.5`. Naming ↔ content cross-check passed (no final whose internal parameters contradict its filename; the "rh20" tag on CRITBETA models is a placeholder — `trat_large` is ignored by critical-beta heating, consistent with the CSV notes column).

**Use-case grades for the 18:** safe for qualitative plotting — yes, all. Safe for quantitative comparison — yes, all (statistical flux error at 230 GHz ≪ the 5 % tuning tolerance given ≥2.4e7 superphotons). Safe for publication tables — yes for pos0; pos1 values quoted in brems-sensitive bands (X-ray) should wait for the physics-consistency decision (§8 A2). Repeat-seed tests were never run (seed −1, unrecorded), so quoted uncertainties should come from the 5 % tuning tolerance, not MC error bars.

---

## 7. Comparison with the known-good output (`5e4_test` + `_qa`)

**A. Structure.** Identical conventions: spectra as `spectrum_<dump-tag>_<model>_<pos>.h5` under `igrmonty_outputs/m87/`, par/logs under `igrmonty/logs/`, suffix-free = final. Differences: the audited campaign adds `rh`, `bc`, `f` tags and pos0/pos1 branches (expected — richer grid); the known-good campaign has a **machine QA report** (`_qa/QA_REPORT.md`, inventory/sanity/pairwise CSVs, plots) that the new campaign lacks; the new campaign has the Slurm `.out/.err` audit trail the old one lacks. Nothing required by downstream tooling is missing from the audited finals.

**B. Parameters.** Ns 5e4 → 1e6 (expected: QA vs paper quality). `fit_bias_ns 50000` / `ratio 1.414` in the reference matches the **May** finals exactly; **June** finals differ (`14000` / `1.05` / `bias_abort_ratio 5`) — operational difference requiring a one-line methods note, not an error (bias tuning is variance reduction; it does not bias the estimator). MBH, dumps, distance, electron modes, wJET parameters all agree. Reference `.par`s predate positron support (no `positron_ratio` key) — expected.

**C. Logs.** Reference logs end with the same `N_superph_made/scatt/recorded` + wallclock block and no fatal patterns; audited final logs add the `run status:` line (an improvement from the June/March code). Case-insensitive sweep of all 18 final logs for error/warning/failed/fatal/nan/inf/abort/segmentation/killed/timeout/missing/cannot open/zero photons/overflow: the only hits are the 3 benign `isnan nu` photon-rejection lines in MAD_RBETA a+0.94 t5000 pos1 and the routine bias-tuning progress lines. No segfaults, no MPI/Slurm errors, no truncation (every final log reaches the summary block).

**D. Numerics.** Reference: 25/25 spectra parse, 0 bad bins, F₂₃₀ = 0.435–0.532 Jy at 5e4 photons. Audited finals: 18/18 parse, 0 bad bins, F₂₃₀ = 0.475–0.523 Jy at 1e6 photons — tighter, as expected with 20× photons; jaggedness near 230 GHz comparable or better. Pair-scaling direction check against `pair_sweep_20260224` (pairf0 → 0.285 Jy, pairf0.5 → 0.469, pairf1 → 0.839 at fixed M_unit): monotonic increase with pair fraction, consistent with every audited pos1 branch needing a **lower** M_unit than its pos0 sibling (ratio ≈ 0.60–0.63 across all 8 pairs — internally consistent and physically plausible). No all-zero interior sections, no isolated spikes beyond MC noise, ordered frequency grid in every file.

---

## 8. Problems and anomalies (ranked)

**CRITICAL**
- **A1. 30 of 48 branches have no final product.** The MAD_RBETA block (10 of 12 branches) and the entire MAD_CRITBETA block (12) are missing, plus 8 SANE branches. Evidence: no suffix-free finals; Slurm exits 75/41; `run status` lines quoted in §5.2. The MAD_CRITBETA (β_crit=0.01) block **cannot** currently run at all: every June attempt dies in `fit_bias` with `fitbias_zero_ratio_stalled` (e.g. `/work/vmo703/igrmonty/logs/MAD_CRITBETA_a-0.5_t4000_rh20_bc0.01_f0.5_pos0_trial01.log`) — the bias fitter finds zero scattering ratio, i.e. these ultra-cold-disk models produce essentially no Compton events at the seeded M_unit under the June guard settings. This needs a code/strategy fix, not more walltime.

**MAJOR**
- **A2. Physics changed mid-campaign for pair models.** Commit `f72bbc1` (between binaries `e6ede55` → `0bfdb56`) added the opposite-sign (e⁻e⁺) pair bremsstrahlung term in `src/jnu_mixed.c`. The term vanishes at `positron_ratio=0`, so the 10 pos0 finals are physics-consistent regardless of binary. But the 8 pos1 finals split 6 (old, incomplete pair-brems) / 2 (new: SANE_CRITBETAwJET a+0.94 t4000 pos1, t5000 pos1). At 230 GHz (synchrotron-dominated) the M_units are essentially unaffected, but brems-sensitive bands of the pos1 SEDs are not mutually consistent across the set. Decide: either re-run the 6 May pos1 finals under `0bfdb56`+, or document that pair brems differs and avoid quoting pos1 X-ray fluxes.
- **A3. `munits_tuning_history.csv` has a stale header.** The header is the Nov 2025 19-column schema, but rows appended since carry 20 then 27 fields (`_append_history` only writes a header when the file is new). Naïve `DictReader` parsing silently mis-assigns every recent column (this audit had to re-map by row length). Any downstream script that read this file naively produced garbage. The file is also uncommitted (git status: modified).
- **A4. Both campaign binaries are `-dirty` builds.** The June build is fully recoverable (`build_archive/` snapshot verified identical to the archived source; `build_archive/main.c` == current `src/main.c`). The **May binary's dirty state was never archived** and the binary itself was overwritten on Jun 11 — the exact code that produced 14 of 18 finals cannot be reconstructed with certainty (git stashes `saving my wjet edits` etc. may contain it, unverified).

**MODERATE**
- **A5. Latent false-convergence path** in `auto_munit_bracket.py:1265-1281` (max-iters → closest trial logged `converged=1` and promoted to final names). Did not fire in this campaign (verified), but will silently bless an out-of-tolerance model in a future run.
- **A6. June bias settings block completion.** `target_ratio 1.05` + `bias_abort_ratio 5` + `fit_bias_ns 14000` caused most June failures (`abort_bias` at ratio≈5; `fitbias_zero_ratio_stalled`), including on models that ran fine in May at ratio √2 / limit 10. The current uncommitted `src/main.c`/`par.c` edits and stash notes ("hard guard on bias …") suggest a fix was in progress but never re-run.
- **A7. Final `.par` `spectrum` field points at the pre-rename trial path** (e.g. `spectrum_..._trial04.h5`), which no longer exists. Harmless for reading results; misleading for re-runs (re-running the final `.par` writes a *trial* file, not the final name).

**MINOR**
- A8. 3 `isnan nu: track_super_photon` photons in `MAD_RBETA_a+0.94_t5000_rh20_pos1.log` (3 of 3.6e7; negligible).
- A9. `seed -1` (time-seeded) and never recorded → bitwise reproduction impossible; only statistical reproduction.
- A10. Stale clutter in the output root: `spectrum_Sa-0.5_4000_CRITBETA_pos0_TEST.h5` (Jun 11 binary test; a *different* file with the same name sits in `test_scrap/` — same name, different checksums), 17 in-progress `_trialNN.h5` spectra intermixed with finals (required by `--resume`, but easy to plot by accident), and an empty `images/` directory.
- A11. A few duplicated identical rows in the history CSV from resume re-logging; and `final_grmonty_paper.csv` says `converged=True` for all 24 rows — that column describes the *February* campaign that produced the seeds, not the current one; the seeds proved 1.5×–70× off under current code.

---

## 9. Missing evidence

- Exact source of the **May** binary (`e6ede55-dirty`): the dirty diff is unarchived (A4). This is the only gap preventing bit-level provenance for 14 of the 18 finals.
- Random seeds (A9) — prevents exact re-runs and formal MC error bars via repeat seeds.
- Slurm `sacct` records were not queried (job age); the `.out/.err` files were sufficient for dispositions.
- No stored mapping from paper figure/table labels to spectra (the paper-side notebooks reference `paper_output*.csv` from the Feb campaigns; nothing yet consumes the 18 new finals, so no figure-level cross-check was possible).

---

## 10. Recommended next actions

**Mandatory before using results in the paper**
1. **Decide the M87 model grid.** If MAD models at all dump times are required, the dataset is incomplete: resume the 16 partial branches (`--resume` works; several are within 10 % already, e.g. MAD_RBETA a+0.94 t6000 at −5.5 %) and fix the MAD_CRITBETA bias-init failure before spending walltime on it (A1, A6).
2. **Resolve the pos1 physics split (A2):** rerun the 6 May-era pos1 finals under the current (archived) binary, or explicitly scope pair results to synchrotron-dominated bands and document the brems-treatment difference.
3. **Freeze provenance:** commit the current `igrmonty` working tree (or tag `build_archive/`), commit `data/munits_tuning_history.csv` and the final `.par`s (several are untracked), and record which finals came from which binary (table in §6).
4. If any June-era value is quoted alongside May-era values, add a methods note that bias-tuning settings (variance-reduction only) differed (§2.3).

**Recommended (not blocking)**
5. Fix the stale header in `munits_tuning_history.csv` (A3) or split per-schema files, before any script reads it again.
6. Patch `auto_munit_bracket.py` so the max-iters fallback writes `converged=0` (or a distinct flag) and does not promote to suffix-free names (A5).
7. Write the final-run `.par`s with the final spectrum path at promotion time (A7).
8. Move `*_TEST.h5` and trial spectra of *finished* models out of the output root; delete or archive `test_scrap/`, empty `images/` (A10).
9. Re-run the `_qa/qa_grmonty_outputs.py` sanity pipeline on the 18 finals so the paper campaign has the same QA artifact as the reference campaign.
10. Record seeds (set `seed` explicitly per trial) in future runs; optionally run 2–3 repeat seeds on one representative model to quote MC scatter.

---

## 11. Reproducibility manifest (proposed)

For each of the 18 finals, the complete chain exists today as:

| Field | Location pattern |
|---|---|
| Model label | `{STATE}_{MODEL}_a{spin}_t{dump}_rh{Rhigh}[_bc{βc}_f{f}]_pos{p}` |
| Input dump | `/work/vmo703/grmhd_dump_samples/{M\|S}a{spin}_{dump}.h5` |
| Final `.par` | `/work/vmo703/igrmonty/logs/<label>.par` |
| Final GRMONTY log | `/work/vmo703/igrmonty/logs/<label>.log` (ends `run status: code=2 label=ok`) |
| Accepted M_unit | `M_unit` line of the final `.par` == last `converged=1` row for the label in `/work/vmo703/data/munits_tuning_history.csv` |
| Convergence history | all rows for the label in `munits_tuning_history.csv` + Slurm `row_{732850,755312}_*_{row}_pos*.{out,err}` in `/work/vmo703/igrmonty_logs/` |
| Spectrum | `/work/vmo703/igrmonty_outputs/m87/spectrum_{M\|S}a{spin}_{dump}_{MODEL}_rh{Rhigh}[_bc..._f...]_pos{p}.h5` |
| Binary/source | June finals: `/work/vmo703/igrmonty/build_archive/` (verified); May finals: `e6ede55` + unarchived dirty edits (**gap**) |
| Environment | module list + venv in `/work/vmo703/igrmonty/run_auto_munit.slurm` |
| Downstream input | 230 GHz flux + inclination cone via `tools/viewing_cone_postprocess.py` (validated in `/work/vmo703/_reports/munit_ipole_vs_grmonty_20260218/`) |

Writing this table out as a small CSV (18 rows, plus binary hash and flux columns from §6) is the single highest-value archival step.

---

## 12. Post-audit actions (2026-07-10, same day)

At the user's request, the MAD_CRITBETA bias-init failure was diagnosed and fixed, and the 30 incomplete branches were resubmitted. This section supersedes recommendation #1 of §10.

**Root cause of the MAD_CRITBETA `fitbias_zero_ratio_stalled` failure.** Not a regression, and not walltime: GRMONTY's fit-bias initialization is structurally unable to run β_crit=0.01 MAD models. At bc=0.01 the disk electrons are exponentially cold (β≫β_crit) and the funnel is cooled by the σ-cut (`Thetae = SMALL` in `model/iharm/model.c`), so nearly every photon in the small fit-bias sample is bremsstrahlung-tagged; the scatter gate at `src/track_super_photon.c:450` (`ph->ratio_brems < 0.9 && …`) excludes such photons from Compton sampling, so `N_scatt ≡ 0` at any bias and the fitter exits with code 41 after 10 futile ×5 escalations (`src/main.c:220`). Supporting evidence: the "converged" MAD_CRITBETA entries in the Jan/Feb history were **bc=1.0** runs (`scratch_logs/MAD_CRITBETA_a-0.5_t4000_pos0.par`), and the bc=0.01 seed M_units in `paper_output_critbeta.csv` came from an **ipole** sweep (`/work/vmo703/ipole_outputs/M87/betacrit/`), which has no Monte Carlo scattering and no bias fitter. GRMONTY had never successfully run this configuration.

**Fix (config-only, no rebuild, no new binary).** Run the MAD_CRITBETA branches with `fit_bias 0` (fixed `bias 0.05`), skipping the fit-bias phase entirely. Compton scattering is physically negligible for these models. Validated by smoke test (scratchpad, Ns=2×10⁴, copy of the previously failing trial-01 par): completed in 225 s with `run status: code=2 label=ok`, 711k superphotons, scatter ratio 0.0096, and F₂₃₀ = 0.27 Jy at the seed M_unit 3.634e25 — the seeds are within a factor ~2 of target, so ~2–3 tuner iterations expected.

**Resubmissions (2026-07-10):**
- **Job 770778** (`auto_munit_madcb`), tasks 36–47 (all 12 MAD_CRITBETA branches): env `AUTO_MUNIT_FIT_BIAS=0`, all other June settings unchanged.
- **Job 770779** (`auto_munit_resume`), tasks 3, 8, 9, 11–19, 22, 23, 27, 32, 34, 35 (the 18 partial branches), with `--resume` (default) and the bias knobs reverted to the May-proven values that produced the 14 May finals: `AUTO_MUNIT_FIT_BIAS_NS=50000`, `AUTO_MUNIT_TARGET_RATIO=√2`, `AUTO_MUNIT_BIAS_ABORT_RATIO=10`. Rationale: the stricter June knobs (14000 / 1.05 / 5) caused the June `abort_bias` and `init_error` failures and the runaway-scatter slowdowns documented in §5.2.
- Task-index → row/pos mapping (task = 2·row + pos) was re-verified against the script's own task-list generator before submission. Both jobs started immediately on `c122` (%5 throttle).

**Caveats carried forward:** (a) new pos1 completions will use the June binary's fuller pair-brems treatment, widening the §8 A2 split unless the 6 May pos1 finals are eventually re-run; (b) the §8 A5 max-iters fallback remains unpatched — after these jobs finish, verify each new final's flux is within tolerance (do not trust `converged=1` alone); MAD_CRITBETA `.par` files from these jobs will show `fit_bias 0`, an intentional, documented operational difference.

## 13. Final verdict

**Partially ready.** The 18 finished branches are trustworthy: convergence is genuine, GRMONTY completed cleanly, the files are internally consistent, numerically clean, and match or exceed the known-good reference in quality. You can use the **10 pos0 finals now** for figures, tables, and quantitative claims, and the **8 pos1 finals** for anything synchrotron-dominated once you document (or eliminate) the pair-bremsstrahlung version split. **Do not** present this as the full 24-model × 2-composition grid: 30 of 48 branches — including nearly all MAD models — have no valid output, and the MAD_CRITBETA block is blocked by a real code failure, not just walltime. Complete or descope those before the paper's model table is drafted.
