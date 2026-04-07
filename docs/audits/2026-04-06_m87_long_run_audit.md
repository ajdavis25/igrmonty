# M87 igrmonty/grmonty Audit

Date: 2026-04-06

Repo: `/work/vmo703/igrmonty`

Branch: `positrons`

HEAD: `e6ede55854baf36ffd7e70ccd30d668ffd6e5865`

Upstream merge-base used for audit: `4a1b1c5a6394a7967b1e901f2a5b15b6e1f5d1a2`

## A. Executive Summary

This fork is an `igrmonty` / `grmonty` workflow for M87, with custom `with-jet` and positron support, plus an automated `M_unit` tuning pipeline. The 20-day Slurm batch I inspected was job `712854` on `anantuabhg`, launched by `run_auto_munit.slurm` with `--time=20-00:01:00`.

Verdict on the 20-day batch:

- The batch was **not wasted overall**.
- It produced **two scientifically usable final spectra**:
  - `igrmonty_outputs/m87/spectrum_Ma-0.5_4000_CRITBETA_rh20_bc1_f0.5_pos0.h5`
  - `igrmonty_outputs/m87/spectrum_Ma-0.5_5000_CRITBETA_rh20_bc1_f0.5_pos0.h5`
- It also produced **two valid positron trial products** that are useful for tuning, but not final converged `M_unit` solutions:
  - `..._4000_..._pos1_trial01.h5`
  - `..._5000_..._pos1_trial01.h5`
- Most of the `wJET` work in this batch was **not scientifically usable as final output**, either because the run explicitly aborted on the bias guard or because it timed out before writing a final spectrum.

Main findings:

1. The clean completed products in this batch are the plain `CRITBETA` runs, not the `wJET` runs.
2. Your current `auto_munit` workflow tunes `M_unit` against a **full `4pi` angular average**, not an IPOLE-like camera angle.
3. The current `with-jet` workflow label is potentially misleading:
   - `with_electrons=4/5` is active in `wJET` runs.
   - But your recent auto-generated `.par` files do **not** set `jet_sigma_cut`, `jet_beta_cut`, `jet_thetae`, or `jet_ne_mult`.
   - In the recent `wJET` runs, the only active extra physics was the additive constant-beta temperature supplement for `sigma >= sigma_transition`.
4. The positron implementation is internally coherent for the main synchrotron and scattering paths, and it was definitely active in the `pos1` runs.
5. The main performance cost is transport plus Compton sampling in highly scattering branches. The worst `wJET` branches are expensive because they both scatter heavily and hit pathological sampling / invalid-frequency paths.

High-priority next steps:

- Treat the completed `CRITBETA` `pos0` spectra as usable.
- Treat the `pos1` trial files as useful tuning evidence, not final science outputs.
- Do **not** treat the recent `wJET` results as validated science products.
- Fix the `with-jet` parameter plumbing in `auto_munit_bracket.py` before trusting future `wJET` labels.
- If you want IPOLE-comparable `M_unit`, move from `4pi` tuning to a detector-side viewing-angle workflow inside `grmonty` or in a tightly coupled postprocess, but do not expect that alone to eliminate all IPOLE vs `grmonty` differences.

## B. Run Audit

### B1. Which long run was inspected

The exact 20-day batch was launched by `run_auto_munit.slurm:2-11`:

```bash
#SBATCH --job-name=auto_munit_bhg
#SBATCH --partition=anantuabhg
#SBATCH --account=anantuabhg
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --time=20-00:01:00
#SBATCH --output=/work/vmo703/igrmonty_logs/%x_%j.out
#SBATCH --error=/work/vmo703/igrmonty_logs/%x_%j.err
```

Scheduler accounting for job `712854` shows:

- start: `2026-03-17T08:15:46`
- end: `2026-04-06T08:17:04`
- state: `TIMEOUT`

The terminating scheduler error is explicit in `igrmonty_logs/auto_munit_bhg_712854.err:1`:

> `slurmstepd-c117: error: *** JOB 712854 ON c117 CANCELLED AT 2026-04-06T08:17:04 DUE TO TIME LIMIT ***`

Important path note:

- Slurm wrapper stdout/stderr for the batch and per-row tasks live in `/work/vmo703/igrmonty_logs`.
- Per-trial `grmonty` `.log` / `.par` files live in `/work/vmo703/igrmonty/logs`.

### B2. Which row tasks in that batch produced useful products

The batch expands CSV rows and positron branches in `run_auto_munit.slurm:59-259`, then launches one `python auto_munit_bracket.py --row ...` worker per row/positron branch.

The only clearly completed final M87 products near the end of this batch were:

| Row log | Model | Status | Final output |
| --- | --- | --- | --- |
| `igrmonty_logs/row_712854_48_pos0.out` | `MAD CRITBETA a=-0.5 dump=4000 rh=20 bc=1 f=0.5 pos=0` | converged on trial 2 | `igrmonty_outputs/m87/spectrum_Ma-0.5_4000_CRITBETA_rh20_bc1_f0.5_pos0.h5` |
| `igrmonty_logs/row_712854_49_pos0.out` | `MAD CRITBETA a=-0.5 dump=5000 rh=20 bc=1 f=0.5 pos=0` | converged on trial 2 | `igrmonty_outputs/m87/spectrum_Ma-0.5_5000_CRITBETA_rh20_bc1_f0.5_pos0.h5` |

Evidence:

`igrmonty_logs/row_712854_48_pos0.out:6-10`

> `[update] trial=01 scale=3.963e-01 P=2.0 raw_M=1.8083e+25 capped_M=1.8083e+25`
>
> `[trial 02] M_unit=1.8083e+25 -> F=5.112127e-01 Jy`
>
> `[done] best trial: #02 M_unit=1.808310e+25 flux=0.5112 Jy (target 0.500 Jy)`

`igrmonty_logs/row_712854_49_pos0.out:6-10`

> `[update] trial=01 scale=2.661e-01 P=2.0 raw_M=2.1260e+25 capped_M=2.1260e+25`
>
> `[trial 02] M_unit=2.1260e+25 -> F=5.249927e-01 Jy`
>
> `[done] best trial: #02 M_unit=2.126029e+25 flux=0.5250 Jy (target 0.500 Jy)`

`auto_munit_bracket.py:995-1031` then renames the winning trial output to the final no-`_trialXX` name. That explains why the final output files are named without a trial suffix even though the stored trial parfiles still reference `..._trial02.h5`.

Two positron branches also produced valid trial outputs before the batch timed out:

| Row log | Model | Status | Output |
| --- | --- | --- | --- |
| `igrmonty_logs/row_712854_48_pos1.out` | same as row 48, but `positron_ratio=1` | trial 1 completed, trial 2 interrupted by walltime | `igrmonty_outputs/m87/spectrum_Ma-0.5_4000_CRITBETA_rh20_bc1_f0.5_pos1_trial01.h5` |
| `igrmonty_logs/row_712854_49_pos1.out` | same as row 49, but `positron_ratio=1` | trial 1 completed, trial 2 interrupted by walltime | `igrmonty_outputs/m87/spectrum_Ma-0.5_5000_CRITBETA_rh20_bc1_f0.5_pos1_trial01.h5` |

Evidence from `igrmonty_logs/row_712854_48_pos1.out:1-7`:

> `[ctx] positron_ratio=1 (source=cli)`
>
> `[note] nonzero positron_ratio: GRMONTY applies pair scaling internally; do not pre-scale M_unit by (1+2*positron_ratio).`
>
> `[trial 01] M_unit=1.6242e+25 -> F=1.233303e+00 Jy`
>
> `[trial 02] running GRMONTY with M_unit=1.0341e+25...`

And the interruption is explicit in `igrmonty_logs/row_712854_48_pos1.err:1-3`:

> `srun: Job step aborted: Waiting up to 32 seconds for job step to finish.`
>
> `slurmstepd-c117: error: *** STEP 712854.61 ON c117 CANCELLED AT 2026-04-06T08:17:04 DUE TO TIME LIMIT ***`

### B3. Parameter choices for the usable completed run

Representative final converged parfile:

`igrmonty/logs/MAD_CRITBETA_a-0.5_t4000_rh20_bc1_f0.5_pos0.par:1-20`

```text
seed -1
Ns 1000000.0
MBH 6500000000.0
M_unit 1.80830987e+25
dump /work/vmo703/grmhd_dump_samples/Ma-0.5_4000.h5
spectrum /work/vmo703/igrmonty_outputs/m87/spectrum_Ma-0.5_4000_CRITBETA_rh20_bc1_f0.5_pos0_trial02.h5

fit_bias 1
fit_bias_ns 50000
bias 0.05
ratio 1.4142135623730951

TP_OVER_TE 3.0
beta_crit 1
beta_crit_coefficient 0.5
with_electrons 3
trat_small 1
trat_large 20
positron_ratio 0
Thetae_max 1e100
```

Interpretation:

- electron model: `with_electrons=3` = critical-beta electron model
- heating / electron prescription: `beta_crit=1`, `beta_crit_coefficient=0.5`
- `with-jet`: **not enabled in this final usable output**
- positrons: **not enabled in this final usable output**
- normalization: `M_unit = 1.80830987e+25 g`
- photon budget: `Ns=1e6`, with bias fitting enabled
- bias fitter: target scatter effectiveness ratio `sqrt(2)`, starting bias `0.05`

The other converged final run is identical in setup except:

- dump: `Ma-0.5_5000.h5`
- `M_unit = 2.12602907e+25 g`

### B4. Camera, binning, and post-processing assumptions

The `iharm` model bins escaped photons by folded polar angle and frequency:

- `model/iharm/model.h:28-30` sets `N_EBINS=200`, `N_THBINS=18`
- `src/decs.h:40-41` sets `N_COMPTBINS=3`, `N_TYPEBINS=8`
- `model/iharm/model.c:140-224` records photons into `spect[type][theta][energy]`
- `model/iharm/model.c:156-169` folds `theta` about the equator before binning
- `model/iharm/model.c:1492` stores `dOmega_buf[j] = 2 * dOmega_func(j)` because of that hemispheric fold
- `model/iharm/model.c:1532-1548` writes `/output/lnu`, `/output/dOmega`, `/output/nuLnu`, `tau_*`, and run status metadata into HDF5

This means the stored spectrum is:

- all-sky in the sense of escaped photons over `4pi`
- azimuth-averaged
- folded north/south about the equator
- not a finite camera image plane
- not a single IPOLE line-of-sight transport solution

This is reinforced by the postprocess tool itself, `tools/viewing_cone_postprocess.py:19-24`:

> `theta is folded about the equator during binning`
>
> `a cone cut is an azimuth-averaged theta cut, not a true camera/FOV`
>
> `cone_renorm_4pi_equiv ... still cannot reproduce finite image-plane/FOV ray tracing`

The auto-tuner does not use that cone tool. `auto_munit_bracket.py:388-505` measures the 230 GHz flux by integrating `nuLnu * dOmega / 4pi` over the full angular block. So the current `M_unit` workflow is a `4pi`-average workflow.

### B5. Runtime behavior

Representative clean completed run:

`igrmonty/logs/MAD_CRITBETA_a-0.5_t4000_rh20_bc1_f0.5_pos0.log:22-25`

> `finding bias (target effectiveness ratio 1.41421)...`
>
> `bias 0.05 ..ratio = 8.47977`
>
> `tuning time 747s`

Tail of the same log:

`igrmonty/logs/MAD_CRITBETA_a-0.5_t4000_rh20_bc1_f0.5_pos0.log:761-779`

> `compute time 461268s, ph made 36.5M, rate 0.0791k/s, scatter 157M, ratio 4.31`
>
> `N_superph_made = 36467887`
>
> `N_superph_scatt = 157197379`
>
> `N_superph_recorded = 92390568`
>
> `Total wallclock time: 462464 s`
>
> `run status: code=2 label=ok detail=ratio=4.31057 limit=10 biasTuning=0.05`

The corresponding final HDF5 file is internally consistent:

- `run_status = ok`
- `run_status_code = 2`
- `run_status_detail = ratio=4.31057 limit=10 biasTuning=0.05`
- `sum(dOmega) = 12.56637061435917`, i.e. `4pi`
- `nuLnu` shape `(8, 200, 18)`
- all checked `nuLnu` values finite
- 230 GHz flux reproduced from the file as `0.5112126908491682 Jy`

The second completed final run at dump `5000` is similarly clean:

- `run_status = ok`
- `run_status_detail = ratio=2.63883 limit=10 biasTuning=0.05`
- 230 GHz flux `0.5249926913982148 Jy`

Angular structure of the clean completed spectra looks plausible rather than pathological. At 230 GHz, both files have nonzero signal across all 18 folded-theta bins, with a smooth rise toward larger polar angle instead of a single-bin spike.

### B6. Did the batch end only because of walltime

No. The batch itself ended because of walltime, but individual tasks inside that batch fell into three classes:

1. **Completed cleanly before walltime**
   - the two final `CRITBETA pos0` products above

2. **Still healthy but incomplete when walltime hit**
   - the `CRITBETA pos1 trial02` branches

3. **Scientifically/numerically problematic independent of the final walltime**
   - many `wJET` branches aborted on the bias guard
   - some long-running `wJET` branches logged repeated invalid-frequency events and never wrote a clean final spectrum

Representative explicit abort:

`igrmonty/logs/MAD_CRITBETAwJET_a-0.5_t4000_rh20_bc1_f0.5_pos0_trial01.log:485-492`

> `bias guard triggered: ratio=10 limit=10 biasTuning=0.05`
>
> `it seems the bias was too high -- aborting (ratio=10 limit=10 biasTuning=0.05).`
>
> `run status: code=3 label=abort_bias detail=ratio=10 limit=10 biasTuning=0.05`

So the correct statement is:

- **the Slurm batch ended because of walltime**
- but **several `wJET` runs were already in bad scientific/numerical territory even before the scheduler killed the batch**

### B7. Warning signs found

#### Clean `CRITBETA` outputs

For the four inspected `CRITBETA` logs:

- `MAD_CRITBETA_a-0.5_t4000_rh20_bc1_f0.5_pos0.log`
- `MAD_CRITBETA_a-0.5_t5000_rh20_bc1_f0.5_pos0.log`
- `MAD_CRITBETA_a-0.5_t4000_rh20_bc1_f0.5_pos1_trial01.log`
- `MAD_CRITBETA_a-0.5_t5000_rh20_bc1_f0.5_pos1_trial01.log`

I did **not** find:

- `isnan nu`
- `abort_bias`
- boundary warnings
- NaN / inf output data in the checked HDF5 products

These are the scientifically strongest products from the batch.

#### Problematic `wJET` branches

Representative long-running `wJET` log:

`igrmonty/logs/MAD_RBETAwJET_a+0.94_t4000_rh80_pos0_trial01.log:4-8`

> `bias guard threshold: effectiveness ratio <= 10`
>
> `composition: positron_ratio=0, n_i=Ne(fluid), n_lep=(1+2*positron_ratio)*n_i`
>
> `using mixed tp_over_te with trat_small = 1, trat_large = 80, sigma_transition = 1, constant_beta_e0 = 0.1, constant_beta_e0_exponent = 1, jet_sigma_cut = -1, jet_beta_cut = -1, jet_thetae = 0, jet_ne_mult = 1`

The same log shows repeated invalid-frequency events during bias fitting:

`igrmonty/logs/MAD_RBETAwJET_a+0.94_t4000_rh80_pos0_trial01.log:23-47`

> `isnan nu: track_super_photon ...`
>
> `ratio = 23.6171`
>
> `tuning time 4763s`

and still shows invalid-frequency events thousands of lines later:

`igrmonty/logs/MAD_RBETAwJET_a+0.94_t4000_rh80_pos0_trial01.log:5943-5969`

> repeated `isnan nu: track_super_photon ...` blocks

This is not a harmless print. `src/track_super_photon.c:404-417` shows that after this diagnostic, the production code sets `ph->w = 0` and drops the photon.

For this representative log I counted:

- `1308` occurrences of `isnan nu`

By contrast, the clean completed `CRITBETA` logs had `0`.

There is another important red flag in the same family of runs. `src/compton.c:267-273` contains:

```c
if (sample_cnt > 10000000) {
    fprintf(stderr, "in sample_electron ...\n");
    // This is a kluge to prevent stalling for large values of \Theta_e
    Thetae *= 0.5;
    sample_cnt = 0;
}
```

So the `in sample_electron ...` lines seen in several `wJET` logs are not harmless diagnostics. They mean the code hit an extreme rejection loop and then locally halved `Thetae` as a stall-avoidance kluge. That fallback is inherited from the older codebase, not introduced by the positron patch, but it is a real science-changing behavior on the affected branches.

### B8. Scientific usability verdict for the batch outputs

Final verdict:

- `CRITBETA pos0` final outputs: **scientifically usable**
- `CRITBETA pos1 trial01` outputs: **valid intermediate tuning products, not final**
- `wJET` outputs from this batch: **not validated science products**

More precisely:

- The clean finished `CRITBETA` runs made meaningful progress and completed successfully.
- The positron trial products made meaningful progress and show the pair hooks are active.
- The `wJET` runs exposed useful diagnostics about bottlenecks and failure modes, but the batch did **not** produce final `wJET` science products I would trust without further code and workflow repair.

## C. Source-Code Change Inventory

### C1. Comparison method

I used:

```bash
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git diff --name-status 4a1b1c5..3e50e52
git diff --name-status 3e50e52..e6ede55
git diff --stat 4a1b1c5..e6ede55
```

Interpretation:

- `4a1b1c5..3e50e52` captures the major `with-jet` era changes
- `3e50e52..e6ede55` captures the later positron-era changes

The full diff from merge-base to current head is very large because it includes many committed run artifacts under `logs/`, `logs/5e4_test/`, `logs/pair_sweep_20260224/`, and `scratch_logs/`. I treat those `.par` files as archived workflow artifacts, not core source.

### C2. With-jet related source changes

#### Core physics / numerics

`model/iharm/model.c`

- adds electron modes `4` and `5`
- adds runtime parameters:
  - `sigma_transition`
  - `constant_beta_e0`
  - `constant_beta_e0_exponent`
  - `jet_sigma_cut`
  - `jet_beta_cut`
  - `jet_thetae`
  - `jet_ne_mult`
- implements `constant_beta_thetae()`
- implements `in_jet_region()`
- modifies `thetae_func()` so that:
  - modes `4/5` inherit the base `RBETA` or `CRITBETA` temperature
  - if `sigma >= sigma_transition`, an additive constant-beta temperature term is added
  - if `in_jet_region()` and `jet_thetae > 0`, `Thetae` is hard-overridden
- modifies `get_fluid_zone()` and `get_fluid_params()` so that `jet_ne_mult` can scale `Ne`
- writes the with-jet parameter metadata into HDF5 output

This is the main `with-jet` implementation.

Important behavior detail:

- `in_jet_region()` only returns true if `jet_sigma_cut > 0` or `jet_beta_cut > 0`
- your recent auto-generated `wJET` parfiles do **not** set either one
- therefore `jet_thetae` override and `jet_ne_mult` are inert in those runs
- the only active extra effect in the recent `wJET` runs was the additive high-sigma temperature supplement

That means your current `wJET` label does **not** imply an explicit funnel cut.

`src/par.c` and `src/par.h`

- add defaults and parsing for the new jet-control parameters

This is necessary plumbing and works correctly for parameters that are actually written into `.par` files.

`src/radiation.c`

- adds `DEBUG_WJET` context capture so invalid-frequency failures can print surrounding fluid state

This is diagnostic support, not new radiation physics.

`src/track_super_photon.c`

- adds `sanitize_bias()`
- adds `try_boundary_recover_nu(...)`
- adds extensive invalid-frequency diagnostics
- production path still drops the photon if invalid `nu` cannot be recovered

This file is central to the observed `isnan nu` behavior in problematic `wJET` runs.

`src/utils.c`

- adds initialization reject counters for bad state / out-of-domain / bad metric / invalid `nu`
- improves runtime summaries
- supports richer run-status reporting

This is numerics and diagnostics support.

`src/compton.c`

- adds several safety caps and fallback guards around scattering
- keeps the pre-existing `sample_electron` stall kluge

This was likely changed because the hotter branches were exposing more extreme scattering pathologies.

`src/grid.c`

- adds `X[0]=t` handling and eKS/MKS3 path adjustments

This is infrastructure / geometry support rather than jet physics proper.

`src/scatter_super_photon.c`

- small additional quality-control checks in scattering

`src/main.c`, `src/decs.h`

- improved run-status accounting
- bias guard threshold logging

These changes support the new workflow and make the batch diagnostics much clearer.

#### Workflow / infrastructure

`auto_munit_bracket.py`

- introduces the automatic `M_unit` tuning workflow
- maps `RBETAWJET -> with_electrons=4`
- maps `CRITBETAWJET -> with_electrons=5`
- measures flux from full `4pi` angular integration

Critical workflow mismatch:

- `write_par_file()` does **not** emit any of the with-jet control parameters
- therefore auto-generated `wJET` runs always use parser defaults unless you inject those parameters some other way

This is the biggest silent-failure risk I found in the `with-jet` workflow.

`run_auto_munit.slurm`

- orchestrates the row-wise automated tuning batch

`tools/viewing_cone_postprocess.py`

- adds post-hoc angle-cone processing of the theta-binned `grmonty` outputs
- explicitly documents that it is not a true IPOLE-equivalent camera treatment

#### Archived artifacts

Large numbers of added `.par` files under `logs/`, `logs/5e4_test/`, and `scratch_logs/` document experiments, but they are not source code.

### C3. Positron related source changes

`src/main.c`

- adds `positron_ratio` validation and runtime logging
- prints the core convention:
  - `n_i = Ne(fluid)`
  - `n_lep = (1 + 2*positron_ratio) * n_i`
- warns downstream tools not to pre-scale `M_unit`

`src/par.c` and `src/par.h`

- add `positron_ratio`
- accept alias `positronRatio`
- add a `run_tests` switch

`src/radiation.c`

- `alpha_inv_scatt()` scales with total leptons `Ne * (1 + 2 f_pos)`
- `alpha_inv_abs()` scales synchrotron absorptivity with total radiating leptons

`src/jnu_mixed.c`

- defines:
  - total lepton density
  - electron-minus density
  - positron-plus density
- scales thermal / kappa / power-law synchrotron emissivities with total radiating leptons
- scales integrated synchrotron emissivities the same way
- adds a minimal pair-aware thermal bremsstrahlung extension

`src/tests.c`

- adds `test_pair_scalings()`
- checks:
  - scattering opacity ratio is `3` between `f_pos=0` and `f_pos=1`
  - synchrotron emissivity increases with pairs
  - brems emissivity increases in a `B=0` test

`model/iharm/model.c`, `model/riaf/model.c`, `model/sphere/model.c`

- write positron metadata into HDF5 output
- `iharm` writes both `positron_ratio` and `positronRatio` for compatibility

`auto_munit_bracket.py`

- resolves positron branches from CSV or CLI
- writes `positron_ratio` into parfiles
- warns not to pre-scale baryonic `M_unit`

`run_auto_munit.slurm`

- expands positron branches in the batch launcher

`README`, `template.par`

- document the pair convention

### C4. Unrelated infrastructure / operational changes

Notable non-physics items include:

- `makefile`
- `batch.slurm`
- `run_spectra.sh`
- `outputparser.py`
- `tune_munit_once.py`
- many archived `.par` files
- `scatter_log.txt`

These support compilation, testing, or workflow bookkeeping rather than changing the main radiation model.

## D. Physics Implementation Notes

### D1. With-jet implementation

#### Plain-language description

Your current `with-jet` implementation is best described as:

- take the base `RBETA` or `CRITBETA` electron temperature model
- optionally add an extra high-sigma temperature component
- optionally define a separate "jet region" using sigma or beta cuts
- optionally override `Thetae` or multiply `Ne` inside that jet region

In the recent auto-generated `wJET` runs, only the first two parts were active. The explicit jet-region override logic was not actually turned on, because the cut parameters were left at their inactive defaults.

#### Code-level description

Key logic lives in `model/iharm/model.c`:

- `constant_beta_thetae()` at `462-511`
  - computes an additive temperature term from magnetic energy density and density
- `in_jet_region()` at `513-545`
  - returns true if `sigma >= jet_sigma_cut` or `beta <= jet_beta_cut`
  - but only when those thresholds are positive
- `thetae_func()` at `547-689`
  - computes the base electron model
  - then either:
    - overrides `Thetae` with `jet_thetae` in the jet region, or
    - adds the constant-beta term in high-sigma regions
- `get_fluid_zone()` and `get_fluid_params()` at `764-777` and `932-946`
  - multiply `Ne` by `jet_ne_mult` inside `in_jet_region()`

#### Was it active in the recent batch

Yes, but only on the `wJET` rows.

Evidence:

- `auto_munit_bracket.py:56-61` maps `RBETAWJET -> 4` and `CRITBETAWJET -> 5`
- representative `wJET` log `MAD_RBETAwJET_a+0.94_t4000_rh80_pos0_trial01.log:7` prints the full mode-4 configuration

However:

- the only final usable products from the latest batch were plain `CRITBETA` runs (`with_electrons=3`)
- so the **usable final outputs from the batch are not with-jet outputs**

#### Internal consistency and silent-failure risks

The implementation itself is coherent, but I see four caveats:

1. `wJET` in the auto-tuner currently means "mode 4 or 5 with defaults", not necessarily "explicit jet/funnel cut".
2. `sigma_transition` heating is triggered by high sigma anywhere, not by a geometric jet mask. That can heat any high-sigma region, not just the funnel.
3. `jet_thetae` and `jet_ne_mult` silently do nothing unless `jet_sigma_cut` or `jet_beta_cut` are explicitly set.
4. Because `write_par_file()` omits the jet-control parameters, future collaborators could think a `wJET` run used tuned funnel cuts when it actually used parser defaults.

Recommended fix:

- extend `auto_munit_bracket.py` so it can read and write `sigma_transition`, `constant_beta_e0`, `constant_beta_e0_exponent`, `jet_sigma_cut`, `jet_beta_cut`, `jet_thetae`, and `jet_ne_mult`
- record those values in the tuning history CSV
- consider renaming the current default mode from `wJET` to something more literal if you keep the defaults-only behavior

### D2. Positron implementation

#### Plain-language description

The code treats the fluid-provided `Ne` as an ion-associated baseline density `n_i`. A positron fraction `f_pos` then increases the total radiating and scattering lepton density to:

`n_lep = (1 + 2 f_pos) n_i`

This affects:

- synchrotron emissivity
- synchrotron absorption
- Compton scattering opacity
- thermal bremsstrahlung normalization

The fluid-side mass scale `M_unit` remains baryonic. That is why the wrapper correctly warns not to pre-scale `M_unit` externally when `positron_ratio > 0`.

#### Code-level description

Main hooks:

- `src/main.c:105-123`
  - validates `positron_ratio`
  - prints the density convention
- `src/radiation.c:163-176`
  - `alpha_inv_scatt()` scales with total leptons
- `src/radiation.c:184-291`
  - `alpha_inv_abs()` uses total radiating leptons in synchrotron absorption
- `src/jnu_mixed.c:47-66`
  - defines total / negative / positive lepton densities
- `src/jnu_mixed.c:247-459`
  - scales thermal, power-law, kappa, and integrated synchrotron emission with total leptons
- `src/jnu_mixed.c:171-245`
  - extends thermal bremsstrahlung with:
    - `e-i` term proportional to `n_i * n_lep`
    - same-sign lepton term proportional to `n_-^2 + n_+^2`

#### Was it active in the recent batch

Yes, on the `pos1` branches.

Evidence:

- `igrmonty_logs/row_712854_48_pos1.out:1-2` shows `positron_ratio=1`
- `src/main.c:115-123` prints the composition convention at runtime
- the HDF5 products `...pos1_trial01.h5` store `/params/electrons/positron_ratio = 1.0`

It was not active in the final converged `pos0` products, by design.

#### Consistency and caveats

What looks good:

- the baryonic `M_unit` convention is consistent through the wrapper, runtime log, and HDF5
- scattering and synchrotron scaling are internally consistent
- the implementation is tested in `src/tests.c`

What is still approximate:

- the pair-aware bremsstrahlung extension is minimal rather than exhaustive
- I do not see a distinct opposite-sign `e^-e^+` brems channel or pair-annihilation physics
- that matters much more for high-energy pair-dominated spectra than for the 230 GHz `M_unit` anchor

My judgment:

- for the main mm-wave `M_unit` workflow, the positron implementation looks physically and numerically self-consistent enough to use
- for high-energy pair-plasma interpretation, document the bremsstrahlung approximation explicitly

## E. Performance Analysis

### E1. Where the runtime is going

Observed from logs:

- clean final `CRITBETA` run:
  - about `36.5M` photons made
  - about `157M` scatters
  - about `462 ks` wallclock
- problematic `RBETAwJET` example:
  - about `26.5M` photons made by `886604 s`
  - about `212M` scatters
  - ratio already `8.02`
  - many invalid-frequency drops

Interpretation:

1. The dominant cost is transport with repeated fluid lookups and scattering, not output I/O.
2. The worse branches are slower because they spend more work per launched photon in Compton interactions.
3. Some `wJET` runs also waste time in pathological rejection loops:
   - `sample_electron` stall fallback
   - `sample_klein_nishina()` is explicitly commented as inefficient for large `k0` in `src/compton.c:193-208`
4. Bias fitting is not the main 20-day cost, but it is nontrivial:
   - `747 s` in a clean completed run
   - `4763 s` in the problematic `RBETAwJET` example

### E2. Concrete bottlenecks and evidence

#### High scattering depth / effectiveness ratio

This is the clearest bottleneck.

- clean final `CRITBETA` run ended at ratio `4.31`
- problematic branches reached `8-10+`
- many low-`rhigh` `wJET` runs explicitly aborted at the ratio limit `10`

This is both a speed problem and, when the guard is hit, a no-output problem.

#### Compton rejection loops

`src/compton.c:201-224` says directly:

> `This routine is inefficient; needs improvement.`

about `sample_klein_nishina()`, especially at large photon energy.

`src/compton.c:267-273` also shows the `Thetae *= 0.5` kluge inside the electron-sampling stall path.

Those two facts line up with the long `wJET` logs that print repeated `in sample_electron ...` messages.

#### Invalid-frequency photon drops

`src/track_super_photon.c:404-417` drops the photon after logging `isnan nu` if recovery fails.

That is wasted transport work and an output-integrity problem. In the clean `CRITBETA` outputs it did not trigger. In some `wJET` branches it triggered hundreds to thousands of times.

### E3. Prioritized recommendations

#### Safe immediately

1. Stop treating low-`rhigh` `wJET` branches that reliably hit `abort_bias` as full production jobs.
   - Run them first as short scouting jobs.
   - If the ratio races upward toward `10` early, kill or skip them.

2. Use low-`Ns` exploratory runs for `M_unit` bracketing, then a single high-`Ns` final run.
   - Your current final products used `Ns=1e6`.
   - For tuning only, much smaller `Ns` is acceptable if you are fitting a local response curve rather than publishing the noisy spectrum itself.

3. Start `grmonty` from the IPOLE-derived `M_unit` as the initial guess, but still do at least one validation bracket around it.
   - That is already compatible with `auto_munit_bracket.py`.

#### Safe after validation

1. Fix the `with-jet` parameter plumbing in `auto_munit_bracket.py`.
   - This is more important than any micro-optimization because it affects what run you think you are doing.

2. Add a detector-side viewing-angle tally inside `record_super_photon()`.
   - Keep the full physical propagation.
   - Restrict only the recorded products.
   - Validate the result against the current cone postprocess.

3. Replace the current fixed-`P` update with a local response fit in `log F_nu` vs `log M_unit`.
   - Two or three low-`Ns` scout points are enough to estimate the local slope.
   - Then jump directly to one final high-`Ns` run.

4. Revisit the Compton samplers.
   - A mathematically equivalent but more efficient Klein-Nishina / electron-sampling method would be a real transport speedup.
   - This is numerically delicate, so validate against current clean `CRITBETA` baselines.

#### Risky / changes interpretation

1. Restricting photon emission or transport to a viewing cone inside `grmonty`.
   - That changes the Monte Carlo problem, not just the detector tally.
   - It can miss photons that were emitted elsewhere and scattered into the line of sight.

2. Raising the bias guard limit just to keep pathological runs alive.
   - This may turn an abort into a timeout without fixing the underlying cost or reliability problem.

3. Accepting `wJET` runs that trigger repeated `isnan nu` drops or `Thetae *= 0.5` stall fallbacks as production science runs.

### E4. Best way to reduce the number of expensive `M_unit` trials

My recommended workflow:

1. Use IPOLE `M_unit` as a prior, not as an automatic replacement.
2. Run two cheap `grmonty` scout points around that value at the actual comparison setup you care about.
3. Fit a local slope:
   - `log F_nu = a + p log M_unit`
4. Predict the target `M_unit` from that fitted slope.
5. Run one high-`Ns` confirmation / final production job.

This should cut the number of expensive `grmonty` trials substantially more reliably than relying on the current fixed-`P=2` update alone.

## F. Recommendation on Viewing-Angle Implementation

### F1. Is the current post-processing approximately equivalent to IPOLE

No, not in the strict sense.

Current status:

- `grmonty` stores an azimuth-averaged, equator-folded theta histogram
- `viewing_cone_postprocess.py` applies a cone cut in that theta histogram
- IPOLE computes line-of-sight radiative transfer for a specific observer orientation and image plane

So the current cone postprocess is:

- better than comparing against the raw `4pi` average if you care about a specific observer angle
- but still only a proxy
- not equivalent to IPOLE's camera treatment

Concrete example from the clean `4000` `CRITBETA pos0` file:

- full `4pi` average at 230 GHz: `0.5112 Jy`
- `17 deg +/- 10 deg` cone, physical power only: `0.0287 Jy`
- same cone, renormalized to `4pi` equivalent: `0.3190 Jy`

That is a large shift, and it proves the current `4pi` tuning and your cone-restricted comparison are not interchangeable.

### F2. Should you add direct viewing-angle restriction inside grmonty

My recommendation is:

- **maybe, but only as a detector-side recording restriction**
- **no, if the plan is to use it as a full substitute for IPOLE matching**

Scientifically justified version:

- keep the full all-sky transport
- record only photons escaping into the desired observer cone or detector geometry
- optionally also keep the existing full-theta output for diagnostics

Scientifically unjustified shortcut:

- restricting photon creation or transport itself to the viewing angle

That shortcut would not capture photons that scatter into the viewing cone from other directions.

### F3. Would that let you use IPOLE-derived M_unit directly with one grmonty run

Usually no.

It might get you closer, but several assumptions would all have to hold:

1. the same electron model is used in both codes
2. the same pair convention is used
3. the same observer angle and effective detector geometry are used
4. the 230 GHz flux is dominated by direct synchrotron and not significantly altered by Compton / brems pathways
5. the IPOLE and `grmonty` transfer differences are small at the anchor frequency for the model family of interest

Even if viewing angle matched perfectly, IPOLE and `grmonty` would still differ in:

- transport method
- Monte Carlo sampling noise
- treatment of scattering
- treatment of bremsstrahlung
- angular averaging history in the current workflow
- possibly other implementation details in emissivity / absorption tables

So the correct answer is:

- **no, this would not by itself solve the underlying mismatch**

### F4. Best recommended workflow going forward

Recommendation:

- implement detector-side viewing-angle tally in `grmonty` if your real comparison target is IPOLE at a specific observer angle
- validate that tally against the current cone postprocess on clean completed runs
- then use IPOLE `M_unit` only as the starting guess for a very small local `grmonty` bracket
- estimate the local flux response and do one final high-`Ns` run

In short:

- adding viewing-angle support is worth doing for comparison fidelity
- it is **not** a license to skip `grmonty` validation entirely

## G. Reproducibility Appendix

### G1. Paths used

- repo: `/work/vmo703/igrmonty`
- scheduler logs: `/work/vmo703/igrmonty_logs`
- trial logs / parfiles: `/work/vmo703/igrmonty/logs`
- outputs: `/work/vmo703/igrmonty_outputs/m87`

### G2. Commands used

Representative commands:

```bash
sacct -j 712854 --format=JobID,JobName%40,Partition,AllocCPUS,Elapsed,State,ExitCode,Start,End%25
find /work/vmo703/igrmonty_outputs/m87 -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TT %12s %f\n' | sort | tail -n 30
rg -n "run status|bias guard|isnan nu|warning|zone|out of bounds|nan|inf|invalid nu|boundary" /work/vmo703/igrmonty/logs/*.log
git diff --name-status 4a1b1c5..3e50e52
git diff --name-status 3e50e52..e6ede55
git diff --stat 4a1b1c5..e6ede55
```

I also used `h5py` scripts to inspect:

- `/output/run_status`
- `/output/run_status_detail`
- `/output/dOmega`
- `/output/nuLnu`
- `/params/electrons/positron_ratio`

### G3. Environment assumptions

- current date during audit: `2026-04-06`
- timezone: `America/Chicago`
- repo branch: `positrons`
- repo head: `e6ede55854baf36ffd7e70ccd30d668ffd6e5865`

### G4. Dirty worktree note

At audit time the working tree was dirty, mainly because of log/parfile churn under `logs/` plus a modified `run_auto_munit.slurm`. I did not revert any of those files.

## Bottom-line verdict

Was the 20-day run wasted?

- **No, but only partially.**

What was salvaged:

- two clean converged `CRITBETA pos0` final spectra
- two valid positron trial outputs
- strong diagnostic evidence about why the current `wJET` production path is unreliable and expensive

What was not salvaged as final science:

- the `wJET` portion of the batch

Most important practical conclusion:

- before spending another 20-day block on `wJET`, fix the parameter plumbing so the run labels actually match the intended jet physics, and do not tune `M_unit` against `4pi` averages if your real comparison target is an IPOLE-like viewing angle.
