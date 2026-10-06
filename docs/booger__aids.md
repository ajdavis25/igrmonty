# Phase-4 production campaign — final writeup (meeting + paper material)

2026-10-05 · ashton · status: **campaign complete — 72/72 branches production-ready**
(~10 months past the soft deadline, but every number the paper needs now exists,
audited, with provenance on disk.)

---

## 1. The bottom line

- **All 72 production branches are tuned, audited, and final** (6 model families ×
  2 spins × 3 dumps × pair content 0/1), each calibrated to the EHT anchor
  (0.5 Jy @ 230 GHz) and checked against the 2017 Chandra+NuSTAR core X-ray
  limit (4.4×10⁴⁰ erg/s) in both the sky-average and 17° observer frames.
- **Headline result: every X-ray gate failure in the campaign is caused by the
  wJET supplement on MADs** — attributable *by design* (completed 2×3 factorial),
  not by inference. Base R-β and Crit-β pass everywhere; SANE passes everywhere.
- **The constant-β_e convention matters an order of magnitude in X-ray and not
  at all at 230 GHz**: the legacy 12π-hot form (traced to a 2021 transcription
  slip, settled by provenance without author input) would have put every MAD
  wJET model ~9× higher still — instantly excluded. Paper-literal makes the
  model class *constrainable*.
- **Imaging is now consistent with the spectra for the first time**: the local
  ipole carries the same two gates (paper-literal P_B, Crit-β floor), 264
  production frames + 24 movies exist in the group's house style.

## 2. What is production-ready, and where

| artifact | where | count |
|---|---|---|
| Final tuned spectra | `igrmonty_outputs/m87/run_2026-09-15/` | 72 |
| Pass/fail table (live) | `_qa/p4_passfail.md` (+ CSV) | 72 rows |
| QA metrics per branch | `_qa/p4_qa_run_2026-09-15.csv` | 72 |
| 2×3 factorial | `_qa/p4_factorial.py` (rerun any time) | 6 cells, n=12 each |
| Confirm-run overrides | `_qa/p4_confirm_notes.md` | 1 (see §4) |
| Per-branch SEDs | `_qa/plots/seds_p4/` | 72 |
| Pair comparisons (pos0 vs pos1) | `_qa/plots/pairs_p4/` | 36 |
| Campaign figures | `_qa/plots/mwl_sed/fig1–3` (July era archived in `mwl_sed_prebugfix_jul/`) | 3 |
| ipole frames (gated, spectra-consistent) | `ipole_outputs/M87/frames_p4/` | 264 |
| Production movies (11-frame, house style) | `ipole_outputs/M87/movies/{MAD,SANE}/` | 24 |

Provenance: every spectrum h5 carries its settings (`paper_literal=1` on wJET,
Ns=10⁶, positron ratio, tuned M_unit); binary githash 388a18d (clean); freeze
commits 65835d9 → 46c1118 → 388a18d (igrmonty), f6efd0a/7515c2d (outer).

## 3. The physics verdicts (what goes in the paper)

**Final gate tally: 55 PASS / 17 FAIL of 72** (raw table says 54/18; one SANE
"failure" was refuted by a dedicated confirm run — §4).

The completed 2×3 factorial — median L_X(2–10 keV, 4π) / 4.4×10⁴⁰, n=12 per cell:

| base \ config | MAD plain | MAD + wJET | SANE + wJET |
|---|---|---|---|
| R-β | PASS (0.098×) | **FAIL 12/12 (3.1×)** | PASS (0.098×) |
| Crit-β | PASS (0.0036×) | **FAIL 5/12 (0.89×)** | PASS (0.27×) |

1. **The jet supplement on MADs overproduces the 2017 core X-ray** once
   calibrated at 230 GHz: MAD R-β wJET by 1.3–5.7× (4π). Both plain columns
   pass 24/24 — the base electron model is not the driver; the supplement's
   P_e = β_e0·P_B injection is, and it only gets loud where MAD-strength
   fields fill σ>2 volume (SANE σ>2 regions are nearly empty: the SANE wJET
   "pass" is largely the supplement never firing).
2. **Frame dependence decides prograde vs retrograde** (all "×" here =
   multiples of the 4.4×10⁴⁰ erg/s X-ray limit; same runs, different angular
   bins): in the 17° observer frame, prograde MAD R-β wJET drops from
   1.3–5.7× the limit (4π) to **0.3–1.2× the limit** (5/6 under), but
   **retrograde stays 1.6–3.3× over the limit in both frames** → a−0.5 MAD wJET at
   β_e0=0.1/rh80 is X-ray-excluded regardless of frame; a+0.94 survives only
   if the observer frame is the standard (fig1: "7 of 72 exceed in the 17°
   frame"). This gives the long-parked tuning-frame question real stakes.
   Caveat to state: the in-frame pass mixes conventions (mm calibrated at 4π,
   X-ray judged at 17°); a fully observer-frame treatment would retune M_unit
   upward and could pull prograde back over.
3. **Base-model interaction**: at fixed calibration the supplement is ~3.5×
   X-ray-louder on an R-β base than a Crit-β base (3.1× vs 0.89× medians) —
   the cold Crit-β disk forces a different mass normalization that
   repartitions the anchor. (The pre-registered prediction was "Crit-β wJET
   fails like R-β wJET"; the design *refined* it instead of confirming it.)
4. **Positrons are a gate-edge discriminator**: pair loading (pos1) raises
   L_X ≈ 2× campaign-wide; inside the marginal MAD Crit-β wJET cell every
   pos0 passes (0.73–0.89×) and every pos1 fails (1.1–1.5×). The project's
   original positron axis becomes observationally testable exactly where the
   supplement puts models near the X-ray line. (Statistical caveat in §4.)
5. **β_e0 pressure**: the simplest reading of (1) is that the X-ray limit
   pushes the supplement's β_e0 down from the group's 0.1 toward ~10⁻² —
   independently converging on Emami+2021's published best-bet value. A small
   β_e0 scan (agenda decision 5) is the natural follow-up and the pipeline is
   push-button for it.
6. **The 12π convention** (methods + appendix): the shipped constant-β_e form
   was 12π ≈ 37.7× hotter than the published P_e = β_e0·B²/8π (Anantua+2020;
   Emami+2021 eq. 25). Provenance settled it as a transcription slip in a
   2021 post-publication port — no published result ever used the hot form
   (full chain: `docs/2026-09-08_12pi_provenance_verdict.md`). Measured
   impact at fixed M_unit (A/B, scratch/pb_fix_ab): **F230 ×0.97** (the mm
   anchor cannot tell the difference), direct synchrotron ×3.3, L_bol ×7.2,
   **L_X ×9.2**, Compton fraction 0.83→0.64. Production = paper-literal.
7. **The Crit-β floor is part of the model**: Θe ≥ 3×10⁻² (state it in
   methods beside β_crit and f). At the ipole-matching 1×10⁻³ the model
   generates no photons at grid M_units (bisect table:
   `docs/2026-09-25_critbeta_floor_decision.md`). ipole harmonized upward,
   not grmonty down.

## 4. Failures — both kinds

**Physics failures (= results, intended):** the 17 X-ray gate exceedances in
§3. These are the gate machinery working; they exclude/constrain models.

**One refuted failure:** SANE Crit-β wJET d5000 pos1 initially read 1.52× over
— a dedicated confirm run (10× Compton sampling, independent seed, 33 min)
measured **0.44× = PASS**. The campaign value was low-bias Monte-Carlo noise
(one heavy packet). Recorded in `_qa/p4_confirm_notes.md`; the production
record stays raw, the override is documented.
**Open statistical caveat:** the five MAD Crit-β wJET pos1 fails (1.1–1.5×)
also ran bias 0.05. Their Compton sampling is far healthier (f_compton
0.33–0.48), but the same ~30-min confirm treatment is recommended before the
paper leans on that cell. *Not yet run.*

**Operational failures (all diagnosed, fixed, and documented — good rigor
story for the meeting):**

| failure | root cause | fix |
|---|---|---|
| 24 Crit-β tasks died in ~3 min (exit 41) | upstream "M1" change set the Crit-β floor to 1e-3 → zero photon generation (silently "ok" at fixed bias) | floor reverted to 3e-2 and promoted to a model parameter; bisect-proven |
| 11+ MAD tasks burned 66-h budgets | bias guard limit 5 tripped on healthy ratio~10 runs, triggering a ×5 M_unit backoff spiral | guard limit 100 (A/B-calibrated); backoff never triggers on healthy runs |
| 60-h single-trial timeouts (several waves) | deep-KN Compton fix made MAD trials ~100+ core-h; runaway cascades at fitter-chosen bias (ratio 49–93 observed) | multicore tasks (8–16 cores) + fixed bias |
| grmonty's internal bias fitter reports ratio=0 on configs that run fine at fixed bias | **unresolved code bug**; confirmed scope: all MAD Crit-β + MAD wJET a+0.94 | production standard = fit_bias 0, bias 0.05 (July's fitted value); fitter bug on the group debt list |
| the campaign's first binary silently ignored the paper-literal flag | grmonty ignores unknown par keys; the tuner's default binary predated the flag | rebuilt + smoke-tested binary; launcher refuses to run without it |
| scatter-counter int32 overflow (−2×10⁹ printed; guard disarmed) | N_scatt is 32-bit; a 2.3×10⁹-scattering run wrapped it | cosmetic for spectra (bins are doubles); one-line `long long` fix recommended |

Closing stat for the fitter saga: the final pair of branches spent 60+ hours
dying under fitter-chosen bias and finished in **30 minutes** at fixed bias on
16 cores.

## 5. Imaging status (for the figures/movies slide)

- Local ipole now carries both gates, parameter-gated and default-off:
  **bit-identical** to the old binary with flags off (Ftot 0.0751926 ≡),
  paper-literal engaging (+45% F230 at fixed M_unit — same direction as the
  spectral A/B), floor engaging (negligible at 230 GHz, as expected).
- 264 frames (24 branches × 11 dumps, Δ200 M), rendered with the *spectra's*
  conventions: MBH 6.5×10⁹, D = 16.8 Mpc (deliberate change from the old
  image set's 6.2/16.9), θcam 163°, 230 GHz, tuned M_units ln-interpolated
  between the three anchors.
- 24 production movies (house two-panel style: Stokes I + EVPA ticks | CP),
  `ipole_outputs/M87/movies/{MAD,SANE}/<MODEL>/`, 11 frames each; the old
  3-frame flipbooks are preserved in `movies/legacy_flipbooks/`.
- Honest caveats: Δ200 M between frames is brisk (smooth movies need
  fine-cadence dumps from the group archive); ipole has no equivalent of the
  grmonty jet Θe=50 override (σ>10 is cut; ~1.5% of MAD emission); the gates
  exist only in OUR local ipole — upstream adoption is a group item.

## 6. Open items / decisions for the group

1. **Tuning frame** (now has teeth): 4π vs 17°-cone calibration decides the
   prograde MAD wJET verdict. Paper must pick and state one.
2. **β_e0 scan** (agenda decision 5): X-rays point at ~10⁻²; a {β_e0} ×
   {MAD} mini-scan is push-button.
3. **MAD Crit-β wJET pos1 confirm runs** (~6 × 30 min) before quoting that
   cell's pair effect.
4. **12π + floor upstreaming**: both one-line gates should land in the
   group's ipole fork (ours has them); FYI material is fully written
   (`2026-09-08_12pi_provenance_verdict.md`, `2026-09-25_critbeta_floor_decision.md`).
5. **σ trust boundary wording** (Ryan+2018 §3.3 precedent) and the
   energy-conservation caveat for the methods section.
6. Fitter bug + N_scatt overflow: group code-debt list.
7. Housekeeping: the ipole gate edits are uncommitted in the local clone;
   recommend a freeze commit before imaging-era work continues.

## 7. One-paragraph version (for the paper)

Calibrated to the 2017 EHT 230 GHz core flux, the full 72-branch model grid
(MAD/SANE × spin × R-β/Crit-β × jet supplement × pair content) was
spectrally post-processed with a provenance-audited Monte-Carlo pipeline and
confronted with the 2017 Chandra+NuSTAR core X-ray limit. Every exceedance
of the limit traces, by factorial design, to the constant-β_e jet supplement
operating on MAD-strength fields: retrograde MAD wJET models are excluded in
any viewing frame, prograde ones survive only in the observer frame, and the
limit drives the supplement's electron-pressure fraction from the fiducial
β_e0 = 0.1 toward the ~10⁻² value independently preferred by Emami et al.
(2021). The X-ray band — invisible to mm-only calibration, which cannot even
distinguish electron-pressure conventions that differ there by an order of
magnitude — emerges as the discriminating observable for jet electron
thermodynamics, and, at the gate boundary, for positron content.
