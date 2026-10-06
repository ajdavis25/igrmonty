# The QA-line glossary

Reference for reading `_qa/p4_qa_live.log` / `p4_production_qa.py` output (and
the run logs behind it). Running example, an actual line set:

```
== spectrum_Sa-0.5_6000_CRITBETAwJET_rh20_bc1_f0.5_pos0.h5 ==
  provenance: M_unit=8.2003e+28 bias=8.03 Ns=1e+06 paper_literal=1.0 pos=0 status=ok -> PASS
  F230: 4pi=0.4835 Jy [PASS]  cone17=0.3461 Jy (x0.72 vs 4pi)
  L_X(2-10): 4pi=2.553e+39 [PASS vs 4.4e+40]  cone17=1.438e+39  L_bol=3.716e+41  f_compton=0.296
  vs July: M_unit x1.000  F230cone x0.912  L_X x0.949
```

---

## 1. Reading the filename (the branch's full identity)

`spectrum_Sa-0.5_6000_CRITBETAwJET_rh20_bc1_f0.5_pos0.h5`

| token | meaning |
|---|---|
| `S` / `M` | **SANE** vs **MAD** — the accretion state of the GRMHD simulation. MAD = "magnetically arrested disk": magnetic field strong enough to shove the inflow around. SANE = "standard and normal evolution": weaker field, more turbulent disk. |
| `a-0.5` / `a+0.94` | Black-hole **spin**. Sign = direction: +0.94 is fast **prograde** (disk orbits the same way the hole spins), −0.5 is **retrograde** (opposite). Spin powers the jet and shapes the inner flow. |
| `4000/5000/6000` | GRMHD **dump index** — which snapshot in time of the simulation. Three dumps ≈ three independent samples of the turbulence, so results aren't a fluke of one moment. |
| `RBETA` | **R-β electron model**: T_i/T_e follows plasma β — dense disk (high β) → electrons cold (ratio → R_high); magnetized regions (low β) → electrons warm (ratio → R_low = 1). |
| `CRITBETA` | **Critical-β electron model** (Anantua+2020): electron share of the temperature = f·exp(−β/β_crit) — electrons get a fixed max share f in magnetized gas, exponentially frozen out in weakly magnetized gas. |
| `wJET` | The **jet supplement** is active on top of the base model: where σ > 2, ADD P_e = β_e0·P_B; where σ ≥ 10 or β ≤ 0.1, hard-override Θe = 50. This is the model family the 12π story lives in. |
| `rh20` / `rh80` / `rh160` | **R_high** — the cold-disk limit of T_i/T_e in the R-β formula (bigger = colder disk electrons). Kept in CRITBETA names for bookkeeping, but the crit-β temperature law itself doesn't use it. |
| `bc1` / `bc0.01` | **β_crit** — how quickly the crit-β model freezes electrons out as β rises. bc0.01 (MADs) = only the most magnetized zones keep hot electrons; bc1 (SANEs) = gentler. |
| `f0.5` | **f = beta_crit_coefficient** — the maximum electron share of the gas temperature in the crit-β model (0.5 = electrons can get at most half). |
| `pos0` / `pos1` | **positron_ratio** — pairs added per electron. pos0 = electron–ion plasma only; pos1 = one positron per electron (doubles the leptons at fixed mass). The paper's pair-content axis. |
| `_trialNN` / absent | A tuner **trial** en route to convergence vs the promoted **final** spectrum (suffix-free = converged and renamed). |

## 2. The dials (provenance line)

- **M_unit** — the mass-scale dial, in grams. GRMHD is scale-free; M_unit sets
  how much actual gas the code-density corresponds to, and thereby all
  densities and field strengths. It is the ONE free knob, tuned per branch
  until the model emits the observed 0.5 Jy at 230 GHz. Everything else the
  spectrum does after that is a prediction. (SANEs need ~10³× more M_unit
  than MADs: weaker fields need more gas for the same brightness.)
- **bias (biasTuning)** — Compton-sampling oversampling factor. The Monte
  Carlo samples scattering events bias× more often than nature, with
  1/bias-weighted photons, so rare scatterings are well measured. Statistical
  quality only — never changes the physics answer, only the noise. Fitted per
  branch (fit_bias 1) or fixed (fit_bias 0 — used where the fitter
  misbehaves; see §6).
- **Ns** — target superphoton count (10⁶ for production). The resolution of
  the Monte Carlo. NOT cost-linear on hot models (weights depend on it).
- **paper_literal** — the 12π flag: 1 means the constant-β supplement uses
  the published convention P_e = β_e0·P_B with P_B = B²/8π
  (Anantua+2020; Emami+2021 eq. 25). 0 would be the legacy as-shipped form,
  12π ≈ 38× hotter, traced to a transcription slip
  (docs/2026-09-08_12pi_provenance_verdict.md). Production = 1, always.
- **pos** — positron_ratio actually recorded inside the file (cross-check of
  the filename).
- **status** — grmonty's run label; `ok` = finished cleanly and wrote the
  spectrum (anything else means the guard or an error stopped it).
- **→ PASS/FAIL** — the *provenance gate*: all of the above are exactly what
  production demands (right flag, right Ns, clean status). Guards against
  silently-wrong-settings data entering the paper.

## 3. Brightness at the EHT band (F230 line)

- **F230** — flux density at 230 GHz (the EHT observing frequency), in
  **janskys** (1 Jy = 10⁻²³ erg s⁻¹ cm⁻² Hz⁻¹). The observed M87 core value
  is **0.5 Jy**; the tuner adjusts M_unit until the model matches it (±5%).
  Measured with the tuner's own convention (nearest frequency bin) so the QA
  gate agrees with the machinery that did the tuning.
- **4pi** — direction-averaged: total luminosity smeared equally over the
  whole sky (4π steradians), as if the source shone the same in every
  direction. The pipeline's historical tuning frame.
- **cone17** — only the light escaping toward directions within ~10° of our
  actual viewing angle (Earth sees M87 ~17° off the jet axis), renormalized
  to 4π-equivalent so the two numbers are directly comparable.
- **(×0.72 vs 4pi)** — the **frame gap**: this branch beams mildly (we see
  72% of the sky average at 230 GHz). Some branches are far more anisotropic
  (×0.15–0.4) — which is why the "tune at 4π or tune in the observer frame?"
  question matters.

## 4. The prediction side (L_X line)

- **L_X(2–10)** — X-ray luminosity integrated over 2–10 keV, in erg/s. The
  model's *prediction* once the mm anchor is set.
- **[PASS vs 4.4e+40]** — the **2017 core X-ray gate**: Chandra+NuSTAR
  measured ≈4.4×10⁴⁰ erg/s from the core during the EHT 2017 campaign. A
  correctly-anchored model must not exceed what was actually seen.
  "Overproduces the X-ray" = predicted L_X above this → the model
  configuration is in tension with data (that's a result, not a bug — it is
  how models get excluded). Current example: MAD wJET branches fail at 4π;
  prograde ones pass in the 17° cone; retrograde fail in both frames.
- **cone17** — same X-ray band, observer-frame version.
- **L_bol** — bolometric luminosity: everything, all frequencies, 4π. For
  scale, M87's Eddington luminosity is ≈8×10⁴⁷ erg/s, so 3.7×10⁴¹ is
  ~5×10⁻⁷ L_Edd — deeply sub-Eddington, as M87 should be.
- **f_compton** — fraction of the emitted energy carried by photons that
  Compton-scattered at least once (here ~30%). High f_compton (≳0.5) means
  the spectrum is scattering-dominated — the regime where the hot sheath and
  the 12π question bite hardest.

## 5. The continuity check (vs July line)

Ratios of this run against the archived July 2026 spectrum of the *same*
branch (`pre_bugfix_2026-07/`):

- **M_unit ×1.000** — the July value still hits 0.5 Jy under the rebuilt
  code: the warm start converged on its first trial.
- **F230cone / L_X ×0.91–0.95** — within ~10% of July = Monte-Carlo noise +
  small code corrections. Big systematic deltas are flagged instead (e.g.
  Sa+0.94_4000's L_X ×5–10 from the deep-KN Compton sampling fix —
  documented physics correction, not an error). `×nan` = no July
  counterpart exists (the new MAD wJET branches).

## 6. Run-mechanics terms in the logs behind the QA line

- **superphoton** — one Monte-Carlo packet standing in for a huge number of
  real photons; its **weight** is how many. Weights below **WEIGHT_MIN**
  (hardcoded 10²⁸) drown in the weight table — the mechanism behind the
  "zero photons generated" failures.
- **scatter ratio (effectiveness ratio)** — N_scattered / N_made during a
  run. Production statistics target ~1; this code sits ~10 on MADs at
  moderate bias. The **bias guard** (`bias_abort_ratio`) kills a run mid-MC
  if the ratio exceeds its limit — now set to 100 after limit-5 discarded
  finished runs and triggered the launcher's ×5 M_unit **backoff spiral**.
- **fit_bias / the fitter** — grmonty's internal search for a good bias
  before the main run. Known open bug: on MAD CRITBETA (both spins) and MAD
  wJET a+0.94 it measures ratio = 0 on configs that run perfectly at fixed
  bias — those branches run `fit_bias 0, bias 0.05` (July's fitted value).
- **warm start** — starting the M_unit tuning from the July-tuned value
  (stored per branch in `data/final_grmonty_paper.csv`), so convergence
  takes 1–2 trials instead of a cold hunt.
- **crit_floor = 3×10⁻²** — the minimum Θe in Crit-β models; PART OF THE
  MODEL (agenda decision 6): ~45% of zones sit at it and it dominates
  Crit-β photon generation (docs/2026-09-25_critbeta_floor_decision.md).
- **Θe (theta-e)** — electron temperature as a pure number,
  kT_e/(m_e c²): 1 ≈ 6×10⁹ K. **σ (sigma)** — magnetic energy per unit
  rest-mass energy, b²/ρ: jet interior σ ≫ 1, disk σ ≪ 1. **β (beta)** —
  gas pressure over magnetic pressure: disk β ≫ 1, jet β ≪ 1. These three
  fields decide which electron-temperature rule applies where.
