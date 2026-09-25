# Zoom agenda — funnel emission & the Θe cap in the wJET model

Prepared 2026-08-06 · for call with Richard (+ postdoc, cc'd) · target ≤ 45 min
Background threads: my two emails to Richard (jet supplement, Θe cap) + postdoc
exchange (numerical-trust framing, entropy inversions, two-temperature check).

## 30-second context

M87 models: EHT-library GRMHD (MAD/SANE, a = +0.94/−0.5) post-processed with
grmonty/ipole; base electrons = R-β (R_high 80–160) / Crit-β; **jet supplement**
where σ > 2 sets P_e = β_e0·P_B (β_e0 = 0.1); hard override Θe = 50 where σ ≥ 10
or β ≤ 0.1; everything clamped at Θe = 10³. Each model retuned to 0.5 Jy @ 230 GHz.

## Facts on the table (all verified on disk this week)

- **The supplement is Θe ≈ 3.5×10³·σ as coded** (exponent = 1; M_unit-independent).
  It crosses the 10³ cap at σ ≈ 0.3 ⇒ **saturated everywhere it activates
  (σ ≥ 2), on every dump, by construction.** Effective model as run:
  Θe = 10³ sheath (2 ≤ σ < 10, β > 0.1) + Θe = 50 jet.
- **⚠ 12π convention question (resolve first — it reframes everything below):**
  both codes (grmonty port ≡ ipole fork, ipole model/iharm/model.c:547) compute
  Θe = β_e0·B_cgs²/(2(γe−1))/(n_e m_e c²). The papers define:
  Anantua+2020 (MNRAS 493, 1404): "β_e = P_e/P_B = (γe−1)u_e/(b²/2) = β_e0",
  i.e. P_B = b²/2 in *code/HL units*; Emami+2021 (ApJ 923, 272) Eq. 25:
  "P_B = B²/8π" in *CGS*. Same physical P_B. The code applies the HL form b²/2
  to the Gaussian B (×4π hot) and divides by (γe−1) — using electron *energy
  density* where *pressure* belongs (×3 hot): total **12π ≈ 38× the published
  Θe**. Paper-literal (implemented 08-07 behind `constant_beta_paper_literal=1`,
  default off; C ≡ numpy to 7×10⁻¹²) ⇒ sheath Θe = 270–1000 (92σ supplement
  + R-β base), **cap binds 11.3% of sheath zones (hot σ→10 edge) instead of
  100%**, sheath emissivity proxy ×3.2 lower at fixed M_unit, T_e ≈ virial.
  As-coded ⇒ everything in this agenda. **Provenance verdict 09-08: transcription
  slip in a post-publication port — settled without author input** (line born
  2021-05-24 in ARRicarte/ipole, months AFTER both papers; Emami+2021 used
  GRTRANS, not ipole; the line contradicts its own "//ARR: See definition in
  Anantua et al. (2020)" citation AND its birth comment; no published result
  uses the hot form; upstream frozen since 2021-08-26). Full chain:
  docs/2026-09-08_12pi_provenance_verdict.md. **Fixed-M_unit A/B measured
  (09-15, scratch/pb_fix_ab/): the 12π lives almost entirely in the Compton
  hump — F230 essentially unchanged (×0.97 4π-avg, ×1.10 17° cone; the mm
  M_unit anchor is insensitive to the choice), direct synchrotron ×3.3
  (matches the ×3.2 proxy), L_bol ×7.2, 2–10 keV ×9.2, Compton fraction
  0.83→0.64. Both variants imply the same retune (M_unit ×0.55 from the
  wJET-untuned start), so the ×9 X-ray gap survives retuning to first order.
  Fig: _qa/plots/zone_breakdown/pbfix_sed_ab.png; numbers:
  scratch/pb_fix_ab/sed_ab_summary.csv.**
- **Postdoc P.S. (agreed):** Θe = 10⁴ ⇔ T_e ≈ 6×10¹³ K — unrealistic, and above
  the single-fluid GRMHD gas temperature anywhere in the domain (virial ceiling
  ≈ few×10² in electron Θe units).
- **MAD (Ma+0.94_4000, rh80 test):** sheath band = 0.5% of zones but ~98% of the
  emissivity proxy; pre-clamp Θe median 1.1×10⁴ (p95 2.9×10⁴). Funnel interior
  (σ ≥ 10) sits at Θe = 50 → only ~1.5%.
- **SANE (Sa−0.5_4000, rh160 production):** supplement nearly inactive (0.8%);
  the **β ≤ 0.1 override paints broad mid-latitude material** far outside the
  funnel and carries ~94% of the proxy.
- **The 10³ cap is NOT a table bound:** hotcross Compton table extends to
  Θe = 10⁴ (src/hotcross.c:26–27); the 10³ ceilings are code choices
  (THETAE_HARD_MAX, model/iharm/model.c:9; SCATTERING_THETAE_MAX, src/decs.h:26).
  Soft constraints above 10³ = emissivity-fit validity + sampling + cost.
- **SED gate exists and passes today:** all 48 tuned (pre-fix) runs sit below the
  2017 core X-ray (worst = 0.22× of 4.4×10⁴⁰ erg/s, Chandra+NuSTAR); but tuning
  is 4π-averaged while a 17° observer sees 1.3–25× less at 230 GHz
  (worst: SANE +0.94 wJET). pos1 runs are ×3.3 the X-ray of pos0.
- **Physics tension to discuss:** Θe = 7×10³–3×10⁴ ⇒ T_e super-virial
  (virial-ion Θe-equivalent ≈ 600 at 3 r_g) — inverts the usual T_e ≲ T_i jet
  hierarchy. t_syn(Θe = 10³, 10 G) ≈ 0.08 r_g/c ⇒ equilibrium set by
  heating–cooling balance (reconnection/dissipation vs synchrotron+IC); no
  first-principles ceiling. Postdoc: entropy-inversion fixups in the funnel break
  energy conservation (our dump samples carry no fail flags — caveat, not number).

## Five decisions

1. **σ trust boundary.** Emit from sheath only (σ-cut at ~5–10), full funnel
   with caveats (status quo), or exclude fixup/floor zones if flags obtainable?
   (Precedent: Ryan+2018, ApJ 864, 126, §3.3 — they *forbid* radiation
   interactions wherever b²/ρ > 1 because "harm-like total energy codes cannot
   accurately represent even total fluid thermodynamics" there.)
2. **Cap justification.** Keep Θe_max = 10³ as an explicit *regularizer* with
   data-gate language (not "table limit"). Any appetite for testing 10⁴ (table
   allows it; fits/cost don't obviously)?
3. **The 12π resolution (absorbs the old "rescale e0" question).**
   ~~Adjudicate intent~~ → **settled by provenance 09-08 (see
   docs/2026-09-08_12pi_provenance_verdict.md): slip in a post-publication
   port; no published result depends on the hot form; production 48-run grid
   unaffected (MADs had no wJET, SANE supplements ~extinct).** Remaining
   decision is sign-off only: adopt paper-literal (sheath 270–1000 with real
   σ-dependence; cap a boundary effect at 11%; ≈ e0 × 1/12π ≈ 2.7×10⁻³) for
   the go-forward MAD wJET configs, methods cite Anantua+2020/Emami+2021
   Eq. 25, fork discrepancy noted in appendix. Same one-line gate goes into
   the ipole copy before P4.3 imaging.
4. **β_cut scope.** β ≤ 0.1 alone qualifying zones as "jet" dominates SANE
   emission with non-funnel material. Tighten to (σ AND β) or lower β_cut —
   keeping the β arm in some form is already decided (ashton, 08-07); dropping
   it entirely is off the table.
5. **Paper framing + Phase-4 plan.** Energy-conservation caveat wording; propose
   small scan ({Θe_max or β_e0} × MAD/SANE) with acceptance = 17° SED gates
   (X-ray 4.4×10⁴⁰, NIR ~mJy) — pipeline is push-button ready.

6. **Crit-β Θe floor = 3×10⁻² is part of the model (decided: ashton,
   09-25).** Finding M1's 1×10⁻³ (ipole-matching) made Crit-β unable to
   generate photons at grid M_units and killed 24 Phase-4 tasks; reverted,
   documented in methods, full chain in
   docs/2026-09-25_critbeta_floor_decision.md. Group item: harmonize by
   raising IPOLE's floor for Crit-β models before P4.3 imaging.

## If time permits

- ipole positron polarization: jV/aV/rV carry 1/(1+f) — looks (1+f) low; needed
  before any polarized pos1 images (question for Richard/Angelo).
- Tuning frame: switch M_unit target from 4π-average to the 17° cone?

## Figures to screen-share

1. `igrmonty_outputs/m87/_qa/plots/zone_breakdown/zones_Map0_94_4000_RBETAwJET_rh80.png`
2. `…/zone_breakdown/zones_Sa-0_5_4000_RBETAwJET_rh160.png`
3. `…/mwl_sed/fig1_lx_gate.png`  4. `…/mwl_sed/fig3_sed_families.png`
(Backup: `…/mwl_sed/fig2_f230_frame_gap.png`; numbers: `_qa/zone_breakdown_summary.csv`,
`_qa/mwl_sed_verdicts.csv`; method notes: `_qa/MWL_SED_CHECK.md`.)
