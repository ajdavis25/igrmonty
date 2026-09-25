# Where things stand after Brandon's email — the long, plain-language version

Written 2026-08-07. Companion to `2026-08-06_zoom_agenda_thetae_cap.md` (the
one-page version for the call). This is the same material at walking pace:
what Brandon raised, what got settled, what is still open, and why each piece
matters. Jargon is unpacked as it appears.

---

## The vocabulary

- **Θe ("theta-e")** — electron temperature written as a pure number:
  Θe = kT_e / (m_e c²), i.e. "thermal energy per electron divided by the
  electron's rest-mass energy." Θe = 1 is about 5.9 billion K. So
  Θe = 1000 ≈ 5.9×10¹² K, and Θe = 10⁴ ≈ 5.9×10¹³ K — Brandon's "6×10¹³ K."
- **σ ("sigma", magnetization)** — magnetic energy density divided by the
  rest-mass energy density of the matter (b²/ρ). Large σ means the magnetic
  field dominates and there is almost no matter. The funnel has large σ; the
  disk has small σ.
- **β ("beta", plasma beta)** — gas pressure divided by magnetic pressure.
  Small β also means "magnetically dominated," but measured with pressure
  instead of rest mass. The code uses BOTH σ and β as tags for "this zone is
  jet material."
- **β_e0** — the model's main knob: electron pressure is set to a fixed
  fraction of magnetic pressure. We use β_e0 = 0.1, i.e. electrons carry 10%
  of the magnetic pressure.
- **Funnel** — the nearly empty, magnetically dominated column along the black
  hole's spin axis where the jet lives. The GRMHD code cannot evolve a true
  vacuum, so it injects artificial "floor" density there to stay stable.
  That floor density is fake — a numerical crutch, not physics.
- **Sheath** — the boundary layer between the funnel and the disk wind
  (σ between about 2 and 10 in our tagging). Mildly relativistic. Radio
  observations of the M87 jet base (limb-brightening) say this layer, not the
  fast spine, is what actually shines there.
- **MAD / SANE** — the two accretion states we simulate. MAD = strong,
  ordered magnetic field ("magnetically arrested disk"); SANE = weaker,
  more tangled field.
- **Post-processing** — grmonty/ipole compute radiation from saved snapshots
  of a finished GRMHD run. The radiation can't push back on the gas, and any
  garbage in the snapshot (floor density, failed-solve zones) flows straight
  into the predicted light.
- **M_unit** — the knob that converts the simulation's unitless density into
  grams. Each model's M_unit is tuned until the model emits 0.5 Jy at
  230 GHz (the EHT-era compact flux), so any change to electron temperatures
  also moves M_unit when we retune, which reshapes the whole spectrum.
- **Entropy inversion / fixup** — when the GRMHD solver fails to recover
  sensible gas values from its conserved variables (which happens mostly in
  the funnel), it falls back to a repair step that keeps the run alive but
  abandons energy conservation in that zone.
- **Virial temperature** — roughly the hottest the gas can honestly be: the
  temperature you get if all the gravitational energy of infalling matter is
  turned into heat. A hard sanity ceiling. Near the black hole (3
  gravitational radii) it works out to about Θe ≈ 600 in electron units.
- **Gaussian vs. code units for B** — two bookkeeping conventions for the
  magnetic field. In textbook CGS ("Gaussian") units, magnetic pressure is
  B²/8π. Simulation codes use "Heaviside–Lorentz" units, which hide the 4π
  inside the definition of the field, so the same physical pressure is
  written b²/2. Both are correct on their own; mixing them in one formula
  silently multiplies the answer by 4π.

## One-paragraph refresher on our model

The base electron temperature comes from the standard EHT recipe
(R_high/R_low, or Critical-β for those branches), applied to single-fluid
GRMHD. On top of that sits the jet supplement: wherever σ ≥ 2, electrons are
instead given pressure P_e = β_e0 × (magnetic pressure); wherever σ ≥ 10 OR
β ≤ 0.1, a hard override sets Θe = 50; and everything is clamped at
Θe = 1000 by a compile-time cap.

---

## What Brandon actually said, decoded

His email made seven points:

1. **Framing:** this is a numerical-trust problem, not a theory problem. The
   real funnel obviously radiates; the issue is that the simulation doesn't
   model the funnel gas credibly, so computing radiation from it is hard to
   justify.
2. **Entropy inversions:** the solver's failure-repair steps concentrate in
   the jet, and they break energy conservation — one more reason not to
   trust funnel gas properties.
3. **Table advice:** if your scattering tables only go to Θe = 1000, stay
   under 1000, and go find out why that's the cutoff.
4. **Direct question:** "are you just assigning the density and temperature
   from the input GRMHD and running grmonty?"
5. **Heating vs. cooling:** viscous heating and magnetic reconnection can
   plausibly keep the funnel hot; he's never seen "flash cooling" of funnel
   regions in the literature, so he believes heating wins.
6. **Direct question:** "your Θe is from GRMHD without two-temperature
   plasma, right?" — and the key physical expectation: jet electrons are
   usually 10–100× COLDER than the ions, which is the entire reason the EHT
   invented R_high/R_low.
7. **Do a Zoom with Richard** (cc him the link).
   **P.S.:** Θe = 10⁴ means T_e ≈ 6×10¹³ K — not realistic, and hotter than
   the GRMHD gas itself gets anywhere.

---

## SOLVED since his email

### 1. The cap is not a table limit — his advice's premise was false

He said "stick to what your table supports, and check why 1000 is the
cutoff." We checked. The Compton scattering lookup table
(`src/hotcross.c`) actually extends to Θe = 10⁴ — ten times past the cap.
The number 1000 comes from two deliberate lines of code:
`THETAE_HARD_MAX = 1e3` (model/iharm/model.c, line 9) and
`SCATTERING_THETAE_MAX = 1000` (src/decs.h, line 26). Someone chose it.

Why this matters: "we stop at 1000 because the tables stop there" would have
been a clean, boring, referee-proof answer. That answer is not available.
If the cap stays, it has to be defended as a deliberate modeling choice — a
regularizer that tames zones the simulation can't constrain — validated by
checking the predicted spectrum against real M87 data. That reframing is
agenda decision #2.

### 2. His "are you just assigning T from the GRMHD?" question — answered precisely

No — and the answer is more interesting than he expected. Density: yes,
straight from GRMHD, artificial floors included. Temperature: NOT in the
jet. Wherever σ ≥ 2 the GRMHD internal energy is ignored entirely and the
electron temperature is prescribed by the supplement formula. So his worry
"the GRMHD temperature is untrustworthy in the funnel" half-misses our
setup: we never use the GRMHD temperature there. The real question is
whether our replacement prescription is physical — and pulling on that
thread is what led to the discovery in item 4 below.

One uncomfortable subtlety survives, though: the supplement's formula
depends on B²/n_e, and in the funnel n_e IS the artificial floor density.
So we dodge the untrustworthy temperature but still inherit the
untrustworthy density. Neither recipe escapes the funnel's fakeness; they
just import different artifacts.

### 3. His P.S. checks out exactly — and our zone census quantifies it

Θe = 10⁴ converts to 5.93×10¹³ K; his 6×10¹³ is spot on. Our breakdown of
the actual MAD snapshot shows the supplement, before the cap flattens it,
wants a median Θe of about 11,000 in the emitting band (95th percentile
29,000). Compare the virial ceiling of roughly Θe ≈ 600: the prescription
is asking for electrons 20–50× hotter than the total gravitational energy
budget allows for the IONS, let alone the electrons. In Brandon's language:
the model doesn't just make electrons warmer than expected, it inverts the
"electrons are 10–100× colder than ions" hierarchy that motivated
R_high/R_low in the first place. His physical instinct was correct, and we
can now put numbers on it.

### 4. The 12π discovery — the biggest thing his P.S. triggered

Asking "WHY does the formula want Θe ≈ 10⁴?" led to an equation-level audit
against the two papers that define this model, and the code disagrees with
both of them by a specific, decomposable factor.

What the papers say (verbatim, fetched this week):

- Anantua, Ressler & Quataert 2020 (MNRAS 493, 1404):
  "β_e = P_e/P_B = (γe−1)u_e/(b²/2) = β_e0 (constant)". Here b is the
  simulation-unit field, so P_B = b²/2 is magnetic pressure in CODE units.
  The chain also states plainly that pressure = (γe−1) × energy density.
- Emami, Anantua, Chael & Loeb 2021 (ApJ 923, 272), Eq. 25:
  "P_B = B²/8π" — the same magnetic pressure, written in CGS units.

The two papers agree with each other. The code (our grmonty port, copied
verbatim from the group's own ipole fork, model/iharm/model.c line 547)
computes instead:

    Θe = β_e0 · [B_cgs² / (2(γe−1))] / (n_e m_e c²)

Two separate slips hide in that bracket:

- **A factor 4π:** the code takes the CODE-units formula b²/2 but feeds it
  the CGS field B (which is √(4π) times larger). Magnetic pressure in CGS
  is B²/8π, not B²/2 — so the code's "pressure" is 4π ≈ 12.6× too big.
  This is exactly the unit-mixing trap described in the vocabulary section.
- **A factor 3:** the code divides by (γe−1) = 1/3, which converts pressure
  into ENERGY DENSITY (u = 3P for a relativistic gas) — but then uses the
  result where pressure belongs. The papers' own definition shows (γe−1)
  multiplying, not dividing. Net effect: 3× too big.

Together: 12π ≈ 37.7× hotter than the published model. Run the published
formula instead and most of the crisis relaxes: the supplement becomes
Θe ≈ 92·σ, and measured on the real MAD snapshot (fix implemented 08-07
behind the `constant_beta_paper_literal` flag, default off, compiled code
verified against the independent numpy port to 7×10⁻¹²) the sheath spans
Θe ≈ 270–1000 — the 92σ term plus the R-β base underneath it. The cap
still catches the hottest 11.3% of sheath zones (the σ→10 edge) but stops
being the model: 89% of the band shows real σ-dependent structure instead
of a flat 1000, the sheath's emissivity proxy drops ×3.2 at fixed M_unit,
and Brandon's colder-than-ions hierarchy is approximately restored. It also
matches, almost exactly, the "rescale β_e0 by ~3×10⁻³" fix we had derived
independently before finding the cause (0.1/12π ≈ 2.7×10⁻³).

Important honesty note: we are NOT declaring this a bug. The ipole lines
are the group's own code, and an intentionally hotter variant isn't
impossible. But the burden of proof has flipped — the code now visibly
disagrees with the group's own published equations, and only Richard/Angelo
can say whether that was deliberate. That is question #1 for the Zoom. The
fix, if wanted, is one line in each code, kept behind a switch so old
results stay reproducible.

### 5. Where the light actually comes from — his trust argument, quantified

His core worry is "you're computing radiation from the least trustworthy
zones." The zone census says: mostly, no.

- **MAD:** the deep funnel he distrusts most (σ ≥ 10) contributes only
  ~1.5% of the emission proxy. About 98% comes from the σ = 2–10 sheath —
  half a percent of the simulation's zones doing nearly all the work. The
  sheath is also the layer the radio observations say should shine, so the
  model's geography is defensible even if its temperature scale isn't.
- **SANE:** a genuine surprise, beyond anything he asked. The "jet" tag
  fires almost entirely through the β ≤ 0.1 arm, which turns out to paint
  broad mid-latitude DISK material — not the funnel at all — and that
  material carries ~94% of the emission proxy. In SANE models, our "jet
  supplement" is mostly not describing a jet. That's agenda decision #4.

### 6. Zoom prep is done

The one-page agenda he suggested exists
(`docs/2026-08-06_zoom_agenda_thetae_cap.md`): five decisions, the four
figures to screen-share, and the 12π question placed first because its
answer reshapes all the others. Scheduling is on you; he asked to be cc'd.

---

## STILL OPEN

### A. Was the 12π deviation intentional? — SETTLED 09-08, without the authors

It turned out this *could* be settled from our chairs, because "intent" was
the wrong frame — the checkable question is *which form produced the
published science*, and the answer is: neither paper ever used the hot form.
The full chain lives in `2026-09-08_12pi_provenance_verdict.md`; the short
version:

- The hot line was born **2021-05-24** in Angelo's ipole fork — four months
  *after* Emami+2021 was submitted, over a year after Anantua+2020 was
  published. Emami+2021's figures were made with **GRTRANS**, a different
  code entirely. No published result depends on the hot form.
- The line rewrote itself three times in ten weeks (first version had no
  n_e at all), the decisive change hid in an unrelated commit, and the final
  form contradicts both the paper it cites in its own comment ("//ARR: See
  definition in Anantua et al. (2020)" — off by 12π) and the author's own
  birth comment (energy-density intent — off by 36π).
- The fork has been frozen since 2021-08-26: never fixed because never
  revisited, not because it was reviewed and kept.

So the "papers win" future below is the real one — adopted for the
go-forward MAD wJET configs, documented in methods, with the fork
discrepancy as an appendix note. Richard/Angelo now sign off in draft review
instead of gatekeeping by email. Bonus: the tuned 48-run production grid is
untouched (production MADs never used wJET; SANE supplements are ~extinct),
so nothing already run needs redoing over this.

- **Papers win (adopted):** the supplement becomes Θe ≈ 92σ. The cap becomes
  a boundary effect (11% of sheath zones), the super-virial problem
  evaporates, the Compton (X-ray) load drops enormously, and Phase-4
  production gets cheaper and physically cleaner. Most of the remaining
  agenda shrinks to bookkeeping.

### B. Entropy inversions — real concern, currently unmeasurable

His energy-conservation point stands, and we cannot put a number on it: the
snapshot files' `extras` group is empty — the GRMHD runs did not record
which zones needed failure-repairs. So we can't mask those zones or report
"X% of the light comes from energy-non-conserving gas." Unless someone
re-runs or re-dumps the GRMHD with fixup flags saved (not realistic on our
timeline), this stays a written caveat in the paper rather than a
measurement. The caveat wording is agenda decision #5.

### C. Where to stop trusting the simulation — the σ boundary

Policy question, informed but not decided by our census: emit from the
sheath only (cut at σ ≈ 5–10), keep the whole funnel with caveats (status
quo — and note the deep funnel barely matters in MAD anyway), or exclude
flagged zones (blocked by B). Richard's call — agenda decision #1.

### D. The β ≤ 0.1 arm mislabels disk material as jet

The SANE surprise from item 5. Options: require σ AND β together instead
of either-or, lower the β threshold, or drop the β arm entirely. Needs a
decision before Phase-4, because it changes what "jet-aware" means in the
paper — agenda decision #4.

### E. Brandon's heating-rate homework — half done

He suggested comparing reconnection/viscous heating rates against cooling
rates. We have the cooling half: at Θe = 1000 in a 10 G field, an electron
radiates its energy away in ~2600 s, which is only 8% of the natural
timescale near the black hole (r_g/c ≈ 3.2×10⁴ s for M87). So hot funnel
electrons cool FAST, and sustained high Θe requires vigorous continuous
heating — his claim, which we neither confirm nor refute without the
heating side. Nobody has a first-principles number for that equilibrium,
which is precisely why it ends up being a parametrized model choice.

Update 2026-08-07 — the Ryan et al. 2018 check is done (ApJ 864, 126;
arXiv:1808.01958; axisymmetric 2D, so cite with that caveat):

- Their heating rates live in §2.2, Eq. 7: electrons receive a fraction
  f_e of the total dissipation, with f_e taken from Howes (2010) — a
  TURBULENT-cascade damping model — plus explicit Coulomb heating
  (ion-electron collisions, Stepney & Guilbert 1983; implementation §3.3).
  Note for Brandon's point: this is turbulence heating, not reconnection.
  The standard reconnection heating fractions are Rowan, Sironi & Narayan
  2017 and Werner et al. 2018 — those are the papers to check for "does
  reconnection keep funnel electrons hot."
- The claim worth citing on the call, now verified with numbers (§4.3):
  with cooling included, near-hole electron temperatures come out a factor
  ~2–3 LOWER than matched nonradiative models at M87-like accretion rates
  (and farther out, Coulomb coupling makes electrons 5–10× hotter than
  models that ignore it). So cooling does materially reshape T_e exactly
  where our supplement operates.
- Directly relevant precedent for the σ-boundary decision (§3.3): Ryan
  et al. FORBID radiation interactions wherever b²/ρ > 1, stating that
  harm-like codes "cannot accurately represent even total fluid
  thermodynamics" there. A published, citable version of Brandon's trust
  argument — and of the sheath-only option (agenda decision #1).

### F. Downstream work (A resolved 09-08 — these now wait only on ashton's go)

- **Phase-4 production reruns** — 12π resolved: go-forward MAD wJET configs
  run paper-literal (`constant_beta_paper_literal 1`); the 48 tuned runs
  stand as-is. Warm-start M_units are preserved inside the archived July
  spectra. The same one-line gate must land in the ipole copy before P4.3.
- **Tuning frame** — we tune to 0.5 Jy averaged over all viewing directions,
  but a real observer at 17° sees 1.3–25× less; whether to retune in the
  observer's frame is an open choice (agenda "if time permits").
- **Positron polarization** — the ipole jV/aV/rV coefficients look like they
  carry a factor 1/(1+f) where (1+f) belongs; needs Richard/Angelo before
  any polarized positron images.

---

## Bottom line

Brandon's email looked like a "be conservative" checklist: trust the tables,
distrust the funnel, expect cold electrons. Working through it flipped every
premise into something sharper: the tables were never the constraint, the
funnel was never the main emitter, and the temperatures he rightly called
unphysical are plausibly one unit-convention slip (12π) away from being
fine. What's left is one factual adjudication only the code's authors can
make, two scope decisions (σ boundary, β arm), one caveat to word (energy
conservation), and one afternoon of literature homework. That's a short
list, and all of it is on the agenda for the call he suggested.
