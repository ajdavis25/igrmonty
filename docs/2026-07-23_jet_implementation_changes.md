# GRMONTY Jet Implementation: Change Log

Date: 2026-07-23
Related: `docs/audits/2026-07-23_jet_electron_temperature_audit.md` (findings),
`docs/2026-07-23_jet_paper_readiness_plan.md` (phased plan/checklist — this file is the
concrete record of what was actually changed and why).

This document records every change made to the repository in the course of executing
Phase 1 of the paper-readiness plan, plus the factual basis for each decision, so the
rationale doesn't have to be reconstructed from git blame later.

---

## 1. `constant_beta_e0_exponent` default reverted from 0.0 to 1.0 (resolves Finding H1)

**File:** `auto_munit_bracket.py:71-80` (`WJET_DEFAULTS` dict)

**Before:**
```python
WJET_DEFAULTS: Dict[str, float] = {
    "sigma_transition": 2.0,
    "constant_beta_e0": 0.1,
    "constant_beta_e0_exponent": 0.0,
    "jet_sigma_cut": 10.0,
    "jet_beta_cut": 0.1,
    "jet_thetae": 50.0,
    "jet_ne_mult": 1.0,
}
```

**After:**
```python
WJET_DEFAULTS: Dict[str, float] = {
    "sigma_transition": 2.0,
    "constant_beta_e0": 0.1,
    # Matches IPOLE's default (model/iharm/model.c, `constant_beta_e0_exponent = 1.0`)
    # and GRMONTY's own compiled-in default (model/iharm/model.c). Was previously 0.0,
    # which zeroed the B-field dependence of the whole jet-temperature supplement --
    # see docs/audits/2026-07-23_jet_electron_temperature_audit.md, Finding H1.
    "constant_beta_e0_exponent": 1.0,
    "jet_sigma_cut": 10.0,
    "jet_beta_cut": 0.1,
    "jet_thetae": 50.0,
    "jet_ne_mult": 1.0,
}
```

**Why:** `constant_beta_thetae()` (`model/iharm/model.c:462-511`) computes
`constant_beta_e0 · (B²/(2(γₑ-1)))^exponent / (nₑ·mₑc²)`. With `exponent=0`, the
`(...)^0 ≡ 1` for any finite positive base, so the entire jet-temperature supplement loses
**all** magnetic-field/σ dependence — it degenerates to a pure `1/nₑ` term, which
contradicts the physical meaning of a "constant electron beta" (electron-pressure /
magnetic-pressure ratio) model.

**On the specific factual question raised when this decision was made** ("I thought IPOLE
was using 0 for the exponent"): direct inspection of the IPOLE source does not support
that recollection. `constant_beta_e0_exponent = 1.0` appears in the reference tree used
throughout this audit (`ARRicarte/ipole.git`, branch `dev`, commit `d88e5f1` — the same
commit that `ipole+e-` was shown to be byte-identical to for this file), specifically:

```c
double constant_beta_e0_exponent = 1.0;  //ARR:  For electronModel == 5
```

The same value (1.0) also appears independently on the separate `origin/master` lineage
of the same tracked repository (commits `f5372c4`→`43ad5c4`), which implements the same
feature with a shifted electron-model numbering. So two independent commit lineages in
the one IPOLE repository examined during this audit agree on `1.0`; neither was found to
use `0`. It's possible a different IPOLE branch/fork elsewhere uses 0, but nothing in the
trees inspected for the audit does. Decision made: revert to `1.0` regardless, since it
matches both available IPOLE evidence and GRMONTY's own pre-existing compiled default.

**Blast radius (from `tools/tag_wjet_provenance.py`, run before this fix):** 26 of the 35
existing `*wJET*.h5` outputs in `igrmonty_outputs/` — specifically the entire `rh20`/`rh160`
production-style batch, not the small `5e4_test` smoke-test batch — were generated with
`constant_beta_e0_exponent=0.0`. Full detail in
`igrmonty_outputs/m87/_qa/wjet_provenance_report.csv`. **These 26 files do not reflect the
corrected physics and should not be used in a paper without being regenerated** (Phase 4
of the plan).

---

## 2. Crit-Beta Θe floor unified with IPOLE's flat 1e-3 (resolves Finding M1)

**Files:** `model/iharm/model.c:551-556`, `model/iharm/model.h:7`

**`model.c`, before:**
```c
const double rb_floor = 1.e-3;
const double crit_floor = 3.e-2;
```

**`model.c`, after:**
```c
const double rb_floor = 1.e-3;
// Matches IPOLE's flat, model-independent floor (ipole model/iharm/model.c,
// "Secret floor", fmax(..., 1.e-3) applied regardless of electronModel).
// Previously 3.e-2 here, which both diverged from IPOLE and disagreed with the
// THETAE_MIN comment in model.h -- see
// docs/audits/2026-07-23_jet_electron_temperature_audit.md, Finding M1.
const double crit_floor = 1.e-3;
```

**`model.h:7`, before:**
```c
#define THETAE_MIN 1e-3 // 1e-3 for R-Beta | 0.3 for Crit-Beta
```

**`model.h:7`, after:**
```c
#define THETAE_MIN 1e-3 // flat floor for all electron models (R-Beta, Crit-Beta, jet variants); matches IPOLE's "secret floor"
```

**Why:** three-way disagreement existed between the code (`3.e-2` = 0.03), the comment
documenting that code (`0.3`), and the IPOLE reference (a single flat `1.e-3` applied to
every `electronModel`, `ipole+e-/model/iharm/model.c:558`). Decision made: adopt IPOLE's
value exactly, since it removes the deviation rather than just reconciling code and
comment to some other number. This affects the plain (non-jet) `CRITBETA` model too, not
only `CRITBETAwJET` — any `CRITBETA`/`CRITBETAwJET` zone whose unfloored Θe previously fell
between `1e-3` and `3e-2` will now report a smaller (more IPOLE-consistent) Θe there.

---

## 3. Precedence comment added for the two jet mechanisms (resolves Finding M2)

**File:** `model/iharm/model.c:673-693` (inside `thetae_func()`)

Added a comment immediately before the `if (in_jet && jet_thetae > 0.0) ... else if
(in_high_sigma_region) ...` block explaining that these are two deliberately separate
mechanisms — a GRMONTY-only hard override (no IPOLE equivalent) and an IPOLE-ported
additive supplement — and that the hard override is intended to take precedence whenever
both would otherwise apply. **No behavior changed** — this is a comment-only edit.

---

## 4. wJET output provenance-tagging tool added (resolves Finding M3)

**File:** `tools/tag_wjet_provenance.py` (new)

Read-only script: for every `*wJET*.h5` file under `igrmonty_outputs/`, reads the embedded
`/params/electrons/*` metadata and classifies the run as pre- or post- the jet-parameter
plumbing fix (i.e., whether `jet_sigma_cut`/`jet_beta_cut` were actually set to reachable
values, per the prior `docs/audits/2026-04-06_m87_long_run_audit.md` finding), and flags
`constant_beta_e0_exponent == 0`. Writes a summary CSV to
`igrmonty_outputs/m87/_qa/wjet_provenance_report.csv` (new file; nothing existing was
modified). Run once so far; results are the "blast radius" numbers quoted in §1 above.

---

## 5. Phase 2 execution and an unexpected (but explained) result

**Build.** Compiled cleanly with all changes above (`h5cc`, no machine-specific override
on this host). The live production `grmonty` binary was busy — a job appears to be
actively running against it on a compute node — so it was left completely untouched;
the freshly linked binary was copied from `build_archive/grmonty` to a scratch location
(`grmonty_paperfix`) instead, and all Phase 2 test runs below use that copy, writing
outputs only under the session scratchpad, never under `igrmonty_outputs/` or `logs/`.

**Test run A vs B — jet-disabled-by-construction (plan item P2.3).** Ran
`with_electrons=4` with both jet mechanisms neutralized (`sigma_transition=1e6`,
`jet_sigma_cut=jet_beta_cut=-1`) against plain `with_electrons=2`, same seed (42), same
tiny `Ns=100`, `fit_bias 0`, same dump (`Sa-0.5_4000.h5`), same `M_unit`/`MBH`/`TP_OVER_TE`.
**Hypothesis was that these should be bit-identical. They were not:**

| | A (`we=4`, jet neutral) | B (`we=2`, reference) |
|---|---:|---:|
| `L` (erg/s) | 2.63367e+38 | 2.26793e+38 |
| `N_superph_scatt` | 124 | 235 |
| `N_superph_recorded` | 1474 | 1742 |
| `nuLnu` sum | 4,684,474 | 4,423,620 |

Both runs completed cleanly (exit 0, no NaN in either `output/nuLnu` array, confirmed via
`h5py`), so this is not a crash or corruption — it's a real, reproducible ~6-16%
(depending on which derived quantity) difference in output between the two configurations,
even with every *new* jet parameter neutralized.

**Root cause, traced back to source (not a flaw in the new jet mechanisms):**
`get_fluid_params()`/`get_fluid_zone()` contain a *pre-existing, non-jet* cut:
```c
double sig_unscaled = pow((*B) / B_unit, 2) / ((*Ne) / Ne_unit);
if (with_electrons < 3 && sig_unscaled > 1.)
  *Thetae = SMALL;
```
(`model.c:767-769`, `935-937`). This applies to `with_electrons ∈ {0,1,2}` and forces Θe
to ~0 in any zone with `sig_unscaled>1` — but it does **not** apply to `with_electrons ∈
{3,4,5}`. This is exactly Finding L4 from the original audit
(`docs/audits/2026-07-23_jet_electron_temperature_audit.md`), which was rated **Low**
severity there ("pre-existing, out of jet-scope"). **This test shows that call was too
generous: the effect size is not negligible** — it alone produces the ~6-16% swing above,
with neither `sigma_transition` nor any `jet_*` parameter involved. The two brand-new jet
mechanisms audited in H1-H3 were confirmed correctly neutralized by their own printed
parameters (`sigma_transition = 1e+06 ... jet_sigma_cut = -1 ...` in `run_A.log`) and by
the fact that `clamp_sigma()` caps σ at 300 (`model.c:390`), which is always far below
`sigma_transition=1e6` regardless — so the additive supplement and the hard override are
both provably inert in run A. The A-vs-B difference is coming entirely from the
**legacy, pre-jet** `with_electrons<3` cut, which silently stops applying the moment you
select `with_electrons≥3` — including plain `CRITBETA`, not just the wJET modes.

**Practical consequence:** "select a non-jet-parameter configuration of a `with_electrons∈
{4,5}` run and it behaves like `with_electrons=2/3`" is **not literally true** — it behaves
like `with_electrons=2/3`'s *disk formula*, but without that legacy funnel cut. Anyone
comparing CRITBETA(3)/RBETAwJET(4)/CRITBETAwJET(5) against RBETA(2)/dump-model(1)/
constant-ratio(0) should expect a real difference in high-σ zones from this alone, on top
of anything the actual jet parameters do. This affects the equivalence-matrix row "Sigma/
emission cuts" in the audit — updating its practical-impact assessment from "not a
jet-introduced difference" (still true) to "not jet-introduced, but empirically
material, and worth being explicit about in any methods section that compares model
numbers across the 3-boundary."

**Follow-up run C (isolating the additive supplement itself, plan item P2.1-adjacent):**
ran `with_electrons=4` again, holding it fixed (so the legacy-cut confound above is
identical in both arms), only turning on **the exact current `WJET_DEFAULTS` values**
from `auto_munit_bracket.py` post-fix — `sigma_transition=2.0`, `constant_beta_e0=0.1`,
`constant_beta_e0_exponent=1.0`, `jet_sigma_cut=10.0`, `jet_beta_cut=0.1` — while keeping
`jet_thetae=0`/`jet_ne_mult=1.0` so the hard-override/density-scaling mechanisms stay
provably inert. This isolates exactly and only the additive Θe supplement, at exactly the
parameters a real production wJET run will use today.

**Result: a ~37-44× increase in total luminosity relative to run A**, not a modest bump:

| | A (additive off) | C (additive on, current `WJET_DEFAULTS`) | ratio |
|---|---:|---:|---:|
| `L` (erg/s) | 2.63367e+38 | 9.63089e+39 | 36.6× |
| `nuLnu` sum | 4,684,474 | 204,190,430 | 43.6× |
| `N_superph_scatt` | 124 | 9,891 | 79.8× |
| `N_superph_made` | 3,464 | 4,991 | 1.44× |
| scatter ratio (`dNscatt/dNmade`) | 0.0358 | 1.982 | 55.4× |

No NaNs, no crash, no negative values in either output — this is a real, finite,
reproducible result, not corruption.

**Mechanism, traced to source:** `constant_beta_thetae()` (`model.c:462-511`) computes
`constant_beta_e0 · (B²/(2(γₑ-1)))^exponent / (nₑ·mₑc²)` — note the `1/nₑ` in the
denominator. Zones with σ=B²/ρ ≥ `sigma_transition` are, by construction, exactly the
zones where ρ (hence nₑ) is small relative to B — so this term is largest precisely where
nₑ→0, i.e. it is structurally most sensitive in the zones with the least density to anchor
it. The user-facing cap (`Thetae_max`) is set to `1e100` in essentially every existing
parfile in this repository (including all of this session's test files, matching the
project's own convention) — i.e. **effectively disabled** — so the only real ceiling
anywhere is the hardcoded `THETAE_HARD_MAX = 1e3` in `model.c:9`. The jump in scatter
ratio (0.036→1.98, more than half the abort threshold of 10, at `Ns=100` with `fit_bias 0`
and `bias` fixed at 1) is consistent with the prior self-audit's and this audit's Finding
H2 observation that wJET branches disproportionately hit the bias-abort guard — this test
suggests at least part of *why*: the additive supplement, at its current default
parameters, can push local Θe (and therefore scattering optical depth) up dramatically in
low-density zones, which then genuinely does need much more aggressive bias tuning to
sample efficiently.

**This is not a porting error** — the formula is verified identical to IPOLE (audit §6/§7),
so IPOLE's own `constant_beta_e0`-model zones would show the same `1/nₑ` sensitivity given
the same inputs. **Whether a ~40× luminosity contribution from the jet supplement is
physically intended** (jets are supposed to be a real, sometimes-dominant contribution —
that may be exactly the point) **or is an artifact of `sigma_transition=2.0` being too
aggressive a threshold for this dump, or of `Thetae_max` being conventionally disabled,
is a scientific judgment call, not a code-correctness question** — flagging for explicit
user review before more compute is spent tuning `M_unit` for the current `WJET_DEFAULTS`
against a mm-band flux target, since a 40× intrinsic luminosity swing this large will
also swing the `M_unit` needed to hit any fixed flux target, potentially making it
misleading to compare that `M_unit` against a non-jet or IPOLE run.

---

## 6. Advisor input on the ~40× finding, and a cone-restricted follow-up check

**Advisor's reply (2026-07-26, verbatim):** "Magnetization Sigma = electromagnetic flux
density/particle flux density. Sigma is typically highest in jet regions which are often
low density due to centrifugal force barrier to particles. Due to low plasma density, jet
emission is typically subdominant due to extreme relativistic beaming effects."

This confirms the mechanism found in §5 is standard, expected physics (high-σ zones being
low-density is not a numerical artifact) but does not by itself say whether ~40× is right
for *this* dump/config. The actionable part is the beaming caveat: it implies the 4π/
isotropic total that Run C reported is not the physically relevant quantity — what matters
is flux at the actual observer viewing angle, since relativistic beaming concentrates
synchrotron emission strongly in angle.

**Follow-up check.** Applied `tools/viewing_cone_postprocess.py` (pure post-processing,
no recompile, no rerun — operates on `/output/nuLnu` + `/output/dOmega` already in the
existing Run A and Run C HDF5 spectra) at this project's own M87 viewing convention
(`--thetacam-deg 163`, which the tool folds to 17° per the equatorial-fold binning
convention; `--cone-half-angle-deg 10`, matching the tool's documented IPOLE-comparison
example). Compared **A (additive off) vs. C (additive on)** — the same isolated-supplement
pair from §5, not B, so the legacy-cut confound (§5) stays controlled out:

| quantity | A (additive off) | C (additive on) | ratio C/A |
|---|---:|---:|---:|
| cone-restricted sum (θ=17°±10°, `nuLnu` integrated over selected θ bins × all freq) | 4.6113e+39 | 2.5705e+41 | **55.7×** |
| all-θ / 4π-equivalent sum, same binning method | 1.7927e+40 | 7.8144e+41 | **43.6×** |
| (for reference) printed bolometric `L` from run logs, §5 | 2.63367e+38 | 9.63089e+39 | 36.6× |

**Result: restricting to the actual M87 viewing angle does not shrink the excess — it is
mildly larger in-cone (55.7×) than the all-sky average (43.6×).** The three ratios (36.6×,
43.6×, 55.7×) are computed three different ways (bolometric `L`, all-θ `nuLnu` sum, cone
`nuLnu` sum) and broadly agree in order of magnitude, which is a useful cross-check of
internal consistency, but the cone number is *not* smaller, which is the opposite of what
a naive reading of "beaming makes jet emission subdominant" would predict.

**Why this might make sense, not just be noise:** this project's own M87 convention
(`thetacam=163°`, folding to 17° from the jet axis) puts the observer close to *pole-on*,
not edge-on. Relativistic beaming concentrates emission from material moving toward the
observer — for a near-pole-on view of an outflow, that is exactly the geometry where
beaming would be expected to *enhance* the apparent jet contribution, not suppress it. The
advisor's comment reads as the general/typical-orientation case; M87's actual near-axis
viewing geometry may be close to the exception rather than the rule here.

**Caveats — this is suggestive, not decisive:**
- **Statistics are thin.** These are `Ns=100` runs (deliberately, for cheap code-correctness
  checks — see §5). Only 75/200 (A) and 121/200 (C) frequency bins have any nonzero signal
  even within the selected θ bins; the cone sums above are built from a few hundred
  superphotons at most. Good enough for an order-of-magnitude read, not a precise ratio.
- **The tool is a coarse proxy, not true ray-traced beaming.** Per its own docstring,
  GRMONTY's output here is binned by escaping photons' final polar angle only, folded
  about the equator — "a cone cut is an azimuth-averaged theta cut, not a true camera/FOV"
  and "cannot reproduce finite image-plane/FOV ray tracing." It cannot capture the
  frequency-boosting/aberration structure that full geodesic ray-tracing (IPOLE, at a
  specific camera position) would show.
- An earlier pass of this same check mistakenly compared **C against B** (plain
  `with_electrons=2`) instead of C against A, and initially looked at only the single
  frequency bin nearest 228 GHz rather than summing across the cone's frequency bins — that
  version showed a spurious "zero flux in cone" for C, which was an `Ns=100` shot-noise
  artifact in one bin, not a real result. The table above is the corrected, apples-to-apples
  version (A vs. C, cone-summed over all frequency bins) and is the one to trust.

**Not resolved.** This check argues against "beaming rescues it," but given the caveats
above it should not be read as confirming the ~40× is wrong either. The decisive version of
this comparison is Phase 4 item **P4.3** (actual IPOLE ray-traced image at the M87 camera
position) — recommended to prioritize that over further `M_unit` tuning spend on the
current `WJET_DEFAULTS`, since a real ray-traced cone comparison is the only way to fully
capture the beaming effect the advisor raised.

**Follow-up: `Ns=100→1000` rerun (2026-07-26), same A-vs-C pair, same method:**

| quantity | Ns=100 | Ns=1000 |
|---|---:|---:|
| bolometric `L` ratio (printed) | 36.6× | 35.1× |
| all-θ / 4π-equivalent ratio | 43.6× | 48.6× |
| cone-restricted ratio (M87 angle) | 55.7× | 91.1× |
| nonzero freq bins in cone, A / C (of 200) | 75 / 121 | 126 / 144 |

The **all-sky and bolometric numbers are stable** across a 10× jump in photon count
(35-49× both times) — that part of the finding is now on solid footing, not a small-`Ns`
fluke. The **cone-restricted number is not converged** — it nearly doubled instead of
settling down, because the M87 cone only captures 4 of 18 θ-bins (~9% of solid angle by
construction), so it's a small subsample of an already-small photon budget and remains
shot-noise-dominated even at `Ns=1000`. Getting it to converge this same way would likely
need another 1-2 orders of magnitude more `Ns`, at which point this GRMONTY-side proxy is
no longer "cheap."

**What does hold up across both `Ns` levels:** in-cone ≥ all-sky both times (55.7≥43.6,
91.1≥48.6) — i.e. neither data point supports "restricting to the M87 viewing angle makes
the excess smaller." The magnitude isn't pinned down, but the direction is consistent.

**Implication for next steps:** this isn't just "P4.3 would be nice" — it's that the
GRMONTY-side angle-binned proxy structurally can't answer the angle-resolved question
cheaply (small-subsample noise scales as the cone's solid-angle fraction, not something
more `Ns` fixes efficiently). IPOLE, by contrast, is not Monte Carlo — a single camera
angle is its native, well-posed use case, with no equivalent noise floor. P4.3 is the
actually-efficient way to resolve this, not just the more "decisive" one.

---

## 7. P4.3 rehearsal — first real IPOLE ray-traced image (2026-07-26)

**Goal:** validate the IPOLE-side pipeline (units, parameter mapping, camera convention)
against the GRMONTY `Ns=1000` Run C spectrum (§6), before committing to a real
production-scale P4.3 comparison. Not a production run, not a resolution of D4 — a
plumbing check.

**Two source-level findings changed the naive parfile before running:**
1. IPOLE's `electronModel` numbering does **not** match GRMONTY's `with_electrons`.
   Per `model/iharm/model.c:74-80`'s own comment block, IPOLE model `4` is Crit-Beta and
   `5` is a standalone "constant-beta-everywhere" model — neither is the R-β analogue.
   The actual mechanism GRMONTY's jet supplement was ported from is IPOLE's *general*
   behavior at `model.c:551-554`: whenever `sigma_m > sigma_transition` **and**
   `electronModel != 5`, the `constant_beta_e0` term is added on top of *whatever* base
   model was selected. So the correct analogue of GRMONTY's `with_electrons=4` (R-β +
   jet) is IPOLE's **`electronModel=2`** (mixed `trat_small`/`trat_large`, IPOLE's R-β)
   **plus** `sigma_transition` — not `electronModel=5`.
2. IPOLE has a second, unrelated `sigma_cut` (default `1.0`) that zeroes emission along
   each geodesic step wherever local σ exceeds it (`model.c:478-480`, always active in
   this build since `USE_GEODESIC_SIGMACUT` is hardcoded to `1`). Left at default, this
   would have masked every zone the jet supplement (`sigma_transition=2.0`) is meant to
   affect, before ray-tracing ever got there — silently producing a "no jet contribution"
   image for reasons having nothing to do with the physics being tested. Set to `1e6` to
   match GRMONTY's `with_electrons>=3` branch, which has no equivalent emission cut.

**Parfile:** `ipole_rehearsal_C.par` (session scratch) — `electronModel=2`,
`sigma_transition=2.0`, `constant_beta_e0=0.1`, `constant_beta_e0_exponent=1.0`,
`sigma_cut=1e6`, `thetacam=17`, `freqcgs=228.e9`, same `dump`/`M_unit`/`MBH` as Run C,
`dsource=16.8e6` pc (matches `viewing_cone_postprocess.py`'s distance convention; IPOLE's
own compiled-in `DM87_PC=16.9e6` is close but not identical — set explicitly rather than
relying on the default, for a clean comparison).

**Result:** ran cleanly, `Total wallclock time: 112.787 s` (init `2.767 s`, so ray-tracing
itself ~110 s) at `nx=ny=160`, 8 threads — confirms the earlier "should be fast, deterministic"
expectation with a real number.

| quantity | IPOLE (ray-traced, θ=17°) | GRMONTY `Ns=1000` cone (θ=17°±10°, nearest 228 GHz bin) |
|---|---:|---:|
| `Fnu` (228 GHz) | 2.749e-4 Jy | 5.874e-4 Jy (`cone_physical`) |
| (for reference) | — | 7.825e-3 Jy (`baseline_4pi`, i.e. all-sky) |

IPOLE and GRMONTY's cone-restricted number agree to **~2.1×** — the same order of
magnitude, not many-orders-of-magnitude off, which is what a wrong unit conversion or a
misconfigured `electronModel`/`sigma_cut` would have produced. Given known differences
(GRMONTY's Ns=1000 cone statistic is not converged per §6; GRMONTY's cone is an
azimuth-averaged θ-band vs. IPOLE's actual single-camera pixel grid; frequency-bin
granularity offsets the GRMONTY sample from exactly 228 GHz by ~4.5%), this level of
agreement is a reasonable pipeline sanity check, not a precision match.

**Jet-off counterpart run (same day, immediately after):** `ipole_rehearsal_A.par`,
identical to the above except `sigma_transition=1.0e6` (neutralized, mirroring
`test_A_wjet4_neutral_ns1000.par`). Ran cleanly, `Total wallclock time: 111.123 s`.

| | IPOLE `Fnu`(228 GHz) | IPOLE `nuLnu` |
|---|---:|---:|
| A (jet off) | 1.31989e-4 Jy | 1.01626e+37 erg/s |
| C (jet on) | 2.74868e-4 Jy | 2.11636e+37 erg/s |
| **ratio C/A** | **2.083×** | **2.082×** |

**This is the actual, real, ray-traced answer to D4 — and it's a very different picture
than every GRMONTY-side estimate:**

| method | ratio |
|---|---:|
| GRMONTY bolometric `L` (Ns=1000) | 35.1× |
| GRMONTY all-sky `nuLnu` (Ns=1000) | 48.6× |
| GRMONTY cone-restricted `nuLnu` (Ns=1000, not converged) | 91.1× |
| **IPOLE ray-traced, M87 camera angle, 228 GHz** | **2.08×** |

At the actual observing geometry and EHT-relevant frequency, the jet supplement roughly
**doubles** the 228 GHz flux — not the 35-91× the isotropic/proxy comparisons suggested.
This is not noise (IPOLE is deterministic, not Monte Carlo — both runs used identical
camera/resolution settings, differing only in `sigma_transition`) and it is the single
most direct confirmation yet of the advisor's original point (§6): the isotropic total
was the wrong quantity to worry about. A real ray-traced comparison at the relevant angle
tells a much less alarming story.

**Caveats, so this isn't over-read either:**
- Single frequency (228 GHz) and single viewing angle (17°) — not a spectrum or an
  angular sweep. The `1/nₑ` mechanism is Θe-driven and could behave differently at other
  frequencies (e.g. 86 GHz, 345 GHz); this doesn't establish "~2× everywhere."
- Still the same non-production dump/`M_unit=1.83e27` as every test run this session, not
  an `M_unit`-tuned production configuration.
- `sigma_cut=1e6` was a deliberate, non-default choice (IPOLE's own default is `1.0`) made
  to match GRMONTY's unrestricted `with_electrons>=3` behavior — anyone rerunning this
  without that context could "correct" it back to default and get a very different,
  wrong-for-this-comparison answer. Worth a comment wherever this parfile choice gets
  reused.

**Net effect:** this substantially de-risks D4. It doesn't fully close it (single
frequency/angle, non-production `M_unit`), but it directly contradicts "the jet
contribution is alarmingly large" as a blanket concern — at the one place that actually
matters for comparing to M87 observations, it isn't. A multi-frequency version of this
same check (86/230/345 GHz, still cheap given ~110s/image) would be the natural way to
firm this up further before treating it as fully resolved.

---

## 8. H2 instrumentation + L2 cleanup (2026-07-26)

**Files:** `src/decs.h`, `src/main.c`, `src/utils.c`, `src/track_super_photon.c`,
`model/iharm/model.c`. 5 files, +71/-10 lines (`git diff --stat`).

### 8a. H2 — isnan-nu drop-fraction reporting (resolves Finding H2)

**Background.** `track_super_photon()` drops a photon (`ph->w = 0.0; return;`) whenever
`get_fluid_nu()` returns an invalid frequency and `try_boundary_recover_nu()` can't fix
it. The original self-audit found this via `grep`, counting 1,308 occurrences of
`"isnan nu: track_super_photon"` in one jet-enabled production log vs. 0 in a matched
non-jet log — but there was no actual counter, only a raw, occasionally-silent
`fprintf`. Investigating before editing turned up two things worth recording:

- There are **two distinct drop sites**, not one: an early one right after entering the
  function (`track_super_photon.c`, pre-loop, silent in non-`DEBUG_WJET` builds — no
  `fprintf` at all), and the mid-loop one the audit's log-grep actually measured (which
  does `fprintf` unconditionally). The pre-loop site was previously invisible to any
  log-based accounting.
- `N_init_reject_nu` (`decs.h:134`) already existed, but counts a *different* thing
  (rejection at photon *initialization*, in `utils.c`, before `track_super_photon()` is
  ever called) and was already reported in the periodic/final `summary()` stderr line —
  it just wasn't written to the output HDF5.

**What was added:**
- `N_track_reject_nu` (`long long`) — event count for the two mid-flight drop sites
  above, incremented via `#pragma omp atomic` immediately before each `ph->w = 0.0;`.
- `W_track_reject_nu` (`double`) — the photon's weight *captured before it's zeroed*,
  summed across all such drops.
- `W_superph_made` (`double`) — summed initial weight of every superphoton made,
  captured in `main.c`'s main loop (`double w_made = ph.w;` right after
  `make_super_photon()`, before `track_super_photon()` can modify it) and accumulated
  the same way `N_superph_made` already is.
- A weight fraction, `W_track_reject_nu / W_superph_made`, computed wherever reported
  rather than stored redundantly — this is the number the plan explicitly asked for
  ("not just event count -- a few dropped low-weight photons matter less than one
  dropped high-weight photon").
- All three reset in `reset_state()` alongside the existing `N_init_reject_*` counters,
  so (like those) they reflect only the final/main loop, not the bias-tuning warm-up.
- Reported in two places: `utils.c`'s `summary()`, right after the existing
  `N_init_reject_*` line (fires under the same `prefix && count > 0` gate, so it's silent
  when there's nothing to report, same as the existing line); and a new
  `/params/diagnostics/` group in the output HDF5 (`model.c`, alongside `/params/electrons/*`)
  containing `N_init_reject_nu`, `N_track_reject_nu`, `W_track_reject_nu`,
  `W_superph_made`, and the precomputed `track_reject_nu_weight_frac` — so the fraction
  is queryable directly from a spectrum file without re-deriving it.

**Verification.** Built cleanly (`make`, no compile errors; live `grmonty` binary still
untouched — same `build_archive/` + scratch-copy pattern as every other build this
session, `grmonty_h2fix`). Ran three smoke tests:
- `Ns=1000`, `with_electrons=4`, current `WJET_DEFAULTS` (`trat_large=20`): clean run,
  no crash, `N_track_reject_nu=0` (nothing to report, correctly silent).
- Same, `trat_large=80` (the audit's own "problem case" convention, `rh80`): still
  `N_track_reject_nu=0` at `Ns=1000` on this dump. **This specific dump/`Ns` combination
  doesn't reproduce the fragility the audit found in a real production-scale log** --
  a negative result, not a failure of the instrumentation; reproducing a live nonzero
  case would need a run closer to production scale (bigger `Ns`, possibly a different
  dump/spin), which is out of scope for a smoke test.
- Inspected the HDF5 output directly (`h5py`): all `/params/diagnostics/*` fields present
  and well-formed. The zero fields are exactly `0.0` (not NaN/garbage, confirming correct
  init/reset). `W_superph_made = 1.65e56` — not a bug: cross-checked against the run's
  printed luminosity (`L=7.98e39 erg/s` at ~radio/mm frequencies), `N_photons/sec ~
  L/(h*nu)` lands at the same ~1e53-1e56 order of magnitude, so a summed photon-number
  weight at that scale is physically expected, not a units error. This also confirms the
  `#pragma omp atomic` accumulation across ~38k superphotons on 8 threads produced a
  clean, non-corrupted sum, not just a crash-free run.

**Residual caveat:** the counting logic itself (the actual `N_track_reject_nu++`/
`W_track_reject_nu += ph->w` lines) was not observed firing in a live run — verified by
code inspection instead (the increments sit immediately beside the pre-existing,
already-reachable `ph->w=0.0` drop code the original audit's log-grep already proved
fires 1,308 times in a real production log; no new conditional logic was introduced, only
bookkeeping on an already-proven path). A production-scale rerun would be the way to see
a live nonzero value end to end, if that confirmation is wanted later.

### 8b. L2 — removed unreachable `bias < 1.0` branch (resolves Finding L2)

**File:** `src/track_super_photon.c` (then-line 509, inside the scatter block).

Traced the data flow directly: `bias = sanitize_bias(bias);` runs earlier in the same
call (then-line 456), and `sanitize_bias()` (top of file) unconditionally forces any
non-finite or `<1` value to exactly `1.0`. `bias` is not reassigned anywhere between that
line and the `if (!isfinite(bias) || bias < 1.0)` check — so the check can never be true.
Removed the dead branch (which incremented `invalid_bias`, reset `bias=1.0`, and reset
`php.w=ph->w` -- all three unreachable) and replaced it with a one-line comment recording
why, rather than leaving future readers to re-derive the same invariant. `invalid_bias`
itself (declared/reset/printed elsewhere, e.g. `main.c`) was left alone — it's now
permanently `0`, which is consistent with it already always being `0` before this change
(the branch that incremented it was already unreachable), and removing the variable
entirely would touch more files for no behavioral gain.

---

## 9. NEW FINDING (2026-07-26, found during P2.5): Crit-Beta produces a silent NaN spectrum

**Not a jet bug — pre-existing, in `with_electrons=3` (plain Crit-Beta) itself — but it
also affects `with_electrons=5` (Crit-Beta+jet), which is directly in scope for this
paper. Found while executing Phase 2, not part of the original audit. Flagging clearly
rather than continuing to dig unilaterally, since scoping the fix is a real decision.**

**How it was found.** P2.5 (deterministic smoke test) used `with_electrons=5` with real
(non-neutralized) production-style parameters, including `jet_thetae=50` — the first time
this session either Crit-Beta or a live `jet_thetae` override had been exercised. Both
runs completed (`exit=0`, `run status: code=2 label=ok`) but printed:
```
dL = -nan
efficiency = -nan
L/Ladv = -nan
L = -nan erg/s ... lum = -nan LEdd
```
**The run reports itself as "ok" while the spectrum is garbage.** No `isnan nu` or `w isnan`
warning fired anywhere in the log — the existing in-flight checks (including this
session's new H2 instrumentation) did not catch this.

**Isolation (3 quick runs, same dump/seed/`Ns=100`, `Sa-0.5_4000.h5`):**
1. `with_electrons=5`, `jet_thetae=50` (real production default): **NaN**.
2. `with_electrons=5`, `jet_thetae=0` (hard override neutralized, additive supplement
   still on): **still NaN** — rules out the jet hard-override.
3. `with_electrons=3`, all jet parameters neutralized (`sigma_transition=1e6`,
   `jet_sigma_cut=jet_beta_cut=-1`, `jet_thetae=0`): **still NaN** — this is plain
   Crit-Beta with *zero* jet code reachable. **Confirms the bug is pre-existing and
   unrelated to any jet-implementation work**, though it affects the jet-enabled variant
   (`with_electrons=5`) that this paper needs.

**Where it isn't.** Added a temporary full-domain scan hook to `model.c` (gated behind
`GRMONTY_DEBUG_SCAN`, calls `get_fluid_zone()` for all 4,718,592 zones and checks
`isfinite` on `Ne`/`Thetae`/`B`) and ran it against the exact config from isolation run 3
above: **`bad=0`**. Every single zone's `Thetae`/`Ne`/`B` is finite. Reading
`thetae_func()`'s Crit-Beta branch (`model.c:631-676`) directly supports this — it's
unusually defensively coded (`beta` is clamped, `Te_over_Ttot` is explicitly checked
`isfinite`/re-clamped into `(0,1)` before the division that uses it, and `thetae` is only
overwritten if the final denominator is finite and positive). **The NaN is not in the
per-zone electron-temperature calculation.** It must be downstream — emissivity,
Compton-scattering cross-section lookup, or the spectral accumulation in
`report_spectrum()` — triggered by something about the specific *distribution* of
Crit-Beta `Thetae` values (even though every individual value is finite) that R-β's
distribution apparently never hits. Not yet root-caused further.

**Blast radius on existing data: none found.** Scanned all 44 existing
`with_electrons∈{3,5}` (CRITBETA/CRITBETAwJET) production `.h5` outputs under
`/work/vmo703/igrmonty_outputs/` for NaN/Inf in `/output/nuLnu`: **0/44 affected**. This
is reassuring but not a clean bill of health — the trigger conditions for real production
configs (different dumps, spins, `M_unit`, `beta_crit`/`beta_crit_coefficient`, and
especially much larger `Ns`, which samples far more of phase space and could be *more*
likely to hit whatever rare condition triggers this) are not understood yet. Existing
`CRITBETA`/`CRITBETAwJET` outputs look clean by this check, but this check only proves
the *final integrated* spectrum has no NaN — it doesn't rule out other forms of
corruption that don't manifest as an outright NaN.

**Debug tooling added (temporary, both gated behind env vars, `model.c`):**
- `GRMONTY_DEBUG_SCAN=1` — full-domain `isfinite` scan (used above).
- `GRMONTY_DEBUG_ZONE=i,j,k` — report one zone's `Ne`/`Thetae`/`B`/`beta`/`sigma` and
  exit (built for P2.1, also useful for this investigation; see sec 10).

**Status: root-caused and fixed same day — see sec 10.**

---

## 10. D5 root cause and fix (2026-07-26)

**Root cause, precisely.** `jnu_thermal()` (`src/jnu_mixed.c`), the thermal-synchrotron
emissivity formula — whose own file header explicitly says it's "good for `Thetae > 1`"
— computes:
```c
K2 = K2_eval(Thetae);
...
j = (M_SQRT2 * M_PI * EE * EE * Ne_lep * nus / (3. * CL * K2)) * f * exp(-xp1);
```
For `Thetae` just barely above `THETAE_MIN` (`1e-3`) — three orders of magnitude below
the formula's documented valid range — two independent underflows co-occur, both driven
by the same tiny `Thetae`:
- `K2_eval(Thetae)` (a tabulated/asymptotic Bessel-K2-based normalization) underflows to
  exactly `0.0`.
- `nus` (the characteristic synchrotron frequency) scales as `Thetae²`, so it's also
  tiny, making `x = nu/nus` huge and `xp1 = x^(1/3)` large enough that `exp(-xp1)` *also*
  underflows to exactly `0.0`.

`j` becomes `(finite / 0.0) * f * 0.0` = `inf * finite * 0.0` = **`inf * 0.0`, the
IEEE754-indeterminate form that evaluates to NaN.** `thetae_in_valid_range()`
(`jnu_mixed.c`) doesn't catch this — it only checks `isfinite(Thetae) && Thetae>0`, which
`Thetae≈1e-3` passes fine; the guard's name promises more than it checks.

That `NaN` propagates: `synch = jnu_thermal(...) = NaN` inside `jnu_ratio_brems()`, whose
own guard `if (synch + brems == 0) return 0.;` **can't catch it** — any comparison with
NaN is `false` in IEEE754, so `NaN == 0` is `false`, the guard doesn't fire, and the
function falls through to `return brems / (synch + brems)` = `finite / NaN` = `NaN`. That
becomes `ph->ratio_brems`, which multiplies into *every* `spect[...] +=` accumulation in
`record_super_photon()` (both the synch bucket via `ratio_synch = 1 - ratio_brems` and
the brems bucket directly) — poisoning the recorded spectrum for that photon, and from
there the final `dL`/`L` sum in `report_spectrum()`.

**Why only Crit-Beta.** This isn't a jet bug, and it isn't really a "Crit-Beta" bug
either — it's a real gap in `jnu_thermal()`'s domain handling that could in principle
affect *any* electron model. It shows up here because **Crit-Beta reaches `Thetae`
values in the pathological `~1e-3` to a few `×1e-3` band far more often than R-β does**:
Crit-Beta's `Te_over_Ttot = beta_crit_coefficient * exp(-beta/beta_crit)` is exponentially
suppressed for large `beta` (weakly-magnetized zones), pushing `Thetae` toward (and
against) the `THETAE_MIN` floor aggressively and often; R-β's `trat = trat_large*b²/(1+b²)
+ trat_small/(1+b²)` is a smooth, bounded rational function of `beta` that rarely produces
`Thetae` that low. Every one of the 13 photons caught in the diagnostic run had `nscatt=0`
(first-generation, i.e. corrupted at emission, not scattering) and `thetae0` in
`1.01e-3`-`1.24e-3` — confirming this exactly.

**Fix (3 files, defense in depth from the actual source outward):**
1. **`src/jnu_mixed.c`, `jnu_thermal()`** — the real fix: `if (!(K2 > 0.)) return 0.;`
   right after computing `K2`, matching the function's own existing pattern of returning
   `0.` for regimes outside its applicability (it already does this for
   `!(Thetae > THETAE_MIN)` and `nu > 1e12*nus`). Also wrapped the final `j` in
   `if (!isfinite(j)) return 0.;` as a second line of defense against any other
   underflow/overflow combination in the formula that wasn't specifically enumerated
   here.
2. **`src/jnu_mixed.c`, `jnu_ratio_brems()`** — broadened `if (synch + brems == 0)` to
   `if (!isfinite(synch + brems) || synch + brems == 0)`, so a NaN/inf sum can't defeat
   the guard the way it did here. This is a second line of defense for the
   kappa/power-law EDF paths (`jnu_kappa`/`jnu_powerlaw`), which weren't specifically
   audited for the same failure mode but share the same guard.
3. **`model/iharm/model.c`, `record_super_photon()`** — broadened the entry check from
   `isnan(ph->w) || isnan(ph->E)` to `isfinite` on `w`, `E`, `ratio_brems`, `tau_abs`,
   `tau_scatt`, `X1i`, `X2i`, `X[3]`, `ne0`, `b0`, `thetae0` — every field that
   multiplies into a `spect[...]` accumulation. This is the last line of defense: even
   with (1) and (2) fixed, this ensures a photon with *any* non-finite field it carries
   gets dropped rather than recorded, instead of only checking two of the eleven
   relevant fields.
4. **`src/utils.c`, `zone_linear_interp_weight()`** — separately found and fixed a
   genuinely unreachable safety check (a `return` one line before its own `isnan(wgt)`
   check, making that check permanent dead code) while investigating this. Not the cause
   of *this* bug (confirmed empirically — see below), but a real bug in its own right,
   left fixed.

**Verification.** Built and reran the exact reproducer (`with_electrons=3`, all jet
params neutralized, `Sa-0.5_4000.h5`, `Ns=100`, `seed=42`) at each step:
- Before any fix: `L = -nan erg/s`, 0 diagnostic hits (checking only `w`/`E`).
- After broadening the `record_super_photon` check: **13 photons caught**, all
  `ratio_brems=-nan`, all `nscatt=0`, all `thetae0` in `1.01e-3`-`1.24e-3` — this is what
  identified the real field and the real regime. `L` became finite
  (`1.48e38 erg/s`) by simply dropping these 13, confirming the diagnosis before the
  root-level fix was even written.
- After the `jnu_thermal`/`jnu_ratio_brems` fix: **0 photons caught** — nothing needs
  dropping anymore, because `ratio_brems` is never NaN in the first place. `L = 9.61e37
  erg/s`, finite and clean.
- Reran the original P2.5 config that started this (`with_electrons=5`, real
  `jet_thetae=50`): **`L = 7.36e38 erg/s`**, finite. Fixed.
- **Regression check:** reran plain R-β (`with_electrons=2`, `Ns=1000`, same dump/seed):
  `L = 2.13e38 erg/s`, consistent with earlier R-β runs this session (`2.19e38`,
  `2.27e38` at similar settings, within a few percent) — the fix doesn't perturb the
  path that was already working.

**Not yet done:** re-scanned all 44 existing production CRITBETA/CRITBETAwJET outputs
before this fix and found 0 affected (sec 9) — that scan doesn't need to be redone, since
it already showed no *evidence of harm* in existing data, but it also doesn't need to be
*re-explained* now: those files were generated before this fix existed and apparently
didn't hit this narrow regime, which is consistent with it being real but statistically
uncommon at whatever `Ns`/configuration those particular runs used. This fix should be
present in any future Crit-Beta production run from this point forward.

---

## 11. P2.5 completed: reproducibility check — not bit-identical, and why

With D5 fixed, reran P2.5 properly: the exact original config (`with_electrons=5`,
`jet_thetae=50`, `sigma_transition=2.0`, `Sa-0.5_4000.h5`, `M_unit=1.83e27`, `seed=42`,
`Ns=100`), twice, identical parfiles differing only in output path.

**Result: both runs finite (D5 fix holds), but not reproducible, and not by a small
margin:**

| | run 1 | run 2 |
|---|---:|---:|
| `L` (erg/s) | 6.73768e+38 | 1.09279e+39 |
| `N_superph_made` | 3737 | 3598 |
| `N_superph_scatt` | 705 | 425 |
| `dNscatt/dNmade` | 0.189 | 0.118 |

`L` differs by ~62%, scatter count by ~40%, between two runs with an identical seed and
identical parameters.

**Not a bug — a structural property of the parallel RNG design.** `src/random.c`:
```c
#pragma omp threadprivate(r)
...
#pragma omp parallel
{
  r = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(r, 139 * omp_get_thread_num() + seed);
}
```
Each OpenMP thread gets its own independent RNG stream, deterministically seeded from
`(thread_num, seed)`. That part is fully reproducible *per thread*. But the
photon-generation loops in `main.c` (`#pragma omp parallel` around
`make_super_photon()`/`track_super_photon()`) don't use `schedule(static)` — each thread
just grabs the next zone/superphoton as it becomes free. **Which thread ends up
processing which zone is left to the OS scheduler and isn't guaranteed to repeat run to
run.** Same seed, same eight per-thread streams, but a different *assignment* of streams
to zones — so the aggregate result differs even though nothing is actually random in an
uncontrolled sense.

**Likely scale-dependent, not confirmed here.** `Ns=100` means only ~3,600 photons split
across 8 threads (~450/thread) — small enough that which specific (rare, tail-sensitive)
zones a given thread happens to draw can swing the aggregate noticeably, especially for
this configuration specifically: `with_electrons=5` combines Crit-Beta (already shown in
sec 9/10 to sit near sharp numerical edges) with a live `jet_thetae` hard override gated
by a sigma/beta threshold — i.e., a config with *more* sharp zone-boundary sensitivity
than a typical run. Earlier R-β checks this session (sec 5-6) stayed within ~4-15% across
a 10× `Ns` change, not 60%+, which is consistent with (but doesn't prove) this being a
small-`Ns` artifact for the *typical* case that's simply more pronounced for this
particular volatile configuration. **Not re-tested at higher `Ns` for this specific
config** — that would be the natural next check if reproducibility at production scale
matters before committing real cluster time (this is exactly the caveat P2.5 was
designed to catch before "committing to a 1e6-photon run," per the original plan).

**Not fixed, not attempted.** Forcing deterministic zone-to-thread assignment
(`schedule(static)` or similar) would change the code's performance characteristics and
wasn't asked for — flagging the finding, not proposing a fix.

---

## 12. Phase 2 completed: P2.4, P2.1, P2.2 (2026-07-26)

### 12a. P2.4 — `constant_beta_e0_exponent` sweep: quantifying H1

`with_electrons=4`, `Ns=1000`, additive supplement isolated (`jet_thetae=0`), current
`WJET_DEFAULTS` otherwise, sweeping only `constant_beta_e0_exponent`:

| exponent | `L` (erg/s) | `N_superph_scatt` |
|---|---:|---:|
| 0.0 (the H1 bug) | 2.32139e+38 | 924 |
| 0.5 | 7.13932e+39 | 52095 |
| 1.0 (correct) | 7.74276e+39 | 52173 |

**The H1 bug understated `L` by ~33×.** Most of that (30.7× of the 33×) is already
present at `exponent=0.5` — the effect is strongly nonlinear near the low end, which
makes sense given the term scales as `B^exponent`: for `B<1` in code units (common in
much of this domain), `B^0=1` is a large relative overstatement of a small `B^1`, so even
a partial exponent closes most of the gap. Directly answers the plan's question ("does
H1 matter as much as the audit thinks") — yes, by a wide margin.

### 12b. P2.1 — single-cell comparison vs. the audited formula

Located a zone with σ just above `sigma_transition` — `(i=50, j=5, k=0)`, σ=2.771 — via
a temporary `GRMONTY_DEBUG_ZONE=i,j,k` env-var hook added to `init_data()` (`model.c`):
loads the dump normally, prints one zone's `rho`/`uu`/`B`/`Ne`/`Thetae`/`beta`/`sigma`,
exits before any photon transport. `with_electrons=4`, `exponent=1.0`:
```
i=50 j=5 k=0: rho_code=7.0119e-06 uu_code=5.6195e-07 B_code=4.4083e-03
Ne_cgs=8.6643e+00 Thetae=1.000000000000000e+03 B_G=6.7369e-01 sigma=2.7714e+00
```
Independently recomputed the audited formula
(`constant_beta_thetae()`/IPOLE `electronModel==5`,
`constant_beta_e0 * (B_cgs²/(2(game-1)))^exponent / (ne_cgs·ME·CL²)`) directly from the
printed `B_cgs`/`Ne_cgs`:

```
B_cgs = |B_code| * B_unit = 0.673687   (matches printed B_G exactly)
energy_density = B_cgs² / (2*(4/3-1)) = 0.680781
raw Thetae = 0.1 * energy_density / (Ne_cgs * ME * CL²) = 9597.16
capped at THETAE_HARD_MAX=1000  →  1000.000000
```

**Exact match to GRMONTY's actual printed output** — not just formula-on-paper agreement
(already established in the original audit) but a real, running-code confirmation at a
concrete zone, including the hard-cap clamping behaving correctly. This is also a nice
illustration of the mechanism behind the ~40× finding from earlier sessions (§5-6): the
raw, uncapped value here is nearly 10× the hard ceiling itself.

### 12c. P2.2 — non-jet regression vs. baseline `4a1b1c5`

**Setup.** `git worktree add` at `4a1b1c5` (the pre-jet baseline the original audit
identified via `merge-base`). Needed build-environment overrides to compile on this
machine — `make HDF5_DIR=/apps/hdf5/1.12.0 CC=/apps/hdf5/1.12.0/bin/h5cc CFLAGS="..."`
(no `-static`) — this baseline commit's makefile predates this machine's HDF5/CFLAGS
customization (visible in the current tree's own makefile diff); this is purely a
toolchain path issue, not a source change, and doesn't touch the checked-out baseline
source at all.

Added a temporary `GRMONTY_DEBUG_DUMP_THETAE=<path>` hook to **both** trees (same
pattern as P2.1's, but dumping the full `N1×N2×N3` `Thetae` field as flat binary
doubles instead of one zone) — this is a pure function of the dump + parameters, no
Monte Carlo/RNG involved, so it's the correct way to test "is the field the same,"
unlike comparing full spectra (which sec 11 already showed aren't reproducible run to
run for unrelated RNG-scheduling reasons).

**Result: not bit-identical for either `with_electrons=2` or `=3`,** contrary to the
plan's original expectation — but every difference traces to one of three fully
understood causes, not a mystery:

| | total differing | `i<9` band (147,456 zones) | `i≥9`, excl. sigma-cut |
|---|---:|---:|---:|
| R-β (`we=2`) | 2,056,706 / 4,718,592 (43.6%) | 100% | max abs diff 9.18e-4 |
| Crit-Beta (`we=3`) | 1,222,268 / 4,718,592 (25.9%) | 100% | max abs diff **7.9e-9** |

**Cause 1 (both models, ~3.1% of the domain): baseline's `i<9` near-horizon floor is
gone.** Baseline's `get_fluid_zone()`:
```c
sig = pow(*B/B_unit,2)/(*Ne/Ne_unit);
if(sig > 1. || i < 9) { *Thetae = SMALL; }
```
— unconditional, applies to every `with_electrons` value. The current tree's
equivalent (the legacy cut already known as **Finding L4**) is:
```c
if (with_electrons < 3 && sig_unscaled > 1.) *Thetae = SMALL;
```
The `i<9` clause is simply gone, and the sigma clause is now gated to
`with_electrons<3`. L4 was already known to be empirically material (§5); this
pins down the *other* half of what changed — the near-horizon floor, not just the
sigma-cut's scope. 100% of the 147,456 `i<9` zones (9 innermost radial shells ×
all θ,φ) differ in both models, as expected.

**Cause 2 (Crit-Beta only): fully closed by this session's M1 fix.** Excluding `i<9`
and sigma-cut zones, Crit-Beta agrees with baseline to **7.9e-9** — floating-point
noise, not a real difference. Baseline's own Crit-Beta branch already reads
`thetae = fmax(1e-3, ...)`, i.e. baseline's original floor *was* `1e-3` — confirming
the `3e-2` this session found and reverted (Finding M1) was itself a regression
introduced sometime after this baseline, and M1's fix correctly restored the original
behavior. Independent confirmation, not previously available.

**Cause 3 (R-β only, new): the current tree's R-β has a floor baseline never had.**
Excluding `i<9`/sigma-cut, R-β differs by up to 9.18e-4 — consistent with a `~1e-3`
floor clamping up values that would otherwise fall below it. Baseline's R-β branch
(`model.c` `thetae_func`, `with_electrons==2`) has no `fmax`/floor at all — only the
final `1./(1./thetae + 1./Thetae_max)` soft cap, which does nothing at the *low* end.
The current tree's R-β does have a floor (`rb_floor=1e-3`, matching IPOLE's own "secret
floor" per the original audit) that baseline's R-β simply didn't have. **Not previously
documented** — the original audit's Finding M1 was specifically about Crit-Beta's floor
*value* being wrong; this shows R-β gained a floor *concept* it never had, likely as
part of the same general refactor. `1e-3` matches IPOLE's convention, so this is very
likely correct/intentional, just newly surfaced by this direct comparison rather than
flagged before.

**Net assessment:** none of this is alarming — every difference has a specific,
understood cause, and two of the three (`i<9` removal, R-β's new floor) look like
deliberate-if-undocumented refinements rather than regressions, each confined to a small
footprint (~3% of the domain, and a sub-1e-3 clamp, respectively). Worth a line in the
methods section if this dump's near-horizon zones matter for the science, but not a
blocker.

---

## 13. Phase 3 (P3.1) status: matched jet/non-jet drop-fraction pairs (2026-07-26/27, complete)

**Goal (plan P3.1):** one matched jet/non-jet pair at identical (spin, dump, `Ns`) using
the §8 instrumentation, reporting isnan-nu event count *and* dropped-weight fraction.

**Pair 1 — SANE `Sa-0.5_4000.h5`, `trat_large=80`, `Ns=5000` (done, login-node direct
execution — predates the switch to sbatch):**

| | non-jet (`we=2`) | jet (`we=4`, full `WJET_DEFAULTS` incl. `jet_thetae=50`) |
|---|---:|---:|
| `L` (erg/s) | 4.93068e+37 | 6.62331e+38 |
| `N_superph_made` | 168,365 | 177,907 |
| `N_superph_scatt` | 170 | 6,399 |
| `N_track_reject_nu` | **0** | **0** |

Zero mid-flight isnan-nu drops in either arm — the third consecutive negative result on
this SANE dump (after the §8 smoke tests), at 5× the photon budget. The audit's original
1,308-drop evidence came from a **MAD a=+0.94 rh80** production log, so the SANE dump
appears simply not to populate the fragile regime. *(Provenance caveat: the raw logs and
spectra for this pair lived in the session scratchpad under login-node `/tmp`, which was
wiped by session restarts after the numbers were extracted — the table above is the
surviving record. All subsequent artifacts are staged under
`/work/vmo703/scratch/p3_jetcheck/` to prevent a recurrence.)*

**Pair 2 — MAD `Ma+0.94_4000.h5`, `trat_large=80`, `Ns=1000` (non-jet arm completed
2026-07-26; jet arm died → became Finding D6, and completed 2026-07-27 post-D6-fix as
job 778206 — see §14 for the completed-pair numbers):**
SLURM jobs **778135** (non-jet) / **778136** (jet), single node, 8 CPUs,
`partition=compute1,compute2`, all files under `/work/vmo703/scratch/p3_jetcheck/`.
(Three corrections were needed relative to the first submission attempt, jobs
778072/778073, canceled before start: staging moved off login-node-local `/tmp` onto
`/work`; `trat_large` corrected 20→80 to match the audit's actual fragile case;
`M_unit=1.3488e25` from the tuning history plus `fit_bias 1`, since the SANE dump's
`M_unit` overflows `init_zone` on this dump and flat `bias=1` trips the abort guard at
ratio≈59 even at `Ns=10`.)

**Non-jet arm (778135): COMPLETED, exit 0, 5m29s.** `L=9.32e40`, 35,810 made / 67,531
scattered (tuned `biasTuning=0.167`, final ratio 1.89). **Zero isnan-nu drops** —
confirmed independently in the stderr log (`grep -c "isnan nu"` = 0) and in the output
HDF5 (`/params/diagnostics/N_track_reject_nu = 0`, `track_reject_nu_weight_frac = 0`) —
which also validates the §8 HDF5 diagnostics plumbing end-to-end on a real SLURM run.

**Jet arm (778136): FAILED, 8m31s — and the failure is the finding.** Two distinct
pathologies, both live, both in the bias-tuning phase (never reached the main loop):
1. **The §8-instrumented isnan-nu drop path fired 4 times** — the first controlled,
   instrumented reproduction of the audit's Finding H2 fragility (the original evidence
   was 1,308 greps in an old production log). Confirms H2 is real, jet-specific, and
   MAD-specific (0 occurrences in the matched non-jet arm; 0 in every SANE test all
   session).
2. **Fatal: the Compton electron sampler stalled and killed the run.**
   `sampling_error: sample_electron_stalled Thetae=1000 mu=0.03 gamma_e=4679 K=9.74e6
   sigma_KN=6.55e-6 x1=0.02 attempts=10000001` — `sample_electron_distr_p()`
   (`src/compton.c:270`) is a rejection sampler; at `Thetae=1000` (i.e. **exactly the
   `THETAE_HARD_MAX` cap the jet supplement saturates** — see §12b, where the same
   dump-family physics produced raw Θe≈9600 capped to 1000) drawing ultrarelativistic
   electrons against a deep-Klein-Nishina photon (`K~1e7`, `σ_KN/σ_T~7e-6`) has a
   per-draw acceptance so small the sampler exhausted its 10-million-attempt cap
   (`SAMPLE_ELECTRON_MAX_ATTEMPTS`, `compton.c:9`), and `fail_sampling()`
   (`compton.c:12-18`) responds by **`exit(-1)` — one un-sampleable photon in one
   Θe-capped zone kills the entire run.**

**Why this was never seen in the existing wJET production corpus:** all 26 "real" wJET
production outputs carry the H1 bug (`constant_beta_e0_exponent=0`, §1), which removes
the supplement's B-field dependence and keeps Θe far below the cap in most zones. **The
H1 fix un-masks this pathology** — post-fix Phase 4 production runs at MAD/rh80-like
configs would hit it. Flagged as new blocking finding **D6** in the plan doc: the
sampler's stall-then-die behavior needs a decision (graceful drop-with-accounting like
the isnan-nu path, an analytic deep-KN sampling branch, a deliberate lower `Thetae_max`
convention, or something else) before Phase 4 is viable for jet configs.

**P3.2 decision (per plan):** for SANE configs, drop-fraction is measured **zero** at
every tested setting — document as a non-issue in the methods section. For MAD+jet at
current post-H1-fix defaults, the question was temporarily **superseded by D6** (the
run did not survive long enough to measure it). *Resolution 2026-07-27:* with the D6
fix in place the jet arm completes, and the measured drop fraction is **1.218e-14**
(2 events of 2.21M photons, job 778206 — see §14). H2's final disposition across every
tested regime: instrumented, measured, negligible — methods-section caveat only, no
further fix warranted.

---

## 14. D6 fix: analytic deep-KN Compton sampling (2026-07-26)

**Decision (user, 2026-07-26):** option (b) from the D6 write-up — an analytic deep-KN
sampling branch, gated at `Thetae >= 100`.

**Scope note:** investigation showed D6 has *two* legs, not one. The observed killer was
`sample_electron_distr_p()` (acceptance ~5e-8 in the stall regime → 10M-attempt
exhaustion → `fail_sampling()` → `exit(-1)`, one photon kills the run). But
`sample_klein_nishina()` — the scattered-photon energy sampler that runs immediately
after — has self-documented limiting efficiency `log(2k0)/(2k0)` (~1e-5 at the
`K0_MAX`-clamped energies this regime produces): it cannot stall a run (the K0_MAX
clamp upstream bounds it), but it burns ~1e5 draws per deep-KN scattering — very
plausibly a main contributor to the historical multi-day wJET walltimes. Both got the
analytic treatment, since fixing only the first would leave the rerun crawling.

**Implementation (`src/compton.c`):**
1. **Electron sampler** — new `sample_electron_mu_deep_kn()`. Change of proposal:
   `gamma ~ Gamma(2, Thetae)` (exact: `-Thetae*log(u1*u2)`), `mu ~ Uniform(-1,1)`.
   Writing the flux factor as `(1-beta*mu) = K/(gamma*k0)`, the exact acceptance
   collapses to `A = beta * K*sigma_KN(K)/sigma_T / G_sup` — the flux factor cancels
   against the deep-KN `1/K` falloff, which is precisely why the legacy proposal
   (which *favors* head-on, high-K draws) dies in this regime. `g(K)=K*sigma_KN/sigma_T`
   verified monotone increasing over 18 decades, so `G_sup = 1.02*g(K_cap)` majorizes
   rigorously. Same `K <= K0_MAX` truncation as the legacy loop. Used when
   `Thetae >= 100` (Maxwell-Jüttner builds only — the proposal is MJ-specific;
   kappa/power-law builds keep the legacy path). **Additionally, the legacy loop's
   attempt-exhaustion path now falls back to this sampler instead of calling
   `fail_sampling()`** — the run-killing `exit(-1)` for electron sampling is gone
   entirely on MJ builds regardless of the gate.
2. **Scattered-energy sampler** — Butcher & Messel (1960) composition-rejection branch
   in `sample_klein_nishina()` for `k0 >= 1` (the standard EGS/Geant technique).
   Algebraic identity check: with `eps=k0p/k0`, the BM target
   `(1/eps+eps)(1 - eps*sin2/(1+eps^2))` equals the code's own `klein_nishina(k0,k0p)`
   numerator exactly — identical distribution, O(1) (~40%) efficiency at any `k0`
   vs. the legacy box sampler's `log(2k0)/(2k0)`.
3. The KN total cross section was factored into a shared `sigma_kn_over_thomson()`
   (bit-identical expression to what was inline).

**Pre-implementation validation (Python prototypes, numbers preserved here):**
- Envelope: `K*sigma_KN(K)/sigma_T` monotone increasing over `K in [1e-10, 1e8]` ✓.
- Electron sampler vs legacy at `Thetae=100`, `k0 = 1e-8 / 0.1 / 3.0` (N=150k each):
  all nine mean-comparisons within Monte-Carlo noise (|z| <= 2.6); legacy needed 266M
  proposals at k0=3 where the new branch needed 400k.
- Stall regime (`Thetae=1000, k0=2146`, the exact job-778136 numbers): new sampler
  efficiency **10%** (~10 draws) vs legacy ~5e-8 (~2e7 draws) — a ~2e6× improvement.
- BM energy sampler vs legacy at `k0 = 0.5 / 1 / 10` (N=200k): |z| <= 1.4; flat 40%
  efficiency at k0 = 1e3 and 1e6.

**In-repo regression tests (`src/tests.c`, run via `--run-tests`):**
`test_deep_kn_compton_sampling()` checks each sampler branch against **ground-truth
numerical integrals** of its target distribution (trapezoid over the code's own
`klein_nishina()` for the energy sampler; 2-D integral of `MJ*(1-beta*mu)*sigma_KN`
with the K0_MAX truncation for the electron sampler), at: `k0 = 0.5/2.0/1e5` and
`(Thetae,k0) = (80,0.5) / (150,0.5) / (1000,2146)` — i.e. both branches of both
samplers plus the exact regime that killed job 778136. The KN cross section is
deliberately reimplemented inside the test so an accidental edit to the shared helper
can't pass its own test. Two latent harness bugs fixed while wiring this in:
`run_all_tests()` executes before `init_model()`, so **the RNG was never initialized
for the pre-existing `--run-tests` suite** (latent segfault); and the `test/` output
dir was assumed to exist. Now: `init_monty_rand(42)` + `mkdir("test",0755)` at harness
start.

**Cluster validation round 1 (job 778171): FAILED — in the test, not the sampler,
and the gate worked as designed.** The `afterok` dependency correctly refused to launch
the physics reruns (they parked as `DependencyNeverSatisfied`). Failure analysis:
`check_kn_energy_sampler(k0=1e5)` rejected the sampler with sampled mean 10628.7 vs
"truth" 8424.29 — but the truth was the broken side: the test integrated the target on
a grid **uniform in eps**, and at k0=1e5 the density's `1/eps` part spreads its mass
evenly over ~5.3 decades, so the first grid cell alone held ~20% of the mass.
Re-verified offline: uniform-grid integral reproduces the bad 8424.3 exactly; a
**log-grid** integral gives 10493.7 (converged, identical at 20k and 2M points); an
independent 2M-draw BM sample gives 10503.3 ± 15.5 — the sampler agrees with the
converged truth to 0.6σ. The k0=0.5 and 2.0 checks passed in the same run (smooth
densities, grid-insensitive), and the electron-sampler checks never ran (harness exits
on first failure). Fix: `check_kn_energy_sampler` now integrates uniform-in-log(eps)
with the Jacobian factor.

**Cluster validation round 2 (COMPLETED overnight 2026-07-26 → 07-27):** corrected
chain — tests job **778184** → `afterok`-gated MAD rh80 reruns **778185** (jet, the
config that died as 778136) / **778186** (non-jet). All success criteria met:

- **Tests (778184): PASSED, rc=42** — all pre-existing `--run-tests` checks plus all
  six deep-KN ground-truth checks, including the corrected log-grid `k0=1e5` energy
  check (sampled 10628.7 vs converged truth 10493.7, 2.7σ at se=49.4) and the
  job-778136 stall regime itself (`Thetae=1000, k0=2146`: ⟨γ⟩ = 1115.62 vs truth
  1115.95; ⟨K⟩ = 499116 vs truth 499075).
- **Non-jet arm (778186): COMPLETED, 4m28s.** `L = 9.4616e40` vs 9.32e40 on the
  pre-D6 binary (778135) — within the known run-to-run envelope (§11); zero isnan-nu
  drops, HDF5 diagnostics all zero. The D6 changes leave the non-jet arm
  statistically unchanged, as expected (Θe never reaches the deep-KN gate there).
- **Jet arm (778185 → 778206): COMPLETED.** First attempt 778185 hit the submission
  script's 30-minute walltime while making normal progress — killed by `slurmstepd`
  mid-main-loop, *not* a sampler stall; the tuning phase had already converged to
  `biasTuning = 0.00274348`. Resubmitted with an extended walltime as **778206**
  (whose parfile also carries `fit_bias_ns 200`, a tuning-phase-only photon budget;
  both attempts converged to the *identical* tuned bias, so physics is unaffected —
  parameter echo lines match verbatim). 778206 ran **3h14m** to completion:

  | | jet arm (778206, `grmonty_d6` binary) |
  |---|---:|
  | `L` (erg/s) | 3.81972e+43 |
  | `N_superph_made` | 2,211,793 |
  | `N_superph_scatt` | 2,421,113 |
  | `N_superph_recorded` | 1,586,157 |
  | final bias-guard ratio | 1.09 (limit 10) |
  | isnan-nu drops | **2 events, weight 2.142e+43 of 1.759e+57 made** |
  | `track_reject_nu_weight_frac` | **1.218e-14** — stderr summary and HDF5 `/params/diagnostics` agree exactly |

**D6 is closed.** The exact configuration that killed job 778136 in the electron
sampler now completes with finite `L`, no stalls, and delivers the P3.1 drop-fraction
measurement D6 had preempted: **1.2e-14 — negligible** (2 photons of 2.21M). Combined
with the SANE zero-drop results, H2's final disposition everywhere tested:
instrumented, measured, negligible; methods-section caveat only.

**Two Phase-4-relevant observations from the completed pair (flags, not blockers):**
1. **Cost:** at matched `Ns=1000` / 8 CPUs, the jet arm made 62× the photons (2.21M
   vs 35.7k) and took ~44× the walltime (3h14m vs 4m28s) of the non-jet arm.
   Post-H1-fix MAD jet branches are drastically more expensive than their historical
   (H1-bugged) counterparts — Phase 4 walltime budgeting must come from post-fix
   pilots like this one, not from historical runtimes, and this cost is *with* the D6
   Butcher–Messel speedup already in place.
2. **Magnitude:** the bolometric jet/non-jet ratio on this MAD rh80 pair is **~404×**
   (3.82e43 / 9.46e40) — far above the SANE pair's ~13× (§13 Pair 1). The standing D4
   caveat applies (isotropic totals are not the observable; the SANE camera-plane
   ratio was 2.08×, §7), but this number belongs in the advisor conversation about
   Θe caps and funnel physics before Phase 4 production choices are locked.

---

## Update log

- 2026-07-23: initial version — items 1-4 completed. Phase 2 run A vs B executed;
  produced an unexpected but fully traced result (legacy `with_electrons<3` cut, not a
  jet-mechanism flaw) — see §5. Run C isolated the additive supplement itself at current
  production defaults and found a ~40× luminosity increase, mechanism traced to the
  `1/nₑ` term in low-density high-σ zones combined with the conventionally-disabled
  `Thetae_max` cap — flagged for explicit scientific review, not resolved unilaterally.
- 2026-07-26: advisor input received on the §5 finding; ran a cone-restricted follow-up
  check (`viewing_cone_postprocess.py` against existing Run A/C spectra, M87 viewing
  convention) — see §6. Restricting to the M87 viewing angle does not shrink the ~40×
  excess (55.7× in-cone vs. 43.6× all-sky); flagged as suggestive but not decisive, and
  P4.3 (real IPOLE ray-traced cone comparison) recommended as the next, decisive step.
  D4 remains open.
- 2026-07-26 (same day, follow-up): reran the A-vs-C pair at `Ns=1000` (10× the photon
  budget) to firm up the cone-restricted statistic. All-sky/bolometric ratios held stable
  (35-49× at both `Ns` levels); the cone-restricted ratio did not converge (55.7×→91.1×),
  revealing that this GRMONTY-side proxy is structurally too statistics-starved in the
  cone to resolve cheaply — the M87 cone only spans ~9% of solid angle, so it's a small,
  noisy subsample regardless of total `Ns`. Direction still consistent (in-cone ≥ all-sky
  both times), but magnitude unresolved. This sharpens the P4.3 recommendation from
  "worth doing" to "necessary" — IPOLE's deterministic single-camera-angle ray tracing
  doesn't have this noise floor. P4.3 prep started same day (Task #21).
- 2026-07-26 (same day, second follow-up): ran a real IPOLE rehearsal image (jet-on,
  `thetacam=17`, `freqcgs=228e9`) against the `Ns=1000` Run C dump/`M_unit` — see §7.
  Two source-level gotchas found and corrected before running (`electronModel` numbering
  mismatch vs. GRMONTY; a second `sigma_cut` that would have silently masked the jet-
  supplement zones). Result: IPOLE's ray-traced flux and GRMONTY's cone-restricted flux
  agree to ~2.1x -- same order of magnitude, a reasonable pipeline sanity check. Task #21
  complete.
- 2026-07-26 (same day, third follow-up): ran the jet-off IPOLE counterpart
  (`ipole_rehearsal_A.par`, `sigma_transition=1e6`). Real, ray-traced jet/non-jet ratio at
  the M87 camera angle, 228 GHz: **2.08x** -- sharply lower than every GRMONTY-side
  estimate (35-91x, all isotropic-total or coarse-proxy methods). This is the strongest
  evidence yet that the isotropic 4pi comparison was the wrong quantity, and that the
  jet's actual observationally-relevant contribution is modest, not alarming. D4
  substantially de-risked but not fully closed (single frequency/angle, non-production
  M_unit) -- see caveats in sec 7.
- 2026-07-26 (Phase 1 resumed): implemented H2 instrumentation and L2 cleanup -- see
  sec 8. New `N_track_reject_nu`/`W_track_reject_nu`/`W_superph_made` counters surface
  the mid-flight isnan-nu drop-fraction (by weight, not just count) in both the run
  summary and `/params/diagnostics/*` in the output HDF5; a genuinely unreachable
  `bias < 1.0` branch removed after data-flow tracing confirmed it dead. Built and smoke
  tested cleanly (no compile errors, no crashes); the specific drop-counting increments
  were not observed firing live (this dump/Ns combo didn't reproduce the audit's
  production-scale fragility), verified by code inspection instead -- see sec 8's
  residual caveat. Tasks #18 (H2) and the bundled L2 item both complete.
- 2026-07-26 (P2.5 attempted): found a new, pre-existing bug instead of confirming
  reproducibility -- see sec 9. `with_electrons=5` (and plain `with_electrons=3`, zero
  jet code involved) produced `L=-nan` while reporting `run status: ok`. Full-domain
  scan showed `thetae_func()` itself is 100% finite; 0/44 existing CRITBETA/CRITBETAwJET
  production outputs affected. Tracked as **D5**, flagged for direction before
  continuing rather than open-ended unilateral debugging.
- 2026-07-26 (same day, D5 root-caused and fixed on request): traced the NaN to
  `jnu_thermal()` (`src/jnu_mixed.c`) hitting an `inf * 0.0 = NaN` indeterminate form for
  `Thetae` just above `THETAE_MIN` -- a regime Crit-Beta's `exp(-beta/beta_crit)`
  suppression reaches far more often than R-β's bounded rational `trat` formula, which is
  why only Crit-Beta showed this. Fixed in `jnu_thermal()`, `jnu_ratio_brems()` (its
  `==0` guard couldn't catch a NaN sum), and `record_super_photon()` (broadened from a
  2-field to an 11-field `isfinite` check). Also fixed an unrelated dead-code safety
  check found along the way (`zone_linear_interp_weight()`). Verified: exact reproducer
  now finite, original P2.5 config now finite, R-β regression check unaffected. Full
  mechanism and verification in sec 10. Task #26 complete; D5 resolved; P2.2 (which uses
  `with_electrons=3`) no longer at risk of hitting the same issue.
- 2026-07-26 (P2.5 finished): reran the original P2.5 pair post-D5-fix -- see sec 11.
  Both runs finite (D5 fix holds), but not bit-identical: `L` differed ~62% between two
  runs with the same seed. Root cause is structural, not a bug -- `random.c` gives each
  OpenMP thread its own deterministic RNG stream, but zone-to-thread assignment isn't
  forced deterministic (no `schedule(static)`), so which stream lands on which zone
  varies run to run. Documented, not fixed (wasn't asked for, would change performance
  characteristics). Likely a small-`Ns` effect given other checks this session stayed
  within ~4-15% across a 10x `Ns` change, but not confirmed at higher `Ns` for this
  specific volatile config. Task #22 complete.
- 2026-07-26 (Phase 2 completed): P2.4, P2.1, P2.2 all done -- see sec 12. P2.4:
  quantified H1's real impact, the exponent=0 bug understated L by ~33x. P2.1: found a
  real zone (sigma=2.771, just above sigma_transition) and confirmed GRMONTY's actual
  compiled output (Thetae=1000, hard-capped) exactly matches an independent
  recomputation of the audited formula (raw value 9597.16). P2.2: built baseline
  4a1b1c5 in an isolated git worktree (build-env overrides only, no source changes) and
  compared full Thetae fields -- not bit-identical, but fully explained: (1) baseline's
  unconditional i<9 near-horizon floor is gone from the current tree (this is Finding
  L4, now more completely characterized), affecting both electron models equally; (2)
  Crit-Beta otherwise matches to floating-point noise (7.9e-9), independently
  confirming M1's fix correctly restored baseline's original 1e-3 floor; (3) R-beta
  has a new ~1e-3 floor baseline's R-beta never had -- not previously documented,
  likely intentional (matches IPOLE's convention) but newly surfaced here. Tasks #23,
  #24, #25 complete. Phase 2 is now fully done (P2.1-P2.5 all complete).
- 2026-07-26 (Phase 3 started + full-session re-verification): P3.1 SANE rh80 pair done
  (zero drops both arms, sec 13); MAD rh80 pair submitted as SLURM jobs 778135/778136
  after canceling a first attempt (778072/778073) that had staged binary/parfiles/logs
  on login-node-local /tmp -- invisible to compute nodes -- and used trat_large=20
  instead of the audit's fragile rh80. Same day: an independent source-level
  re-verification of every session change (all git diffs re-read hunk by hunk; K2
  underflow band confirmed analytically at Thetae<~1.4e-3, exactly bracketing the 13
  caught photons; document restructured so this update log sits last). The two staging
  defects above were the only errors found; all physics changes stand as documented.
- 2026-07-26 (P3.1 MAD pair landed -- D6 found): non-jet arm clean (zero drops,
  HDF5 diagnostics validated end-to-end); jet arm fired the instrumented isnan-nu
  path live (4 events, first controlled H2 reproduction) then DIED in the Compton
  electron sampler at Thetae=1000 (the hard cap the post-H1-fix jet supplement
  saturates) -- fail_sampling() exit(-1), one photon kills the run. The H1 exponent
  bug had been masking this; Phase 4 jet configs are blocked on it. Recorded as D6
  (sec 13); P3.2 decided: SANE = zero-drop non-issue, MAD+jet superseded by D6.
- 2026-07-26 (D6 fixed per user decision, option b): analytic deep-KN sampling in
  src/compton.c -- new Gamma(2,Thetae)-proposal electron sampler (Thetae>=100 gate +
  exhaustion fallback, so the electron-sampling exit(-1) is gone entirely on MJ
  builds) and a Butcher-Messel branch in sample_klein_nishina (k0>=1). Both
  prototype-validated against the legacy samplers (all |z|<2.6) and against ground
  truth; ~2e6x efficiency gain at the exact stall point. Ground-truth regression
  tests added to --run-tests (which also fixes the harness's latent uninitialized-RNG
  segfault). Cluster validation chain submitted: tests 778171 -> gated MAD rh80
  rerun 778172 (jet, the config that died as 778136) + 778173 (non-jet). See sec 14.
- 2026-07-27 (D6 validation round 2 COMPLETED -- D6 closed, Phase 3 closed): tests job
  778184 PASSED rc=42 (all six deep-KN ground-truth checks, incl. the corrected
  log-grid k0=1e5 case and the exact 778136 stall regime). Non-jet rerun 778186 on the
  D6 binary: statistically unchanged (L=9.46e40 vs 9.32e40), zero drops. Jet rerun --
  the config that died as 778136 -- COMPLETED as 778206 (first attempt 778185 was
  walltime-killed at 30m while progressing normally; identical converged bias): 3h14m,
  L=3.82e43, 2.21M made / 2.42M scattered, bias ratio 1.09, and the long-sought
  MAD+jet drop fraction = 1.218e-14 (2 events; stderr and HDF5 diagnostics agree). H2
  final disposition: negligible everywhere tested, methods-section caveat only. Phase
  4 is now unblocked code-side (its own go/no-go gate stands). Two flags for Phase 4
  planning recorded in sec 14: post-H1-fix jet arms cost ~44x matched non-jet
  walltime, and the MAD bolometric jet/non-jet ratio is ~404x (D4-style caveat
  applies -- isotropic totals are not the observable).

