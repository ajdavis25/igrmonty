# GRMONTY Jet Implementation: Paper-Readiness Plan

Date created: 2026-07-23
Source audit: `docs/audits/2026-07-23_jet_electron_temperature_audit.md`
Owner: ashton davis
Status: **Phase 0/1 in progress** (this document is a living checklist — update the boxes as items land)

This plan sequences the work needed to move the jet-electron-temperature implementation
(`with_electrons ∈ {4,5}`, i.e. RBETAwJET/CRITBETAwJET) from "correct in its core physics but
incompletely validated" to results you can put in a paper. It is organized so cheap,
low-risk, non-ambiguous work happens first, physics judgment calls are made explicitly
(not silently assumed by whoever is executing this plan), and expensive cluster compute is
gated behind everything else.

Every item below is tagged with the audit finding it resolves (H1/H2/H3 = High severity,
M1-M3 = Medium, L1-L4 = Low; see the audit for full evidence/citations).

---

## Phase 0 — Blocking decisions (zero compute; nothing downstream should proceed without these)

- [x] **D1 (resolves H1) — DECIDED 2026-07-23.** `auto_munit_bracket.py:74`'s
  `WJET_DEFAULTS["constant_beta_e0_exponent"]` was `0.0`. User's recollection was that
  IPOLE itself used 0 for this exponent; direct source citation shows otherwise —
  IPOLE's `model/iharm/model.c` sets `constant_beta_e0_exponent = 1.0` (confirmed
  independently in both the `dev`-branch/`ipole+e-` copy and the separate `origin/master`
  lineage of the same tracked repo — see audit §4/§7). Decision: **reverted to 1.0**,
  matching both IPOLE and GRMONTY's own compiled default. Applied in
  `auto_munit_bracket.py`.
- [x] **D2 (resolves M1) — DECIDED 2026-07-23.** Decision: **match IPOLE's flat 1e-3**
  for all electron models. Applied: `model.c:552` `crit_floor` changed from `3.e-2` to
  `1.e-3`; `model.h:7` comment corrected to describe the unified floor instead of the
  stale/incorrect "0.3 for Crit-Beta" claim.
- [ ] **D3 (resolves M2, informational).** Confirm that the `jet_thetae` hard override
  taking precedence over the additive `constant_beta_e0` supplement (the `else if` at
  `model.c:678`) is the intended semantics. (A clarifying comment has been added regardless
  — see Phase 1 — but flag here if the *behavior* itself should change, not just the
  documentation of it.)

---

## Phase 1 — Cheap code fixes (no recompile/run required, or read-only against existing data)

- [x] **M2** — precedence-clarifying comment added at `model.c:673-685` (done this session).
- [x] **M3** — provenance-tagging script (`tools/tag_wjet_provenance.py`) written and run
  against all existing `*wJET*.h5` outputs (done this session). Results, full detail in
  `igrmonty_outputs/m87/_qa/wjet_provenance_report.csv`:
  - **35** wJET outputs total.
  - **8** files (all under `logs/5e4_test`-era outputs, `sigma_transition=1`,
    `jet_sigma_cut=-1`) are pre-fix: additive-supplement-only, and correctly carry
    `constant_beta_e0_exponent=1.0`.
  - **26** files (the `rh20`/`rh160` production-style batch, `sigma_transition=2`,
    `jet_sigma_cut=10`) are post-fix: hard-override reachable — **and every one of them
    carries `constant_beta_e0_exponent=0.0`** (Finding H1's blast radius is precisely
    this set — i.e., essentially the entire corpus anyone would consider "the real"
    wJET results today).
  - 1 file under `test_scrap/` is a one-off with its own settings.
  - 0 filename/metadata mismatches, 0 read errors.
- [x] **H1 fix** — applied 2026-07-23, see D1 above.
- [x] **M1 fix** — applied 2026-07-23, see D2 above.
- [x] **H2 instrumentation** — DONE 2026-07-26. Added `N_track_reject_nu` (count) and
  `W_track_reject_nu`/`W_superph_made` (weight lost / weight made, so a *weight fraction*
  is reportable, not just an event count) for the mid-flight isnan-nu drops in
  `track_super_photon()` — distinct from the pre-existing `N_init_reject_nu`, which
  counts a different (photon-*initialization*-time) rejection and was already being
  counted but not surfaced to HDF5. Both now print in the run summary (`utils.c`'s
  `summary()`, alongside the existing init-reject line) and are written to
  `/params/diagnostics/*` in the output HDF5 (`model.c`). Full detail, code, and
  verification approach in `docs/2026-07-23_jet_implementation_changes.md` §8.
- [x] **L2 cleanup** — DONE 2026-07-26, bundled with H2's compile/test cycle as planned.
  Removed the dead `if (!isfinite(bias) || bias < 1.0)` branch at (then-)
  `track_super_photon.c:509` after confirming via data-flow tracing that `bias` is
  unconditionally sanitized to ≥1.0 earlier in the same call and never reassigned before
  this check — see §8 for the exact reasoning.

---

## Phase 2 — Cheap validation runs (small `Ns`, short wall-clock; requires compiling + running the `grmonty` binary)

**Gate: do not start until Phase 0 decisions have landed in code, and only after an explicit
go-ahead — these still touch the shared cluster/login node even at small scale.**

- [x] **P2.3 (partial) — jet-disabled-by-construction, run A vs B.** Done 2026-07-23.
  Result was **not** bit-identical as hypothesized; traced to a pre-existing, non-jet
  legacy cut (`with_electrons<3 && sig_unscaled>1`), not a jet-mechanism flaw — see
  `docs/2026-07-23_jet_implementation_changes.md` §5 for full detail. Finding L4's
  practical severity revised upward as a result (empirically ~6-16% effect, not
  negligible).
- [x] **P2.1-adjacent — isolated additive-supplement test, run C.** Done 2026-07-23,
  using the exact current (post-H1-fix) `WJET_DEFAULTS`. **Found a ~37-44× luminosity
  increase** from the additive supplement alone, traced to the formula's `1/nₑ` term in
  low-density high-σ zones combined with `Thetae_max` being conventionally set to `1e100`
  (effectively disabled) in every parfile in this repo, leaving only the hardcoded
  `THETAE_HARD_MAX=1e3` as a ceiling. **Not resolved as a code defect — flagged as an
  open scientific-judgment question** (see D4 below) since the formula matches IPOLE
  exactly; whether this magnitude is intended jet physics or an artifact of the chosen
  `sigma_transition`/`Thetae_max` convention needs your call before further tuning spend.
- [ ] **D4 (new, blocking further Phase 2/4 spend) — STILL OPEN, now with advisor input +
  a cone-restricted check.** Is a ~40× jet-supplement luminosity contribution at
  `sigma_transition=2.0`, `constant_beta_e0=0.1` physically expected for this dump, or
  should `sigma_transition`/`Thetae_max` be reconsidered before any further M_unit tuning
  is run against the current `WJET_DEFAULTS`?
  - **Advisor input (2026-07-26):** confirmed the mechanism itself is standard physics
    ("Sigma is typically highest in jet regions which are often low density due to
    centrifugal force barrier to particles") and flagged that isotropic/4π totals can
    overstate what's actually observed ("jet emission is typically subdominant due to
    extreme relativistic beaming effects") — i.e., the right comparison is angle-resolved
    flux at the actual viewing geometry, not the 4π total Run C reported.
  - **Follow-up check run same day:** applied `tools/viewing_cone_postprocess.py` (no
    recompile, operates on existing HDF5 spectra) to the Run A/C pair at this project's
    M87 convention (`thetacam=163°`→folds to 17°, ±10° cone). Restricting to that cone
    does **not** shrink the excess — cone-restricted ratio **55.7×** vs. all-sky
    **43.6×** (consistent with the printed bolometric 36.6×). See
    `docs/2026-07-23_jet_implementation_changes.md` §6 for full numbers and caveats
    (Ns=100 statistics are thin; the tool is an azimuth-averaged polar-angle proxy, not
    true ray-traced beaming — see caveats below).
  - **Important nuance:** M87 is conventionally modeled near-pole-on (~17° from the jet
    axis, not edge-on), which is the geometry where relativistic beaming would be expected
    to *enhance* apparent jet-aligned emission toward the observer, not suppress it — so
    the advisor's general caveat may not straightforwardly apply here the way a first read
    suggests.
  - **`Ns=100→1000` rerun (2026-07-26):** all-sky/bolometric ratios stable (35-49× at
    both levels); cone-restricted ratio did *not* converge (55.7×→91.1×) — the M87 cone
    is too small a solid-angle subsample (~9%) to resolve cheaply via this GRMONTY-side
    proxy, regardless of `Ns`. Direction still consistent (in-cone ≥ all-sky both times).
  - **IPOLE rehearsal image run (2026-07-26):** Task #21 (P4.3 prep) completed; first
    real ray-traced IPOLE image at the M87 camera angle (`thetacam=17`, `sigma_transition
    =2.0`, matching Run C) produced `Fnu(228 GHz) = 2.75e-4 Jy`, ~2.1× from GRMONTY's
    cone-restricted value — same order of magnitude, a clean pipeline sanity check. Two
    IPOLE-side gotchas found and corrected in the process (electronModel numbering
    doesn't match GRMONTY's with_electrons; a second `sigma_cut` that defaults to masking
    the exact zones being tested) — see
    `docs/2026-07-23_jet_implementation_changes.md` §7 for full detail.
  - **Jet-off IPOLE counterpart run (2026-07-26):** real ray-traced jet/non-jet ratio at
    the M87 camera angle, 228 GHz: **2.08×** — sharply lower than every GRMONTY-side
    estimate (35-91×, all isotropic-total or coarse-proxy methods). Strongest evidence
    yet that the isotropic 4π comparison was the wrong quantity and the jet's actual
    observationally-relevant contribution is modest. Full detail and caveats in
    `docs/2026-07-23_jet_implementation_changes.md` §7.
  - **Status: substantially de-risked, not fully closed.** Caveats: single frequency
    (228 GHz) and single viewing angle (17°), not a spectrum or angular sweep; still the
    non-production `M_unit=1.83e27` test configuration, not a tuned production run. A
    multi-frequency version of this same check (86/230/345 GHz, ~110s/image) would firm
    this up further. Recommendation: safe to resume Phase 2 (P2.1/P2.2/P2.4/P2.5) and
    plan toward Phase 4 without D4 as a hard blocker, but flag the single-frequency/angle
    caveat explicitly if `M_unit` tuning decisions get made before a fuller check runs.
- [x] **P2.1 — Single-cell comparison vs. IPOLE. DONE 2026-07-26.** Found a real zone
  (`i=50,j=5,k=0`, σ=2.771, just above `sigma_transition=2.0`) via a temporary
  `GRMONTY_DEBUG_ZONE=i,j,k` hook. GRMONTY's actual compiled code printed
  `Thetae=1000.000000000` there (`with_electrons=4`, exponent=1.0). Independently
  recomputed the audited `constant_beta_thetae()`/IPOLE-`electronModel==5` formula by
  hand from the same printed `B`/`Ne` — raw (uncapped) value **9597.16**, correctly
  saturating at `THETAE_HARD_MAX=1000` — **exact match** to GRMONTY's real output, not
  just the formula on paper. Full numbers: `docs/2026-07-23_jet_implementation_changes.md`
  §12.
- [x] **P2.2 — Non-jet regression. DONE 2026-07-26 — NOT bit-identical, with a real,
  well-explained reason.** Built baseline `4a1b1c5` in an isolated `git worktree`
  (needed `HDF5_DIR`/`CC`/`CFLAGS` overrides to build on this machine — pure
  build-environment fix, no source changes). Dumped full `Thetae` fields for
  `with_electrons=2` and `=3` from both trees (a temporary zone-loop hook, no Monte
  Carlo/RNG involved, so this is a clean deterministic-field comparison). **Not
  bit-identical for either model** — but fully explained, not a mystery:
  1. **A hard floor on the innermost 9 radial zones (`i<9`) that baseline applied
     unconditionally (`if (sig>1 || i<9) Thetae=SMALL`) is gone from the current tree**
     — replaced by a sigma-only check gated to `with_electrons<3` (this is Finding L4
     from the audit, now more fully characterized: it didn't just get scoped to
     `with_electrons<3`, the `i<9` clause was dropped entirely). Accounts for 100% of
     the `i<9` band (147,456 zones, ~3.1% of the domain) in both models.
  2. **Crit-Beta (`with_electrons=3`), excluding `i<9` and the sigma-cut: differences
     are floating-point noise** (~8e-9, i.e. functionally identical) — this is a clean,
     independent confirmation that this session's M1 fix (reverting `crit_floor`
     3e-2→1e-3) correctly restored baseline's original behavior (baseline's own
     Crit-Beta already had `fmax(1e-3, ...)` — the 3e-2 was itself a regression, and
     M1 correctly reverted it).
  3. **R-β (`with_electrons=2`), excluding `i<9` and the sigma-cut: a real, new
     difference** (~9e-4 max, consistent with a floor operation) — the current tree's
     R-β picked up an `rb_floor=1e-3` that baseline's R-β never had at all (baseline's
     R-β branch has no `fmax`/floor, only the final harmonic `Thetae_max` soft-cap).
     Not previously documented anywhere. `1e-3` matches IPOLE's own "secret floor"
     convention (per the original audit), so this is likely intentional/correct, just
     newly surfaced by this comparison rather than something to fix.
  Full numbers and breakdown: `docs/2026-07-23_jet_implementation_changes.md` §12.
- [x] **P2.3 — Jet-disabled-by-construction check.** Already done 2026-07-23 (Run A vs
  B) — see Phase 1 log above; found the same `i<9`-adjacent legacy-cut story (Finding
  L4) empirically before P2.2 explained its exact mechanism.
- [x] **P2.4 — `constant_beta_e0_exponent` sweep. DONE 2026-07-26.** {0, 0.5, 1.0} at
  fixed everything else (`with_electrons=4`, `Ns=1000`, additive-supplement-isolated):
  `L` = 2.32e38 / 7.14e39 / 7.74e39 erg/s. **H1 mattered enormously** — the buggy
  `exponent=0.0` value understates `L` by **~33×** relative to the correct `1.0`. Most
  of the effect (30.7× of the 33×) already appears by `exponent=0.5` — highly
  nonlinear near the low end, consistent with the term's `B^exponent` structure.
  Full numbers: `docs/2026-07-23_jet_implementation_changes.md` §12.
- [x] **P2.5 — Deterministic smoke test. DONE 2026-07-26 (with a real finding, not a
  clean pass).** First attempt found and fixed D5 (see below) instead of confirming
  reproducibility. Re-ran post-fix: both runs finite (D5 fix holds), but **not
  bit-identical — `L` differed by ~62% between two runs with the same seed**
  (`6.74e38` vs `1.09e38` erg/s). Root cause: `src/random.c` gives each OpenMP thread
  its own deterministically-seeded RNG stream (reproducible per thread), but the
  photon-generation loops don't force a fixed zone-to-thread assignment
  (no `schedule(static)`), so which stream lands on which zone varies run to run. Not a
  bug, a structural property of the parallel design — not fixed, not proposed as a fix,
  just documented. Likely a small-`Ns` effect (this test used `Ns=100`; R-β checks
  elsewhere in this plan stayed within ~4-15% across a 10× `Ns` change, not 60%+) but
  **not confirmed at higher `Ns` for this specific config** — worth checking before
  trusting a single production run's numbers for a volatile config like
  `with_electrons=5` + a live `jet_thetae` override. Full detail:
  `docs/2026-07-23_jet_implementation_changes.md` §11.

- [x] **D5 (new, found during P2.5) — Crit-Beta produced a silent NaN spectrum. ROOT
  CAUSED AND FIXED 2026-07-26.** `with_electrons=5` (Crit-Beta+jet) at real production
  parameters (`jet_thetae=50`) printed `L = -nan erg/s` while reporting
  `run status: code=2 label=ok` — no crash, no in-flight warning, just a garbage spectrum
  reported as fine. Isolated to **plain Crit-Beta** (`with_electrons=3`, zero jet
  parameters active) — **pre-existing, not caused by any jet-implementation work**,
  though it directly affected `with_electrons=5`.
  **Root cause:** `jnu_thermal()` (`src/jnu_mixed.c`, documented "good for `Thetae > 1`")
  hits an `inf * 0.0 = NaN` indeterminate form when `Thetae` is barely above
  `THETAE_MIN=1e-3` — both `K2_eval(Thetae)` (denominator) and `exp(-xp1)` underflow to
  exactly `0.0` in that regime. That `NaN` defeated `jnu_ratio_brems()`'s `==0` zero-guard
  (a NaN comparison is always `false` in IEEE754) and poisoned `ph->ratio_brems`, which
  multiplies into every spectrum accumulation. **Crit-Beta reaches this narrow `Thetae`
  band far more often than R-β** because `exp(-beta/beta_crit)` suppresses `Thetae`
  toward the floor much more aggressively than R-β's bounded rational `trat` formula —
  this is why the bug is real but R-β never showed it.
  **Fixed in 3 files** (`jnu_mixed.c`'s `jnu_thermal()` and `jnu_ratio_brems()`,
  `model.c`'s `record_super_photon()`), verified with the exact reproducer (`L` went
  `-nan` → finite), the original P2.5 config (`with_electrons=5`, `jet_thetae=50`: now
  `L=7.36e38`, finite), and an R-β regression check (unaffected, values consistent with
  earlier runs). Also fixed an unrelated but genuinely broken dead-code safety check
  found along the way (`zone_linear_interp_weight()` in `utils.c`). Full mechanism,
  fix, and verification: `docs/2026-07-23_jet_implementation_changes.md` §10.
  **0/44 existing CRITBETA/CRITBETAwJET production outputs showed NaN** before this fix
  (checked in §9) — not evidence of harm in past data, but this fix should be present in
  any Crit-Beta production run going forward. **P2.2 (`with_electrons=3`) is now safe to
  run** — it was flagged as likely to hit this same issue; that's resolved.

---

## Phase 3 — Quantify H2 before deciding whether it needs a real fix

- [x] **P3.1 — DONE 2026-07-26** (two matched pairs; full detail in
  `docs/2026-07-23_jet_implementation_changes.md` §13):
  - SANE `Sa-0.5_4000` rh80, `Ns=5000`: **zero drops both arms** (third consecutive
    zero-drop result on SANE at any setting).
  - MAD `Ma+0.94_4000` rh80, `Ns=1000`, via sbatch (jobs 778135/778136): non-jet arm
    clean, zero drops (log + HDF5 diagnostics agree — instrumentation validated
    end-to-end). **Jet arm: 4 live isnan-nu drops (first instrumented reproduction of
    Finding H2), then died in the Compton electron sampler** — see D6 below.
    **Post-D6 completion (2026-07-27, job 778206): the jet arm runs to completion;
    drop fraction measured 1.218e-14 (2 events of 2.21M photons) — negligible.**
- [x] **P3.2 — DECIDED 2026-07-26 by P3.1's outcome; finalized 2026-07-27:** for SANE
  configs, measured drop fraction is exactly 0 → methods-section note only, no fix
  needed. For MAD+jet post-H1-fix configs the question was temporarily superseded by
  D6 (the run died before a fraction could be measured); with D6 fixed, the measured
  fraction is **1.218e-14** (job 778206) — likewise a methods-section note. No
  further H2 fix warranted anywhere.
- [x] **D6 — CLOSED 2026-07-27 (option b implemented + cluster-validated).**
  Implementation covers *both* deep-KN legs — the fatal electron-sampler stall AND
  the ~1e5-draws-per-event `sample_klein_nishina` inefficiency (a likely contributor
  to historical wJET walltimes) — with ground-truth regression tests added to
  `--run-tests`. Validation (round 2, after round 1 caught a grid-integration bug in
  the *test harness itself*): tests job 778184 PASSED → non-jet rerun 778186
  statistically unchanged with zero drops → jet rerun **778206** (the exact config
  that died as 778136) COMPLETED in 3h14m with finite `L=3.82e43`, no stalls, bias
  ratio 1.09, and measured isnan-nu drop fraction **1.218e-14**. Full derivation,
  prototype numbers, test design, and completed-pair tables:
  `docs/2026-07-23_jet_implementation_changes.md` §14. Original finding record below.
  **Compton electron sampler stalls fatally at the Θe cap.** `sample_electron_distr_p()` (`src/compton.c:270`) rejection
  sampling has vanishing acceptance at `Thetae=1000` (= `THETAE_HARD_MAX`, which the
  post-H1-fix jet supplement *saturates* in MAD funnel zones — raw Θe≈9600 capped to
  1000, see §12b of the changes doc) against deep-Klein-Nishina photons
  (`K~1e7`, `σ_KN/σ_T~7e-6`); after 10M attempts `fail_sampling()` calls `exit(-1)` —
  **one un-sampleable photon kills the whole run** (observed live, job 778136).
  Crucially: the 26 existing wJET production outputs never hit this because the H1
  exponent bug kept Θe far below the cap — **the H1 fix un-masks D6**, so Phase 4 as
  currently scoped would fail on MAD jet branches. Options needing a decision:
  (a) graceful degradation — drop the photon with H2-style count/weight accounting
  instead of exiting; (b) an analytic deep-KN sampling branch for Θe≳100; (c) a
  deliberate, physics-motivated lower `Thetae_max` (parfile-level, no code change —
  ties into the D4/`Thetae_max` discussion and the advisor thread); (d) some
  combination. Recommendation: (a) is the minimal safe unblock and makes the failure
  *quantifiable* via the same weight-fraction machinery; (c) is worth raising with the
  advisor since Θe=1000 electrons (γ~5000) in the funnel is itself a physics-plausibility
  question.

---

## Phase 4 — Production reruns (expensive — separately gated)

**Gate: explicit confirmation required before any SLURM submission. This is real cluster
allocation and multi-hour-to-multi-day wall-clock time (cf. the 20-day batch in the prior
self-audit) — needs its own go/no-go with agreed scope (which branches, how many, expected
walltime) before anything is queued.**

- [ ] **P4.1** — Rerun the M_unit tuning campaign for wJET branches under the final,
  Phase-0-decided parameter set. Do not mix with pre-fix outputs (M3) — regenerate so every
  wJET `.h5` in the eventual paper corpus comes from one consistent configuration.
- [ ] **P4.2** — Resolve the `munits_tuning_history.csv` schema drift (§8 of the audit —
  parsing it during the audit produced impossible values like `converged=1.000e+06`,
  indicating column misalignment across the file's 8-month history) *after* the current
  in-progress campaign finishes — do **not** touch this file while it's still being
  actively written.
- [ ] **P4.3** — Apply the existing `tools/viewing_cone_postprocess.py` to at least one
  wJET output per (spin, dump) and produce the angle-matched IPOLE comparison (resolves H3)
  — this has never been done for a jet-enabled spectrum; it is the centerpiece validation,
  not the 4π M_unit convergence work.

---

## Phase 5 — Paper packaging

- [ ] Methods section: final `sigma_transition`/`constant_beta_e0(_exponent)`/`jet_sigma_cut`/
  `jet_beta_cut`/`jet_thetae` values, with justification.
- [ ] Cite the symbolic equivalence to IPOLE (audit §6/§7) *and* the P2.1 numeric single-cell
  check as validation evidence — not the 4π M_unit-tuning convergence, which this
  repository's own prior reports already warn is not a substitute.
- [ ] Include the P4.3 cone-restricted IPOLE comparison figure.
- [ ] State the P3.1/P3.2 drop-fraction result as a caveat if non-negligible.
- [ ] Consider adding a `test_jet_scalings()`-style regression test to `tests.c` (mirroring
  the existing `test_pair_scalings()` for positrons) so this doesn't silently regress again.

---

## Change log

- 2026-07-23: Plan created. M2 comment and M3 provenance script completed same day
  (see this file's Phase 1 checkboxes and the accompanying session transcript for exact
  diffs/output). D1/D2 decisions requested from the user, not yet resolved as of this
  writing.
