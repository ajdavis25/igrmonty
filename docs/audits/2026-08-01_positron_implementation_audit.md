# Positron (pair-plasma) implementation audit — 2026-08-01

Audit of the `positron_ratio` implementation on branch `positrons` (working tree at
a6e1b40) against the IPOLE reference it was backported from
(`/work/vmo703/ipole/ipole/src/`, ARR positron fork) and against the thermal
pair-brems literature (Svensson 1982; Stepney & Guilbert 1983; Straub+ 2012;
Narayan & Yi 1995). Companion record to the user's implementation report at
`/work/vmo703/_reports/positrons_20260224/implementation_report.md` (823dca8).

## Verdict

The implementation is structurally sound. The composition convention, the
(1+2f) synchrotron/absorption/Compton scaling, and the photon-generation
bookkeeping are all correct, mutually consistent, and match the IPOLE reference
where the physics overlaps. One real bug was found, in the channel added last:
the **opposite-sign (e+e-) bremsstrahlung coefficient is understated by
sigma_T/r_e^2 = 8*pi/3 ~ 8.38x in the Thetae < 1 branch** (Finding PP-1).
Everything else audited clean.

## Convention (verified consistent everywhere)

`Ne` from the fluid model is baseline ion-associated density n_i; with
`positron_ratio = f`:

    n_+ = f n_i,   n_- = (1 + f) n_i,   n_lep = (1 + 2f) n_i

Identical to the IPOLE fork's convention ("positronRatio ... fraction times the
initial number density of electrons added in positrons. Both positrons and
electrons are added now" — model_radiation.c:128), where the unpolarized
transfer coefficients are scaled `jnuinv *= (1+2f)`, `knuinv *= (1+2f)`
(model_radiation.c:646-650). Pairs are thermal, co-moving, and share Thetae
(isothermal pair addition — same convention as the reference; u_e is *not*
re-partitioned over the enlarged lepton population, and Ne_unit stays baryonic,
`model/iharm/model.c:1288`). Downstream rule (README, auto_munit_bracket.py):
keep M_unit baryonic, never pre-scale by (1+2f). Parfile accepts both
`positron_ratio` and IPOLE's `positronRatio` spelling (src/par.c:126-128);
value is validated finite and >= 0 at startup (src/main.c:119-132) and written
to output HDF5 under both keys (provenance).

## Verified-correct scaling chain

Every density-dependent path was traced; all call sites pass **raw n_i** and
scaling is applied exactly once, inside the microphysics functions:

| Path | Location | Scaling | Check |
|---|---|---|---|
| Synch emissivity (MJ/kappa/powerlaw) | jnu_mixed.c:284,338,372 | x(1+2f) internal | correct; e+ emit identically |
| Integrated emissivity int_jnu_* (all EDFs + brems) | jnu_mixed.c:423,485,502,517 | x(1+2f) internal | consistent with jnu |
| Absorption, MJ branch | radiation.c:277-283 | Kirchhoff j/B_nu with **raw Ne** | no double count (Ne_lep at :190 unused by this branch) |
| Absorption, kappa/powerlaw branches | radiation.c:238,265 | explicit Ne_lep | single scaling; Kirchhoff-consistent with their scaled j's |
| Compton scattering opacity | radiation.c:170-175 | nu * kappa_es * (n_lep * MP) | kappa_es = sigma_hot/MP (radiation.c:296) so MP cancels: = sigma_hot * n_lep. Extension beyond IPOLE (no Compton there); correct since KN cross-section is charge-even |
| Scattering electron sampling | compton.c | none needed | e+/e- same MJ distribution at shared Thetae, same KN kernel |
| Photon spawn: weight table | utils.c:242-293 | int_jnu(raw Ne) | spawn rate and photon weight use the same scaled emissivity |
| Photon spawn: per-zone counts | utils.c:302-373 (init_zone) | int_jnu(raw Ne) | consistent |
| Photon spawn: angular sampling | utils.c:531+ | jnu(th)/jnu(pi/2) rejection | density cancels in ratio |
| Scattering bias | model/iharm/model.c:1358 | bias_norm = <Thetae^2> | density-free; importance sampling unaffected by f |
| f = 0 limit | all of the above | helpers return 1x, 1x, 0 | reduces exactly to baseline (P2.2 regression confirmed independently) |

Brems density structure (jnu_mixed.c:227-230): e-i term `n_i * n_lep`
(e+ brems off ions Born-identical to e-), same-sign term `n_-^2 + n_+^2`,
opposite-sign term `n_- * n_+`. Structure correct; f=0 reduces to Straub's
`n^2` forms exactly.

## Finding PP-1 (bug, moderate): e+e- brems NR coefficient missing sigma_T/r_e^2

`bremss_pair_coeffs` (jnu_mixed.c:250-277) builds the three dimensionless
coefficients. Straub+/NY95 conventions differ **per term**: the e-i rate
carries a SIGMA_THOMSON prefactor while both lepton-lepton rates carry r_e^2
(jnu_mixed.c:227-230 preserves this split). The Svensson asymptote the code
comment cites — e+e- brems -> 2*sqrt(2) x e-i in the NR limit — is a
**rate-level** statement. Placing `Fee_opp = 2*sqrt(2)*Fei` directly into the
r_e^2-prefactor slot therefore drops the conversion factor
sigma_T/r_e^2 = 8*pi/3 ~ 8.3776.

Two independent confirmations (numeric check, ipole_venv python, 2026-08-01):

1. **Asymptote:** coded NR rate ratio q_opp/q_ei (per density product)
   = 2*sqrt(2) * r_e^2/sigma_T = **0.338**, vs Svensson's 2*sqrt(2) = 2.828.
2. **Continuity:** at the Thetae = 1 formula switch, Fei is continuous to
   0.02% and Fee_same to 0.4% (designed to be patched there), but Fee_opp
   jumps x**8.387** — numerically equal to 8*pi/3. Restoring the factor makes
   the NR side meet the (correct) relativistic side `2*Fee_same` to 0.11%.

The relativistic branch is correct as written: cross-sections converge at high
energy and the factor 2 vs Fee_same is the identical-particle double-counting
bookkeeping (same-sign F absorbs the 1/2; distinct-species product has none).

**Impact:** only f > 0 runs, only Thetae < 1 zones, only the brems component.
At f = 1 in the NR limit the correct e+e- channel is comparable to e-i
(47.4 vs 25.1 in units of n_i^2 r_e^2 alpha m_e c^3 F_ei), so total brems
emissivity from cold zones is understated by up to ~2.4x — relevant to X-ray
brems predictions of pos1 models, irrelevant to mm synchrotron/IC. f = 0
results completely unaffected.

**Fix (one line)**, jnu_mixed.c:264:

    Fee_opp_local = 2. * sqrt(2.) * (8. * M_PI / 3.) * Fei_local;

**Status: FIXED + cluster-validated 2026-08-01.** Staged build+test jobs:
781662 (first round) confirmed the fix but exposed a 0.26% constants-rounding
subtlety in the new test's expectation — jnu_bremss derives r_e from a
locally-rounded e_charge = 4.80e-10, so (8 pi/3) r_e^2 differs from
constants.h SIGMA_THOMSON by 0.26% and the measured NR ratio is
2.820886815852... (matches the constants-exact prediction to 9e-13). Test
re-anchored to the implemented identity in the code's own constants (tight,
1e-9) plus the physical 2*sqrt(2) at 1% tolerance; job 781663 passed the full
suite rc=42 in 51 s.

## Finding PP-2 (test gap, minor)

`check_pair_brems_channel` (src/tests.c) asserts only that the measured brems
ratio *exceeds* the minimal no-opp-channel model — a presence test — so it
passes with the channel 8.4x low. After fixing PP-1, tighten it to assert
(a) the NR rate-level ratio fee_opp/fei -> 2*sqrt(2) within tolerance
(prefactor-aware), and (b) Fee_opp continuity across Thetae = 1 to ~1%,
matching the continuity the other two coefficient pairs already exhibit.

## Finding PP-3 (documentation, minor)

The frozen 2026-02-24 implementation report states the opposite-sign channel is
deliberately absent ("minimal model"). f72bbc1 (2026-04-07) superseded that by
adding the channel + channel-presence tests. The report is accurate for its
date; this audit doc is the current record. The report's sweep table
(flux x1.64/x2.94 and tau_scatt x2.06/x2.97 at f = 0.5/1.0 vs the x2/x3
lepton factors) is consistent with self-absorbed transfer + MC noise and needed
no correction.

## Production-corpus note

The 14 campaign logs built at e6ede55 (including the final positron branch,
MAD_RBETA a+0.94 t5000 pos1) had the correct (1+2f) synchrotron + Compton
scaling and the e-i + same-sign brems terms, but no e+e- brems channel at all,
and retained the old sample-electron stall kluge (silently halves Thetae after
1e7 rejection attempts — replaced by fail_sampling in f72bbc1, then by the
deep-KN sampler in a6e1b40). The 42 logs at 0bfdb56 carried the PP-1
understated (not absent) channel. Neither changes the corpus verdict
(§15 of docs/2026-07-23_jet_implementation_changes.md) — pos1 spectra were
already non-paper-usable on independent grounds — but Phase 4 pos1 reruns
should carry the PP-1 fix so the pair sweep is run against the intended
Svensson physics.

## Side observation: reference IPOLE fork V-coefficient scaling (not igrmonty)

For the polarized coefficients the reference fork scales jV, aV, rV by
`(1 - f/(1+f)) = 1/(1+f)` (model_radiation.c:134,138,141). Under the fork's
own additive convention, the circular/rotation coefficients scale as
(n_- - n_+)/n_i = ((1+f) - f) = **1** — unchanged — because the base
coefficient is proportional to fluid n_e = n_i. The 1/(1+f) factor looks like
(n_- - n_+)/n_- applied to an n_i-normalized quantity, i.e., a factor (1+f)
too small (2x suppression of Stokes V and Faraday rotation at f = 1). Does not
affect igrmonty (unpolarized; only jnuinv/knuinv scaling was ported, and that
part is correct). Flag to Richard/Angelo before any polarized pos1 IPOLE images
are used in the paper.

## References

Svensson 1982, ApJ 258, 335 (pair brems asymptotes); Stepney & Guilbert 1983,
MNRAS 204, 1269; Narayan & Yi 1995, ApJ 452, 710; Straub+ 2012, A&A 543, A83
(coefficient conventions followed by jnu_bremss); Emami+ 2023, ApJ 950, 38
(IPOLE positron treatment).
