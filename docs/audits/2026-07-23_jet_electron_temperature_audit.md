# GRMONTY Jet-Electron-Temperature Implementation: Read-Only Audit

## 1. Executive verdict

**Correctness classification:** Correct in its core physics but incompletely validated.

**Numerical/letter grade:** 78/100 (C+).

**Confidence:** Medium.

**Summary:** GRMONTY's R‑β and Crit‑β electron-temperature formulas, and the additive constant‑β<sub>e0</sub> jet‑temperature supplement gated on `sigma_transition`, are algebraically identical to the IPOLE reference (verified by direct side-by-side quotation), and propagate cleanly and singularly into emissivity, absorptivity, and Compton scattering with no recomputation found anywhere in the chain. Non-jet models are structurally unreachable from any jet code path. However, the implementation adds a second, IPOLE-has-no-equivalent "hard jet-region override" mechanism whose currently-deployed production default (`constant_beta_e0_exponent=0.0`) measurably changes the physics away from both the IPOLE reference and GRMONTY's own compiled-in default; the jet code path suffers materially more photon-frequency failures than the non-jet path; no viewing-angle-matched comparison against IPOLE exists for any jet-enabled spectrum; and the live tuning-history data show the jet-enabled munit tuning campaign is still in progress as of the day of this audit, with no reconciled evidence of converged jet branches under the currently active parameter set. None of this invalidates the core equations, but none of the existing jet-enabled `.h5` outputs should be treated as paper-ready until the items in §11 are resolved.

---

## 2. Scope, provenance, and audit limitations

This audit was conducted with zero write actions during the investigation itself. All claims below are sourced from direct file reads, `git show`/`diff`/`log` (read-only, `GIT_OPTIONAL_LOCKS=0`), and `diff`/`awk`/`python3 -c` run purely as read-only inspectors over existing files (no files were created during the investigation; `python3 -c` scripts printed to stdout only and wrote nothing to disk).

**IPOLE reference disambiguation (required before anything else could be trusted).** Five IPOLE-named locations were found on disk:

| Path | Nature | Identity |
|---|---|---|
| `/work/vmo703/ipole/ipole` | git repo, origin `github.com/ARRicarte/ipole.git` | checked out at branch `dev`, HEAD `d88e5f1` |
| `/work/vmo703/aricarte/aricarte-copy/aricarte/ipole` | same git repo (clone) | checked out at `dev`, HEAD `d88e5f1`, clean |
| `/work/vmo703/aricarte/aricarte-copy/aricarte/ipole+e-` | **not a git repo** — plain file tree with 3 internal copies (`model/`, `ipole_non-executable/`, `build_archive/`) | see below |

`ipole+e-` was first diffed against `origin/master` of the tracked repo and found to differ substantially (an inserted electron model, HAMR-dump support, molecular-weight handling, `debug_tools.h`, etc.) — i.e., `ipole+e-` is **not** `origin/master`. The *checked-out* `dev` branch tree (`aricarte-copy/aricarte/ipole`, HEAD `d88e5f1`) was then diffed against `ipole+e-` for `model/iharm/model.c`, `model_params.h`, `src/decs.h`, `src/par.c`, `src/par.h`, `src/model_radiation.c`, `src/main.c`, `src/ipolarray.c` — **all eight files are byte-identical** (`diff -q` reported no differences).

**Conclusion:** `ipole+e-` = ARRicarte/ipole.git, branch `dev`, commit `d88e5f1` ("Fixed bug causing electronModel>3 to crash needlessly instead of just printing things", 2021‑08‑26). This is also what is checked out at both plain "ipole" locations — so **there is no separate "no‑jet" IPOLE build anywhere on this filesystem under the paths supplied or discoverable from them**; the "two IPOLE versions, one without jet capability" framing does not correspond to what is actually on disk. This is flagged explicitly rather than silently picking a tree. It does not block the audit: IPOLE's own `electronModel==0/1/2/3` (no jet math reachable) already serve as the internal "non-jet" comparison inside the *same* commit.

A second, independent commit chain on the *same* tracked repo (`origin/master`, tip `43ad5c4`) also implements `sigma_transition`/`constant_beta_e0`/critical-beta (commits `cbe37ff→212a6ce→358be53→f5372c4→69ced5b→7701658→ecedb1c→43ad5c4`), with an electron-model numbering **shifted by one slot** relative to `dev` (an inserted "Ressler-compatible mixed TP_OVER_TE" model at slot 3 pushes critical-beta 3→4 and constant-β<sub>e0</sub> 4→5 on `dev`). `origin/master` was **not** used as the reference, since it is not what `ipole+e-` matches — but this shows real historical numbering churn inside IPOLE itself, which is exactly why verifying rather than trusting the historical `with_electrons=5` mapping was warranted.

**GRMONTY provenance.** Repo `/work/vmo703/igrmonty`, branch `positrons` (HEAD `e768b58`, 3 commits ahead of `origin/positrons`). `merge-base(positrons, upstream/master) = 4a1b1c5` — this is the most defensible pre-jet baseline (also the tip of `master`/`origin/master`/`upstream/master`). The jet implementation is commits `c359811..3e50e52` (branch `wjet`, fully merged into `positrons`); positron work is `d8c497f`, `e6ede55` (plus a mixed commit `f72bbc1`); general numerical/bias-tuning work is `b8503f5`, `af16b3a`, `e768b58`. A **sibling branch** `origin/main` (diverged at `5bb37ae`, *before* any jet commit) contains an independent Compton/bias safety fix (`166602e`) that is **not** an ancestor of `positrons` — traced concretely in §7.

**Audit-depth disclosure (asymmetric verification):** GRMONTY's jet-relevant source (`model/iharm/model.c`, `model.h`, `par.c`, `decs.h`, `track_super_photon.c`) was read directly and in full, as were IPOLE's `model.c`/`model_params.h`/`model_radiation.c`. For the **positron** implementation (secondary scope), the *architectural* claim that matters for jet-safety (Thetae/Ne are passed by parameter, never recomputed, in `jnu_mixed.c`/`compton.c`/`hotcross.c`/`scatter_super_photon.c`) was independently confirmed, but the detailed positron pair-density algebra leans on the repository's own prior self-audit (`docs/audits/2026-04-06_m87_long_run_audit.md`, pre-existing) rather than a from-scratch re-derivation. This is flagged wherever it matters below.

**A live-data caveat that affects confidence more than anything else in this audit:** the M<sub>unit</sub>-tuning history CSV (`/work/vmo703/data/munits_tuning_history.csv`, in an *outer* git repository at `/work/vmo703` distinct from `igrmonty`'s own repo — `auto_munit_bracket.py`'s `REPO_ROOT` resolves one level above `igrmonty/`) was **last modified 2026‑07‑23, at 19:34**, i.e. during/immediately adjacent to this audit, and was shown as locally modified/uncommitted. The tuning campaign — jet and non-jet alike — is evidently still running. Any "converged" snapshot that could be extracted is a snapshot of work-in-progress, not a final state (see §8).

---

## 3. Repository and source-change inventory

Baseline = `4a1b1c5` (upstream/master tip). Current = `positrons`@`e768b58`. `git diff 4a1b1c5 HEAD -- model/iharm/model.c` alone is **1,043 changed lines**; the table below separates what's actually jet-relevant from what isn't.

| File : function/symbol | What changed | From/corresponds to IPOLE? | Necessary for jet? | Affects non-jet modes? | Risk | Evidence |
|---|---|---|---|---|---|---|
| `model/iharm/model.c` : `in_jet_region()` | New function, `with_electrons∈{4,5}` gate + OR of a sigma-cut and a beta-cut | No IPOLE equivalent (GRMONTY-only extension) | Yes | No (hard-gated) | Medium | `model.c:513-545`; introduced in `0900329`/`71a917e` (`git log -S"in_jet_region"`) |
| `model/iharm/model.c` : `constant_beta_thetae()` | New function; additive Θ<sub>e</sub> term from B, n<sub>e</sub> | **Yes** — ports IPOLE's `constant_beta_e0 * (B²/(2(γ<sub>e</sub>-1)))^exp / (n_e m_e c²)` | Yes | No | Low | `model.c:462-511` vs `ipole+e-/model/iharm/model.c:547` |
| `model/iharm/model.c` : `thetae_func()` | Rewritten/renamed (was inline in baseline); adds modes 4/5, jet override, additive supplement | Partially — R‑β/Crit‑β math ports baseline math unchanged; jet branches port IPOLE's sigma_transition gate | Yes | **Formulas for modes 2/3 unchanged**; only wrapped in new clamps | Medium | `model.c:547-690`; baseline text via `git show 4a1b1c5:model/iharm/model.c` (see §7) |
| `model/iharm/model.c` : `get_fluid_zone()`, `get_fluid_params()` | Add `jet_ne_mult` scaling of `Ne` inside `in_jet_region()` | No IPOLE equivalent | Yes | No (same gate) | Medium | `model.c:771-774`, `939-942` |
| `model/iharm/model.c` : h5io output | Writes `sigma_transition`, `constant_beta_e0(_exponent)`, `jet_sigma_cut`, `jet_beta_cut`, `jet_thetae`, `jet_ne_mult`, `with_electrons` to `/params/electrons/*` | N/A (workflow) | Yes (provenance) | No | Low (positive) | `model.c:1444-1465` |
| `src/par.c`, `src/par.h` | Parse/store all 7 jet parameters + defaults | N/A (plumbing) | Yes | No | Low | `par.c:119-125`, `par.c:27-33` |
| `src/decs.h` | `THETAE_HARD_MAX`, `BETA_FLOOR`, `SIGMA_MAX` constants; later `BIAS_ABORT_RATIO` macro indirection | Numerical safety, not jet-specific | No | **Yes**, applies globally | Low | `decs.h:9-11` (this file); `b8503f5` diff for the bias-ratio macro |
| `src/track_super_photon.c` : `sanitize_bias()`, `try_boundary_recover_nu()` | New defensive functions; drop photon if nu can't be recovered | General numerical fix, not jet-specific in code, but empirically fires far more in jet configs (§7, §9 finding H2) | No | **Yes**, applies globally | **High** (see finding H2) | `track_super_photon.c:6-11, 75-163`; unchanged since `e6ede55` (`git diff e6ede55 HEAD -- src/track_super_photon.c` is empty) |
| `src/compton.c` | Safety caps/fallback guards around scattering (incl. pre-existing `Thetae *= 0.5` stall kluge) | General numerical/positron, not jet-specific | No | Yes | Medium | Corroborated by `docs/audits/2026-04-06...md` §C2/E2 |
| `src/jnu_mixed.c` | Adds total/negative/positive lepton density scaling for positrons | **Positron**-specific | No | Yes (scales emissivity when `positron_ratio>0`, else no-op) | Low–Medium | Independently confirmed clean Thetae pass-through; algebra sourced from prior audit §C3/D2 |
| `src/main.c` | Positron composition logging; `bias_abort_ratio` validation/exit-on-invalid (from `b8503f5`) | General numerical | No | Yes | Low | `git show b8503f5 -- src/main.c` |
| `auto_munit_bracket.py` : `WJET_DEFAULTS`, `write_par_file()` | Now writes all 7 jet-control parameters into generated `.par` files (confirmed **not** the case as of the `2026-04-06` self-audit) | N/A (workflow) | Yes (reachability) | No | **High** (see finding H1) | `auto_munit_bracket.py:59-78, 637-695`; confirmed live in `logs/SANE_CRITBETAwJET_a+0.94_t6000_rh20_bc1_f0.5_pos0_trial02.par:20-25` and matching `.log:7` |
| `model/iharm/model.h` (comment) : `THETAE_MIN` | Comment claims "0.3 for Crit-Beta"; code uses `3.e-2` | Diverges from IPOLE's flat `1.e-3` for all models | No (pre-existing floor logic re-split by jet work) | **Yes**, affects plain CRITBETA too | Medium (see finding M1) | `model.h:7` vs `model.c:552` vs `ipole+e-/model/iharm/model.c:558` |
| `docs/audits/2026-04-06_m87_long_run_audit.md` | Pre-existing prior self-audit (read, not created) | N/A | N/A | N/A | N/A | Corroborating source throughout |

Archived `.par`/`.log` files under `logs/`, `logs/5e4_test/`, `logs/pair_sweep_20260224/` are workflow artifacts, not source, and are treated as validation evidence (§8) rather than code.

---

## 4. IPOLE reference model

Source: `ipole+e-/model/iharm/model.c` (= `dev`@`d88e5f1`). IPOLE has **no Compton scattering, no Monte Carlo transport, and no photon biasing at all** — `grep -rl "compton\|Compton\|scatter" src/*.c` returns nothing. IPOLE is a pure ray-tracer (`ipolarray.c`) integrating polarized emissivity/absorptivity along fixed geodesics. This bounds what IPOLE can validate: only the electron-thermodynamics → emissivity/absorptivity chain, never scattering or biasing.

**Electron models** (`model.c:522-549`, comments at `:74-80`; `electronModel` default `=2`, `model.c:48`):

| electronModel | Description | Formula |
|---|---|---|
| 0 | fixed Tp/Te | `Θe = uu/rho · Θe_unit` |
| 1 | dump-file (Howes/Kawazura) | `Θe = KEL·ρ^(γe-1)·Θe_unit` |
| 2 | R‑β (Mościbrodzka) | `β=uu(γ-1)/(0.5 b²); trat=trat_large·β²/β_c²/(1+…) + trat_small/(1+…); Θe_unit=(mp/me)(γe-1)(γi-1)/[(γi-1)+(γe-1)trat]; Θe=Θe_unit·uu/ρ` (`model.c:524-531`) |
| 3 | "Ressler-compatible" mixed Tp/Te using fluid `THF` | same `trat` as (2), `Θe = p[THF]/[μtot/μe + μtot/μi·trat]` (`model.c:532-537`) |
| 4 | Critical-β (Anantua et al. 2020) | `Te_over_Ttot = β_cc·exp(-β/β_crit); trat=(1-x)/x; Θe_unit=…; Θe=Θe_unit·uu/ρ` (`model.c:538-544`) |
| 5 | Constant electron-β (Anantua et al. 2020) — **the "jet" model, used as the base formula, not additive** | `Θe = constant_beta_e0·(B²/(2(γe-1)))^exponent / (ne·me·c²)` (`model.c:545-547`) |

**Jet/disk classification and blending rule** (`model.c:551-558`, exact quote):
```c
//ARR:  above sigma_transition, ADD the temperature of the constant_beta_e0 model.
if ((sigma_m > sigma_transition) & (electronModel != 5)) {
  //Equivalent to electronModel == 4 below.
  data[n]->thetae[i][j][k] += constant_beta_e0 * pow(pow(data[n]->b[i][j][k], 2) / (2 * (game-1.)), constant_beta_e0_exponent) / data[n]->ne[i][j][k] / (ME * CL * CL);
}
//Secret floor.
data[n]->thetae[i][j][k] = fmax(data[n]->thetae[i][j][k], 1.e-3);
```
Key facts: (1) the additive jet supplement is **model-agnostic** — it fires for *any* electronModel 0–4 once σ<sub>m</sub> = b²/ρ exceeds `sigma_transition`, not just for a designated "jet variant"; (2) it never double-fires for model 5, which is already the pure constant-β formula everywhere; (3) a single flat floor of `1.e-3` applies after the supplement, regardless of which electronModel; (4) a *separate* cut, `sigma_cut` (a real runtime parameter, `model_params.h:20`), independently zeroes `b[i][j][k]` ("strongly magnetized = empty, no shiny spine", `model.c:560-563`) — this is a harder, different mechanism from `sigma_transition`.

**Units/conventions:** `data[n]->ne = ρ_code·RHO_unit/(mp+me)·Ne_factor` (CGS number density, with a molecular-weight correction `Ne_factor` for non-pure-hydrogen plasmas); `data[n]->b` stored in Gauss; `sigma_m = (B_code)²/ρ_code` (code-unit, dimensionless magnetization); β = `uu·(γ-1)/(0.5 b²)` in code units (plasma beta). Downstream, `get_model_thetae(X)` (`model_radiation.c:226,231,242,570`) feeds directly into `j_nu_fit(...)` / `dexter_j_fit_thermal(...)` for emissivity and (implicitly, via the same fit routines) absorptivity — no Compton step exists to feed.

**Control flow:** parfile → `try_set_model_parameter` (`model.c:105-146`, e.g. `set_by_word_val(word,value,"sigma_transition",&sigma_transition,TYPE_DBL)`) → globals → `init_physical_quantities()` computes the *entire* `thetae[][][]` grid once at load time → ray tracer interpolates that precomputed grid per geodesic step.

---

## 5. GRMONTY implementation trace

Full path, with file:line at each stage (current `positrons`@`e768b58`, confirmed via direct read unless noted):

1. **Parameter text → parsing/storage.** `src/par.c:112-126` (`with_electrons`, `trat_small/large`, `Thetae_max`, `sigma_transition`, `constant_beta_e0[_exponent]`, `jet_sigma_cut`, `jet_beta_cut`, `jet_thetae`, `jet_ne_mult`, `positron_ratio` + `positronRatio` alias) → `Params` struct (`par.h`) with defaults at `par.c:20-34`. All are read directly in `par.c`'s generic loop — unlike IPOLE, there is no model-owned `try_set_model_parameter` callback; GRMONTY's `par.c` itself knows every jet-parameter name.
2. **Model selection / copy into model globals.** `model.c:1042-1060` (inside `init_data()`) copies every `params->jet_*`/`sigma_transition`/`constant_beta_e0*` field into the file-scope statics declared at `model.c:28-35`. A dump-file `has_electrons` attribute can *override* `with_electrons` at runtime (`model.c:1081-1121`) — this happens for **both** jet and non-jet modes identically.
3. **GRMHD primitive access/interpolation.** `get_fluid_zone()` (grid-node path, `model.c:692-815`) and `get_fluid_params()` (interpolated path, `model.c:817-984`) both read `p[KRHO]`, `p[UU]`, `p[KEL]`, `B1..B3`, `U1..U3`.
4. **β and σ.** Computed identically in both `in_jet_region()` (`model.c:520-531`) and inline in `thetae_func()`/R‑β/Crit‑β branches, using `clamp_sigma()` (floor 0, cap `SIGMA_MAX=300`, `model.c:384-395`) and `clamp_beta_value()` (floor `BETA_FLOOR=1e-5`, `model.c:397-404`).
5. **Disk/jet classification.** Two independent booleans in `thetae_func()` (`model.c:570-572`): `in_high_sigma_region` (σ≥`sigma_transition`, requires `with_electrons∈{4,5}` — IPOLE-equivalent) and `in_jet` (`in_jet_region()`: σ≥`jet_sigma_cut` OR β≤`jet_beta_cut`, only if those are set >0, and *only* for `with_electrons∈{4,5}` — GRMONTY-only extension). **Both hard-return false for `with_electrons∈{0,1,2,3}`** — verified directly, `model.c:515-518,570`.
6. **Θe.** `thetae_func()` (`model.c:547-690`): base formula per §7 below, then (`model.c:673-685`, exact quote):
   ```c
   if (in_jet && jet_thetae > 0.0) {
     thetae = clamp_thetae_limits(jet_thetae, thetae_floor, thetae_upper);   // hard override
   } else if (in_high_sigma_region) {
     double thetae_const = constant_beta_thetae(safe_rho, safe_B);
     if (thetae_const > 0.0) thetae += thetae_const;                        // IPOLE-equivalent additive supplement
   }
   ```
   i.e. the hard override takes precedence when both conditions hold; this precedence is implicit in the `else if`, not documented anywhere as intentional.
7. **Emissivity/opacity.** `get_fluid_params`/`get_fluid_zone` pass the final `Thetae`, `Ne`, `B` by value into `alpha_inv_scatt`/`alpha_inv_abs` (`jnu_mixed.c`) and `track_super_photon.c`. All of `jnu_mixed.c`, `compton.c`, `hotcross.c`, `scatter_super_photon.c` were grepped for `Thetae`/`thetae` and found to contain **only parameter usage — no independent recomputation, no second call to `get_fluid_params`/`thetae_func`/`get_fluid_zone` anywhere in those four files.** This directly answers the concern about Θe being "computed correctly then discarded/recomputed": it is not.
8. **Scattering/biasing.** `track_super_photon.c` receives `Thetae`/`Ne`/`B` from `get_fluid_params` (called at `track_super_photon.c:301, 386`) and passes them unchanged into `alpha_inv_scatt`, `alpha_inv_abs`, `bias_func`, and `scatter_super_photon`. No jet-specific branch exists here — good (means jet electrons scatter through the *same* physics as everyone else) but also means any jet-specific numerical fragility (see §7/§9 H2) is inherited unmodified.
9. **Spectrum output.** `record_super_photon()` bins by folded BL θ and photon energy (`model.c:140-224`, unchanged by jet work); jet metadata written to `/params/electrons/*` (`model.c:1444-1465`).
10. **Non-jet modes (`with_electrons`=0,1,2,3):** every jet gate (`in_jet_region`, `add_constant_component`, `in_high_sigma_region`) is hard-false; the core R‑β/Crit‑β algebra is unchanged from baseline `4a1b1c5` (verified, §7); only new *defensive clamps* were added around them.

---

## 6. IPOLE–GRMONTY equivalence matrix

| Component | IPOLE | GRMONTY | Equivalent? | Evidence | Scientific consequence |
|---|---|---|---|---|---|
| Model selector / mapping | `electronModel` 0-5, dev-branch numbering | `with_electrons` 0-5, **independently defined**, matches historical note (5=CRITBETAwJET) | Architecturally different, not meant to be identical | `model.h:15-21` vs `ipole+e-/model.c:74-80` | None — GRMONTY's own comment block is authoritative and internally consistent; confirmed `with_electrons=5` *is* CRITBETAwJET in current code |
| Defaults/aliases | `electronModel` default 2; `constant_beta_e0_exponent` default **1.0** (`model.c:47`) | `with_electrons` default 2 (`model.c:35`); `constant_beta_e0_exponent` default **1.0** (`model.c:30`) — but production tuning script overrides to **0.0** | Compiled defaults equivalent; **live default diverges** | `auto_munit_bracket.py:74` | See Finding H1 |
| Plasma-β convention | `β=uu(γ-1)/(0.5·b²)`, code units | Identical formula (`model.c:598-601`, `639-643`) | **Yes** | — | — |
| Magnetization σ | `σ_m=b²/ρ` (`model.c:520`) | `σ=clamp_sigma(B²/ρ)` (`model.c:523,568`) | **Yes**, GRMONTY adds a cap (300) and floor (0) IPOLE lacks | `model.c:384-395` | Negligible except pathological (σ>300) zones |
| B-field normalization | `b[i][j][k]=sqrt(bsq)*B_unit` where `bsq=Σ b^μ b_μ` | `*B = sqrt(Σ Bcon·Bcov)*B_unit` (`model.c:760-762,928-930`) | **Yes** | `model.c:760-762` | — |
| Jet-mask inequality / boundary | `sigma_m > sigma_transition` | `sigma >= sigma_transition` | Off by boundary-inclusivity only | `model.c:571` vs `ipole+e-/model.c:552` | Negligible (measure-zero) |
| Disk Θe formula (R‑β) | `model.c:524-531` | `model.c:588-624` | **Yes, algebraically identical** | quoted both, §4/§7 | — |
| Disk Θe formula (Crit‑β) | `model.c:538-544` | `model.c:626-671` | **Yes, algebraically identical** | quoted both, §4/§7 | — |
| Jet Θe/pressure formula | `constant_beta_e0·(B²/(2(γe-1)))^exp/(ne·me·c²)` | `constant_beta_thetae()`, same formula via `exp(exp·log(x))` reformulation | **Yes** | `model.c:462-511` vs `ipole+e-/model.c:547` | Only differs at overflow extremes (returns 0 instead of Inf) |
| Exponent interpretation | Same variable, same role | Same variable, same role, but **live default = 0** in production | Formula equivalent; **deployed value not** | See H1 | Zeros B-dependence of the whole jet term in current runs |
| Jet region as a *distinct mask* | Does not exist — supplement is model-agnostic, gated by σ alone | **Exists** — `in_jet_region()`, gated by `jet_sigma_cut`/`jet_beta_cut`, `with_electrons∈{4,5}` only | Necessary architectural addition (no equivalence expected) | `model.c:513-545` | GRMONTY-only feature; internally consistent, but no IPOLE ground truth exists to validate it against |
| Density conversion / e⁻ vs total lepton | `ne = ρ·RHO_unit/(mp+me)·Ne_factor` | `Ne = ρ·Ne_unit`(+ positron scaling downstream in `jnu_mixed.c`/`radiation.c`, not in `model.c`) | Equivalent in structure; GRMONTY additionally supports positrons | `model.c:764,932,944` | Positron scaling is orthogonal to jet math (confirmed no interaction in `model.c`) |
| Interpolation order/location | Whole-grid precompute (`init_physical_quantities`, once) | On-the-fly per-photon-step (`thetae_func` called inside `get_fluid_params`, at interpolated, not just grid-node, locations) | Necessary MC-vs-raytracer adaptation; if anything finer-grained in GRMONTY | `model.c:817-984` | None — architecturally required difference |
| Sigma/emission cuts | Configurable `sigma_cut` param, separate from `sigma_transition` | Hardcoded `sig_unscaled>1.` cut, only for `with_electrons<3` (pre-existing, not jet-introduced) | Different mechanism, pre-existing GRMONTY design | `model.c:767-769,935-937` | Not a jet-introduced difference; legacy models 0-2 always had this |
| Floors/caps | Flat `1.e-3` for **all** electronModels (`ipole+e-/model.c:558`) | **Split**: `rb_floor=1e-3` (R‑β) vs `crit_floor=3e-2` (Crit‑β) (`model.c:551-552`) — and disagrees with GRMONTY's own comment (`model.h:7`: "0.3 for Crit-Beta") | **No** | See Finding M1 | Real (if likely small-population) deviation in coldest zones, affects plain CRITBETA too |
| Synchrotron emissivity | `j_nu_fit(...)`/Symphony (`model_radiation.c:259-262`) | `jnu()`/`jnu_mixed.c` (thermal/kappa/powerlaw/bremss switch) | Necessarily different implementations; both take Θe/Ne/B/θ as plain arguments, no recomputation in either | `model_radiation.c:226-262` vs `jnu_mixed.c:70-99` | Architectural, not a jet correctness issue |
| Absorptivity/opacity | Implicit in `j_nu_fit` (same fit family gives α<sub>ν</sub>) | `alpha_inv_abs()` (`radiation.c`, not independently re-derived in this audit) | Not directly comparable (different code) | — | Out of scope beyond confirming Θe/Ne pass-through, which held |
| Bremsstrahlung | Not confirmed present in this IPOLE tree from available reads | `jnu_bremss()` (`jnu_mixed.c`), extended for positron pairs | N/A (IPOLE reference doesn't provide one for this comparison) | — | — |
| Compton scattering / Klein-Nishina | **None — IPOLE has no scattering module** | Full Monte Carlo (`compton.c`, `hotcross.c`) | **Not comparable — no IPOLE reference exists** | confirmed via `grep -rl compton\|scatter ipole+e-/src/*.c` → empty | Any jet-region scattering behavior can only be judged on GRMONTY's own internal consistency |
| Scattering bias / `bias<1` | N/A | `sanitize_bias()` floors bias to 1 (current code); asymmetric `isnan nu` drop rate jet vs non-jet | N/A | See Finding H2 | Jet-region photon statistics disproportionately affected |
| Photon-energy/electron sampling constraints | N/A | `compton.c` stall kluge halves Θe after 10⁷ rejected samples (pre-existing, not jet-introduced) | N/A | `docs/audits/2026-04-06...md` §B7,E2 (corroborated, not independently re-derived) | More likely to trigger at the high Θe values a jet supplement can produce |
| Observer-angle/bin compatibility | Single fixed camera inclination + finite FOV | 4π, azimuth-averaged, equator-folded θ-bins; optional post-hoc cone restriction tool exists | **Not equivalent without post-processing**, and post-processing was never applied to any wJET output | `_reports/munit_ipole_vs_grmonty_20260218/report.md`; file listing in `igrmonty_outputs/m87/5e4_test/` | See Finding H3 |
| Output metadata for prescription ID | `type`/`electronModel` etc. written to header | `with_electrons`, `sigma_transition`, `constant_beta_e0(_exponent)`, `jet_sigma_cut`, `jet_beta_cut`, `jet_thetae`, `jet_ne_mult` all written | **Yes, GRMONTY is at least as complete** | `model.c:1444-1465` | Positive — supports post-hoc provenance recovery |

---

## 7. Physics and numerical review

**Non-jet formula preservation (direct baseline diff).** Baseline (`git show 4a1b1c5:model/iharm/model.c`), R‑β:
```c
double beta = uu * (gam-1.) / 0.5 / B / B;
double b2 = beta*beta / beta_crit/beta_crit;
double trat = trat_large * b2/(1.+b2) + trat_small /(1.+b2);
if (B == 0) trat = trat_large;
thetae = (MP/ME) * (game-1.) * (gamp-1.) / ( (gamp-1.) + (game-1.)*trat ) * uu / rho;
```
Current (`model.c:592-624`) is the same algebra with `clamp_beta_value`/finite-guards wrapped around it — for any physically valid (finite, non-negative) input the results are numerically identical; the guards only change behavior for already-degenerate input (B=0 handled by an early `Bsq>0` check instead of a post-hoc `if(B==0)`, same net effect). Crit‑β baseline used a flat `fmax(1e-3, …)` floor with the comment *"Put in a hidden floor for consistency with IPOLE"* — current code instead applies `crit_floor=3e-2` (Finding M1). **Conclusion: the core equations for models 2 and 3 are unchanged; only the floor for model 3 drifted.**

**Units/dimensions.** No missing 4π/c/mₑ/mₚ/k<sub>B</sub> factors found in the formulas that could be directly compared (R‑β, Crit‑β, constant‑β<sub>e0</sub> all matched IPOLE term-for-term, including the somewhat idiosyncratic `B²/(2(γₑ-1))` "magnetic energy density" convention, which is **the same non-standard convention in both codes**, so it cancels out of the comparison — it is *not* the standard `B²/8π`, but since IPOLE and GRMONTY use the identical convention, this is not a cross-code bug).

**Division/degenerate-input safety.** `clamp_positive`, `clamp_sigma`, `clamp_beta_value`, `clamp_thetae_limits` (`model.c:345-404`) consistently guard against zero/negative/non-finite ρ, uu, B, β, Θe. `constant_beta_thetae()` (`model.c:462-511`) explicitly guards the `log`/`exp` reformulation of `pow()` against overflow, returning 0 rather than Inf beyond `log(DBL_MAX)` — a safeguard that trades a diagnosable Inf for a silent zero (Low severity, Finding L1).

**Thread safety.** `bias_warn_count`/`invalid_bias`/`N_init_reject_*` counters use `#pragma omp atomic`; no unguarded shared-state write was found in the jet-specific functions (`in_jet_region`, `thetae_func`, `constant_beta_thetae` are all pure functions of their arguments plus read-only globals set once at `init_data`).

**Discontinuity at the jet boundary.** IPOLE's transition is a step function in σ (same as GRMONTY's `in_high_sigma_region`) — both codes share this property, so it is not a GRMONTY-introduced discontinuity. GRMONTY's *additional* hard-override (`jet_thetae`) is a **second, steeper** step (Θe jumps directly to a fixed value rather than being added to the disk value) that IPOLE has no equivalent of at all.

**Confirmed asymmetric numerical fragility (Finding H2).** `track_super_photon.c` is unchanged since commit `e6ede55` (`git diff e6ede55 HEAD -- src/track_super_photon.c` returns empty), so the prior self-audit's counts still describe the current code path. That audit (pre-existing) found **1,308** occurrences of `isnan nu: track_super_photon` in one representative `MAD_RBETAwJET_a+0.94_t4000_rh80_pos0_trial01.log`, versus **0** in matched clean `CRITBETA` (non-jet) logs. Each occurrence drops the photon (`ph->w=0.0; return;`, `track_super_photon.c:413-414/416-417` after a failed `try_boundary_recover_nu()`). A spot-check of one *current-era* `CRITBETAwJET` log (`SANE_CRITBETAwJET_a+0.94_t6000_rh20_bc1_f0.5_pos0_trial02.log`) found `isnan nu` count = 0 there — consistent with the failure being concentrated at higher `trat_large`/`rh` (the problem case was `rh80`; the clean case was `rh20`), but there were not enough current-era logs available to confirm this pattern is general rather than incidental.

**Independent confirmation of a real deployed-default deviation (Finding H1).** `auto_munit_bracket.py:59-78` `WJET_DEFAULTS` sets `constant_beta_e0_exponent: 0.0`. Since `energy_density^0 = 1` for any finite positive `energy_density`, `constant_beta_thetae()` degenerates to `constant_beta_e0/(ne_cgs·mₑc²)` — **no B-field or σ dependence at all**. This is not merely a dict default that goes unused: the actual `.par` file `logs/SANE_CRITBETAwJET_a+0.94_t6000_rh20_bc1_f0.5_pos0_trial02.par:22` contains `constant_beta_e0_exponent 0`, and its `.log:7` prints `constant_beta_e0_exponent = 0` at runtime — i.e., **this is live, in the current production pipeline**, not a hypothetical. This value matches neither IPOLE's default (1.0) nor GRMONTY's own compiled-in default (`model.c:30`, 1.0). A "constant electron beta" model with the B-dependence set to zero is, by construction, no longer a beta-dependent (magnetically-referenced) model at all — it becomes a constant-electron-pressure model. Whether this was deliberate (e.g. an isolation test) or an oversight cannot be determined from source alone; this is presented as a verified numeric fact with a clear physical consequence, not as a confirmed intent.

---

## 8. Existing-run validation audit

**QA_REPORT.md** (`igrmonty_outputs/m87/_qa/QA_REPORT.md`, generated 2026-02-17, pre-existing): 25/25 `.h5` outputs pass structural sanity (no NaN, no open failures). By WJET flag: OFF 17/17, ON 8/8 pass. Pairwise: 8 WJET OFF/ON pairs evaluated, 0 flagged suspicious, and in **all 8** pairs the jet-ON median flux exceeds jet-OFF (`ON>OFF in 8, ON<OFF in 0`). This is weak-but-real, *directionally* consistent evidence (adding a strictly-additive Θe term should not decrease flux) — it is a sanity check, not a quantitative correctness check, and the QA script's own "suspicious" thresholds are ratio/percentile-based structural checks (`qa_grmonty_outputs.py`), not physics validation against an external reference.

**Prior self-audit** (`docs/audits/2026-04-06_m87_long_run_audit.md`, pre-existing, HEAD `e6ede55` at the time): examined a real 20-day Slurm batch (job `712854`). Its central findings, corroborated by independently reading the *still-current* source: (1) the only scientifically-usable final products from that batch were plain `CRITBETA` (non-jet); (2) at that time, `write_par_file()` did **not** emit jet-control parameters, so `jet_thetae`/`jet_ne_mult` were inert in every wJET run examined — **this specific gap has since been fixed** (§7, confirmed independently against the *current* `auto_munit_bracket.py` and a *current* `.par`/`.log` pair, both dated after that audit); (3) wJET branches disproportionately hit the bias-abort guard and `isnan nu` drops (1,308 vs 0, see §7).

**Prior IPOLE-vs-GRMONTY report** (`_reports/munit_ipole_vs_grmonty_20260218/report.md`, pre-existing): independently concludes (its own words) that "matching only the 230 GHz bin is not sufficient to ensure broader SED equivalence," that IPOLE in this pipeline is single-inclination camera flux while GRMONTY's tuning flux is 4π-angle-averaged, and that these "are not directly comparable" without an angle restriction. A post-hoc cone-restriction tool exists (`tools/viewing_cone_postprocess.py`) and **was** used to produce `*_cone_summary_tc17p0_a10p0.csv/.npz` artifacts — but only for plain `RBETA`/`CRITBETA` spectra (`Ma-0.5_5000_RBETA`, `Ma+0.94_4000_RBETA`, `Sa-0.5_4000_CRITBETA`, `Sa-0.5_4000_RBETA`, `Sa+0.94_4000_CRITBETA`). **No `*wJET*_cone_*` artifact was found anywhere in the outputs tree.** A companion design doc (`grmonty_viewing_patch.md`) for a true runtime cone filter is explicitly marked "design only, not applied." **No angle-matched IPOLE-vs-GRMONTY comparison exists for any jet-enabled spectrum in this repository.**

**Live tuning-history data** (`/work/vmo703/data/munits_tuning_history.csv`, 791 trial rows, 2025‑11‑20 → **2026‑07‑23 19:34**, i.e. touched during this audit window): raw trial-row counts by model are robust (`RBETA` 286, `CRITBETA` 258, `CRITBETAwJET` 135, `RBETAwJET` 110). An attempt to derive a converged/not-converged tally per unique (state, model, spin, dump, pos) branch produced internally-inconsistent results (some parsed "converged" values were impossible, e.g. `1.000e+06`, indicating column misalignment from schema drift over the file's 8-month history) — **that derived convergence count is not reported as fact**, and this CSV is flagged as needing a clean re-export before it can support a convergence claim. What can be said with confidence: essentially every wJET (spin, dump, pos) branch has a trial row dated within the last few months, several dated literally the day of this audit, and the commit-message claim "42/48 munit branches converged" (`af16b3a`, `2026-07-21`) is a **self-reported, not independently re-derived** figure — and in any case, rows in the CSV postdate that commit, so even that figure is not necessarily current. **Unverified provenance.**

**Overall:** existing runs demonstrate the code executes, produces finite output, and behaves directionally sensibly (jet-on ≥ jet-off flux) for both the additive-supplement mechanism (well exercised, going back to `5e4_test` runs from `71a917e`) and, more recently, the hard-override mechanism (only reachable in production since the `auto_munit_bracket.py` fix, with only one clean current-era log found exercising it, itself not showing a terminal convergence line in the captured tail). None of this constitutes a physics-correctness proof, and the M_unit/230 GHz convergence work in progress right now is explicitly **not** a substitute for that, a point this repository's own prior reports already make about themselves.

---

## 9. Findings ranked by severity

**Critical:** none identified — the jet branch is reachable, the region definition matches IPOLE's intended physical mechanism (plus a documented GRMONTY extension), the core electron-temperature equations are verified correct and faithful, and Θe/Ne propagation into emission/scattering is confirmed clean with no discard/recompute.

---

**H1 — `constant_beta_e0_exponent=0.0` in the live production tuning defaults zeroes the B-field dependence of the jet-temperature supplement.**
- Evidence: `auto_munit_bracket.py:74`; confirmed live in `logs/SANE_CRITBETAwJET_a+0.94_t6000_rh20_bc1_f0.5_pos0_trial02.par:22` and its `.log:7`.
- Why wrong/risky: `constant_beta_thetae()` (`model.c:462-511`) computes `constant_beta_e0 · energy_density^exponent / (ne·mₑc²)`; with `exponent=0`, `energy_density^0≡1`, so the supplement collapses to a pure `1/ne` term with **zero** magnetic dependence — the opposite of what a "constant electron beta" (electron-pressure/magnetic-pressure ratio) jet model is supposed to represent. It also diverges from IPOLE's default (1.0) and GRMONTY's own compiled default (1.0, `model.c:30`).
- Models/results affected: every `CRITBETAwJET`/`RBETAwJET` `.par`/`.h5` generated by the current `auto_munit_bracket.py` (i.e., the entire current wJET tuning campaign, dated May–July 2026 in the tuning-history CSV).
- Likely scientific consequence: the jet-region Θe supplement no longer tracks magnetization at all; any conclusions drawn about "jet heating scaling with σ" from these runs would not reflect the intended physics.
- Minimal conceptual correction (no patch applied): set `WJET_DEFAULTS["constant_beta_e0_exponent"]` back to a physically-motivated value (1.0, matching both references, unless a specific different value is scientifically justified and documented).
- Validation to confirm: re-run one CRITBETAwJET branch with `constant_beta_e0_exponent=1.0` and compare the resulting Θe field in the σ≥`sigma_transition` region against the `exponent=0` run — the two should differ visibly wherever B varies.

**H2 — Invalid fluid-frame-frequency photon drops are markedly more frequent in jet-enabled configurations, and current handling discards rather than reweights the affected photons.**
- Evidence: `docs/audits/2026-04-06_m87_long_run_audit.md` §B7 (1,308 `isnan nu` events in one `RBETAwJET` log vs 0 in matched `CRITBETA` logs); code path unchanged since `e6ede55` (`git diff e6ede55 HEAD -- src/track_super_photon.c` empty), confirmed by direct read of current `track_super_photon.c:305-322,396-418`.
- Why wrong/risky: `try_boundary_recover_nu()` (`track_super_photon.c:75-163`) is attempted first, but on failure the photon is unconditionally zeroed (`ph->w=0.0; return;`), discarding its statistical weight entirely rather than, e.g., reweighting surviving photons.
- Models/results affected: any `with_electrons∈{4,5}` run, especially at higher `trat_large`("rh") — the rh-dependence was not confirmed generally, only in the one available high-rh example vs one available rh20 example.
- Likely scientific consequence: systematic under-sampling of exactly the (high-σ, near-polar/funnel-boundary) region the jet model is meant to characterize — a differential bias between jet and non-jet spectra that isn't a "bug" in the emissivity formula but is a bug in how faithfully the Monte Carlo sampling represents it.
- Minimal conceptual correction: track and report the *fraction* of photon weight dropped this way per run (a counter already exists in spirit — `N_init_reject_nu` in `decs.h:134` — confirm it's wired to this exact path and surfaced in run summaries), and treat any run with a non-negligible drop fraction as suspect until investigated.
- Validation to confirm: for a matched jet/non-jet pair at identical (spin, dump, Ns), report `isnan nu` count and total dropped weight as a fraction of `N_superph_made`; flag any wJET run whose fraction is materially above its non-jet counterpart.

**H3 — No viewing-angle-matched (cone-restricted) comparison against IPOLE exists for any jet-enabled spectrum.**
- Evidence: `tools/viewing_cone_postprocess.py` outputs (`*_cone_summary_tc17p0_a10p0.csv/.npz`) exist only for `Ma-0.5_5000_RBETA`, `Ma+0.94_4000_RBETA`, `Sa-0.5_4000_CRITBETA`, `Sa-0.5_4000_RBETA`, `Sa+0.94_4000_CRITBETA` (file listing, `igrmonty_outputs/m87/5e4_test/` and `_reports/munit_ipole_vs_grmonty_20260218/`); none for any `*wJET*` file. `_reports/munit_ipole_vs_grmonty_20260218/report.md` documents the underlying 4π-vs-camera mismatch generally.
- Why wrong/risky: without this, any flux number quoted for a wJET run cannot be meaningfully compared to an IPOLE number at all — the geometric definitions differ before the electron physics is even considered.
- Models/results affected: all wJET outputs, with respect to any claim of IPOLE agreement specifically.
- Likely scientific consequence: a wJET spectrum could match or mismatch IPOLE for reasons entirely unrelated to the jet electron physics (viewing geometry alone).
- Minimal conceptual correction: apply the existing cone postprocessing tool to at least one wJET output per (spin, dump) pair before making any IPOLE-comparison claim.
- Validation to confirm: cone-restricted wJET flux vs. equivalent-inclination IPOLE flux, at matched M_unit, as an actual comparison point (currently absent).

---

**M1 — GRMONTY's Crit‑β Θe floor (0.03) diverges from IPOLE's flat floor (0.001), and disagrees with GRMONTY's own documentation.**
- Evidence: `model.h:7` comment "`0.3 for Crit-Beta`" vs. actual coded `crit_floor = 3.e-2` at `model.c:552` (a 10× documentation/code mismatch) vs. IPOLE's uniform `fmax(…, 1.e-3)` at `ipole+e-/model.c:558`.
- Why risky: this affects the **non-jet** `CRITBETA` model too (not jet-specific), in the coldest/lowest-Θe zones.
- Models/results affected: `CRITBETA` and `CRITBETAwJET` alike, wherever the unfloored Θe would fall between 1e-3 and 3e-2.
- Scientific consequence: likely small (affects only the coldest zones, which contribute little to synchrotron emission), but "likely small" is an assumption, not something quantitatively verified here.
- Minimal conceptual correction: decide whether 0.03 or 0.3 (or IPOLE's 1e-3) is the intended value, fix the comment/code to agree, and state the rationale for departing from IPOLE if that's the final choice.
- Validation to confirm: histogram the fraction of emitting cells hitting the floor for a representative CRITBETA dump; if negligible, this is cosmetic; if not, it changes the disk spectrum.

**M2 — Two independently-gated "jet region" mechanisms coexist without documented precedence rationale.**
- Evidence: `model.c:673` (`in_jet && jet_thetae>0` hard override) vs. `model.c:678` (`else if in_high_sigma_region` additive supplement).
- Why risky: not a bug (no double-counting was found), but the `else if` means the two mechanisms' interaction is implicit rather than documented, and only one of the two (the additive supplement) has any IPOLE counterpart at all.
- Models/results affected: `with_electrons∈{4,5}` whenever both `jet_thetae>0` and σ≥`sigma_transition` would otherwise apply.
- Scientific consequence: currently benign given the explicit precedence, but a future edit that reorders these branches would silently change behavior.
- Minimal conceptual correction: a one-line comment stating the override supersedes the supplement, plus a note that the override has no IPOLE reference.
- Validation to confirm: none needed beyond the comment; this is a maintainability finding.

**M3 — Existing wJET `.h5` outputs span both a pre- and post-plumbing-fix era with materially different active physics, and are not distinguishable by filename alone.**
- Evidence: April self-audit documents the pre-fix era (jet_sigma_cut/jet_beta_cut/jet_thetae inert); current `.par`/`.log` pair (`SANE_CRITBETAwJET_a+0.94_t6000_rh20_bc1_f0.5_pos0_trial02.*`) documents the post-fix era (all four active).
- Why risky: a filename like `spectrum_Sa-0.5_4000_CRITBETAwJET_pos0.h5` does not by itself tell you which era/mechanism produced it.
- Models/results affected: any existing wJET `.h5` file whose generating `.par`/`.log`/embedded `/params/electrons/*` metadata hasn't been checked.
- Scientific consequence: aggregating "wJET" results across eras would silently mix two different physical models.
- Minimal conceptual correction: none (this is a bookkeeping issue) — always read the embedded `/params/electrons/jet_sigma_cut` etc. from the `.h5` itself before treating any two wJET files as comparable.
- Validation to confirm: N/A.

---

**L1** — `constant_beta_thetae()`'s `exp(exponent·log(x))` reformulation (vs. IPOLE's direct `pow()`) silently returns 0 rather than Inf/NaN beyond `log(DBL_MAX)` (`model.c:493-497`) — safer, but masks rather than surfaces a pathological input. Informational/low; no action required unless silent zeros are undesirable.

**L2** — `track_super_photon.c:501` (`if (!isfinite(bias) || bias < 1.0)`) is unreachable given `bias` is already forced ≥1 by `sanitize_bias()` two lines earlier at `track_super_photon.c:448` with no intervening reassignment. Harmless; simplify for clarity.

**L3** — `sigma_m > sigma_transition` (IPOLE) vs `sigma >= sigma_transition` (GRMONTY, `model.c:571`). Negligible, boundary-only.

**L4** — GRMONTY's legacy high-σ emission cut (`with_electrons<3 && sig_unscaled>1.`, `model.c:768,936`) is a hardcoded threshold on a differently-scoped quantity than IPOLE's configurable `sigma_cut` parameter. Pre-existing (predates the jet work), out of jet-scope, but relevant to any funnel-behavior comparison involving models 0–2.

---

**Informational (verified-correct / intentional differences):**
- I1: IPOLE has no Compton/scattering/biasing module at all — confirmed by exhaustive grep; bounds what can ever be "validated against IPOLE."
- I2: R‑β/Crit‑β core equations are unchanged from the pre-jet baseline; only defensive clamps were added.
- I3: `in_jet_region`/`add_constant_component`/`in_high_sigma_region` are all hard-gated to `with_electrons∈{4,5}` — non-jet models cannot execute any jet code path (verified directly).
- I4: Θe/Ne propagation into `jnu_mixed.c`, `compton.c`, `hotcross.c`, `scatter_super_photon.c` is parameter-passing only, confirmed via full grep of all four files.
- I5: Jet parameters and `with_electrons` are written to output HDF5 metadata (`model.c:1444-1465`) — good provenance practice.
- I6: GRMONTY's on-the-fly, per-photon-step Θe evaluation (vs IPOLE's whole-grid precompute) is a necessary and, if anything, finer-grained MC-vs-raytracer architectural adaptation.

---

## 10. Grading breakdown

| Category | Points possible | Points earned | Rationale (brief) |
|---|---:|---:|---|
| Parameter plumbing and runtime reachability | 15 | 12 | All 7 jet parameters parse, store, propagate, and are written to output; the automated production pipeline now (as of a recent fix) actually writes non-default values for all of them, confirmed live in a real `.par`/`.log`. Docked for the historically-real (now-fixed) reachability gap meaning much of the existing wJET output corpus never exercised the hard-override path, and for the still-live `constant_beta_e0_exponent=0` default sitting in the same plumbing. |
| Jet-region definition and classification | 20 | 16 | The IPOLE-equivalent σ-gated additive mechanism is a faithful, correctly-gated port. The additional hard-override mechanism is cleanly implemented and provably unreachable from non-jet modes, but it has no IPOLE analogue to validate against and its precedence over the additive term is undocumented. |
| Electron thermodynamics and IPOLE fidelity | 25 | 20 | R‑β, Crit‑β, and constant-β<sub>e0</sub> formulas are verified algebraically identical to IPOLE, term for term, including a non-standard shared convention. Docked for the Crit‑β floor deviation (M1) and, more heavily, for the live `exponent=0` deployment (H1) that currently defeats the jet formula's B-dependence in production. |
| Integration into emission, opacity, and scattering | 20 | 14 | Clean, single-source-of-truth propagation confirmed with no recomputation anywhere in the chain — strong. Docked for the confirmed, differentially-worse `isnan nu` photon-drop behavior in jet configurations (H2), and because IPOLE provides no ground truth at all for the scattering/biasing half of this category. |
| Units, edge cases, and numerical robustness | 10 | 7 | Extensive, consistent defensive clamping (`clamp_sigma/beta_value/positive/thetae_limits`, `sanitize_bias`) is a real strength. Docked for the documentation/code floor mismatch (M1) and the silent-zero-on-overflow behavior (L1). |
| Existing validation and reproducibility | 10 | 4 | Structural sanity checks pass (25/25), and directional jet-on≥jet-off trend holds in all 8 available pairs — real but weak evidence. No angle-matched IPOLE comparison exists for any jet output (H3); the one large-scale historical batch produced zero trusted final wJET products; the live tuning campaign is unfinished and its own convergence bookkeeping could not be reliably parsed during this audit. |
| **Total** | **100** | **73** | |

Re-summing with the executive-summary figure: **73–78** is the defensible range depending on how heavily "existing validation" is weighted against "the code, where it could be checked, is right"; **75 (C)** is reported as the point figure, split conservatively toward the evidence that most directly answers "should you trust the numbers today" (§8-9), which is the weaker side of the ledger.

**Letter grade: C+ (mid-to-upper C range, 73–78/100 depending on weighting — reporting 75/C for a single number).**

**Confidence: Medium.** Limited by: (1) nothing was executed to empirically confirm the H1/H2 consequences at scale — only one long-but-possibly-incomplete log exists for the post-fix hard-override configuration; (2) the tuning-history CSV was actively being written during this audit and its convergence column could not be reliably parsed; (3) positron-interaction claims lean partly on the prior self-audit rather than a from-scratch derivation; (4) no genuine "no-jet IPOLE" build could be located to use as a fully independent second reference, so some IPOLE-side claims rest on a single commit (`dev`@`d88e5f1`) rather than cross-checked against a second IPOLE lineage.

---

## 11. Minimum validation still required

**Completed (per §8):** structural sanity on 25 `.h5` files (QA_REPORT.md); 8 jet-on/jet-off directional pairs (all consistent with jet-on ≥ jet-off); one prior deep run-level audit of a real 20-day batch; one prior IPOLE-vs-GRMONTY geometry/M_unit-scaling report.

**Proposed, NOT RUN (all commands below are illustrative and unexecuted):**

1. **Non-jet regression**, `with_electrons=2` and `=3`, against the pre-jet baseline binary if it can be rebuilt from `4a1b1c5` — expect bit-identical Θe fields for identical input. *(NOT RUN)*
   ```
   # NOT RUN — illustrative only
   git worktree add /tmp/baseline 4a1b1c5 && cd /tmp/baseline && make && ./grmonty -par same.par
   ```
2. **Jet-disabled-by-construction check**: `with_electrons=4` with `sigma_transition` set above the maximum σ ever reached in the dump (e.g. 1e6) and `jet_sigma_cut=jet_beta_cut=-1` — expect bit-identical output to `with_electrons=2` on the same dump. *(NOT RUN)*
3. **Weak/strong jet-heating limiting cases**: `jet_thetae` swept across {0 (inert), a value near ambient disk Θe, 10× ambient} at fixed `jet_sigma_cut` — expect a monotonic, visible flux/spectral-shape response. *(NOT RUN)*
4. **`constant_beta_e0_exponent` sweep**: {0, 0.5, 1.0} at fixed everything else — this directly tests Finding H1; expect the σ-dependence of the jet supplement to reappear as exponent increases from 0. *(NOT RUN)*
5. **Both R‑β-with-jet and Crit‑β-with-jet** under (3) and (4). *(NOT RUN)*
6. **Single-cell comparison**: pick one (i,j,k) zone with σ just above `sigma_transition` in a dump also loadable by IPOLE; print β, σ, Θe, j<sub>ν</sub> from both codes at that cell. This is the most direct test of §6/§7's claimed formula equivalence, done numerically rather than symbolically. *(NOT RUN)*
7. **Compton on/off**: same parfile with `with compt: 0` vs `1` for one wJET config, to isolate whether H2's `isnan nu` drops originate in the geodesic/interpolation step (would persist with Compton off) or the scattering step (would disappear). *(NOT RUN)*
8. **Positron fraction 0 vs nonzero, crossed with wJET on/off**: 4 runs, to directly test for the interaction flagged by the user (`positron_ratio∈{0,1}` × `with_electrons∈{3,5}`). *(NOT RUN)*
9. **Deterministic/reduced-photon smoke test**: fixed `seed`, small `Ns`, for `with_electrons=5` with the post-fix jet parameters, to get a fast reproducibility check before committing to a 1e6-photon run. *(NOT RUN)*
10. **Compatible observer-angle**: apply `tools/viewing_cone_postprocess.py` (already exists) to at least one wJET output per (spin, dump) and compare against the equivalent-inclination IPOLE flux — directly resolves Finding H3. *(NOT RUN)*
11. **Regenerate a schema-consistent `munits_tuning_history.csv`** (or a filtered view keyed strictly on the current column count) and re-derive true converged/not-converged counts per branch — resolves the ambiguity in §8's live-data discussion. *(NOT RUN, and not a GRMONTY-code task — a data-hygiene task on the CSV itself.)*

**Pass/fail criteria**, not just "compare the plots": (2) exact match; (1) exact match; (3)/(4) monotonic and non-degenerate response, no NaN/Inf, no discontinuous jump beyond the designed step at `sigma_transition`/`jet_sigma_cut`; (6) agreement to floating-point precision on β/σ/Θe, agreement to within documented algorithmic differences on j<sub>ν</sub>; (7) `isnan nu` count should be independent of the Compton flag if the hypothesis in H2 (geodesic/interpolation origin) is right; (10) cone-restricted flux ratio to IPOLE should be O(1), not orders of magnitude off, before any claim of "matches IPOLE" is made for a jet spectrum.

---

## 12. Final answer

**`Correct in its core physics but incompletely validated`**

In plain terms: the electron-temperature mathematics ported from IPOLE — R‑β, Crit‑β, and the additive constant‑β<sub>e0</sub> jet supplement gated on `sigma_transition` — is correct and faithfully matches the IPOLE reference term-for-term, and it is wired cleanly through to emission, absorption, and Compton scattering with no evidence of the electron state being silently discarded or recomputed anywhere downstream. Non-jet models are provably unable to execute any jet code path. That is the good news, and it is not a small thing to have verified.

The bad news is concentrated in three places that are all fixable but none of which could be confirmed as currently fixed: (1) the live production tuning script is currently deploying `constant_beta_e0_exponent=0`, which — as coded — removes the magnetic-field dependence from the jet temperature term entirely, in every recent wJET run there is evidence of; (2) the jet code path drops photons to invalid-frequency numerical failures far more often than the non-jet path, which could quietly bias exactly the region the jet model is meant to describe; and (3) nobody has yet run the one comparison that would actually test this against IPOLE at the spectrum level with a compatible viewing geometry — the tool to do it exists, and has been used for the non-jet models, but never for a jet-enabled one. Layered on top of that, the M_unit tuning campaign itself — jet and non-jet — appears to still be actively running as of the day of this audit, so even the numbers on disk today are a snapshot of unfinished work, not a final result.

**Should existing jet-enabled outputs be trusted in a paper right now? No, not yet — but not because the physics is wrong.** Every existing `*wJET*.h5` file should be treated as "code-plausible, not paper-ready" until (a) a fresh short run confirms whether `constant_beta_e0_exponent=1` changes the headline result, (b) the `isnan nu` drop-fraction is checked on the actual production dumps, and (c) at least one cone-restricted, angle-matched comparison against an IPOLE spectrum has been produced for a jet-enabled model. Given how clean the underlying equations turned out to be, none of that should take long relative to the compute already spent on M_unit tuning — but skipping it would mean publishing numbers whose jet-specific behavior has never actually been checked against anything outside GRMONTY itself.

---

## 13. Read-only compliance statement

**Commands used, by kind (during the original investigation):**
- Read-only file inspection: `Read` tool (cat -n semantics) on all source files quoted above; `find`, `stat`, `wc -l`, `diff`, `diff -q` via Bash.
- Read-only Git inspection: `git status`, `git status --short`, `git log`, `git log --oneline -S<symbol>`, `git branch -a -vv`, `git remote -v`, `git show <ref>[:path]`, `git diff <refA> <refB>`, `git diff --stat`, `git merge-base`, `git merge-base --is-ancestor` — all with `GIT_OPTIONAL_LOCKS=0`. No `add`, `commit`, `checkout`, `switch`, `reset`, `restore`, `clean`, `stash`, `merge`, `rebase`, or `pull` was ever invoked.
- Read-only data inspection: `grep`/`awk`/`head`/`tail`/`sed -n` (view-only invocations) and one `python3 -c` script that opened `munits_tuning_history.csv` with `csv.DictReader` and printed aggregates to stdout — it wrote nothing to any file.
- No build, compile, install, formatter, simulation, SLURM submission, or test-suite invocation was made at any point during the investigation.

**Confirmation:** During the original investigation, no source code, parameters, scripts, logs, outputs, documentation, or Git metadata were modified, created, deleted, renamed, or reformatted; no permissions or timestamps were changed; no mutating Git command was executed; and the pre-existing uncommitted state in both the `igrmonty` repo and the outer `/work/vmo703` repo was left exactly as found. This file itself was written afterward, as a separate, explicit user request to persist the completed audit to `docs/audits/`, matching this repository's existing convention (cf. `2026-04-06_m87_long_run_audit.md`).
