# The 12π question, settled by provenance — no author input required

2026-09-08 · evidence assembled from the local ARRicarte/ipole clone
(/work/vmo703/ipole/ipole, refreshed from GitHub today), the published papers,
and this repo's history. Companion to docs/2026-08-06_zoom_agenda_thetae_cap.md.

## The question

The constant-β_e supplement as coded (grmonty port ≡ ipole fork,
model/iharm/model.c:547) computes

    Θe = β_e0 · B_cgs² / (2(γe−1)) / (n_e m_e c²)        [12π ≈ 37.7× the papers]

while the defining papers say P_e = β_e0·P_B with P_B = B²/8π. Was the coded
form an intentional deviation or a transcription slip? We had framed this as
"Richard/Angelo adjudicate." It turns out the operationally decisive facts are
all checkable without them.

## Evidence chain

**1. The papers' definitions are unambiguous and pre-date the code.**
- Anantua+2020 (MNRAS 493, 1404; published 2020 February):
  "β_e = P_e/P_B = (γe−1)u_e/(b²/2) = β_e0" (code/HL units).
- Emami+2021 (ApJ 923, 272 = arXiv:2101.05327; **submitted 2021 January**,
  published 2021 December): Eq. 25 "P_B = B²/8π" (CGS), Eq. 34 "β_e = β_e0";
  β_e0 ∈ {10⁻⁶, 10⁻⁴, 10⁻²} surveyed, best-bet 10⁻².

**2. Neither paper's published science used the fork's line — it didn't exist
yet, and they used a different code.** Emami+2021's images and spectra were
produced with **GRTRANS (Dexter 2016)**, not ipole. The fork's constant-β_e
implementation was born 2021-05-24 — four months after Emami+2021 was
submitted, fifteen months after Anantua+2020 was published. **There is no
published result anywhere that depends on the 12π-hot form.**

**3. The fork's line is a post-publication port that churned through three
inconsistent formulas in ten weeks** (all commits by Angelo,
ARRicarte/ipole):

| commit | date | formula (× β_e0/(m_e c²)) | note |
|---|---|---|---|
| f5372c4 | 2021-05-24 | B²/(6(γe−1)) — **no n_e** | birth; comment: "Set the internal energy to be equal to a fraction of the magnetic energy density. Note that when ultrarelativistic, u = 3kT, not 3/2kT." Dimensionally broken (no density); the /6 carries the u=3kT factor and the /(γe−1) re-cancels it. |
| 7701658 | 2021-06-17 | B²/(2(γe−1))/n_e — **current form** | commit message mentions only "Added sigma_threshold"; the formula change (6→2, +n_e) rides along **silently**. |
| ecedb1c | 2021-07-28 | same, wrapped in exponent | attaches the comment "**//ARR: See definition in Anantua et al. (2020).**" |
| d88e5f1 | 2021-08-26 | unchanged | **last commit ever pushed to the fork** (verified by fetch 2026-09-08: upstream frozen since). |

**4. The line contradicts both of its author's own stated intents.**
- vs. the cited paper (pressure ratio, P_B = B²/8π): **hot by 12π ≈ 37.7**.
- vs. the birth comment (energy-density ratio u_e = β_e0·u_B, with u = 3kT):
  should be Θe = β_e0B²/(24π n_e m_e c²) — coded form is **hot by 36π ≈ 113**.
No reading of the author's own comments reproduces the coded line.

**5. Internal inconsistency within the same function.** The R-β branch a few
lines up computes plasma β correctly in code units
(`beta = p[UU]*(gam-1.)/0.5/bsq`, where P_B = bsq/2 is right). The
constant-β_e line applies that same code-units magnetic form to
`b = sqrt(bsq)*B_unit` — a **Gauss-converted** field (the ×4π), then divides
by (γe−1) where a pressure identity needs no factor (the ×3). Two textbook
unit slips, stacked: 4π × 3 = 12π.

**6. Independent corroborations already in hand.**
- The empirical e0 rescale found months ago (~2.7×10⁻³ needed to make the
  supplement sane) equals 0.1/(12π) = 2.65×10⁻³.
- As-coded Θe ≈ 3461σ ⇒ T_e up to ~10¹⁴ K, super-virial, cap-saturated at
  every σ ≥ 0.3 by construction. The EHT-context electron temperatures Emami+2021
  quote are (5–35)×10¹⁰ K. Paper-literal gives 91.8σ (sheath 270–1000 with the
  R-β base; cap a boundary effect at 11.3%).
- Fixed-M_unit spectrum A/B in flight (scratch/pb_fix_ab/, jobs 807327/807328).

**7. Chain of custody into this repo.** grmonty's copy is ashton's faithful
port of Angelo's line (igrmonty c359811, 2025-11-21) — the 12π was inherited,
not introduced here.

## Verdict

**Transcription slip in a post-publication port, to the strength provenance
can deliver.** The "intentional deviation" hypothesis now requires Angelo to
have cited a specific published definition while silently coding something
37.7× different that also contradicts his own birth comment, in a formula he
rewrote three times without ever remarking on the physics — and for that
choice to matter, some published result would have to depend on it, and none
does. The subjective-intent question is thereby **moot**: whatever was in
anyone's head in June 2021, the published definition is P_e = β_e0·B²/8π and
no publication ever used anything else.

## What this means for the M87 paper

- **The tuned 48-run production grid is untouched by the fix.** Production
  MADs ran plain R-β/Crit-β (no wJET); production SANE wJET supplements are
  nearly extinct (38 zones in the rh160 census). The 12π only bites the
  go-forward MAD wJET configurations — exactly the ones still being designed.
- **Adopt the paper-literal form** for those: `constant_beta_paper_literal 1`
  (implemented, verified C ≡ numpy to 7×10⁻¹²), cite Anantua+2020 Eq. and
  Emami+2021 Eq. 25 in methods, and note the fork discrepancy in an appendix
  or footnote. The identical one-line gate must go into the ipole working copy
  before any P4.3 imaging comparison.
- (Context, not verdict: our β_e0 = 0.1 is itself 10× Emami's best-bet 10⁻² —
  a separate, legitimate model choice to state explicitly.)

## What still needs the group (and when)

- Nothing *blocks* on Richard now. Coauthor sign-off on the adopted definition
  happens naturally in draft review, against this written evidence, instead of
  as a pre-emptive email ruling. The drafted email can be repurposed as an FYI
  ("we traced it; here's the provenance; adopting paper-literal") or dropped.
- Only genuinely irreducible unknown: whether other *unpublished* group work
  built on the hot line since 2021 (their concern, not this paper's).
