P2.4 (exponent sweep): L = 2.32e38 / 7.14e39 / 7.74e39 erg/s at exponent 0.0/0.5/1.0. H1 understated luminosity by ~33× — confirms the audit's concern was not overstated. Most of the effect (30.7× of the 33×) already shows up at exponent=0.5, highly nonlinear near the low end.

P2.1 (single-cell vs. formula): found a real zone at σ=2.771 (just above sigma_transition). GRMONTY's actual compiled code printed Thetae=1000.000000000. Independently recomputing the audited formula by hand from the same printed B/Ne gives a raw value of 9597.16, correctly saturating at THETAE_HARD_MAX=1000 — exact match. This is a real-code confirmation, not just the formula-on-paper agreement the original audit already established.

P2.2 (regression vs. pre-jet baseline): built commit 4a1b1c5 in an isolated worktree and diffed full Θe fields — not bit-identical, contrary to the plan's original assumption, but every difference is now understood, not a mystery:
1. Baseline had an unconditional floor on the innermost 9 radial zones (i<9) that's simply gone from the current tree — this refines Finding L4 (previously just "the sigma-cut got scoped to with_electrons<3"; now we know the near-horizon floor was also dropped entirely).
2. Excluding that, Crit-Beta matches baseline to floating-point noise — a clean independent confirmation that this session's M1 fix correctly restored baseline's original behavior.
3. Excluding that, R-β has a new ~1e-3 floor baseline never had — not previously documented anywhere, likely intentional (matches IPOLE's convention) but newly surfaced by this comparison.




1. The verified strong boosting lives downstream — HST-1 and the optical knots sit at ~10⁵–10⁶ r_g. At the jet base, the region grmonty actually images, measured apparent speeds are sub-relativistic and accelerate to Γ ~ a few only over ~10³–10⁵ r_g, and the base is limb-brightened with a dim spine. So your "230 GHz funnel emission comes from slower sheath material" reading is the standard interpretation, and it's supported by the base data specifically, not just inferred.
2. At θ ≈ 17°, the Doppler factor peaks for Γ ≈ 1/sin θ ≈ 3.4 — a much faster spine deboosts again. Fastest ≠ brightest, which is another reason the mildly relativistic sheath can dominate at the base even pole-on.

What it does not answer: your Θe cap question. Boosting redistributes apparent brightness with viewing angle; it cannot produce your ~40× (SANE) or ~400× (MAD) jet/nonjet ratios, because those are angle-integrated luminosities in grmonty — intrinsic fluid-frame emissivity increases from hotter funnel electrons. Beaming neither explains nor excuses them.




Why this one parameter has outsized importance — the causal chain:

- It binds exactly where the simulation gives no guidance. The capped zones are floor-density zones: Θe there comes from b²/n_e with an arbitrary n_e, so nothing in the GRMHD constrains the value. It's a pure model choice — which is why a referee will probe it.
- It sets where the funnel emits, not just how much. Thermal synchrotron power goes roughly as n_e Θe² B² and peaks near ν ∝ B Θe². At 10 G, Θe = 10³ puts the funnel's synchrotron bump in the IR; Θe = 10⁴ puts it in the UV. Compton is steeper still: photons gain up to ~16Θe² per scattering, so a single scatter off cap-level electrons moves a mm photon six-plus decades. The cap therefore controls the entire high-energy half of the SED — precisely the bands (optical, X-ray) where M87's faint core is well measured.
- M_unit retuning propagates the choice into every band. Because each model is renormalized to 0.5 Jy at 230 GHz, a hotter funnel doesn't just add luminosity — it pulls M_unit down and dims the disk, reshaping the whole SED toward jet dominance. Your 400× jet/nonjet MAD measurement means the post-fix retune will move M_units a lot, and where the X-ray lands afterward is exactly what our new gate measures.
- It sets the price of production. Compton dominance is what makes the jet arm cost 40× — your own phrase, "the cap sets the price of production," is literally true: raising it to 10⁴ would multiply the scattering load again.
- It's not a free dial in the code. The ceiling is 10³ partly for numerical reasons: the hotcross tables (the file you had open) and at least one other ceiling (GAMMACUT = 1000) are built to the same scale, and our deep-KN sampling fix kicks in at Θe ≥ 100. Raising the cap is a code-audit item, not a parameter edit.

Implications by scenario: Lower it (~few hundred): funnel dims, IR/X-ray comfortably safe, compute drops — but the jet-aware feature that motivates the paper weakens. Keep 10³: current behavior; the pre-fix corpus sits ≥4.6× under the Chandra+NuSTAR core, but the post-fix MAD runs will land much closer, and the gate decides. Raise toward 10⁴: synchrotron bump into the UV, Compton into γ-rays — after retuning this almost certainly overshoots the 4.4×10⁴⁰ erg/s X-ray core and plausibly the ~mJy NIR nucleus even sooner (the hottest existing family already reaches ~0.5 mJy at 2.2 μm), on top of the code-table and cost problems. Effectively ruled out by data, and we can demonstrate that cheaply rather than argue it.






My read: this is the most useful email of the thread — it settles your question 2 (with a twist your code makes concrete), reframes question 1 with a new argument, pushes back fairly on one piece of my framing, and asks two questions you can now answer better than he expects. Point by point:

1. The numerical-trust framing of Q1, plus a genuinely new point. His argument — the funnel obviously radiates in reality, but the simulation doesn't model its gas credibly, so post-processing radiation there is unjustified — is the clean version of the σ-cut rationale. The entropy-inversion point is new to our discussion and strengthens it: iharm-family solvers fall back to an entropy update when the primitive inversion fails, which happens preferentially in the funnel, so energy isn't even conserved there — u is doubly unreliable. Our zone breakdown adds the sharp corollary: your supplement avoids u but keeps ρ, and with exponent=1 it's Θe ≈ 3.5×10³·σ where σ = b²/ρ — so in its active band its value is set directly by the floor density. Both u-based and b²/ρ-based prescriptions inherit funnel arbitrariness; they just inherit different artifacts. I checked whether we could quantify his point: your dump samples' extras group is empty — no inversion-fail flags recorded — so the fixup fraction can't be measured from these files; it stays caveat language unless you get full dumps with fail masks.

2. His table advice — I looked, and the table is not the bottleneck. He said stick to Θe ≤ 1000 if that's what your table supports and to check why. The check: the hotcross Compton cross-section table extends to Θe = 10⁴ (MINT=1e-4, MAXT=1e4, src/hotcross.c:26-27). What sits at 10³ is two deliberate code clamps — THETAE_HARD_MAX = 1e3 (model/iharm/model.c:9) and SCATTERING_THETAE_MAX = 1000 (src/decs.h:26). So "the cap is where the table ends" is not true of your code — the table leaves headroom to 10⁴, and the real soft constraints above 10³ are emissivity-fit validity and sampling (the regime our deep-KN fix at Θe ≥ 100 patched). This also corrects something I said earlier — I'd guessed the tables were built to the 10³ scale; they're not. His conclusion (stay at 10³) remains sound as conservative practice, but the justification should be "model-level regularizer, validated against data," not "table bound" — worth having straight before the zoom so the discussion doesn't anchor on a limit that doesn't exist.

3. His two direct questions — you can answer both precisely, with figures. On "are you just assigning density and temperature from the GRMHD?": density yes, straight from GRMHD ρ, floors included; temperature no in the jet — GRMHD u is unused where σ ≥ 2; Θe there is the parametrization (supplement ≈ 3.5×10³σ, capped at 10³, in the 2 ≤ σ < 10 sheath; hard Θe = 50 where σ ≥ 10 or β ≤ 0.1). The two zone-breakdown panels are this answer — attach them. On "two-temperature, R_high/R_low?": yes — your base model literally is R_high/R_low (R-β with R_high = 80/160, R_low = 1) on single-fluid GRMHD. But his question surfaces the sharpest self-critique yet: the supplement inverts the usual hierarchy. He says jet electrons are usually 10–100× colder than ions; your supplement's pre-clamp Θe of 7×10³–3.5×10⁴ corresponds to T_e ≈ 4×10¹⁶ K — hotter than the virial ion temperature (T_i ≈ T_vir at r = 3 r_g is Θe-equivalent ~600). The constant-β_e model makes funnel electrons super-virial by tapping magnetic pressure against a floor density. Expect that to be the first thing probed on the call.

4. On "flash cooling" — his pushback is fair, and the disagreement is smaller than it looks. That was my phrase, not literature language. The defensible core survives: t_syn(Θe = 10³, B ≈ 10 G) ≈ 0.08 r_g/c, so hot funnel electrons are fast-cooling and high Θe persists only if reconnection/dissipation heating balances it — which is exactly what he says happens. Nobody disputes heating exists; the open question is the equilibrium value, and no first-principles number pins it — which is precisely why it ends up a parametrization. Two-temperature radiative GRMHD work on M87 (Ryan et al. 2018 is the one to name) found electron cooling materially reshapes T_e near the jet base at M87-like accretion rates; I can verify the exact claim before you cite it on the call.

The zoom: strongly agree, and you're unusually well-armed for it. The decision list practically writes itself: (1) σ trust boundary — emit from the sheath only, or the whole funnel with caveats; (2) cap justification — regularizer + SED-gate language, not table language; (3) the e0 coherence choice our breakdown exposed — keep the clamp-saturated model (effectively "Θe = 10³ sheath") or rescale e0 to ~3×10⁻³ so the supplement expresses real σ-dependence below the cap; (4) the β_cut scope problem — in SANE, β ≤ 0.1 paints mid-latitude material far outside the funnel and dominates the emission proxy; (5) energy-conservation caveat wording for the paper. If you want, I'll assemble a one-page agenda for the call with those five items and the four figures (two zone breakdowns, two SED gates) so you and Richard are looking at the same evidence — say the word.








Epistemic care: I'm not declaring this a bug. The ipole lines are your group's own hand, and Anantua+2020's abstract mentions "generalizing equipartition of particle and magnetic energies," so an intentionally hotter energy-density variant isn't impossible. But the burden has flipped — the code disagrees with the paper's stated definition by 38×, and that's now question #1 for the Zoom, ahead of everything else, because its answer decides whether decisions 2–4 even exist. (Also to keep separate: this is not the B_unit scaling bug you already fixed — that was about code-vs-cgs field units; this 12π is a residual convention gap that survives that fix.)

