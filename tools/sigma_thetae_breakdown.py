#!/usr/bin/env python3
"""sigma/thetae zone breakdown for iharm dumps under the wJET electron model.

Faithful numpy port of model/iharm/model.c (thetae_func + get_fluid_zone paths,
2026-08 state): R-Beta base prescription, additive constant-beta-e0 supplement
where sigma >= sigma_transition, hard jet override (thetae = jet_thetae) where
sigma >= jet_sigma_cut OR beta <= jet_beta_cut, clamps THETAE_MIN=1e-3 /
rb_floor=1e-3 / SIGMA_MAX=300 / BETA_FLOOR=1e-5 / THETAE_HARD_MAX=1e3.
b^2 is built from prims through the FMKS metric exactly as get_fluid_zone does.

Outputs (per dump) under igrmonty_outputs/m87/_qa/:
  plots/zone_breakdown/zones_<tag>.png   2D hist + assigned/pre-clamp maps
  zone_breakdown_summary.csv             fractions + percentiles per dump

Run:  /work/vmo703/ipole_venv/bin/python tools/sigma_thetae_breakdown.py
"""

import argparse
import csv
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

# CGS constants (constants.h values)
ME = 9.1093826e-28
MP = 1.67262171e-24
CL = 2.99792458e10
GNEWT = 6.6742e-8
MSUN = 1.989e33

# model.c clamps
THETAE_MIN = 1e-3
RB_FLOOR = 1e-3
SIGMA_MAX = 300.0
BETA_FLOOR = 1e-5
THETAE_HARD_MAX = 1.0e3

# dataviz palette (light mode)
INK = "#0b0b0b"
INK2 = "#52514e"
SURFACE = "#fcfcfb"

QA_DIR = "/work/vmo703/igrmonty_outputs/m87/_qa"
OUT_PLOTS = os.path.join(QA_DIR, "plots", "zone_breakdown")
OUT_CSV = os.path.join(QA_DIR, "zone_breakdown_summary.csv")

# Configs mirror the runs the email numbers come from (current code defaults:
# constant_beta_e0_exponent = 1.0, as in par.c and the P3.1 test par).
PRESETS = [
    dict(
        tag="Ma+0.94_4000_RBETAwJET_rh80",
        dump="/work/vmo703/grmhd_dump_samples/Ma+0.94_4000.h5",
        note="P3.1 MAD test config (scratch/p3_jetcheck par): M_unit = April frozen value",
        MBH=6.5e9, M_unit=1.3488e25,
        trat_small=1.0, trat_large=80.0,
        sigma_transition=2.0, constant_beta_e0=0.1, constant_beta_e0_exponent=1.0,
        jet_sigma_cut=10.0, jet_beta_cut=0.1, jet_thetae=50.0,
    ),
    dict(
        tag="Sa-0.5_4000_RBETAwJET_rh160",
        dump="/work/vmo703/grmhd_dump_samples/Sa-0.5_4000.h5",
        note="SANE production config, July-corpus M_unit (pre_bugfix pos0 file)",
        MBH=6.5e9, M_unit=3.726281e28,
        trat_small=1.0, trat_large=160.0,
        sigma_transition=2.0, constant_beta_e0=0.1, constant_beta_e0_exponent=1.0,
        jet_sigma_cut=10.0, jet_beta_cut=0.1, jet_thetae=50.0,
    ),
]


def load_dump(path):
    with h5py.File(path, "r") as f:
        h = f["header"]
        d = dict(
            a=float(h["geom/fmks/a"][()]),
            hslope=float(h["hslope"][()]),
            mks_smooth=float(h["mks_smooth"][()]),
            poly_alpha=float(h["poly_alpha"][()]),
            poly_xt=float(h["poly_xt"][()]),
            poly_norm=float(h["poly_norm"][()]),
            gam=float(h["gam"][()]),
            n1=int(h["n1"][()]), n2=int(h["n2"][()]), n3=int(h["n3"][()]),
            startx=np.array([0.0, float(h["geom/startx1"][()]),
                             float(h["geom/startx2"][()]), float(h["geom/startx3"][()])]),
            dx=np.array([0.0, float(h["geom/dx1"][()]),
                         float(h["geom/dx2"][()]), float(h["geom/dx3"][()])]),
        )
        prims = np.array(f["prims"], dtype=np.float64)  # (n1,n2,n3,8) RHO UU U1-3 B1-3
    assert abs(d["poly_alpha"] - round(d["poly_alpha"])) < 1e-12 and int(d["poly_alpha"]) % 2 == 0, \
        "theta-stretch port assumes even integer poly_alpha"
    return d, prims


def bl_coord(hdr, X1, X2):
    """FMKS (with_derefine_poles) bl_coord, vectorized. X1,X2 broadcastable."""
    r = np.exp(X1)
    hs = hdr["hslope"]
    thG = np.pi * X2 + ((1.0 - hs) / 2.0) * np.sin(2.0 * np.pi * X2)
    y = 2.0 * X2 - 1.0
    yx = y / hdr["poly_xt"]
    alpha = hdr["poly_alpha"]
    thJ = hdr["poly_norm"] * y * (1.0 + np.abs(yx) ** alpha / (alpha + 1.0)) + 0.5 * np.pi
    fac = np.exp(hdr["mks_smooth"] * (hdr["startx"][1] - X1))
    th = thG + fac * (thJ - thG)
    # analytic derivatives for the Jacobian
    dr_dX1 = r
    dthG_dX2 = np.pi + np.pi * (1.0 - hs) * np.cos(2.0 * np.pi * X2)
    dthJ_dX2 = 2.0 * hdr["poly_norm"] * (1.0 + np.abs(yx) ** alpha)
    dth_dX1 = -hdr["mks_smooth"] * fac * (thJ - thG)
    dth_dX2 = dthG_dX2 + fac * (dthJ_dX2 - dthG_dX2)
    return r, th, dr_dX1, dth_dX1, dth_dX2


def ks_gcov(r, th, a):
    """Kerr-Schild gcov(t,r,th,phi), shapes broadcast to r/th."""
    cth, sth = np.cos(th), np.sin(th)
    s2 = sth * sth
    rho2 = r * r + a * a * cth * cth
    g = np.zeros(r.shape + (4, 4))
    tfac = 2.0 * r / rho2
    g[..., 0, 0] = -1.0 + tfac
    g[..., 0, 1] = g[..., 1, 0] = tfac
    g[..., 0, 3] = g[..., 3, 0] = -tfac * a * s2
    g[..., 1, 1] = 1.0 + tfac
    g[..., 1, 3] = g[..., 3, 1] = -a * s2 * (1.0 + tfac)
    g[..., 2, 2] = rho2
    g[..., 3, 3] = s2 * (rho2 + a * a * s2 * (1.0 + tfac))
    return g


def fmks_geometry(hdr):
    """gcov/gcon/sqrt(-g) and (r,th) at zone centers, shape (n1,n2,...)."""
    i = np.arange(hdr["n1"])
    j = np.arange(hdr["n2"])
    X1 = hdr["startx"][1] + (i + 0.5) * hdr["dx"][1]
    X2 = hdr["startx"][2] + (j + 0.5) * hdr["dx"][2]
    X1g, X2g = np.meshgrid(X1, X2, indexing="ij")
    r, th, dr1, dth1, dth2 = bl_coord(hdr, X1g, X2g)
    gks = ks_gcov(r, th, hdr["a"])
    J = np.zeros(r.shape + (4, 4))  # dx^mu/dX^nu, x=(t,r,th,phi)
    J[..., 0, 0] = 1.0
    J[..., 1, 1] = dr1
    J[..., 2, 1] = dth1
    J[..., 2, 2] = dth2
    J[..., 3, 3] = 1.0
    gcov = np.einsum("...ma,...mn,...nb->...ab", J, gks, J)
    gcon = np.linalg.inv(gcov)
    detg = -np.linalg.det(gcov)
    return r, th, gcov, gcon, np.sqrt(np.clip(detg, 0.0, None))


def fluid_quantities(hdr, prims, gcov, gcon):
    """Port of get_fluid_zone: bsq in code units, per (n1,n2,n3)."""
    Vcon = prims[..., 2:5]  # U1 U2 U3
    Bp = prims[..., 5:8]    # B1 B2 B3
    gc = gcov[:, :, None, :, :]
    VdotV = np.einsum("...lm,...l,...m->...", gc[..., 1:, 1:], Vcon, Vcon)
    g00 = gcon[:, :, None, 0, 0]
    Vfac = np.sqrt(-1.0 / g00 * (1.0 + np.abs(VdotV)))
    Ucon = np.zeros(prims.shape[:3] + (4,))
    Ucon[..., 0] = -Vfac * g00
    Ucon[..., 1:] = Vcon - Vfac[..., None] * gcon[:, :, None, 0, 1:]
    Ucov = np.einsum("...ab,...b->...a", gc, Ucon)
    UdotBp = np.einsum("...l,...l->...", Ucov[..., 1:], Bp)
    Bcon = np.zeros_like(Ucon)
    Bcon[..., 0] = UdotBp
    Bcon[..., 1:] = (Bp + Ucon[..., 1:] * UdotBp[..., None]) / Ucon[..., 0:1]
    Bcov = np.einsum("...ab,...b->...a", gc, Bcon)
    bsq = np.einsum("...a,...a->...", Bcon, Bcov)
    return np.clip(bsq, 0.0, None)


def thetae_wjet(cfg, rho, uu, b_code, game=4.0 / 3.0, gamp=5.0 / 3.0, gam=4.0 / 3.0,
                Ne_unit=None, B_unit=None):
    """Port of thetae_func (with_electrons=4, R-Beta + supplement + override).

    Returns dict with sigma, beta, base, supplement, pre-clamp, assigned, masks.
    """
    safe_rho = np.clip(rho, 1e-30, None)
    safe_uu = np.clip(uu, 1e-30, None)
    safe_B = np.abs(b_code)

    sigma = np.clip(safe_B**2 / safe_rho, 0.0, SIGMA_MAX)
    denom_beta = 0.5 * safe_B**2
    beta = np.where(denom_beta > 0, safe_uu * (gam - 1.0) / np.where(denom_beta > 0, denom_beta, 1.0), np.inf)
    beta_cl = np.clip(beta, BETA_FLOOR, None)

    # R-Beta base (with_electrons==4 branch)
    b2 = (beta_cl / cfg["beta_crit"]) ** 2 if cfg.get("beta_crit") else beta_cl**2
    inv = 1.0 / (1.0 + b2)
    trat = cfg["trat_large"] * b2 * inv + cfg["trat_small"] * inv
    denom = (gamp - 1.0) + (game - 1.0) * trat
    thetae_base = (MP / ME) * (game - 1.0) * (gamp - 1.0) / denom * safe_uu / safe_rho
    thetae_floor = max(THETAE_MIN, RB_FLOOR)

    # constant-beta supplement (energy_density = B_cgs^2 / (2(game-1)))
    ne_cgs = safe_rho * Ne_unit
    B_cgs = safe_B * B_unit
    energy_density = B_cgs**2 / (2.0 * (game - 1.0))
    term = energy_density ** cfg["constant_beta_e0_exponent"]
    thetae_const = cfg["constant_beta_e0"] * term / (ne_cgs * ME * CL * CL)
    thetae_const = np.where(np.isfinite(thetae_const) & (thetae_const > 0), thetae_const, 0.0)

    in_high_sigma = sigma >= cfg["sigma_transition"]
    in_jet = (sigma >= cfg["jet_sigma_cut"]) | (beta_cl <= cfg["jet_beta_cut"])

    # precedence: hard override wins; else additive supplement; then clamps
    pre_clamp = np.where(
        in_jet, cfg["jet_thetae"],
        np.where(in_high_sigma, thetae_base + thetae_const, thetae_base),
    )
    assigned = np.clip(pre_clamp, thetae_floor, THETAE_HARD_MAX)

    return dict(
        sigma=sigma, beta=beta_cl, thetae_base=thetae_base,
        thetae_const=thetae_const, pre_clamp=pre_clamp, assigned=assigned,
        in_jet=in_jet, in_supplement=in_high_sigma & ~in_jet,
        clamped=(pre_clamp > THETAE_HARD_MAX),
    )


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8.5)


def make_figure(tag, cfg, q, r, th, sqrtg, ne, B_cgs, out_png):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), facecolor=SURFACE)

    # emissivity proxy weight per zone: ne * thetae^2 * B^2 * sqrt(-g)
    w_em = (ne * q["assigned"] ** 2 * B_cgs**2 * sqrtg[:, :, None]).ravel()

    # panel 1: sigma vs pre-clamp thetae, emissivity-weighted
    ax = axes[0]
    style_ax(ax)
    sig = np.clip(q["sigma"].ravel(), 1e-4, None)
    thp = np.clip(q["pre_clamp"].ravel(), 1e-3, None)
    hb = ax.hist2d(
        np.log10(sig), np.log10(thp), bins=140,
        range=[[-4, np.log10(SIGMA_MAX)], [-3, 5]],
        weights=w_em, norm=LogNorm(), cmap="Blues",
    )
    fig.colorbar(hb[3], ax=ax, label="emissivity proxy  neΘe²B²√g")
    ax.axvline(np.log10(cfg["sigma_transition"]), color=INK, lw=1.0, ls=(0, (4, 3)))
    ax.axvline(np.log10(cfg["jet_sigma_cut"]), color=INK, lw=1.0, ls=(0, (1, 2)))
    ax.axhline(3.0, color="#e34948", lw=1.2)
    ax.text(np.log10(cfg["sigma_transition"]) + 0.05, 4.6, "σ_transition", fontsize=8, color=INK)
    ax.text(np.log10(cfg["jet_sigma_cut"]) + 0.05, 4.1, "σ jet cut", fontsize=8, color=INK)
    ax.text(-3.85, 3.12, "Θe hard cap 10³", fontsize=8, color="#e34948")
    ax.set_xlabel("log₁₀ σ", fontsize=9.5, color=INK)
    ax.set_ylabel("log₁₀ Θe (pre-clamp)", fontsize=9.5, color=INK)
    ax.set_title("where the model wants to go", fontsize=10, color=INK, loc="left")

    # panels 2+3: phi-averaged maps, assigned vs pre-clamp
    x = r * np.sin(th)
    z = r * np.cos(th)
    for ax, field, label in (
        (axes[1], np.mean(q["assigned"], axis=2), "Θe assigned (clamped)"),
        (axes[2], np.mean(np.clip(q["pre_clamp"], THETAE_MIN, 1e5), axis=2), "Θe pre-clamp"),
    ):
        style_ax(ax)
        pm = ax.pcolormesh(x, z, field, norm=LogNorm(vmin=1e-2, vmax=1e5),
                           cmap="Blues", shading="auto", rasterized=True)
        sig2d = np.mean(q["sigma"], axis=2)
        ax.contour(x, z, sig2d, levels=[cfg["sigma_transition"]], colors=[INK], linewidths=0.9, linestyles="dashed")
        ax.contour(x, z, sig2d, levels=[cfg["jet_sigma_cut"]], colors=[INK], linewidths=0.9, linestyles="dotted")
        ax.set_xlim(0, 40)
        ax.set_ylim(-40, 40)
        ax.set_aspect("equal")
        ax.set_xlabel("x  [r_g]", fontsize=9.5, color=INK)
        ax.set_title(label, fontsize=10, color=INK, loc="left")
        fig.colorbar(pm, ax=ax, label="Θe (φ-avg)")
    axes[1].set_ylabel("z  [r_g]", fontsize=9.5, color=INK)

    fig.suptitle(
        f"{tag} — σ/Θe zone breakdown (dashed σ={cfg['sigma_transition']:g}, dotted σ={cfg['jet_sigma_cut']:g})",
        fontsize=11.5, color=INK, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print("[fig]", out_png)


def main():
    ap = argparse.ArgumentParser(description="sigma/thetae zone breakdown")
    ap.add_argument("--preset", default="all", help="preset tag substring or 'all'")
    args = ap.parse_args()

    os.makedirs(OUT_PLOTS, exist_ok=True)
    rows = []
    for cfg in PRESETS:
        if args.preset != "all" and args.preset not in cfg["tag"]:
            continue
        print(f"[run] {cfg['tag']}  dump={cfg['dump']}")
        hdr, prims = load_dump(cfg["dump"])
        cfg = dict(cfg, beta_crit=1.0)

        Mbh_cgs = cfg["MBH"] * MSUN
        L_unit = GNEWT * Mbh_cgs / CL**2
        RHO_unit = cfg["M_unit"] / L_unit**3
        B_unit = CL * np.sqrt(4.0 * np.pi * RHO_unit)
        Ne_unit = RHO_unit / (MP + ME)

        r, th, gcov, gcon, sqrtg = fmks_geometry(hdr)
        bsq = fluid_quantities(hdr, prims, gcov, gcon)
        rho = prims[..., 0]
        uu = prims[..., 1]
        b_code = np.sqrt(bsq)

        q = thetae_wjet(cfg, rho, uu, b_code, gam=hdr["gam"],
                        Ne_unit=Ne_unit, B_unit=B_unit)
        ne = np.clip(rho, 1e-30, None) * Ne_unit
        B_cgs = b_code * B_unit

        w_em = ne * q["assigned"] ** 2 * B_cgs**2 * sqrtg[:, :, None]
        w_tot = float(np.sum(w_em))
        n_tot = q["sigma"].size

        def frac(mask, w=None):
            if w is None:
                return float(np.count_nonzero(mask)) / n_tot
            return float(np.sum(w_em[mask])) / w_tot if w_tot > 0 else float("nan")

        sup = q["in_supplement"]
        row = dict(
            tag=cfg["tag"], dump=os.path.basename(cfg["dump"]), note=cfg["note"],
            M_unit=cfg["M_unit"], B_unit_G=B_unit, Ne_unit=Ne_unit,
            trat_large=cfg["trat_large"],
            f_zones_supplement=frac(sup),
            f_zones_override=frac(q["in_jet"]),
            f_zones_clamped=frac(q["clamped"]),
            f_emis_supplement=frac(sup, w=1),
            f_emis_override=frac(q["in_jet"], w=1),
            f_emis_clamped=frac(q["clamped"], w=1),
            supp_thetae_preclamp_p50=float(np.median(q["pre_clamp"][sup])) if sup.any() else float("nan"),
            supp_thetae_preclamp_p95=float(np.percentile(q["pre_clamp"][sup], 95)) if sup.any() else float("nan"),
            supp_thetae_preclamp_max=float(np.max(q["pre_clamp"][sup])) if sup.any() else float("nan"),
            f_supp_zones_above_cap=float(np.count_nonzero(q["pre_clamp"][sup] > THETAE_HARD_MAX)) / max(1, np.count_nonzero(sup)),
        )
        rows.append(row)
        print(
            f"  zones: supplement {row['f_zones_supplement']:.2%}, override {row['f_zones_override']:.2%}, "
            f"clamped {row['f_zones_clamped']:.2%} | emissivity-weighted: supplement {row['f_emis_supplement']:.2%}, "
            f"override {row['f_emis_override']:.2%}, clamped {row['f_emis_clamped']:.2%}"
        )
        print(
            f"  supplement-band pre-clamp Θe: median {row['supp_thetae_preclamp_p50']:.3g}, "
            f"p95 {row['supp_thetae_preclamp_p95']:.3g}, max {row['supp_thetae_preclamp_max']:.3g}; "
            f"{row['f_supp_zones_above_cap']:.2%} of band zones above the 10³ cap"
        )
        make_figure(cfg["tag"], cfg, q, r, th, sqrtg, ne, B_cgs,
                    os.path.join(OUT_PLOTS, f"zones_{cfg['tag'].replace('+','p').replace('.','_')}.png"))

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print("[done]", OUT_CSV)


if __name__ == "__main__":
    main()
