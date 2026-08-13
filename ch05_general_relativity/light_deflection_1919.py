"""29 May 1919: a factor of two that made Einstein famous overnight.

Newton's theory is not silent about light bending. If light is corpuscles moving at
speed c, gravity deflects them, and Soldner worked out the answer in 1801:

    Newton / Soldner:   α = 2GM / (c²b)      = 0.87″ at the Sun's limb
    Einstein 1911:      α = 2GM / (c²b)      — he got the same wrong answer first!
    Einstein 1915:      α = 4GM / (c²b)      = 1.75″   ← exactly twice

The factor of 2 is the entire physical content. In the Newtonian picture only TIME
is warped (that is the 1911 calculation, which is really just the equivalence
principle). In full general relativity SPACE is curved too, and the spatial
curvature contributes an equal second half. Measuring 1.75″ rather than 0.87″ is a
direct measurement that space itself is curved.

Einstein was lucky twice. His 1911 paper prompted an eclipse expedition to Crimea in
1914 which was interrupted by the outbreak of war — had they succeeded they would
have refuted his (then wrong) prediction. By 1919 he had the right answer.

    phenomenon:   stars near the eclipsed Sun appear shifted outward
    simulation:   integrate a photon's path in the Sun's field, both theories
    dissection:   measure the total bend angle as a function of impact parameter
    formula:      α = 4GM/(c²b), and the 1919 Sobral/Príncipe plates that chose
                  between the two candidate numbers.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

G, C = 6.67430e-11, 2.99792458e8
M_SUN, R_SUN = 1.98892e30, 6.957e8
GM = G * M_SUN
RAD2ARCSEC = 180 / np.pi * 3600


def deflection(b, relativistic=True, n_pts=200_000, phi_span=None):
    """Integrate the photon orbit  u'' + u = f·(3GM/c²)·u²  and measure the bend.

    For light there is no GM/h² source term (a photon has no Newtonian orbit to
    perturb); the entire equation IS the correction. Setting f = 1/2 reproduces the
    Newtonian corpuscle answer, f = 1 the full general-relativistic one.
    """
    f = 1.0 if relativistic else 0.5
    phi_span = phi_span or np.pi * 1.2
    dphi = phi_span / n_pts
    # Start at infinity, moving inward, with impact parameter b: u = 0, u' = 1/b.
    # u must be EXACTLY zero here. Seeding it with a small epsilon instead would
    # place the photon a finite distance in, offsetting phi by ~u0·b radians —
    # which for u0 = 1e-12 is 7e-4 rad, eighty times larger than the effect we
    # are trying to measure.
    u = torch.tensor(0.0, dtype=torch.float64)
    du = torch.tensor(1.0 / b, dtype=torch.float64)

    def acc(u_):
        return f * 3 * GM * u_**2 / C**2 - u_

    us = torch.empty(n_pts + 1, dtype=torch.float64)
    us[0] = u
    a = acc(u)
    for i in range(n_pts):
        du = du + 0.5 * dphi * a
        u = u + dphi * du
        a = acc(u)
        du = du + 0.5 * dphi * a
        us[i + 1] = u
        if u < 0:                      # photon has escaped back to infinity
            us = us[:i + 2]
            break
    phis = torch.arange(len(us), dtype=torch.float64) * dphi

    # The ray comes in from u = 0 and leaves at u = 0. A straight line would span
    # exactly π of azimuth between those two points; the excess IS the deflection.
    # Find the outgoing zero crossing by linear interpolation on the last two samples.
    j = len(us) - 2
    phi_out = phis[j].item() + dphi * (us[j] / (us[j] - us[j + 1])).item()
    return phi_out - np.pi, us, phis


b = R_SUN
alpha_gr, us_gr, phis_gr = deflection(b, relativistic=True)
alpha_nt, us_nt, phis_nt = deflection(b, relativistic=False)

print("Photon grazing the Sun's limb (b = R_sun):")
print(f"  integrated, full GR       α = {alpha_gr*RAD2ARCSEC:.4f}″")
print(f"  closed form 4GM/(c²b)     α = {4*GM/(C**2*b)*RAD2ARCSEC:.4f}″")
print(f"  integrated, Newton/Soldner α = {alpha_nt*RAD2ARCSEC:.4f}″")
print(f"  closed form 2GM/(c²b)     α = {2*GM/(C**2*b)*RAD2ARCSEC:.4f}″")
print(f"  ratio GR / Newton = {alpha_gr/alpha_nt:.4f}   ← the famous factor of 2\n")

print("Where does the factor of 2 come from?")
print("  half from the warping of TIME   (gravitational redshift / equivalence")
print("                                   principle — Einstein had this in 1911)")
print("  half from the curvature of SPACE (only in the full 1915 field equations)")
print("  Measuring 1.75″ instead of 0.87″ is therefore a direct detection of")
print("  spatial curvature. There is no other way to read the number.\n")

# --- the 1919 data ---
print("The 1919 eclipse expeditions (Dyson, Eddington & Davidson, 1920):")
results_1919 = [("Sobral, Brazil (4-inch)", 1.98, 0.16),
                ("Príncipe, W. Africa", 1.61, 0.40),
                ("Sobral, astrographic*", 0.93, 0.40)]
gr_pred, nt_pred = 4 * GM / (C**2 * R_SUN) * RAD2ARCSEC, 2 * GM / (C**2 * R_SUN) * RAD2ARCSEC
for nm, val, err in results_1919:
    z_gr = abs(val - gr_pred) / err
    z_nt = abs(val - nt_pred) / err
    print(f"  {nm:26s} {val:.2f} ± {err:.2f}″   "
          f"→ {z_gr:.1f}σ from Einstein, {z_nt:.1f}σ from Newton")
print("  * the astrographic plates were ruined by heat distortion of the mirror and")
print("    were set aside — a decision still argued about a century later.")
print()

# --- modern measurements ---
print("Modern versions of the same measurement:")
for nm, val, err in [("VLBI radio, 1970s", 1.0002, 0.0020),
                     ("Hipparchos astrometry 1997", 0.9970, 0.0030),
                     ("Cassini time delay 2003", 1.000021, 0.000023),
                     ("Gaia (ongoing)", 1.00000, 0.00001)]:
    print(f"  {nm:28s} α/α_GR = {val:.6f} ± {err:.6f}")
print("  General relativity now holds to a part in 10⁵ on this one prediction alone.")

# --- visualization ---
fig = plt.figure(figsize=(15, 8.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])

# Panel 1: the two bent rays, exaggerated
ax = fig.add_subplot(gs[0, :2])
for alpha_x, us_x, phis_x, col, lab in [
        (alpha_gr, us_gr, phis_gr, "crimson", f"Einstein 1915: {alpha_gr*RAD2ARCSEC:.2f}″"),
        (alpha_nt, us_nt, phis_nt, "steelblue", f"Newton/Soldner: {alpha_nt*RAD2ARCSEC:.2f}″")]:
    EXAG = 60000
    ang = phis_x * 1.0
    # re-plot with the bend amplified so it is visible at all
    r_x = 1.0 / us_x.clamp(min=1e-14)
    frac = (ang - ang[0]) / (ang[-1] - ang[0])
    bend = alpha_x * EXAG * frac
    xs = r_x * torch.cos(ang + bend)
    ys = r_x * torch.sin(ang + bend)
    keep = r_x < 40 * R_SUN
    ax.plot(xs[keep] / R_SUN, ys[keep] / R_SUN, color=col, lw=2.2, label=lab)
sun = plt.Circle((0, 0), 1.0, color="orange", alpha=0.9)
ax.add_patch(sun)
ax.plot([-40, 40], [1, 1], "k:", lw=1, alpha=0.5, label="undeflected straight line")
ax.set_aspect("equal")
ax.set_xlim(-32, 32)
ax.set_ylim(-6, 12)
ax.set_xlabel("distance (solar radii)")
ax.set_title("Starlight grazing the Sun (bend exaggerated 60 000×).\n"
             "Both theories bend it. They disagree by exactly a factor of two, and\n"
             "that factor is the difference between curved time and curved spacetime.")
ax.legend(fontsize=9, loc="upper right")
ax.grid(alpha=0.3)

# Panel 2: deflection vs impact parameter
ax = fig.add_subplot(gs[0, 2])
bs = np.linspace(1.0, 8.0, 200) * R_SUN
ax.plot(bs / R_SUN, 4 * GM / (C**2 * bs) * RAD2ARCSEC, color="crimson", lw=2.4,
        label="Einstein  4GM/c²b")
ax.plot(bs / R_SUN, 2 * GM / (C**2 * bs) * RAD2ARCSEC, color="steelblue", lw=2.4,
        ls="--", label="Newton  2GM/c²b")
for nm, val, err in results_1919[:2]:
    ax.errorbar(1.0, val, yerr=err, fmt="o", color="black", ms=8, capsize=5)
ax.text(1.15, 1.98, "Sobral", fontsize=8)
ax.text(1.15, 1.55, "Príncipe", fontsize=8)
ax.set_xlabel("impact parameter (solar radii)")
ax.set_ylabel("deflection (arcsec)")
ax.set_title("The 1919 measurements land on the\nupper curve. The observation had to\n"
             "choose between two numbers, and did.")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel 3: the 1919 result as a significance plot
ax = fig.add_subplot(gs[1, 0])
ys = np.arange(len(results_1919))
ax.errorbar([r[1] for r in results_1919], ys, xerr=[r[2] for r in results_1919],
            fmt="o", color="black", ms=9, capsize=6)
ax.axvline(gr_pred, color="crimson", lw=2.4, label=f"Einstein {gr_pred:.2f}″")
ax.axvline(nt_pred, color="steelblue", lw=2.4, ls="--", label=f"Newton {nt_pred:.2f}″")
ax.set_yticks(ys)
ax.set_yticklabels([r[0].split(",")[0] + ("*" if "astro" in r[0] else "")
                    for r in results_1919], fontsize=8.5)
ax.set_xlabel("measured deflection at the limb (arcsec)")
ax.set_title("6 November 1919, Royal Society.\nThe Times next morning:\n"
             "'REVOLUTION IN SCIENCE'.")
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3, axis="x")

# Panel 4: what the plates looked like
ax = fig.add_subplot(gs[1, 1])
rng = np.random.default_rng(4)
n_star = 13
ang = rng.uniform(0, 2 * np.pi, n_star)
rad = rng.uniform(1.6, 6.0, n_star)
sx, sy = rad * np.cos(ang), rad * np.sin(ang)
shift = (gr_pred / rad) * 1.2                # outward radial push, exaggerated
ax.scatter(sx, sy, s=26, color="black", label="true position (night plate)")
ax.scatter(sx * (1 + shift / rad), sy * (1 + shift / rad), s=26, facecolors="none",
           edgecolors="crimson", label="eclipse-day position")
for i in range(n_star):
    ax.annotate("", xy=(sx[i] * (1 + shift[i] / rad[i]), sy[i] * (1 + shift[i] / rad[i])),
                xytext=(sx[i], sy[i]),
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1.1))
ax.add_patch(plt.Circle((0, 0), 1.0, color="black"))
ax.add_patch(plt.Circle((0, 0), 1.06, color="gold", alpha=0.5))
ax.set_aspect("equal")
ax.set_xlim(-7, 7); ax.set_ylim(-7, 7)
ax.set_title("What Eddington actually measured:\nevery star pushed radially OUTWARD,\n"
             "more strongly the closer to the limb.")
ax.legend(fontsize=7.5, loc="lower right")
ax.set_xticks([]); ax.set_yticks([])

# Panel 5: a century of tightening
ax = fig.add_subplot(gs[1, 2])
years = [1919, 1922, 1973, 1997, 2003, 2020]
vals = [1.13, 1.04, 1.0002, 0.9970, 1.000021, 1.00000]
errs = [0.10, 0.09, 0.0020, 0.0030, 0.000023, 0.00001]
ax.errorbar(years, vals, yerr=errs, fmt="o-", color="crimson", ms=7, capsize=4)
ax.axhline(1.0, color="black", ls="--", lw=1.4, label="general relativity")
ax.set_yscale("log")
ax.set_ylim(0.9, 1.3)
ax.set_xlabel("year")
ax.set_ylabel("measured / predicted deflection")
ax.set_title("A century of the same test.\nThe error bar shrank by 10⁴.\n"
             "The centre never moved.")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which="both")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
