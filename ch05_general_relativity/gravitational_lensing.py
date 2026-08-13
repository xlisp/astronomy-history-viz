"""From a curiosity Einstein thought useless to the main tool for weighing the universe.

If mass bends light by 4GM/c²b, then a massive object is a LENS. Einstein worked
this out and published it in 1936 only because a Czech engineer, Rudi Mandl, kept
pestering him. He wrote to the editor of Science: "it is of little value, but it
makes the poor guy happy." He believed the alignment would never be observed.

He was wrong in the most useful way possible. Zwicky pointed out within a year that
GALAXIES, not stars, would make observable lenses. The first was found in 1979
(the "Twin Quasar" QSO 0957+561). Today gravitational lensing is how we:

    · weigh galaxy clusters, including the dark matter (ch08)
    · find exoplanets by microlensing
    · measure H₀ independently, from time delays between images
    · see galaxies at z > 10 that would otherwise be far too faint

Note the epistemic upgrade: light bending began as a TEST of relativity (1919), and
within 60 years became an INSTRUMENT that assumes relativity and measures something
else. That transition is how a theory stops being controversial.

    phenomenon:   a mass between us and a distant source produces multiple,
                  distorted, brightened images
    simulation:   the thin-lens equation for a point mass
    dissection:   solve for image positions and magnifications with torch; check
                  the analytic total magnification against the image-by-image sum
    formula:      β = θ − θ_E²/θ,  θ_E = sqrt(4GM/c² · D_LS/(D_L·D_S))
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

G, C = 6.67430e-11, 2.99792458e8
M_SUN = 1.98892e30
PC = 3.0856775814913673e16
RAD2ARCSEC = 180 / np.pi * 3600


def einstein_radius(M, D_L, D_S):
    """Angular radius of the Einstein ring, in radians."""
    D_LS = D_S - D_L                     # flat-space approximation, fine for a demo
    return np.sqrt(4 * G * M / C**2 * D_LS / (D_L * D_S))


# --- three regimes, six orders of magnitude apart in angle ---
systems = [
    ("Star lensing a star\n(microlensing)", 0.3 * M_SUN, 4e3 * PC, 8e3 * PC),
    ("Galaxy lensing a quasar\n(strong lensing)", 1e11 * M_SUN, 5e8 * PC, 2e9 * PC),
    ("Cluster lensing a galaxy\n(giant arcs)", 1e15 * M_SUN, 1e9 * PC, 3e9 * PC),
]
print("  system                          M (M_sun)   θ_E")
for nm, M, D_L, D_S in systems:
    tE = einstein_radius(M, D_L, D_S) * RAD2ARCSEC
    unit = f"{tE*1e3:.3f} mas" if tE < 0.05 else f"{tE:.2f}″"
    print(f"  {nm.splitlines()[0]:32s} {M/M_SUN:.1e}   {unit}")
print()
print("Microlensing rings are milliarcseconds — unresolvable. But you can still see")
print("the BRIGHTENING as the alignment drifts, and that is how thousands of")
print("exoplanets and dark objects have been found. → ch08\n")

# --- the lens equation ---
M, D_L, D_S = 1e11 * M_SUN, 5e8 * PC, 2e9 * PC
theta_E = einstein_radius(M, D_L, D_S)
print(f"Working example: a 10¹¹ M_sun galaxy, θ_E = {theta_E*RAD2ARCSEC:.3f}″\n")

# beta = theta - theta_E^2/theta  →  a quadratic with two roots, always
beta = torch.linspace(0.001, 3.0, 400, dtype=torch.float64) * theta_E
theta_plus = 0.5 * (beta + torch.sqrt(beta**2 + 4 * theta_E**2))
theta_minus = 0.5 * (beta - torch.sqrt(beta**2 + 4 * theta_E**2))

# magnification of each image: mu = 1/|1 - (theta_E/theta)^4|
mu_plus = 1.0 / torch.abs(1 - (theta_E / theta_plus) ** 4)
mu_minus = 1.0 / torch.abs(1 - (theta_E / theta_minus) ** 4)
mu_total = mu_plus + mu_minus

# the textbook closed form, in units u = beta/theta_E
u = beta / theta_E
mu_closed = (u**2 + 2) / (u * torch.sqrt(u**2 + 4))
print("check: sum of the two image magnifications vs the closed form (u²+2)/(u√(u²+4))")
print(f"  max |difference| = {(mu_total - mu_closed).abs().max():.2e}   ✓\n")

# --- lensing conserves surface brightness, so magnification is pure area gain ---
print("Two facts that make lensing an instrument rather than a curiosity:")
print("  1. It is ACHROMATIC — gravity bends every wavelength identically, unlike")
print("     a glass lens. So a lensed source keeps its colours exactly.")
print("  2. The deflection depends only on MASS, not on whether that mass emits")
print("     light. This is why lensing sees dark matter directly. → ch08\n")

# --- the microlensing light curve: the observable when you cannot resolve images ---
t = torch.linspace(-3, 3, 600, dtype=torch.float64)          # in Einstein-radius crossings
print("microlensing light curve — peak magnification vs impact parameter u_min:")
for u_min in (1.0, 0.5, 0.2, 0.05):
    uu = np.sqrt(u_min**2)
    peak = (uu**2 + 2) / (uu * np.sqrt(uu**2 + 4))
    print(f"  u_min = {u_min:.2f}  →  peak magnification {peak:6.2f}×  "
          f"({2.5*np.log10(peak):.2f} magnitudes)")
print()
print("Einstein, 1936, in the paper itself:")
print('  "there is no great chance of observing this phenomenon."')
print("As of today, lensing has produced tens of thousands of measurements and")
print("two Nobel-adjacent research programmes. It is the standard cautionary tale")
print("about a theorist predicting what will never be measurable.")

# --- visualization ---
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3)

# Panel 1: image positions vs source position
ax = fig.add_subplot(gs[0, 0])
ax.plot(u, theta_plus / theta_E, color="crimson", lw=2.4, label="image + (outside ring)")
ax.plot(u, theta_minus / theta_E, color="steelblue", lw=2.4, label="image − (inside, flipped)")
ax.plot(u, u, "k:", lw=1.4, label="true source position")
ax.axhline(1, color="darkgreen", ls="--", lw=1.4, label="Einstein ring θ_E")
ax.axhline(-1, color="darkgreen", ls="--", lw=1.4)
ax.set_xlabel("source offset β / θ_E")
ax.set_ylabel("image position θ / θ_E")
ax.set_title("Every source has TWO images —\none outside the ring, one inside\n"
             "and inverted. Perfect alignment ⇒ a ring.")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel 2: the sky picture
ax = fig.add_subplot(gs[0, 1])
for b_show, col, alpha in [(0.15, "crimson", 1.0), (0.6, "darkorange", 0.75),
                           (1.5, "steelblue", 0.6)]:
    tp = 0.5 * (b_show + np.sqrt(b_show**2 + 4))
    tm = 0.5 * (b_show - np.sqrt(b_show**2 + 4))
    mp = 1 / abs(1 - (1 / tp) ** 4)
    mm = 1 / abs(1 - (1 / tm) ** 4)
    ax.scatter([b_show], [0], s=40, marker="*", color=col, alpha=alpha)
    ax.scatter([tp], [0], s=60 * mp, color=col, alpha=alpha, edgecolors="k", lw=0.5)
    ax.scatter([tm], [0], s=60 * mm, color=col, alpha=alpha, edgecolors="k", lw=0.5)
    ax.text(b_show, 0.13, f"β={b_show}", fontsize=7, ha="center", color=col)
ring = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(ring), np.sin(ring), color="darkgreen", ls="--", lw=1.5)
ax.plot(0, 0, "x", color="black", ms=12, mew=2.5)
ax.text(0.05, -0.22, "lens", fontsize=8)
ax.text(1.02, 0.75, "Einstein ring", fontsize=8, color="darkgreen")
ax.set_aspect("equal")
ax.set_xlim(-2.6, 2.8); ax.set_ylim(-1.6, 1.6)
ax.set_xlabel("θ / θ_E")
ax.set_title("On the sky: ★ is where the source really is,\n● are the images "
             "(size ∝ brightness).\nThe closer the alignment, the brighter and\n"
             "more separated the pair.")
ax.grid(alpha=0.3)

# Panel 3: magnification blows up at perfect alignment
ax = fig.add_subplot(gs[0, 2])
ax.semilogy(u, mu_total, color="crimson", lw=2.6, label="total magnification")
ax.semilogy(u, mu_plus, color="darkorange", lw=1.4, ls="--", label="image +")
ax.semilogy(u, mu_minus.abs(), color="steelblue", lw=1.4, ls="--", label="image −")
ax.axhline(1, color="black", lw=1, ls=":")
ax.set_xlabel("source offset β / θ_E")
ax.set_ylabel("magnification μ")
ax.set_title("μ → ∞ as β → 0.\nA lens is a free telescope, and it is how\n"
             "we see the faintest galaxies ever found.")
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3, which="both")

# Panel 4: microlensing light curves
ax = fig.add_subplot(gs[1, 0])
for u_min, col in [(1.0, "#c6dbef"), (0.5, "#6baed6"), (0.2, "#2171b5"), (0.05, "#08306b")]:
    uu = torch.sqrt(u_min**2 + t**2)
    mu = (uu**2 + 2) / (uu * torch.sqrt(uu**2 + 4))
    ax.plot(t, mu, color=col, lw=2.2, label=f"u_min = {u_min}")
ax.set_xlabel("time  (Einstein radius crossings)")
ax.set_ylabel("magnification")
ax.set_yscale("log")
ax.set_title("What you actually observe when the images\ncannot be resolved: a "
             "symmetric, achromatic\nbrightening. Thousands found; this is also\n"
             "how free-floating planets are detected.")
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3, which="both")

# Panel 5: an extended source lensed into an arc
ax = fig.add_subplot(gs[1, 1])
rng = np.random.default_rng(11)
n = 4000
src_r = 0.35 * np.sqrt(rng.uniform(0, 1, n))
src_a = rng.uniform(0, 2 * np.pi, n)
sx = 0.45 + src_r * np.cos(src_a)
sy = src_r * np.sin(src_a)
b_mag = np.sqrt(sx**2 + sy**2)
for sign in (+1, -1):
    th = 0.5 * (b_mag + sign * np.sqrt(b_mag**2 + 4))
    ix, iy = sx / b_mag * th, sy / b_mag * th
    ax.scatter(ix, iy, s=1.2, color="crimson" if sign > 0 else "steelblue", alpha=0.35)
ax.scatter(sx, sy, s=1.2, color="gray", alpha=0.35)
ax.plot(np.cos(ring), np.sin(ring), color="darkgreen", ls="--", lw=1.3)
ax.plot(0, 0, "x", color="black", ms=11, mew=2.5)
ax.set_aspect("equal")
ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
ax.set_title("An extended source (gray disc) becomes two\nstretched arcs. This is "
             "what Hubble sees around\ngalaxy clusters — and the arc shape is what\n"
             "we invert to map the dark matter.")
ax.set_xticks([]); ax.set_yticks([])

# Panel 6: the scale ladder
ax = fig.add_subplot(gs[1, 2])
nms = [s[0] for s in systems]
tEs = [einstein_radius(s[1], s[2], s[3]) * RAD2ARCSEC for s in systems]
ax.barh(nms, tEs, color=["#08306b", "#2171b5", "#6baed6"], alpha=0.9)
ax.set_xscale("log")
ax.set_xlabel("Einstein radius θ_E (arcsec)")
ax.axvline(0.05, color="crimson", ls="--", lw=1.8, label="resolvable from the ground")
for i, v in enumerate(tEs):
    ax.text(v * 1.25, i, f"{v:.3g}″", va="center", fontsize=8.5)
ax.set_title("Einstein was right that stellar lensing is\nunresolvable — and wrong "
             "that this made it\nuseless. Zwicky saw galaxies would work.")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="x", which="both")
ax.tick_params(axis="y", labelsize=8)

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
