"""The 8 arcminutes that destroyed 2000 years of circles.

Kepler fitted a circular orbit to Tycho's Mars observations and got agreement to
within 8 arcminutes — a quarter of the Moon's width, better than any astronomer
before Tycho could even measure. Every predecessor would have published and gone
home. Kepler knew Tycho's instruments were good to ~2′, so 8′ could not be noise.

He wrote, in the Astronomia Nova (1609):

    "Since divine goodness has given to us Tycho Brahe, a most diligent observer,
     from whose observations the 8′ error in this Ptolemaic computation is
     revealed, it is fitting that we recognise and honour this benefit of God...
     Because these 8′ could not be ignored, they alone have led the way towards
     the reformation of all of astronomy."

Eight arcminutes. That is the entire distance between the medieval universe and
the modern one. The lesson is about RESIDUALS: a fit is only as good as your
knowledge of your own error bar.

    phenomenon:   Mars is observed at a set of positions over many oppositions
    simulation:   generate positions from the true ellipse (e = 0.0934)
    dissection:   fit the best possible circle (torch, least squares) and the
                  best ellipse; look at what is left over in each case
    formula:      the circle's residual maxes at ~e²/2 · a in distance, which for
                  Mars projects to ≈ 8′ on the sky — the number Kepler saw.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

RAD2ARCMIN = 180 / np.pi * 60

# --- truth: Mars' real orbit, Sun at one focus ---
A_MARS, E_MARS = 1.5237, 0.09341
b_mars = A_MARS * np.sqrt(1 - E_MARS**2)
c_mars = A_MARS * E_MARS

theta = torch.linspace(0, 2 * np.pi, 361, dtype=torch.float64)[:-1]
# polar form about the FOCUS (where the Sun is) — this is the physical orbit
r_true = A_MARS * (1 - E_MARS**2) / (1 + E_MARS * torch.cos(theta))
xy_true = torch.stack([r_true * torch.cos(theta), r_true * torch.sin(theta)], dim=1)

# --- dissection 1: best-fit CIRCLE (what everyone before Kepler assumed) ---
# Free the centre too: this is the "eccentric circle" Ptolemy and Copernicus both used,
# so we are being maximally generous to the old model.
centre = torch.zeros(2, dtype=torch.float64, requires_grad=True)
radius = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
opt = torch.optim.Adam([centre, radius], lr=0.01)
for _ in range(4000):
    opt.zero_grad()
    d = torch.linalg.norm(xy_true - centre, dim=1)
    loss = ((d - radius) ** 2).mean()          # distance from the circle, squared
    loss.backward()
    opt.step()
centre_f, radius_f = centre.detach(), radius.detach()

d_circ = torch.linalg.norm(xy_true - centre_f, dim=1)
resid_circ = d_circ - radius_f                          # radial miss, in AU
print(f"best eccentric circle:  centre = ({centre_f[0]:.4f}, {centre_f[1]:.4f}) AU, "
      f"R = {radius_f:.4f} AU")
print(f"  max radial residual = {resid_circ.abs().max():.5f} AU")

# --- convert the radial miss into an ANGLE on the sky, as seen from Earth ---
# Earth is ~1 AU away; a miss of dr at distance D subtends dr/D radians.
# Taking the mean Earth-Mars distance over oppositions, D ~ 0.7 AU at closest,
# ~1.5 AU typical. Kepler worked in heliocentric longitude, so use the orbit itself:
# the angular error in longitude as seen from the Sun.
ang_err_circ = torch.atan2(resid_circ.abs(), r_true) * RAD2ARCMIN
print(f"  max angular residual as seen from the Sun = {ang_err_circ.max():.2f} arcmin")

# Where does that number come from? Expand the orbit equation to second order:
#   r = a(1−e²)/(1+e·cosθ) ≈ a[1 − e·cosθ − (e²/2)·(1 − cos2θ)]
# An offset circle absorbs the whole e·cosθ term (that is what an eccentric IS) and
# absorbs the constant part of the e² term into its radius. What it cannot absorb is
# the cos2θ ripple of amplitude a·e²/2 — and least squares splits that in half:
predicted = E_MARS**2 / 4 * RAD2ARCMIN
print(f"  closed form  e²/4 = {E_MARS**2/4:.6f} rad = {predicted:.2f} arcmin"
      f"   (measured {ang_err_circ.max():.2f}′)")
print("  → an irreducible ~8′, set purely by Mars' eccentricity. No circle can remove it.\n")

# --- dissection 2: best-fit ELLIPSE, Sun forced to sit at a focus ---
a_p = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
e_p = torch.tensor(0.02, dtype=torch.float64, requires_grad=True)
phi_p = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)   # orientation
opt2 = torch.optim.Adam([a_p, e_p, phi_p], lr=0.01)
for _ in range(4000):
    opt2.zero_grad()
    r_model = a_p * (1 - e_p**2) / (1 + e_p * torch.cos(theta - phi_p))
    loss = ((r_model - r_true) ** 2).mean()
    loss.backward()
    opt2.step()
r_ell = (a_p * (1 - e_p**2) / (1 + e_p * torch.cos(theta - phi_p))).detach()
resid_ell = r_ell - r_true
ang_err_ell = torch.atan2(resid_ell.abs(), r_true) * RAD2ARCMIN
print(f"best ellipse (Sun at focus): a = {a_p.item():.4f} AU, e = {e_p.item():.5f}  "
      f"(truth a = {A_MARS}, e = {E_MARS})")
print(f"  max angular residual = {ang_err_ell.max():.4f} arcmin")
print()
print(f"Circle:  {ang_err_circ.max():.2f}′   vs   Tycho's precision ~2′  →  RULED OUT")
print(f"Ellipse: {ang_err_ell.max():.4f}′   vs   Tycho's precision ~2′  →  survives")
print()
print("Kepler had no theory of gravity, no calculus, and no reason to prefer an")
print("ellipse. He had one thing: he trusted Tycho's error bar more than he trusted")
print("Aristotle. That is the whole of the scientific method in one decision.")

# --- visualization ---
fig = plt.figure(figsize=(15, 6.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1])

# Panel 1: the two orbits, visually identical
ax = fig.add_subplot(gs[0, 0])
ax.plot(xy_true[:, 0], xy_true[:, 1], color="black", lw=2.6, label="true orbit (ellipse)")
circ_t = torch.linspace(0, 2 * np.pi, 400, dtype=torch.float64)
ax.plot(centre_f[0] + radius_f * torch.cos(circ_t),
        centre_f[1] + radius_f * torch.sin(circ_t),
        color="crimson", lw=1.8, ls="--", label="best-fit circle")
ax.plot(0, 0, "*", color="orange", ms=20, label="Sun (at the focus)")
ax.plot(centre_f[0], centre_f[1], "x", color="crimson", ms=10, mew=2.5,
        label="circle's centre")
ax.set_aspect("equal")
ax.set_xlabel("AU")
ax.set_title("To the eye, a circle and Mars' ellipse\nare the SAME curve. e = 0.093.")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)

# Panel 2: residuals in angle — the 8 arcminutes
ax = fig.add_subplot(gs[0, 1])
deg = torch.rad2deg(theta)
sign_resid = torch.atan2(resid_circ, r_true) * RAD2ARCMIN
ax.plot(deg, sign_resid, color="crimson", lw=2.2, label="circle model residual")
ax.plot(deg, torch.atan2(resid_ell, r_true) * RAD2ARCMIN, color="darkgreen", lw=2.2,
        label="ellipse model residual")
ax.axhspan(-2, 2, color="steelblue", alpha=0.2, label="Tycho's error bar (±2′)")
ax.axhline(0, color="k", lw=0.5)
ax.annotate(f"{ang_err_circ.max():.1f}′", xy=(deg[int(sign_resid.argmax())],
                                              sign_resid.max()),
            xytext=(15, 12), textcoords="offset points", fontsize=13, color="crimson",
            weight="bold", arrowprops=dict(arrowstyle="->", color="crimson"))
ax.set_xlabel("true anomaly (deg)")
ax.set_ylabel("residual (arcmin)")
ax.set_title("The residual Kepler refused to ignore.\n"
             "It sticks 4× out of Tycho's error bar.")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel 3: what happens if you trust the wrong error bar
ax = fig.add_subplot(gs[0, 2])
precisions = np.array([30, 10, 8, 5, 2, 1, 0.5])
detectable = ang_err_circ.max().item() / precisions
ax.bar([f"{p:g}′" for p in precisions], detectable,
       color=["gray" if d < 1 else "crimson" for d in detectable], alpha=0.85)
ax.axhline(1, color="black", ls="--", lw=2, label="detection threshold")
ax.set_yscale("log")
ax.set_xlabel("observer's precision")
ax.set_ylabel("residual / precision")
ax.set_title("Same sky, different eyes.\nCopernicus (10′) could NOT have found this.\n"
             "Tycho (2′) handed Kepler a 4σ signal.")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
