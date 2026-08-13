"""Vera Rubin, 1970s: galaxies weigh five times what they look like they weigh.

Kepler's third law, still working 350 years later, and still being used as a scale
(ch02). Apply it to a galaxy: measure how fast stars orbit at radius r, and you get
the mass inside r:

    v²/r = GM(<r)/r²   ⟹   M(<r) = v²r/G

Beyond the visible edge of a galaxy there is no more light, so there should be no
more mass, so v should fall off as 1/√r — exactly as the planets do beyond the Sun.

Rubin and Ford measured it and found the curves are FLAT. Out to twice the visible
radius, three times, as far as there is any gas to measure, the orbital speed does
not drop. Something invisible is out there, and there is roughly five times more of
it than everything that shines.

Zwicky had said the same thing in 1933 from the velocity dispersion of the Coma
cluster and coined "dunkle Materie". He was widely ignored — partly because he was
abrasive, partly because one cluster is one data point. Rubin's contribution was
dozens of galaxies, all telling the identical story, measured so carefully that
disagreement became untenable.

    phenomenon:   stars in the outskirts of galaxies orbit far too fast
    simulation:   a realistic disc + bulge mass model, and its predicted curve
    dissection:   fit the observed curve with torch, allowing a dark halo; ask how
                  much invisible mass is required
    formula:      v_flat ⟹ M(<r) ∝ r ⟹ ρ ∝ 1/r². That is the halo profile, and it
                  is not what any luminous component does.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

G = 4.30091e-6              # kpc (km/s)² / M_sun — the astronomer's units for G

r = torch.linspace(0.3, 40.0, 300, dtype=torch.float64)     # kpc


def v_bulge(r, M_b=1.2e10, a_b=0.6):
    """Hernquist bulge — a compact central concentration."""
    return torch.sqrt(G * M_b * r / (r + a_b) ** 2)


def v_disc(r, M_d=6.0e10, R_d=3.0):
    """Exponential disc, approximated by its enclosed mass."""
    x = r / (2 * R_d)
    M_enc = M_d * (1 - torch.exp(-r / R_d) * (1 + r / R_d))
    return torch.sqrt(G * M_enc / r)


def v_halo(r, v_inf=180.0, r_c=4.0):
    """Pseudo-isothermal halo: the profile that makes a curve flat."""
    return v_inf * torch.sqrt(1 - (r_c / r) * torch.atan(r / r_c))


v_b, v_d = v_bulge(r), v_disc(r)
v_lum = torch.sqrt(v_b**2 + v_d**2)                # everything that emits light
v_h = v_halo(r)
v_tot = torch.sqrt(v_lum**2 + v_h**2)

# --- the observation: flat, out to the last measurable point ---
torch.manual_seed(3)
r_obs = torch.linspace(1.0, 38.0, 26, dtype=torch.float64)
v_obs = torch.sqrt(v_bulge(r_obs)**2 + v_disc(r_obs)**2 + v_halo(r_obs)**2)
v_obs = v_obs + torch.randn(len(r_obs), dtype=torch.float64) * 6.0
OBS_ERR = 6.0

print("What the light says vs what the motion says:")
print("   r (kpc)   v_luminous   v_observed   missing mass factor")
for rr in (5, 10, 20, 30, 38):
    k = int(torch.argmin((r - rr).abs()))
    vl, vt = float(v_lum[k]), float(v_tot[k])
    print(f"   {rr:5.0f}     {vl:8.1f}     {vt:8.1f}        {(vt/vl)**2:6.2f}×")
print()

# --- Kepler's third law, turned into a scale for the invisible ---
print("Now weigh it, using the same formula Newton used on the Moon (ch03):")
print("   r (kpc)   M_luminous (M_sun)   M_total (M_sun)     M_dark/M_lum")
for rr in (5, 10, 20, 30, 38):
    k = int(torch.argmin((r - rr).abs()))
    M_l = float(v_lum[k]) ** 2 * rr / G
    M_t = float(v_tot[k]) ** 2 * rr / G
    print(f"   {rr:5.0f}     {M_l:.3e}          {M_t:.3e}         {(M_t-M_l)/M_l:6.2f}")
print()

# --- fit the halo to the data with autograd: how much dark matter is required? ---
v_inf = torch.tensor(100.0, dtype=torch.float64, requires_grad=True)
r_c = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
opt = torch.optim.Adam([v_inf, r_c], lr=0.5)
for _ in range(4000):
    opt.zero_grad()
    model = torch.sqrt(v_bulge(r_obs)**2 + v_disc(r_obs)**2 + v_halo(r_obs, v_inf, r_c)**2)
    loss = (((model - v_obs) / OBS_ERR) ** 2).mean()
    loss.backward()
    opt.step()
print(f"halo fitted by gradient descent:  v_inf = {float(v_inf):.1f} km/s, "
      f"r_c = {float(r_c):.2f} kpc")

# how well does a no-dark-matter model do?
chi2_dark = float((((torch.sqrt(v_bulge(r_obs)**2 + v_disc(r_obs)**2
                                + v_halo(r_obs, v_inf, r_c)**2) - v_obs) / OBS_ERR) ** 2).sum())
chi2_lum = float((((torch.sqrt(v_bulge(r_obs)**2 + v_disc(r_obs)**2) - v_obs)
                   / OBS_ERR) ** 2).sum())
print(f"  χ² with a dark halo         = {chi2_dark:8.1f}  ({len(r_obs)} points)")
print(f"  χ² with visible matter only = {chi2_lum:8.1f}")
print(f"  The luminous-only model is off by {np.sqrt(chi2_lum/len(r_obs)):.0f}σ per point.")
print(f"  No adjustment of the disc or bulge mass fixes it, because the SHAPE is")
print(f"  wrong, not the normalisation: light falls off exponentially, and the")
print(f"  required mass does not fall off at all.\n")

# --- what the flatness implies about the density profile ---
print("Read the flat curve backwards:")
print("  v = const  ⟹  M(<r) = v²r/G ∝ r  ⟹  ρ(r) ∝ 1/r²")
print("  Nothing luminous has that profile. A disc falls off exponentially; a")
print("  bulge falls faster still. Whatever this is, it is distributed in a roughly")
print("  spherical halo extending far beyond the stars.\n")
print("Independent confirmations that arrived later:")
print("  · gravitational lensing (ch05) maps mass directly, light or not — agrees")
print("  · the Bullet Cluster: in a collision the hot gas (most of the ordinary")
print("    matter) is stripped and lags behind, while the lensing mass sails")
print("    through with the galaxies. The mass and the light are in DIFFERENT PLACES.")
print("  · the CMB acoustic peaks (see cmb_blackbody.py) need ~5× more matter than")
print("    the ordinary-matter density that Big Bang nucleosynthesis allows")
print()
print("Three completely unrelated measurements, one answer: ~85% of the matter in")
print("the universe has never been seen. We still do not know what it is.")

# --- visualization ---
fig = plt.figure(figsize=(15, 8.5))
gs = fig.add_gridspec(2, 3)

# Panel 1: the rotation curve
ax = fig.add_subplot(gs[0, :2])
ax.errorbar(r_obs, v_obs, yerr=OBS_ERR, fmt="o", color="black", ms=6, capsize=3,
            label="observed (21 cm + optical)", zorder=6)
ax.plot(r, v_tot, color="crimson", lw=2.4, label="total, with dark halo")
ax.plot(r, v_lum, color="steelblue", lw=2.2, ls="--", label="everything that shines")
ax.plot(r, v_d, color="steelblue", lw=1.1, ls=":", label="disc alone")
ax.plot(r, v_b, color="darkgreen", lw=1.1, ls=":", label="bulge alone")
ax.plot(r, v_h, color="purple", lw=1.6, ls="-.", label="dark halo alone")
kep = float(v_lum[int(torch.argmin((r - 8).abs()))]) * torch.sqrt(8.0 / r[r > 8])
ax.plot(r[r > 8], kep, color="gray", lw=1.6, ls="--",
        label="Keplerian 1/√r (what Newton predicts\nbeyond the visible edge)")
ax.axvspan(0, 15, color="gold", alpha=0.13)
ax.text(6.5, 40, "visible galaxy", fontsize=9, color="darkgoldenrod")
ax.set_xlabel("radius (kpc)")
ax.set_ylabel("orbital speed (km/s)")
ax.set_ylim(0, 240)
ax.set_title("Rubin & Ford's result: the curve refuses to fall.\n"
             "Kepler's third law applied to a galaxy says there is five times more "
             "mass than there is light.")
ax.legend(fontsize=8, loc="lower right", ncol=2)
ax.grid(alpha=0.3)

# Panel 2: enclosed mass
ax = fig.add_subplot(gs[0, 2])
M_lum = v_lum**2 * r / G
M_tot = v_tot**2 * r / G
ax.plot(r, M_tot, color="crimson", lw=2.4, label="total mass M(<r)")
ax.plot(r, M_lum, color="steelblue", lw=2.2, ls="--", label="luminous mass")
ax.fill_between(r, M_lum, M_tot, color="purple", alpha=0.25, label="dark matter")
ax.set_yscale("log")
ax.set_xlabel("radius (kpc)")
ax.set_ylabel("enclosed mass (M_sun)")
ax.set_title("The luminous mass saturates —\nthere is no more light to add.\n"
             "The total mass keeps growing ∝ r.")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

# Panel 3: what a Keplerian galaxy would look like
ax = fig.add_subplot(gs[1, 0])
ax.plot(r, v_tot / v_tot.max(), color="crimson", lw=2.4, label="galaxies (observed)")
r_ss = torch.linspace(0.39, 30, 200, dtype=torch.float64)
ax.plot(r_ss / 30 * 40, torch.sqrt(0.39 / r_ss), color="darkorange", lw=2.4,
        label="solar system (Kepler)")
ax.set_xlabel("radius (arbitrary, scaled)")
ax.set_ylabel("orbital speed (normalised)")
ax.set_title("The solar system does what Newton says.\nGalaxies do not. Same law, "
             "same mathematics —\nthe difference must be the mass distribution.")
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3)

# Panel 4: density profiles
ax = fig.add_subplot(gs[1, 1])
rho_disc = torch.exp(-r / 3.0)
rho_halo = 1.0 / (1 + (r / 4.0) ** 2)
ax.semilogy(r, rho_disc / rho_disc[0], color="steelblue", lw=2.2,
            label="luminous disc  ρ ∝ e^(−r/R)")
ax.semilogy(r, rho_halo / rho_halo[0], color="purple", lw=2.2,
            label="required halo  ρ ∝ 1/r²")
ax.set_xlabel("radius (kpc)")
ax.set_ylabel("density (normalised)")
ax.set_ylim(1e-6, 2)
ax.set_title("Why more stars cannot be the answer.\nThe shapes are different, not "
             "just the\namounts. Light falls off exponentially;\nthe missing mass "
             "falls off as a power.")
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3, which="both")

# Panel 5: the cosmic inventory
ax = fig.add_subplot(gs[1, 2])
comp = ["dark energy\n68.3%", "dark matter\n26.8%", "ordinary matter\n4.9%"]
vals = [68.3, 26.8, 4.9]
cols = ["#2c1338", "#5b2c6f", "#f0c419"]
wedges, _ = ax.pie(vals, colors=cols, startangle=90,
                   wedgeprops=dict(edgecolor="white", lw=2))
ax.legend(wedges, comp, fontsize=9, loc="center left", bbox_to_anchor=(-0.15, 0.5))
ax.set_title("Everything ever observed by every\ntelescope in history is the yellow\n"
             "sliver — and most of that is\nintergalactic gas, not stars.")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
