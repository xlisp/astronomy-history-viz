"""Bessel 1838: the first rung of the ladder, and the death of a 2000-year objection.

Aristarchus proposed a heliocentric universe around 270 BC. It was rejected for
one *good* scientific reason: if the Earth moves, nearby stars must shift against
the far ones over a year. Nobody could see any shift. The Greeks concluded the
Earth doesn't move. The correct conclusion — the one that took 2100 years to
verify — was that the stars are unimaginably far away.

Bessel finally measured it for 61 Cygni in 1838: 0.314 arcsec, an angle the size
of a coin seen from 15 km. That single number turned the sky from a dome into a
volume, and it is still the ONLY rung of the cosmic distance ladder that requires
no astrophysical assumptions — just trigonometry.

    phenomenon:   a nearby star traces a tiny ellipse on the sky over one year
    simulation:   generate a year of noisy (RA, Dec) measurements of 61 Cygni
    dissection:   fit the parallax ellipse with torch — the amplitude IS the
                  parallax angle; also fit the proper motion drifting underneath
    formula:      d(parsec) = 1 / p(arcsec).  That reciprocal DEFINES the parsec.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.manual_seed(0)
rng = np.random.default_rng(3)

# --- truth (unknown to the fitter): 61 Cygni, the star Bessel chose ---
P_TRUE = 0.2860         # arcsec, modern value of the parallax
PM_RA, PM_DEC = 4.10, 3.14   # arcsec/yr proper motion — 61 Cygni is a "flying star",
                             # which is exactly why Bessel guessed it was nearby
NOISE = 0.06            # arcsec, roughly Bessel's heliometer precision

# --- phenomenon: one year of observations ---
n = 60
t = np.sort(rng.uniform(0, 3.0, n))            # 3 years of campaign, in years
# Parallax makes the star swing opposite to Earth's position; proper motion drifts it.
ra_obs = P_TRUE * np.sin(2 * np.pi * t) + PM_RA * t + rng.normal(0, NOISE, n)
dec_obs = P_TRUE * np.cos(2 * np.pi * t) * 0.6 + PM_DEC * t + rng.normal(0, NOISE, n)
#                                          ^ the ellipse is squashed by the star's
#                                            ecliptic latitude; the RA swing is the
#                                            full parallax, so we fit on that.

# --- dissection: separate the yearly wobble from the straight-line drift ---
# Model:  ra(t) = p·sin(2πt) + μ_ra·t + ra0      (three unknowns, linear in all of them)
tt = torch.tensor(t)
design = torch.stack([torch.sin(2 * np.pi * tt), tt, torch.ones_like(tt)], dim=1)
sol = torch.linalg.lstsq(design, torch.tensor(ra_obs).unsqueeze(1)).solution.squeeze(1)
p_fit, pm_fit, ra0_fit = sol.tolist()

# Uncertainty on the parallax: sigma_p = sigma_noise * sqrt((AᵀA)^-1)_00
cov = torch.linalg.inv(design.T @ design)
resid = torch.tensor(ra_obs) - design @ sol
sigma = torch.sqrt((resid**2).sum() / (n - 3))
p_err = (sigma * torch.sqrt(cov[0, 0])).item()

print(f"fitted parallax  p = {p_fit:.4f} ± {p_err:.4f} arcsec   (truth {P_TRUE})")
print(f"fitted proper motion = {pm_fit:.3f} arcsec/yr  (truth {PM_RA})")
print(f"detection significance = {p_fit / p_err:.1f} sigma "
      f"— Bessel needed exactly this argument to be believed")
print()

# --- formula: the parsec is DEFINED so that this is a reciprocal ---
d_pc = 1.0 / p_fit
d_ly = d_pc * 3.26156
d_km = d_pc * 3.0857e13
print(f"d = 1/p = {d_pc:.2f} parsec = {d_ly:.2f} light-years = {d_km:.3e} km")
print(f"Greek upper bound on parallax (naked eye ~ 120\") implied d > {1/120:.4f} pc.")
print("They were not wrong to look. They were wrong to assume 'too small to see' = 'zero'.")

# --- visualization ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: the raw measurement, wobble hidden inside the drift
ax1.errorbar(t, ra_obs, yerr=NOISE, fmt="o", ms=4, color="black", alpha=0.7,
             lw=0.8, label="observed RA offset")
tf = np.linspace(0, 3, 400)
ax1.plot(tf, p_fit * np.sin(2 * np.pi * tf) + pm_fit * tf + ra0_fit, "-",
         color="crimson", lw=2, label="fit: parallax wobble + proper motion")
ax1.plot(tf, pm_fit * tf + ra0_fit, "--", color="steelblue", lw=1.6,
         label="proper motion alone (the star's own travel)")
ax1.set_xlabel("years")
ax1.set_ylabel("RA offset (arcsec)")
ax1.set_title("What the telescope sees:\na straight drift with a 1-year ripple on it")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# Right: subtract the drift — the parallax ellipse appears
ax2.errorbar(t, ra_obs - (pm_fit * t + ra0_fit), yerr=NOISE, fmt="o", ms=5,
             color="black", alpha=0.75, label="observation − proper motion")
ax2.plot(tf, p_fit * np.sin(2 * np.pi * tf), color="crimson", lw=2.2,
         label=f"parallax p = {p_fit:.3f}″ ± {p_err:.3f}″")
ax2.axhline(p_fit, ls=":", color="crimson", lw=1)
ax2.axhline(-p_fit, ls=":", color="crimson", lw=1)
ax2.annotate("", xy=(2.25, p_fit), xytext=(2.25, -p_fit),
             arrowprops=dict(arrowstyle="<->", color="darkgreen", lw=2))
ax2.text(2.32, 0, f"2p\n↓\nd = 1/p\n= {d_pc:.2f} pc\n= {d_ly:.1f} ly",
         fontsize=9.5, color="darkgreen", va="center")
ax2.set_xlabel("years")
ax2.set_ylabel("RA offset with drift removed (arcsec)")
ax2.set_title("Bessel 1838, 61 Cygni:\nthe Earth's orbit, seen reflected in a star")
ax2.legend(fontsize=8, loc="lower left")
ax2.grid(alpha=0.3)

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
