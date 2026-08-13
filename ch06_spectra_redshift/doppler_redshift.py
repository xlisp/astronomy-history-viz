"""The spectral line as a speedometer — and the three different things a shift can mean.

Once Kirchhoff showed that each line sits at a fixed laboratory wavelength, any
observed displacement of that line becomes a velocity measurement. Huggins did it
first in 1868, getting Sirius' motion to within a few tens of km/s using a visual
spectroscope and a great deal of patience.

This is the single most productive measurement in astronomy. It gave us:
    · binary star masses (and therefore all stellar masses)
    · galaxy rotation curves → dark matter (ch08)
    · exoplanets by radial velocity → the 2019 Nobel Prize
    · the expansion of the universe (ch07)

But the last of those requires care, because THREE physically different things all
shift a line, and telling them apart is most of modern cosmology:

    1. classical Doppler    — the source moves through space
    2. relativistic Doppler — plus time dilation, matters above ~0.1c
    3. cosmological redshift — space itself expands while the light is in flight;
                               nothing is "moving" at all

Hubble's galaxies are case 3, but the small-z formula happens to look like case 1,
which caused (and still causes) enormous confusion. z > 1 galaxies are not
travelling faster than light; they are not travelling at all.

    phenomenon:   a known spectral line is observed away from its rest wavelength
    simulation:   shift a hydrogen spectrum by each of the three mechanisms
    dissection:   invert the shift for velocity; compare the three formulas
    formula:      1 + z = λ_obs/λ_rest, and what z means depends on the physics.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

C_KMS = 299792.458

# rest wavelengths (nm) of lines every astronomer knows by heart
REST = {"Ca⁺ K": 393.37, "Ca⁺ H": 396.85, "H δ": 410.17, "H γ": 434.05,
        "H β": 486.13, "Mg b": 517.27, "Na D": 589.29, "H α": 656.28}


def z_from_v_classical(v):
    return v / C_KMS


def z_from_v_relativistic(v):
    b = v / C_KMS
    return torch.sqrt((1 + b) / (1 - b)) - 1


def v_from_z_relativistic(z):
    return C_KMS * ((1 + z) ** 2 - 1) / ((1 + z) ** 2 + 1)


print("The same observed redshift z, read by three different formulas:")
print("     z      classical v      relativistic v     recession 'speed'")
print("            = cz (km/s)      (km/s)             at that z (km/s)")
for z in (0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 7.0):
    zt = torch.tensor(z, dtype=torch.float64)
    v_cl = z * C_KMS
    v_rel = float(v_from_z_relativistic(zt))
    flag = "  ← exceeds c!" if v_cl > C_KMS else ""
    print(f"  {z:5.3f}   {v_cl:12.1f}   {v_rel:14.1f}      {v_cl:12.1f}{flag}")
print()
print("At z = 7 the naive cz gives 2.1 million km/s — seven times the speed of light.")
print("Nothing is wrong with relativity; the formula is simply the wrong one. Those")
print("photons were not outrun by anything. The space they crossed grew underneath")
print("them by a factor of 1+z = 8 while they were in transit.\n")

# --- a worked example: measuring a galaxy's redshift from its spectrum ---
Z_TRUE = 0.0335                     # roughly the Coma cluster
lam = torch.linspace(370, 700, 4000, dtype=torch.float64)


def spectrum(z, depths=None):
    depths = depths or {k: 0.55 for k in REST}
    s = torch.ones_like(lam) * (1 + 0.0006 * (lam - 500))     # a sloped continuum
    for name, l0 in REST.items():
        s = s * (1 - depths[name] * torch.exp(-((lam - l0 * (1 + z)) / 1.6) ** 2))
    return s


torch.manual_seed(2)
obs = spectrum(Z_TRUE) + 0.012 * torch.randn(len(lam), dtype=torch.float64)

# --- dissection: cross-correlate against a rest-frame template ---
# This is exactly what every redshift pipeline in the world does.
z_grid = torch.linspace(0.0, 0.08, 1600, dtype=torch.float64)
template_0 = spectrum(0.0)
scores = []
for zz in z_grid:
    t = spectrum(float(zz))
    a, b = obs - obs.mean(), t - t.mean()
    scores.append(float((a * b).sum() / torch.sqrt((a * a).sum() * (b * b).sum())))
scores = torch.tensor(scores)
z_best = float(z_grid[int(torch.argmax(scores))])

print(f"cross-correlation redshift measurement:")
print(f"  recovered z = {z_best:.5f}   (truth {Z_TRUE:.5f})")
print(f"  → cz = {z_best*C_KMS:.0f} km/s")
print(f"  every line moves by the SAME FACTOR (1+z), not the same amount — that is")
print(f"  how you tell a redshift from an instrument error or a blended line.\n")

print("  line       rest λ (nm)   observed λ (nm)   shift (nm)")
for name, l0 in REST.items():
    print(f"  {name:9s}  {l0:10.2f}   {l0*(1+z_best):13.2f}   {l0*z_best:+9.2f}")

# --- visualization ---
fig = plt.figure(figsize=(15, 8.5))
gs = fig.add_gridspec(2, 3)

# Panel 1: rest vs observed spectrum
ax = fig.add_subplot(gs[0, :])
ax.plot(lam, spectrum(0.0) + 0.55, color="steelblue", lw=1.2,
        label="laboratory / rest frame")
ax.plot(lam, obs, color="black", lw=1.0, label=f"observed galaxy (z = {z_best:.4f})")
for name, l0 in REST.items():
    ax.annotate("", xy=(l0 * (1 + z_best), 0.30), xytext=(l0, 0.85),
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2, alpha=0.8))
    ax.text(l0, 1.48, name, fontsize=7.5, ha="center", color="steelblue", rotation=90)
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("relative flux (offset)")
ax.set_title("Every line slides redward by the same FACTOR 1+z — long wavelengths "
             "move farther in nm than short ones.\n"
             "That proportionality is the signature of a redshift, and it is why "
             "one spectrum gives one unambiguous number.")
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.3)

# Panel 2: the cross-correlation peak
ax = fig.add_subplot(gs[1, 0])
ax.plot(z_grid, scores, color="crimson", lw=1.6)
ax.axvline(Z_TRUE, color="darkgreen", ls="--", lw=1.8, label=f"true z = {Z_TRUE}")
ax.plot(z_best, scores.max(), "o", color="black", ms=9, label=f"recovered {z_best:.4f}")
ax.set_xlabel("trial redshift z")
ax.set_ylabel("correlation with template")
ax.set_title("How every redshift in every survey\nis actually measured: slide a "
             "template\nuntil it locks on.")
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3)

# Panel 3: the three formulas diverging
ax = fig.add_subplot(gs[1, 1])
zs = torch.linspace(0.001, 8, 400, dtype=torch.float64)
ax.plot(zs, zs * C_KMS / C_KMS, color="steelblue", lw=2.2, label="classical  v = cz")
ax.plot(zs, v_from_z_relativistic(zs) / C_KMS, color="crimson", lw=2.2,
        label="relativistic Doppler")
ax.axhline(1.0, color="black", ls=":", lw=1.6, label="speed of light")
ax.set_xlabel("redshift z")
ax.set_ylabel("inferred v / c")
ax.set_title("Above z ≈ 0.1 the two Doppler formulas\npart company. Above z = 1 the "
             "classical\none is nonsense — yet it is what 'v = cz'\nmeans in Hubble's law.")
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3)

# Panel 4: which mechanism applies where
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
ax.text(0.0, 1.0, "Three shifts, three meanings", fontsize=11.5, weight="bold", va="top")
ax.text(0.0, 0.87, (
    "1. DOPPLER (source moves)\n"
    "   stars in our galaxy, binaries,\n"
    "   exoplanet wobbles, galaxy rotation\n"
    "   → real motion through space\n\n"
    "2. GRAVITATIONAL (Einstein 1916)\n"
    "   light climbing out of a potential\n"
    "   well loses energy: white dwarfs,\n"
    "   Pound–Rebka 1959, GPS clocks\n"
    "   → no motion at all\n\n"
    "3. COSMOLOGICAL (Lemaître 1927)\n"
    "   space expands during the flight\n"
    "   1 + z = a(now)/a(then)\n"
    "   → also no motion at all\n\n"
    "A z = 7 galaxy is not receding at 7c.\n"
    "It is not receding. The universe was\n"
    "8× smaller when that light set out.\n\n"
    "Hubble measured case 3 and described\n"
    "it with case 1's formula. We have been\n"
    "explaining the difference ever since."),
    fontsize=8.4, va="top", family="monospace")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
