"""v = H₀d — twenty-four galaxies that turned the universe into an event.

Hubble published in 1929 with 24 galaxies, a distance range of barely 2 Mpc, and
scatter so large that a modern referee would reject the paper. His value of H₀ was
about 500 km/s/Mpc — seven times too big, because his Cepheid calibration was wrong
(he had confused two types of Cepheid, a mistake Baade only caught in 1952). Running
his number backwards gave a universe two billion years old, younger than rocks
already dated on Earth. The "age paradox" was a serious argument against the whole
picture for twenty years.

He was wrong in the number and right in the structure — and the structure is what
mattered: recession velocity is PROPORTIONAL to distance. No other relation would
have implied expansion. If v were constant with d, or fell off, we would be at the
centre of an explosion. Proportionality, and only proportionality, is what you get
when space itself stretches uniformly (→ expanding_space_no_center.py).

Credit where due: Lemaître published the same relation, with a better H₀, in 1927 —
in French, in an obscure Belgian journal. The 1931 English translation had the
relevant paragraphs removed, apparently by Lemaître himself.

    phenomenon:   nearly every galaxy's spectrum is redshifted, and more distant
                  galaxies more so
    simulation:   Hubble's actual 1929 table plus a modern sample
    dissection:   fit v = H₀·d with torch, both datasets, and propagate the error
    formula:      1/H₀ is a TIME — the age of the universe if expansion were
                  constant. Getting that number right took 70 years.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

MPC_KM = 3.0856775814913673e19
YR_S = 3.155815e7

# --- Hubble's 1929 data, as published (distances in Mpc, velocities in km/s) ---
hubble_d = np.array([0.032, 0.034, 0.214, 0.263, 0.275, 0.275, 0.45, 0.5, 0.5,
                     0.63, 0.8, 0.9, 0.9, 0.9, 0.9, 1.0, 1.1, 1.1, 1.4, 1.7,
                     2.0, 2.0, 2.0, 2.0])
hubble_v = np.array([170, 290, -130, -70, -185, -220, 200, 290, 270, 200, 300,
                     -30, 650, 150, 500, 920, 450, 500, 500, 960, 500, 850,
                     800, 1090], dtype=float)

# --- a modern sample reaching a thousand times farther ---
rng = np.random.default_rng(23)
H0_TRUE = 70.0
mod_d = np.sort(rng.uniform(10, 400, 60))
mod_v = H0_TRUE * mod_d + rng.normal(0, 350, len(mod_d))     # peculiar velocities


def fit_H0(d, v):
    """Fit v = H0·d through the origin, and get an honest error bar."""
    X = torch.tensor(d, dtype=torch.float64).unsqueeze(1)
    y = torch.tensor(v, dtype=torch.float64).unsqueeze(1)
    H = float(torch.linalg.lstsq(X, y).solution)
    resid = y.squeeze(1) - H * X.squeeze(1)
    dof = len(d) - 1
    sigma = float(torch.sqrt((resid**2).sum() / dof))
    H_err = sigma / float(torch.sqrt((X**2).sum()))
    return H, H_err, sigma


H_1929, e_1929, s_1929 = fit_H0(hubble_d, hubble_v)
H_mod, e_mod, s_mod = fit_H0(mod_d, mod_v)


def age_gyr(H0):
    """1/H0 in billions of years. H0 arrives as km/s/Mpc."""
    return (MPC_KM / H0) / YR_S / 1e9


print("HUBBLE 1929  (24 galaxies, all within 2 Mpc)")
print(f"  H₀ = {H_1929:.1f} ± {e_1929:.1f} km/s/Mpc")
print(f"  scatter about the fit = {s_1929:.0f} km/s — comparable to the signal itself")
print(f"  → age of the universe 1/H₀ = {age_gyr(H_1929):.2f} Gyr")
print(f"  The Earth was already known to be ~2 Gyr old from radioactive dating.")
print(f"  Hubble's own number made the universe YOUNGER THAN THE ROCKS IN IT.\n")

print("MODERN SAMPLE  (60 galaxies out to 400 Mpc)")
print(f"  H₀ = {H_mod:.1f} ± {e_mod:.1f} km/s/Mpc")
print(f"  → age 1/H₀ = {age_gyr(H_mod):.2f} Gyr")
print(f"  (the true age, 13.8 Gyr, needs the full expansion history — ch08)\n")

print(f"Hubble's error factor: {H_1929/H_mod:.1f}×.  Two causes, both calibration:")
print("  1. He used Type II Cepheids to calibrate Type I ones (Baade, 1952).")
print("  2. What he took for bright stars in distant galaxies were often whole")
print("     H II regions — much more luminous, so he placed them far too close.")
print("  Neither error touched the LINEARITY, which is the physics.\n")

# --- the history of H0: a century of halving ---
history = [(1929, 500, 100, "Hubble"), (1936, 526, 80, "Hubble & Humason"),
           (1956, 180, 40, "Humason, Mayall & Sandage"), (1958, 75, 25, "Sandage"),
           (1974, 55, 7, "Sandage & Tammann"), (1974, 100, 10, "de Vaucouleurs"),
           (1996, 73, 10, "HST Key Project (early)"), (2001, 72, 8, "HST Key Project"),
           (2011, 73.8, 2.4, "SH0ES"), (2018, 67.4, 0.5, "Planck CMB"),
           (2022, 73.0, 1.0, "SH0ES (Cepheid)"), (2023, 67.8, 0.7, "CMB + BAO")]
print("A century of measuring one number:")
print("  year   H₀ (km/s/Mpc)      who")
for yr, h, e, who in history:
    print(f"  {yr}   {h:6.1f} ± {e:4.1f}     {who}")
print()
print("Note the last four entries. Two methods, each claiming ~1% precision, that")
print("disagree by 5σ. That is the HUBBLE TENSION: the distance-ladder route (ch00)")
print("and the CMB route (ch08) do not meet. It is currently the most important")
print("unexplained number in physics — and it is, once again, astronomy producing a")
print("residual that no existing theory absorbs. The pattern from ch04 repeats.")

# --- visualization ---
fig = plt.figure(figsize=(15, 8.5))
gs = fig.add_gridspec(2, 3)

# Panel 1: Hubble's original plot
ax = fig.add_subplot(gs[0, 0])
ax.plot(hubble_d, hubble_v, "o", color="black", ms=6)
dd = np.linspace(0, 2.1, 20)
ax.plot(dd, H_1929 * dd, color="crimson", lw=2,
        label=f"H₀ = {H_1929:.0f} km/s/Mpc")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("distance (Mpc)")
ax.set_ylabel("recession velocity (km/s)")
ax.set_title("Hubble 1929, the actual data.\n24 galaxies, huge scatter, and some\n"
             "moving TOWARD us. He was right anyway.")
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3)

# Panel 2: the modern version, same axes ratio
ax = fig.add_subplot(gs[0, 1])
ax.plot(mod_d, mod_v, "o", color="steelblue", ms=5, alpha=0.8)
dd = np.linspace(0, 410, 20)
ax.plot(dd, H_mod * dd, color="crimson", lw=2, label=f"H₀ = {H_mod:.1f} km/s/Mpc")
# Hubble's 1929 box, to scale
ax.add_patch(plt.Rectangle((0, -250), 2.1, 1400, fill=False, edgecolor="black",
                           lw=1.5, ls="--"))
ax.text(12, 900, "all of Hubble's 1929 data\nfits in this box", fontsize=8)
ax.set_xlabel("distance (Mpc)")
ax.set_ylabel("recession velocity (km/s)")
ax.set_title("The same law, 200× farther out.\nThe scatter that dominated Hubble's\n"
             "plot is now a rounding error.")
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3)

# Panel 3: why proportionality specifically
ax = fig.add_subplot(gs[0, 2])
dd = np.linspace(0, 400, 100)
ax.plot(dd, H_mod * dd, color="crimson", lw=2.4, label="v ∝ d  → uniform expansion")
ax.plot(dd, np.full_like(dd, 8000), color="gray", lw=1.6, ls="--",
        label="v = const → we're in a shell")
ax.plot(dd, 20000 * (1 - np.exp(-dd / 120)), color="gray", lw=1.6, ls=":",
        label="v saturating → edge of an explosion")
ax.plot(mod_d, mod_v, "o", color="steelblue", ms=4, alpha=0.6)
ax.set_xlabel("distance (Mpc)")
ax.set_ylabel("velocity (km/s)")
ax.set_title("Only strict proportionality implies\nexpanding SPACE. The other curves\n"
             "would put us at a centre.")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel 4: the century of H0
ax = fig.add_subplot(gs[1, :2])
yrs = [h[0] for h in history]
vals = [h[1] for h in history]
errs = [h[2] for h in history]
cols = ["crimson" if "SH0ES" in h[3] or "Cepheid" in h[3] else
        "steelblue" if "CMB" in h[3] else "black" for h in history]
for yr, v, e, c in zip(yrs, vals, errs, cols):
    ax.errorbar(yr, v, yerr=e, fmt="o", color=c, ms=7, capsize=4)
ax.axhspan(67.0, 68.5, color="steelblue", alpha=0.2, label="CMB / early-universe route")
ax.axhspan(72.0, 74.5, color="crimson", alpha=0.2, label="distance-ladder route")
ax.set_yscale("log")
ax.set_xlabel("year")
ax.set_ylabel("H₀ (km/s/Mpc)")
ax.set_title("Ninety years of measuring one number. It fell by a factor of seven, "
             "then split in two.\n"
             "The gap between the two shaded bands is the Hubble tension — a live, "
             "unexplained 5σ residual.")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which="both")

# Panel 5: 1/H0 as an age
ax = fig.add_subplot(gs[1, 2])
H_range = np.linspace(45, 550, 300)
ax.plot(H_range, [age_gyr(h) for h in H_range], color="black", lw=2.2)
ax.axhline(4.54, color="darkorange", ls="--", lw=1.8, label="age of the Earth (4.54 Gyr)")
ax.axhline(13.8, color="darkgreen", ls="--", lw=1.8, label="accepted age (13.8 Gyr)")
ax.plot(H_1929, age_gyr(H_1929), "o", color="crimson", ms=11)
ax.annotate("Hubble 1929:\nuniverse younger\nthan the Earth", (H_1929, age_gyr(H_1929)),
            xytext=(-95, 42), textcoords="offset points", fontsize=8, color="crimson",
            arrowprops=dict(arrowstyle="->", color="crimson"))
ax.plot(H_mod, age_gyr(H_mod), "o", color="darkgreen", ms=11)
ax.set_xscale("log")
ax.set_xlabel("H₀ (km/s/Mpc)")
ax.set_ylabel("1/H₀  (Gyr)")
ax.set_title("The Hubble constant is an inverse TIME.\nThat is the whole reason "
             "this measurement\nmatters — it dates the universe.")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
