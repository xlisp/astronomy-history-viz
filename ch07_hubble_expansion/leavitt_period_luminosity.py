"""Henrietta Leavitt, 1912: the ruler that made cosmology possible.

Leavitt was employed at Harvard College Observatory as a "computer" at 30 cents an
hour, cataloguing variable stars on photographic plates. She was not permitted to
operate a telescope. Examining 1777 variables in the Small Magellanic Cloud, she
noticed something nobody had asked her to look for:

    "It is worthy of notice that... the brighter variables have the longer periods."

The critical insight is WHY this is usable. All the stars in the SMC are at
essentially the same distance, so their apparent brightnesses are their relative
true brightnesses. The period–luminosity relation she found turns a Cepheid into a
standard candle: measure how fast it pulses, and you know how bright it truly is;
compare to how bright it looks, and you have the distance.

Everything above the bottom rung of the ladder (ch00) rests on this. Hubble used
it to prove Andromeda is another galaxy (1924), and then to find the expansion of
the universe (1929). Leavitt died of cancer in 1921 at 53. Mittag-Leffler wrote in
1925 hoping to nominate her for the Nobel Prize and was informed she had been dead
four years.

    phenomenon:   Cepheid variables pulse; brighter ones pulse more slowly
    simulation:   Leavitt's SMC sample plus modern calibrating Cepheids
    dissection:   fit M = a·log₁₀P + b with torch; the slope is the law
    formula:      distance modulus  μ = m − M = 5·log₁₀(d/10 pc)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

rng = np.random.default_rng(17)

# --- Leavitt's SMC sample: apparent magnitudes, all at one (unknown) distance ---
N = 40
logP = np.sort(rng.uniform(0.2, 1.8, N))                  # log10(period in days)
MU_SMC = 18.96                                            # true SMC distance modulus
A_TRUE, B_TRUE = -2.81, -1.43                             # the real P-L relation, V band
m_smc = A_TRUE * logP + B_TRUE + MU_SMC + rng.normal(0, 0.16, N)

# --- fit the relation Leavitt could see: apparent mag vs log period ---
X = torch.tensor(np.stack([logP, np.ones(N)], axis=1))
y = torch.tensor(m_smc)
slope, icept = torch.linalg.lstsq(X, y.unsqueeze(1)).solution.squeeze(1)
resid = y - X @ torch.stack([slope, icept])
scatter = float(resid.std())

print("Leavitt's fit (apparent magnitude, one galaxy, distance unknown):")
print(f"  m = {slope:.3f}·log₁₀P + {icept:.3f}      scatter = {scatter:.3f} mag")
print(f"  the SLOPE is already the physics; only the ZERO POINT needs a distance.\n")

# --- calibrate the zero point with the one thing that needs no assumptions: parallax ---
# Real Galactic Cepheids: (name, log P, mean m_V, parallax in mas, extinction A_V).
# Interstellar dust dims them, so A_V must be subtracted or every star looks too far.
cal = [("δ Cephei",    0.7297, 3.95, 3.66,  0.23),
       ("η Aquilae",   0.8559, 3.90, 3.33,  0.49),
       ("ζ Geminorum", 1.0065, 3.92, 2.78,  0.06),
       ("β Doradus",   0.9931, 3.76, 3.13,  0.09),
       ("ℓ Carinae",   1.5509, 3.72, 2.00,  0.51),
       ("RS Puppis",   1.6174, 6.95, 0.524, 1.03)]

# The standard division of labour, and the reason it works: the SMC sample fixes
# the SLOPE (many stars, one common distance, so the unknown distance cancels),
# while the parallax stars fix the ZERO POINT (few stars, but absolute distances).
print("Zero-point calibration from parallax (the only assumption-free rung):")
print("  star           log P   m_V    A_V    π (mas)    d (pc)     M_V")
M_cal, logP_cal = [], []
for nm, lp, mv, plx, av in cal:
    d_pc = 1000.0 / plx
    M = mv - av - 5 * np.log10(d_pc / 10)
    M_cal.append(M); logP_cal.append(lp)
    print(f"  {nm:13s} {lp:6.3f} {mv:5.2f}  {av:5.2f}  {plx:7.3f}  {d_pc:8.1f}  {M:6.2f}")

a_cal = slope                                    # slope inherited from the SMC fit
b_cal = torch.tensor(np.mean(np.array(M_cal) - float(a_cal) * np.array(logP_cal)))
zp_err = float(np.std(np.array(M_cal) - float(a_cal) * np.array(logP_cal))) / np.sqrt(len(cal))
print(f"\n  slope from the SMC sample:  {float(a_cal):.3f}   (accepted {A_TRUE:.3f})")
print(f"  zero point from parallax:   {float(b_cal):.3f} ± {zp_err:.3f}   "
      f"(accepted {B_TRUE:.3f})")
print(f"  → M_V = {float(a_cal):.3f}·log₁₀P + {float(b_cal):.3f}")
print(f"  The {abs(float(b_cal)-B_TRUE):.2f} mag zero-point offset from only six calibrators")
print(f"  translates into a {100*(10**(abs(float(b_cal)-B_TRUE)/5)-1):.0f}% distance error. "
      f"This is not a toy problem:")
print(f"  the Cepheid zero point is the single largest systematic in H₀ today.\n")

# --- now the payoff: the SMC's distance falls out of the offset ---
mu_smc_fit = float(icept - b_cal)
d_smc = 10 ** (mu_smc_fit / 5 + 1)
print(f"distance modulus of the SMC  μ = {mu_smc_fit:.3f}  (accepted {MU_SMC:.2f})")
print(f"  → d = {d_smc/1000:.1f} kpc = {d_smc*3.26156/1000:.0f} thousand light-years\n")

# --- and the payoff's payoff: Andromeda, and the end of the "island universe" debate ---
print("Hubble, 1924: he found Cepheids in the Andromeda 'nebula'.")
m_and, logP_and = 19.0, 1.4
M_and = float(a_cal * logP_and + b_cal)
mu_and = m_and - M_and
d_and = 10 ** (mu_and / 5 + 1)
print(f"  a Cepheid with log P = {logP_and} appears at m = {m_and}")
print(f"  its true magnitude must be M = {M_and:.2f}")
print(f"  μ = m − M = {mu_and:.2f}  →  d = {d_and/1e3:.0f} kpc = "
      f"{d_and*3.26156/1e6:.2f} million light-years")
print(f"  The Milky Way is ~30 kpc across. Andromeda is {d_and/30000:.0f}× farther than")
print(f"  our entire galaxy is wide — so it is a galaxy of its own.")
print()
print("That single calculation ended the Great Debate (Shapley vs Curtis, 1920) and")
print("multiplied the known size of the universe by a factor of a hundred thousand.")
print("It was made possible by a woman who was not allowed to use the telescope.")

# --- visualization ---
fig = plt.figure(figsize=(15, 8.5))
gs = fig.add_gridspec(2, 3)

# Panel 1: Leavitt's plot, essentially as she drew it
ax = fig.add_subplot(gs[0, 0])
ax.plot(logP, m_smc, "o", color="black", ms=6)
lp = np.linspace(0.1, 1.9, 50)
ax.plot(lp, float(slope) * lp + float(icept), color="crimson", lw=2,
        label=f"m = {slope:.2f} log P + {icept:.2f}")
ax.invert_yaxis()
ax.set_xlabel("log₁₀ (period in days)")
ax.set_ylabel("apparent magnitude (brighter ↑)")
ax.set_title("Leavitt 1912, redrawn.\n1777 variables in one small galaxy —\n"
             "so 'apparent' equals 'relative true'.")
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3)

# Panel 2: pulsation curves
ax = fig.add_subplot(gs[0, 1])
t = np.linspace(0, 3, 500)
for P, off, col in [(3.0, 0, "#9ecae1"), (10.0, 2.2, "#4292c6"), (30.0, 4.4, "#08519c")]:
    ph = t * 30 / P
    # Cepheids have a sawtooth light curve: fast rise, slow decline
    lc = 0.5 * (np.abs(((ph % 1) * 2 - 1)) ** 1.6) * 2 - 0.5
    M = A_TRUE * np.log10(P) + B_TRUE
    ax.plot(t * 30, lc + off, color=col, lw=1.8,
            label=f"P = {P:g} d   M_V = {M:.2f}")
ax.set_xlabel("time (days)")
ax.set_ylabel("brightness (offset)")
ax.set_yticks([])
ax.set_title("Longer period ⇒ intrinsically brighter.\n"
             "The star is a pulsating heat engine;\nbigger stars ring more slowly.")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel 3: calibrated absolute relation
ax = fig.add_subplot(gs[0, 2])
ax.plot(logP_cal, M_cal, "s", color="darkgreen", ms=10, label="Galactic, Gaia parallax")
ax.plot(lp, float(a_cal) * lp + float(b_cal), color="darkgreen", lw=2)
ax.plot(logP, m_smc - mu_smc_fit, "o", color="black", ms=5, alpha=0.6,
        label="SMC, shifted by μ")
ax.invert_yaxis()
ax.set_xlabel("log₁₀ (period in days)")
ax.set_ylabel("absolute magnitude M_V")
ax.set_title("Anchor the zero point with parallax and\nthe relation becomes a "
             "distance meter\nfor anything you can resolve Cepheids in.")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel 4: how far this ruler reaches
ax = fig.add_subplot(gs[1, :2])
targets = [("LMC", 49.97e3), ("SMC", d_smc), ("Andromeda (M31)", d_and),
           ("M81", 3.6e6), ("Virgo cluster", 16.5e6), ("NGC 4258", 7.6e6),
           ("HST limit for Cepheids", 3.5e7)]
names = [t[0] for t in targets]
dists = [t[1] for t in targets]
ax.barh(names, dists, color=plt.cm.viridis(np.linspace(0.15, 0.85, len(targets))),
        alpha=0.9)
ax.set_xscale("log")
ax.set_xlabel("distance (parsec)")
ax.axvline(3e4, color="crimson", ls="--", lw=2, label="diameter of the Milky Way")
for i, d in enumerate(dists):
    ax.text(d * 1.2, i, f"{d/1e6:.2f} Mpc" if d > 1e6 else f"{d/1e3:.0f} kpc",
            va="center", fontsize=8)
ax.set_title("The reach of Leavitt's ruler. Everything to the right of the red line "
             "is outside our galaxy —\nand before 1924 nobody knew there was such a "
             "place.")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="x", which="both")

# Panel 5: the chain of consequences
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
ax.text(0.0, 1.0, "What one 30-cent-an-hour\njob produced", fontsize=11.5,
        weight="bold", va="top")
ax.text(0.0, 0.82, (
    "1912  Leavitt: P–L relation\n"
    "        ↓\n"
    "1924  Hubble: Andromeda is a\n"
    "      separate galaxy. The universe\n"
    "      is 10⁵× bigger than believed.\n"
    "        ↓\n"
    "1929  Hubble: v = H₀d. The universe\n"
    "      is expanding.\n"
    "        ↓\n"
    "1931  Lemaître: run it backwards —\n"
    "      the 'primeval atom'\n"
    "        ↓\n"
    "1965  the CMB is found (ch08)\n"
    "        ↓\n"
    "1998  Type Ia SNe, calibrated on\n"
    "      Cepheids → dark energy\n"
    "        ↓\n"
    "2020s the Hubble tension, still a\n"
    "      fight about Cepheid calibration\n\n"
    "Every one of these rests on her\n"
    "bottom rung."),
    fontsize=8.4, va="top", family="monospace")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
