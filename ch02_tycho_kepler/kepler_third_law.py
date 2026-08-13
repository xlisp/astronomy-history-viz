"""T² = a³ — one line that turned the solar system from a list into a system.

Kepler hunted this for 22 years. He tried T ∝ a, T ∝ a², sphere-packing, the five
Platonic solids (his Mysterium Cosmographicum, 1596, is a beautiful catastrophe).
On 15 May 1618 he tried logarithms — Napier's tables had reached him only two years
earlier — and log T vs log a fell on a straight line of slope 3/2.

This is the archetypal act of data analysis: TAKE THE LOG. A power law is a
straight line in log space, and nothing else is. The slope is the exponent. That
trick is the reason the log shows up everywhere in science (see the math-history-viz
companion project, Chapter 0.5).

And the constant matters more than the exponent. T² = a³ holds for all six planets
with ONE constant — which means whatever holds Mars is the same thing that holds
Saturn. Ptolemy's model has a separate machine per planet and no way to say that.

    phenomenon:   distant planets take longer per orbit — but how much longer?
    simulation:   none needed. Use the real modern values for all 8 planets,
                  plus the Galilean moons of Jupiter as an independent test.
    dissection:   fit log T = m·log a + b with torch; read off m
    formula:      m = 3/2 exactly ⟹ T² ∝ a³.  Newton later shows the constant is
                  4π²/GM, which turns Kepler's law into a way to WEIGH the Sun.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# --- the data Kepler had (six planets), plus the two found later ---
# Semi-major axes are JPL osculating elements (AU); periods are sidereal (yr).
# Note: do NOT substitute the "mean distance to the Sun" quoted in popular tables —
# the time-average of r is a(1+e²/2), not a, and for Saturn that alone would fake
# a 0.5% violation of a law that is actually exact.
planets = [
    ("Mercury",  0.387098,   0.2408467, True),
    ("Venus",    0.723332,   0.6151973, True),
    ("Earth",    1.000000,   1.0000174, True),
    ("Mars",     1.523710,   1.8808476, True),
    ("Jupiter",  5.202887,  11.8620         , True),
    ("Saturn",   9.536676,  29.4571         , True),
    ("Uranus",  19.189165,  84.0148         , False),     # Herschel 1781
    ("Neptune", 30.069923, 164.7890         , False),     # Le Verrier 1846 — see ch04
]
names = [p[0] for p in planets]
a = torch.tensor([p[1] for p in planets], dtype=torch.float64)
T = torch.tensor([p[2] for p in planets], dtype=torch.float64)
kepler_had = torch.tensor([p[3] for p in planets])

# --- dissection: take the log, fit a line. That is the whole method. ---
loga, logT = torch.log10(a), torch.log10(T)
design = torch.stack([loga, torch.ones_like(loga)], dim=1)
m, b = torch.linalg.lstsq(design, logT.unsqueeze(1)).solution.squeeze(1)

print("fit  log10(T) = m · log10(a) + b     over all 8 planets")
print(f"  m = {m.item():.9f}      (exact value 3/2 = 1.5)")
print(f"  b = {b.item():.2e}      (0 because we chose Earth's units)")
print(f"  deviation from 3/2 = {abs(m.item() - 1.5):.2e}")
print()

# --- how good is it, planet by planet? ---
print("  planet      a (AU)     T (yr)      T²/a³        residual (%)")
ratio = T**2 / a**3
for i, nm in enumerate(names):
    pred = 10 ** (m * loga[i] + b)
    resid = (T[i] - pred) / T[i] * 100
    tag = "" if kepler_had[i] else "   ← not yet discovered in 1618"
    print(f"  {nm:9s} {a[i]:9.5f} {T[i]:10.4f}   {ratio[i]:.7f}    {resid:+.4f}{tag}")
print(f"\n  T²/a³ is constant to {(ratio.std()/ratio.mean()):.2e} across a 78× range in distance.")
print("  Uranus and Neptune were found 160 and 230 years later — and landed exactly")
print("  on Kepler's line without adjustment. THAT is what a real law does.")
print()

# --- an independent system: Jupiter's moons (Galileo, Jan 1610) ---
# Same law, different central mass ⟹ same slope, different intercept.
# Orbital radii in AU (from km / 1.495979e8) and periods in years (from days / 365.25).
moons = [("Io",        421_700 / 1.495979e8,  1.769138 / 365.25),
         ("Europa",    671_034 / 1.495979e8,  3.551181 / 365.25),
         ("Ganymede", 1_070_412 / 1.495979e8,  7.154553 / 365.25),
         ("Callisto", 1_882_709 / 1.495979e8, 16.689018 / 365.25)]
am = torch.tensor([x[1] for x in moons], dtype=torch.float64)
Tm = torch.tensor([x[2] for x in moons], dtype=torch.float64)
dm = torch.stack([torch.log10(am), torch.ones_like(am)], dim=1)
mm, bm = torch.linalg.lstsq(dm, torch.log10(Tm).unsqueeze(1)).solution.squeeze(1)
print(f"Jupiter's four Galilean moons:  slope = {mm.item():.6f}   (again 3/2)")

# --- formula: Newton's constant lets us WEIGH the central body ---
# T² = 4π²a³/(GM)  ⟹  M = 4π²a³/(G T²)
G = 6.67430e-11
AU, YR = 1.495978707e11, 3.155815e7
M_sun = 4 * np.pi**2 * (a[3].item() * AU)**3 / (G * (T[3].item() * YR)**2)
M_jup = 4 * np.pi**2 * (am[2].item() * AU)**3 / (G * (Tm[2].item() * YR)**2)
print(f"\ninvert the constant:  M = 4π²a³/(G·T²)")
print(f"  from Mars' orbit      → M_sun     = {M_sun:.4e} kg   (accepted 1.989e30)")
print(f"  from Ganymede's orbit → M_jupiter = {M_jup:.4e} kg   (accepted 1.898e27)")
print(f"  Sun / Jupiter mass ratio = {M_sun/M_jup:.0f}")
print("\nA law about TIMING became a scale for weighing objects we can never touch.")
print("Every exoplanet mass, every black hole mass, every galaxy mass still uses this.")

# --- visualization ---
fig = plt.figure(figsize=(15, 6.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.1, 1])

# Panel 1: linear axes — Kepler's 22 years of failure
ax = fig.add_subplot(gs[0, 0])
ax.plot(a, T, "o", color="black", ms=9)
xs = torch.linspace(0.2, 32, 300, dtype=torch.float64)
ax.plot(xs, xs, "--", color="gray", lw=1.4, label="T ∝ a   (tried, failed)")
ax.plot(xs, xs**2 / 8, ":", color="gray", lw=1.4, label="T ∝ a²  (tried, failed)")
ax.plot(xs, xs**1.5, "-", color="crimson", lw=2, label="T ∝ a^1.5  (correct)")
for i, nm in enumerate(names):
    ax.annotate(nm, (a[i], T[i]), fontsize=7.5, xytext=(5, -4),
                textcoords="offset points")
ax.set_xlabel("semi-major axis a (AU)")
ax.set_ylabel("period T (yr)")
ax.set_ylim(0, 180)
ax.set_xlim(0, 32)
ax.set_title("On linear axes the exponent is invisible.\n"
             "Kepler burned 22 years here.")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel 2: log-log — the entire discovery, in one straight line
ax = fig.add_subplot(gs[0, 1])
mask = kepler_had.numpy()
ax.plot(loga[mask], logT[mask], "o", color="black", ms=10, label="the 6 planets Kepler had")
ax.plot(loga[~mask], logT[~mask], "s", color="crimson", ms=10,
        label="Uranus (1781), Neptune (1846)")
xl = torch.linspace(-0.5, 1.55, 50, dtype=torch.float64)
ax.plot(xl, m * xl + b, "-", color="crimson", lw=1.8,
        label=f"fit: slope = {m.item():.6f}")
# the slope triangle, drawn explicitly
ax.plot([0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.5, 0.0], color="darkgreen", lw=1.6, ls="--")
ax.text(0.52, -0.13, "Δ log a = 1", fontsize=9, color="darkgreen", ha="center")
ax.text(1.05, 0.75, "Δ log T = 3/2", fontsize=9, color="darkgreen", rotation=90, va="center")
for i, nm in enumerate(names):
    ax.annotate(nm, (loga[i], logT[i]), fontsize=7.5, xytext=(6, -3),
                textcoords="offset points")
ax.set_xlabel("log₁₀ a")
ax.set_ylabel("log₁₀ T")
ax.set_title("15 May 1618: take the logarithm.\nA power law is a straight line — "
             "and only a power law is.")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)

# Panel 3: two systems, same slope, different constant = different mass
ax = fig.add_subplot(gs[0, 2])
ax.plot(loga, logT, "o-", color="crimson", ms=8, lw=1.4, label="planets → the Sun")
ax.plot(torch.log10(am), torch.log10(Tm), "s-", color="steelblue", ms=8, lw=1.4,
        label="Galilean moons → Jupiter")
for i, (nm, _, _) in enumerate(moons):
    ax.annotate(nm, (torch.log10(am[i]), torch.log10(Tm[i])), fontsize=7.5,
                xytext=(6, -3), textcoords="offset points", color="steelblue")
ax.set_xlabel("log₁₀ a (AU)")
ax.set_ylabel("log₁₀ T (yr)")
ax.set_title(f"Same slope 3/2, different intercept.\n"
             f"The gap IS the mass ratio: M_sun/M_jup ≈ {M_sun/M_jup:.0f}.\n"
             "Kepler's law became a weighing scale.")
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3)

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
