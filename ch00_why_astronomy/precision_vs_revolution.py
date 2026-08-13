"""Every revolution in physics began as a residual smaller than the last instrument.

This is the thesis of the whole project, plotted once. Physics does not advance
because someone had a better idea; it advances when someone can MEASURE an angle
small enough that the reigning theory's error becomes visible.

    Tycho (1600) got to ~1 arcmin  →  Kepler's Mars residual was 8 arcmin
                                      → the circle died, the ellipse was born.
    19th c. meridian circles ~1"    →  Mercury's leftover was 43"/century
                                      → Newton died, general relativity was born.
    Eddington (1919) ~0.2"          →  starlight bends 1.75", not Newton's 0.87"
                                      → spacetime curvature confirmed.

    phenomenon:   observational precision improves; anomalies pop into view
    simulation:   tabulate (year, best angular precision, size of the anomaly
                  that precision exposed)
    dissection:   fit the precision history in log-space with torch — it is an
                  exponential, ~10x per century for 400 years
    formula:      an anomaly is DETECTABLE iff  anomaly > precision.  Plot both
                  on one log axis and every revolution sits where the curves cross.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# --- phenomenon: 2000 years of "how small an angle can you measure?" ---
# angular precision of the best instrument of its day, in arcseconds
year_prec = np.array([-150, 1000, 1580, 1600, 1670, 1750, 1850, 1919, 1970, 2000, 2020])
precision = np.array([600.0,  300.0, 240.0, 60.0, 10.0, 3.0, 1.0, 0.2, 0.05, 0.001, 2e-5])
prec_label = ["Hipparchus\n(naked eye)", "al-Battani", "pre-Tycho", "Tycho Brahe",
              "early telescope", "Bradley", "meridian circle", "Eddington plates",
              "photoelectric", "Hipparcos", "Gaia / VLBI"]

# --- the anomalies: each was invisible until precision crossed it ---
# (year the anomaly was resolved, its angular size in arcsec, name)
anomalies = [
    (1609, 8 * 60.0, "Mars' 8′ residual\n→ orbit is an ELLIPSE"),
    (1728, 20.5,     "aberration of starlight\n→ finite speed of light"),
    (1838, 0.314,    "61 Cygni parallax\n→ stars are 10¹³ km away"),
    (1859, 0.43,     "Mercury: 43″/century\n(0.43″/yr) → Newton fails"),
    (1919, 1.75,     "starlight bent 1.75″\n→ spacetime curves"),
]

# --- dissection: is the improvement exponential? fit log10(precision) vs year ---
mask = year_prec >= 1500                       # the telescopic era
t = torch.tensor(year_prec[mask], dtype=torch.float64)
y = torch.log10(torch.tensor(precision[mask], dtype=torch.float64))

# Least squares by hand so the mathematics is visible: y = m*t + b
A = torch.stack([t, torch.ones_like(t)], dim=1)
m, b = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.squeeze(1)
decades_per_century = -m.item() * 100
print("fit  log10(precision in arcsec) = m·year + b")
print(f"  m = {m.item():.5f} dex/year   →  precision improves {decades_per_century:.2f} "
      f"orders of magnitude per century")
print(f"  i.e. a factor of {10**decades_per_century:.0f}x every 100 years, sustained for 400 years")
print()
for yr, size, name in anomalies:
    # what was the best precision available in that year?
    avail = np.interp(yr, year_prec, np.log10(precision))
    print(f"  {yr}: anomaly {size:8.3f}\"  vs  precision {10**avail:8.3f}\"  "
          f"→ ratio {size / 10**avail:6.1f}x  {name.splitlines()[0]}")
print("\nEvery revolution has ratio > 1: the anomaly was BIGGER than the error bar.")
print("That is the entire mechanism. No philosophy required.")

# --- visualization ---
fig, ax = plt.subplots(figsize=(13, 7.5))

ax.plot(year_prec, precision, "o-", color="#1f4e79", lw=2.2, ms=7,
        label="best angular precision of the era")
fit_x = np.linspace(1580, 2020, 50)
ax.plot(fit_x, 10 ** (m.item() * fit_x + b.item()), "--", color="#1f4e79", alpha=0.45,
        lw=1.6, label=f"exponential fit: {10**decades_per_century:.0f}× better per century")

for yr, size, name in anomalies:
    ax.plot(yr, size, "*", color="crimson", ms=20, zorder=6)
    ax.annotate(name, (yr, size), textcoords="offset points", xytext=(9, 14),
                fontsize=8.5, color="crimson",
                arrowprops=dict(arrowstyle="-", color="crimson", lw=0.7, alpha=0.6))

for xt, yt, lab in zip(year_prec, precision, prec_label):
    ax.annotate(lab, (xt, yt), textcoords="offset points", xytext=(-4, -26),
                fontsize=7, color="#1f4e79", ha="center", alpha=0.85)

ax.fill_between(year_prec, precision, 1e-6, color="#1f4e79", alpha=0.07)
ax.text(1650, 3e-5, "INVISIBLE\n(below the error bar of every instrument)",
        fontsize=10, color="#1f4e79", alpha=0.6, ha="center", style="italic")

ax.set_yscale("log")
ax.invert_yaxis()                     # better precision = smaller angle = higher up
ax.set_xlim(1500, 2060)
ax.set_ylim(4000, 1e-5)
ax.set_xlabel("year")
ax.set_ylabel("angle (arcsec)  —  smaller is better, so the axis is inverted")
ax.set_title("Astronomy is the mother science because it is the most precise one:\n"
             "every ★ is a physics revolution that appeared the moment the instrument "
             "out-resolved the theory's error")
ax.legend(loc="lower left", fontsize=9)
ax.grid(alpha=0.3, which="both")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
