"""Halley, 1705: the first time anyone told the future and was believed.

Comets were portents. They appeared without warning, hung in the sky for weeks, and
left. Nothing about them was periodic, and nothing about them was predictable —
that was practically their definition.

Halley, using Newton's brand-new theory and his own laborious reduction of old
observations, computed orbits for 24 comets. Three of them — 1531, 1607 and 1682 —
came out with almost identical elements. He drew the conclusion nobody had dared:

    "I may venture to foretell, that it will return again in the year 1758."

He was 49 and knew he would not live to see it (he died in 1742, aged 85). This was
the first quantitative prediction of a specific future astronomical event that was
not simply the continuation of an already-known cycle, and it is the moment
astronomy became genuinely predictive rather than merely descriptive.

It very nearly failed. The intervals between returns are NOT constant — they range
from 74.4 to 76.7 years — because Jupiter and Saturn tug the comet on every pass.
In 1757 Clairaut, with Lalande and Nicole-Reine Lepaute, spent six months computing
those perturbations by hand, working in shifts, "sometimes not even stopping to
eat." They announced perihelion for mid-April 1759, ±1 month. It arrived 13 March
1759 — a 33-day error on a 76-year prediction.

    phenomenon:   a bright comet returns at irregular intervals near 76 years
    simulation:   the real perihelion dates, 1P/Halley's real elements
    dissection:   the two-body period from Kepler III, then why it is wrong;
                  the sensitivity dT/da that makes tiny tugs matter so much
    formula:      T = 2π√(a³/GM), so δT/T = (3/2)·δa/a — a 0.1% nudge to the
                  semi-major axis moves the return by a month.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

MU_YR = 4 * np.pi**2                # AU³/yr²

# --- 1P/Halley's osculating elements (J2000-ish, epoch 1994) ---
A_H, E_H = 17.834, 0.96714
Q_H, Q_APH = A_H * (1 - E_H), A_H * (1 + E_H)
I_H = 162.26                        # degrees — retrograde!

# --- the observed perihelion passages ---
# datetime.date cannot represent BC years, so dates are carried as decimal years.
_CUM = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]


def dec_year(y, m, d):
    """Calendar date → decimal year. Works for negative (BC) years too."""
    return y + (_CUM[m - 1] + d - 1) / 365.25


PERIHELIA = [
    (dec_year(-239, 5, 25), "240 BC     ", "first certain record, Chinese chronicles"),
    (dec_year(1066, 3, 20), "1066 Mar 20", "before Hastings; woven into the Bayeux Tapestry"),
    (dec_year(1301, 10, 25), "1301 Oct 25", "Giotto paints it as the Star of Bethlehem"),
    (dec_year(1531, 8, 26), "1531 Aug 26", "observed by Apian"),
    (dec_year(1607, 10, 27), "1607 Oct 27", "observed by Kepler"),
    (dec_year(1682, 9, 15), "1682 Sep 15", "observed by HALLEY himself"),
    (dec_year(1759, 3, 13), "1759 Mar 13", "the predicted return — Clairaut said mid-April"),
    (dec_year(1835, 11, 16), "1835 Nov 16", ""),
    (dec_year(1910, 4, 20), "1910 Apr 20", "Earth passed through the tail; cyanogen panic"),
    (dec_year(1986, 2, 9), "1986 Feb 09", "Giotto probe flew 596 km from the nucleus"),
    (dec_year(2061, 7, 28), "2061 Jul 28", "next return — predicted"),
]

print("1P/Halley: the orbit Halley computed from three apparitions")
print(f"  semi-major axis a = {A_H:.3f} AU        (out past Neptune at aphelion)")
print(f"  eccentricity   e = {E_H:.5f}")
print(f"  perihelion     q = a(1−e) = {Q_H:.3f} AU   (inside Venus)")
print(f"  aphelion       Q = a(1+e) = {Q_APH:.2f} AU  (beyond Neptune, 30.1 AU)")
print(f"  inclination    i = {I_H:.2f}°  — retrograde, it orbits the wrong way\n")

# --- Kepler III gives the two-body period ---
T_kepler = A_H ** 1.5
print(f"Kepler III:  T = a^(3/2) = {A_H:.3f}^1.5 = {T_kepler:.3f} years")
print(f"  (equivalently T = 2π√(a³/GM) with GM = 4π² in AU³/yr²)\n")

# --- but the observed intervals are not constant ---
modern = [p for p in PERIHELIA if p[0] >= 1531]
print("The observed returns — and the problem:")
print("  perihelion date     interval from previous     note")
intervals = []
for k, (yv, label, note) in enumerate(modern):
    if k == 0:
        print(f"  {label:12s}          —              {note}")
        continue
    dt_yr = yv - modern[k - 1][0]
    intervals.append(dt_yr)
    print(f"  {label:12s}      {dt_yr:7.2f} yr          {note}")

iv = np.array(intervals)
print(f"\n  mean interval   = {iv.mean():.2f} yr")
print(f"  spread          = {iv.min():.2f} to {iv.max():.2f} yr  "
      f"— a range of {iv.max()-iv.min():.2f} years!")
print(f"  two-body Kepler = {T_kepler:.2f} yr")
print()
print(f"  Kepler III is off from the observed mean by {abs(iv.mean()-T_kepler)*365.25:.0f} days, and no single")
print(f"  fixed period works at all: predicting from the mean interval would still be")
print(f"  wrong by up to {max(abs(iv-iv.mean()))*365.25:.0f} days. Halley knew this. He explicitly warned that")
print("  Jupiter would delay the return, and left the exact date 'to posterity'.\n")

# --- why: the period is brutally sensitive to the semi-major axis ---
a_t = torch.tensor(A_H, dtype=torch.float64, requires_grad=True)
T_t = a_t ** 1.5
T_t.backward()
dT_da = float(a_t.grad)
print("WHY SO SENSITIVE — differentiate Kepler III (autograd):")
print(f"  dT/da = (3/2)·a^(1/2) = {dT_da:.4f} yr per AU")
print(f"  δT/T = (3/2)·δa/a — a relative nudge to a is amplified 1.5× into the period")
print()
print("   change in a       change in a      resulting shift in return date")
for frac in (1e-4, 1e-3, 3e-3, 1e-2):
    da = frac * A_H
    dT = dT_da * da
    print(f"   {frac*100:6.2f}%          {da:8.5f} AU        {dT*365.25:8.1f} days "
          f"({dT:.2f} yr)")
print()
print("  Jupiter's mass is 1/1047 of the Sun's. Halley's comet crosses Jupiter's")
print("  orbit twice per revolution and spends decades out beyond Neptune moving")
print("  slowly, where even a weak tug has a long time to act. Changing the comet's")
print("  orbital ENERGY by a fraction of a per cent is easy; and energy fixes a,")
print("  and a fixes the period, with a 3/2 lever on top.\n")

# --- what does the observed scatter imply about the energy kicks? ---
dT_obs = (iv - iv.mean())
da_implied = dT_obs / dT_da
print("Inverting the observed scatter: what Δa does each return imply?")
print("   interval (yr)    ΔT from mean (d)     implied Δa (AU)     Δa/a")
for k, (dt_yr, dd) in enumerate(zip(iv, dT_obs)):
    print(f"   {dt_yr:9.2f}      {dd*365.25:+12.1f}       {dd/dT_da:+12.5f}   "
          f"{dd/dT_da/A_H:+.2e}")
print(f"\n  So the planets are shifting Halley's semi-major axis by a few times 10⁻³ AU")
print(f"  per orbit — a few hundred thousand kilometres out of 2.7 billion. That is")
print(f"  what Clairaut had to compute, by hand, in 1757, to beat a 2.5-year window")
print(f"  down to one month.\n")

# --- Clairaut's result, scored ---
predicted = dec_year(1759, 4, 15)
actual = dec_year(1759, 3, 13)
err_days = round((predicted - actual) * 365.25)
print("CLAIRAUT'S PREDICTION, SCORED")
print(f"  announced to the Académie, 14 November 1758: perihelion 1759 Apr 15")
print(f"  stated uncertainty: about one month")
print(f"  actual perihelion:  1759 Mar 13")
print(f"  error: {err_days} days out of a {iv[2]:.1f}-year prediction "
      f"= {abs(err_days)/(iv[2]*365.25)*100:.2f}%")
print()
print("  Clairaut computed that Jupiter would delay the return by 518 days and")
print("  Saturn by a further 100 — 618 days of correction, without which the")
print("  prediction would have been badly wrong. Nicole-Reine Lepaute did a large")
print("  share of the arithmetic; Lalande said so publicly, Clairaut omitted her")
print("  from his published account.")
print()
print("  The comet was actually first spotted on Christmas night 1758 by Johann")
print("  Palitzsch, a Saxon farmer with a home-made telescope — one more amateur")
print("  optician (→ ch00_5) at a decisive moment.\n")

print("THE GENERAL RECIPE — how to do this for any comet:")
print("  1. get ≥3 observations, solve for a preliminary orbit   → gauss_three_observations.py")
print("  2. refine on all observations by least squares          → differential_correction.py")
print("  3. read off a and e; if e < 1 the orbit is closed and   → T = 2π√(a³/GM)")
print("  4. add the planetary perturbations, which shift a by ~10⁻³ and the return")
print("     date by months (Clairaut by hand; today by numerical integration)")
print("  5. for non-gravitational forces — outgassing jets on an active comet —")
print("     add an empirical acceleration; Halley's own return drifts by ~4 days")
print("     per orbit from this alone")
print("  → comet_orbit_zoo.py applies steps 3 to a dozen real comets.")

# --- visualization ---
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])

# Panel 1: the orbit, to scale
ax = fig.add_subplot(gs[0, 0])
nu = np.linspace(0, 2 * np.pi, 600)
r_h = A_H * (1 - E_H**2) / (1 + E_H * np.cos(nu))
ax.plot(r_h * np.cos(nu), r_h * np.sin(nu), color="crimson", lw=1.8,
        label="1P/Halley")
for a_p, nm, col in [(1.0, "Earth", "steelblue"), (5.20, "Jupiter", "darkorange"),
                     (9.54, "Saturn", "goldenrod"), (30.07, "Neptune", "navy")]:
    ax.plot(a_p * np.cos(nu), a_p * np.sin(nu), color=col, lw=1.0, alpha=0.8)
    ax.text(0, a_p, nm, fontsize=7, ha="center", color=col)
ax.plot(0, 0, "*", color="orange", ms=18)
ax.set_aspect("equal")
ax.set_xlabel("AU")
ax.set_title(f"e = {E_H:.3f}. Perihelion inside Venus,\naphelion past Neptune. It "
             "crosses every\ngiant planet's orbit, twice per revolution.")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel 2: the intervals
ax = fig.add_subplot(gs[0, 1:])
yrs = [int(m[0]) for m in modern[1:]]
ax.bar([str(y) for y in yrs], iv, color="crimson", alpha=0.85)
ax.axhline(T_kepler, color="steelblue", lw=2.4,
           label=f"two-body Kepler III = {T_kepler:.2f} yr")
ax.axhline(iv.mean(), color="black", ls="--", lw=2,
           label=f"observed mean = {iv.mean():.2f} yr")
ax.set_ylim(72, 80)
ax.set_ylabel("interval since previous perihelion (yr)")
ax.set_title(f"The returns are NOT evenly spaced. Two-body Kepler misses the mean by "
             f"{abs(iv.mean()-T_kepler)*365.25:.0f} days,\nand the returns themselves scatter over "
             f"{iv.max()-iv.min():.2f} years — all of it Jupiter and Saturn.")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")

# Panel 3: the sensitivity
ax = fig.add_subplot(gs[1, 0])
frac = np.logspace(-5, -1.5, 100)
ax.loglog(frac * 100, dT_da * frac * A_H * 365.25, color="crimson", lw=2.4)
ax.axhline(30, color="darkgreen", ls="--", lw=1.8, label="one month")
ax.axhline(365, color="steelblue", ls=":", lw=1.8, label="one year")
for dd in np.abs(dT_obs):
    ax.plot(abs(dd / dT_da / A_H) * 100, dd * 365.25, "o", color="black", ms=7)
ax.set_xlabel("relative change in a (%)")
ax.set_ylabel("shift in return date (days)")
ax.set_title("δT/T = (3/2)·δa/a.\nBlack dots are the actual observed returns.\n"
             "A 0.01% nudge moves the date by weeks.")
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3, which="both")

# Panel 4: 2000 years of returns
ax = fig.add_subplot(gs[1, 1])
allyrs = [p[0] for p in PERIHELIA]
ax.plot(allyrs, [1] * len(allyrs), "|", color="crimson", ms=26, mew=2.5)
for p in PERIHELIA:
    if p[2]:
        ax.annotate(p[1].strip(), (p[0], 1), fontsize=6.5, rotation=90,
                    xytext=(0, 12), textcoords="offset points", ha="center")
ax.axvline(1705, color="steelblue", ls="--", lw=2)
ax.text(1705, 0.86, "Halley predicts\n(1705)", fontsize=8, ha="right", color="steelblue")
ax.set_xlim(-350, 2150)
ax.set_ylim(0.8, 1.25)
ax.set_yticks([])
ax.set_xlabel("year")
ax.set_title("Recorded since 240 BC, recognised as ONE\nobject only in 1705. The "
             "data had been\nsitting in the chronicles for 1900 years.")

# Panel 5: Clairaut scored
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
ax.text(0.0, 1.0, "The first prediction, scored", fontsize=11.5, weight="bold", va="top")
ax.text(0.0, 0.86, (
    "1705  Halley: 'it will return in 1758'\n"
    "        from three apparitions and\n"
    "        Newton's brand-new theory\n\n"
    "1757  Clairaut, Lalande and Lepaute\n"
    "        compute Jupiter + Saturn by\n"
    "        hand for six months:\n"
    "          Jupiter delays it  +518 d\n"
    "          Saturn  delays it  +100 d\n"
    "          ─────────────────────────\n"
    "          total correction   +618 d\n\n"
    "        prediction: 15 April 1759\n"
    "        stated error: ± 1 month\n\n"
    "1758  25 Dec: found by Palitzsch,\n"
    "        a farmer with a home-made\n"
    "        telescope\n\n"
    "1759  13 March: perihelion\n"
    f"        error = {abs(err_days)} days\n"
    f"              = {abs(err_days)/(iv[2]*365.25)*100:.2f}% of the interval\n\n"
    "Comets stopped being portents and\n"
    "became clocks."),
    fontsize=8.3, va="top", family="monospace")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
