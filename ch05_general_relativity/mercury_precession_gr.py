"""November 1915: Einstein computes 43″ and gets heart palpitations for three days.

General relativity does not add a force. It changes the geometry the planet moves
through, and the orbit equation picks up exactly one extra term:

    Newton:      d²u/dφ² + u = GM/h²
    Einstein:    d²u/dφ² + u = GM/h² + 3GM·u²/c²        where u = 1/r

That 3GM u²/c² is the whole of it. It is a tiny correction — for Mercury it is
~1e-7 of the main term — but it is NOT of the form 1/r², so by Bertrand's theorem
(ch03) it must make the orbit precess. Integrate it and out comes 43″/century.

Einstein had no free parameters. He could not tune anything. The number 43 had been
sitting unexplained in the literature since Le Verrier in 1859, and the theory
either produced it or it didn't. On 18 November 1915 he presented the result to the
Prussian Academy; the final field equations came a week later.

    phenomenon:   Mercury's perihelion advances 43″/century more than Newton allows
    simulation:   integrate the relativistic orbit equation in u = 1/r vs φ
    dissection:   measure the angle between successive perihelia
    formula:      Δφ = 6πGM / (c²·a·(1−e²)) per orbit — derive it, then check the
                  integration against it, then convert to arcsec/century.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# --- SI constants and Mercury's orbit ---
G = 6.67430e-11
C = 2.99792458e8
M_SUN = 1.98892e30
GM = G * M_SUN
A = 5.790905e10           # m, Mercury's semi-major axis
E = 0.205630
T_ORB = 87.9691 * 86400   # s
ORBITS_PER_CENTURY = 100 * 365.25 * 86400 / T_ORB
RAD2ARCSEC = 180 / np.pi * 3600

h = np.sqrt(GM * A * (1 - E**2))            # specific angular momentum
eps = 3 * GM**2 / (h**2 * C**2)             # the dimensionless size of the GR term

print("The relativistic correction, sized up before we integrate anything:")
print(f"  h = sqrt(GM·a(1−e²)) = {h:.4e} m²/s")
print(f"  GR term / Newton term  =  3(GM/hc)² = {eps:.3e}")
print(f"  → a correction in the 8th significant figure. Nothing on Earth could")
print(f"    measure that in 1915. Mercury, integrated over a century, could.\n")


def integrate_orbit(gr_factor=1.0, n_rev=3, n_pts=400_000):
    """Integrate  u'' + u = GM/h² + gr_factor·3GM·u²/c²  in φ, by velocity-Verlet.

    Working in u = 1/r versus φ (rather than r versus t) is the classical trick:
    it turns the orbit into a driven harmonic oscillator, and the GR term becomes
    a small nonlinear stiffening that shifts the oscillator's frequency slightly
    below 1 — which is precisely a slow rotation of the perihelion.
    """
    phi_end = 2 * np.pi * n_rev
    dphi = phi_end / n_pts
    u = torch.tensor(1.0 / (A * (1 - E)), dtype=torch.float64)      # start at perihelion
    du = torch.tensor(0.0, dtype=torch.float64)                     # u' = 0 at perihelion

    def acc(u_):
        return GM / h**2 + gr_factor * 3 * GM * u_**2 / C**2 - u_

    us = torch.empty(n_pts + 1, dtype=torch.float64)
    us[0] = u
    a = acc(u)
    for i in range(n_pts):
        du = du + 0.5 * dphi * a
        u = u + dphi * du
        a = acc(u)
        du = du + 0.5 * dphi * a
        us[i + 1] = u
    return us, torch.linspace(0, phi_end, n_pts + 1, dtype=torch.float64)


def perihelion_shift(us, phis):
    """Angle between successive perihelia (maxima of u = 1/r), minus 2π."""
    interior = us[1:-1]
    peaks = torch.where((interior > us[:-2]) & (interior > us[2:]))[0] + 1
    # refine each peak with a parabolic fit through its three samples
    refined = []
    for p in peaks.tolist():
        y0, y1, y2 = us[p - 1].item(), us[p].item(), us[p + 1].item()
        delta = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
        refined.append(phis[p].item() + delta * (phis[1] - phis[0]).item())
    return np.array(refined)


# --- Newton first: does the orbit close? ---
us_n, phis = integrate_orbit(gr_factor=0.0)
per_n = perihelion_shift(us_n, phis)
print(f"NEWTON  (gr term switched off)")
print(f"  perihelion spacing = {np.mean(np.diff(per_n)):.10f} rad")
print(f"  minus 2π           = {np.mean(np.diff(per_n)) - 2*np.pi:+.3e} rad "
      f"({(np.mean(np.diff(per_n))-2*np.pi)*RAD2ARCSEC*ORBITS_PER_CENTURY:+.3f}″/century)")
print(f"  → closes, as it must. This is our numerical noise floor.\n")

# --- now Einstein ---
us_e, _ = integrate_orbit(gr_factor=1.0)
per_e = perihelion_shift(us_e, phis)
shift_sim = np.mean(np.diff(per_e)) - 2 * np.pi
noise = np.mean(np.diff(per_n)) - 2 * np.pi
shift_sim_corrected = shift_sim - noise

# --- the closed form Einstein derived ---
shift_theory = 6 * np.pi * GM / (C**2 * A * (1 - E**2))

print("EINSTEIN  (gr term on)")
print(f"  simulated shift per orbit = {shift_sim_corrected:.6e} rad")
print(f"  closed form 6πGM/(c²a(1−e²)) = {shift_theory:.6e} rad")
print(f"  agreement: {100*shift_sim_corrected/shift_theory:.3f}%\n")

arcsec_century = shift_theory * RAD2ARCSEC * ORBITS_PER_CENTURY
sim_century = shift_sim_corrected * RAD2ARCSEC * ORBITS_PER_CENTURY
print(f"  Mercury completes {ORBITS_PER_CENTURY:.1f} orbits per century, so:")
print(f"    from the simulation   {sim_century:7.2f}″/century")
print(f"    from the formula      {arcsec_century:7.2f}″/century")
print(f"    OBSERVED (ch04)         43.11 ± 0.45 ″/century")
print()

# --- and the other planets, as a consistency check ---
print("The same formula, applied to everything else (it is not tuned to Mercury):")
print("  body        a (AU)     e       GR precession    measured")
print("                                 (″/century)      (″/century)")
bodies = [("Mercury", 0.387098, 0.20563, 87.9691, "43.11 ± 0.45"),
          ("Venus",   0.723332, 0.00677, 224.701, " 8.62 ± 0.10"),
          ("Earth",   1.000000, 0.01671, 365.256, " 3.84 ± 0.04"),
          ("Mars",    1.523710, 0.09341, 686.980, " 1.35 ± 0.05"),
          ("Icarus",  1.077900, 0.82680, 408.780, "10.05 ± 1.0 ")]
AU = 1.495978707e11
gr_vals = []
for nm, a_au, e_b, per_d, meas in bodies:
    a_m = a_au * AU
    sh = 6 * np.pi * GM / (C**2 * a_m * (1 - e_b**2))
    n_orb = 100 * 365.25 / per_d
    val = sh * RAD2ARCSEC * n_orb
    gr_vals.append(val)
    print(f"  {nm:9s} {a_au:8.4f}  {e_b:.5f}   {val:10.2f}      {meas}")
print()
print("One formula, no adjustable parameters, five bodies. Note Icarus: a tiny")
print("asteroid with a wild e = 0.83, measured in 1968 — the theory had no idea it")
print("existed when it was written.")
print()
print("Einstein to Ehrenfest, January 1916:")
print('  "For a few days I was beside myself with joyous excitement."')

# --- visualization ---
fig = plt.figure(figsize=(15, 8.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])

# Panel 1: the rosette, exaggerated so it is visible
ax = fig.add_subplot(gs[0, 0])
us_big, phis_big = integrate_orbit(gr_factor=4e5, n_rev=6, n_pts=200_000)
r_big = 1.0 / us_big
ax.plot(r_big * torch.cos(phis_big) / AU, r_big * torch.sin(phis_big) / AU,
        color="crimson", lw=0.8)
ax.plot(0, 0, "*", color="orange", ms=18)
ax.set_aspect("equal")
ax.set_xlabel("AU")
ax.set_title("The GR term, amplified 400 000×.\nThe ellipse does not close — it "
             "rotates.\nThat is all general relativity does here.")
ax.grid(alpha=0.3)

# Panel 2: u(φ) — Newton vs Einstein, phase slipping
ax = fig.add_subplot(gs[0, 1])
ax.plot(phis / (2 * np.pi), us_n * A, color="steelblue", lw=1.8, label="Newton")
ax.plot(phis / (2 * np.pi), us_e * A, color="crimson", lw=1.2, ls="--", label="Einstein")
for p in per_n:
    ax.axvline(p / (2 * np.pi), color="steelblue", lw=0.6, alpha=0.5)
ax.set_xlabel("φ / 2π  (orbits)")
ax.set_ylabel("u·a = a/r")
ax.set_title("u = 1/r is a harmonic oscillator in φ.\nGR lowers its frequency very "
             "slightly,\nso perihelion arrives a little LATE each turn.")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel 3: perihelion angle drifting linearly
ax = fig.add_subplot(gs[0, 2])
k = np.arange(len(per_e))
ax.plot(k, (per_e - per_e[0] - 2 * np.pi * k) * RAD2ARCSEC, "o-",
        color="crimson", lw=2, ms=7, label="simulation")
ax.plot(k, shift_theory * k * RAD2ARCSEC, "--", color="black", lw=1.6,
        label="6πGM/(c²a(1−e²)) · n")
ax.set_xlabel("orbit number")
ax.set_ylabel("cumulative perihelion advance (arcsec)")
ax.set_title(f"Per orbit: {shift_theory*RAD2ARCSEC:.4f}″.\n"
             f"× {ORBITS_PER_CENTURY:.0f} orbits/century = "
             f"{arcsec_century:.1f}″/century.")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel 4: the budget, closed at last
ax = fig.add_subplot(gs[1, :2])
labels = ["observed\nexcess", "Venus", "Earth", "Mars", "Jupiter", "Saturn",
          "GENERAL\nRELATIVITY", "residual"]
contrib = [574.10, -277.9, -90.0, -2.5, -153.6, -7.3, -43.03, 0]
running = 574.10
bottoms, heights, colors = [0], [574.10], ["black"]
for v in contrib[1:-1]:
    running += v
    bottoms.append(running)
    heights.append(-v)
    colors.append("crimson" if abs(v) == 43.03 else "steelblue")
bottoms.append(0); heights.append(running); colors.append("darkgreen")
ax.bar(labels, heights, bottom=bottoms, color=colors, alpha=0.9)
for i, (b, hh) in enumerate(zip(bottoms, heights)):
    ax.text(i, b + hh + 12, f"{hh:.1f}″", ha="center", fontsize=9,
            weight="bold" if i in (0, 6, 7) else "normal")
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("arcsec / century")
ax.set_title("The ledger that stayed open for 56 years, closed in one week of "
             "November 1915.\nThe red bar was not fitted. It was computed from a "
             "theory built for entirely different reasons.")
ax.tick_params(axis="x", rotation=15)
ax.grid(alpha=0.3, axis="y")

# Panel 5: GR precession across the solar system
ax = fig.add_subplot(gs[1, 2])
nms = [b[0] for b in bodies]
ax.bar(nms, gr_vals, color=["crimson" if n == "Mercury" else "steelblue" for n in nms],
       alpha=0.9)
ax.set_ylabel("GR precession (″/century)")
ax.set_yscale("log")
ax.set_title("Why Mercury and not Jupiter:\nthe effect scales as 1/(a(1−e²)) and\n"
             "you get more orbits per century when\nyou are close in. Mercury wins twice.")
ax.tick_params(axis="x", rotation=25)
ax.grid(alpha=0.3, axis="y", which="both")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
