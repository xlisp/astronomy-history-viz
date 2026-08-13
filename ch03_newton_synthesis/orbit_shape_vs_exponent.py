"""Why orbits close: 1/r² is not one law among many, it is one of only two.

Bertrand's theorem (1873): among all central forces F ∝ r^n, ONLY two produce
orbits that close on themselves for every initial condition —

    n = −2   (gravity / Coulomb)      and     n = +1   (the harmonic spring)

Every other exponent gives a rosette that precesses forever and never repeats.
This is why the planets look like tidy stationary ellipses on human timescales,
which is why Kepler could find his laws at all. Had gravity gone as 1/r^2.1, the
solar system would look like a spirograph and there would have been no ellipse to
discover — and quite possibly no Newton.

The theorem also sets up everything in ch04–ch05. A precessing orbit is the
FINGERPRINT of a deviation from 1/r². So when Mercury's perihelion was found to
creep forward by 43″ per century, it was a direct measurement that the true force
is not exactly inverse-square. Einstein's correction adds precisely a small 1/r⁴
term (see ch05).

    phenomenon:   planetary ellipses appear frozen; some orbits do precess
    simulation:   integrate the same initial condition under F ∝ r^n for many n
    dissection:   measure the apsidal angle — the angle between successive perihelia
    formula:      for a near-circular orbit,  Δφ = 2π/√(3+n).  n = −2 gives 2π
                  (closed) and n = +1 gives π (closed). Everything else is irrational.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

GM = 1.0


def integrate_all(n_exps, t_end=60.0, n_steps=120000, e0=0.45):
    """Velocity-Verlet under central forces a = −GM·r^n·r̂, all exponents at once.

    Batching the exponents into one tensor turns six slow Python loops into one.
    Verlet (symplectic) is essential here: a non-symplectic integrator leaks energy
    and fakes a precession, which is precisely the signal we are trying to measure.
    """
    n_exp = torch.tensor(n_exps, dtype=torch.float64).unsqueeze(1)     # (B, 1)
    B = len(n_exps)
    dt = t_end / n_steps
    pos = torch.zeros(B, 2, dtype=torch.float64)
    vel = torch.zeros(B, 2, dtype=torch.float64)
    pos[:, 0] = 1.0 - e0
    vel[:, 1] = np.sqrt(GM * (1 + e0) / (1 - e0))     # perihelion speed for that e

    def accel(p):
        r = torch.linalg.norm(p, dim=1, keepdim=True)
        return -GM * (p / r) * r ** n_exp

    traj = torch.empty(n_steps + 1, B, 2, dtype=torch.float64)
    traj[0] = pos
    a = accel(pos)
    for i in range(n_steps):
        vel = vel + 0.5 * dt * a
        pos = pos + dt * vel
        a = accel(pos)
        vel = vel + 0.5 * dt * a
        traj[i + 1] = pos
    return traj


def apsidal_angle(traj):
    """Angle swept between two successive perihelion passages."""
    r = torch.linalg.norm(traj, dim=1)
    th = torch.from_numpy(np.unwrap(torch.atan2(traj[:, 1], traj[:, 0]).numpy()))
    # local minima of r
    mins = torch.where((r[1:-1] < r[:-2]) & (r[1:-1] < r[2:]))[0] + 1
    if len(mins) < 2:
        return float("nan"), 0
    return float(np.mean(np.diff(th[mins].numpy()))), len(mins)


exponents = [-2.0, -1.5, -1.0, 1.0, -2.5, -3.0]
all_traj = integrate_all(exponents)
print("force law  F ∝ r^n      apsidal angle Δφ      Δφ/2π        verdict")
print("                        measured   theory*")
results = {}
for bi, n_exp in enumerate(exponents):
    tj = all_traj[:, bi, :]
    dphi, n_peri = apsidal_angle(tj)
    theory = 2 * np.pi / np.sqrt(3 + n_exp) if (3 + n_exp) > 0 else float("nan")
    frac = dphi / (2 * np.pi)
    if np.isnan(frac):
        # Fewer than two perihelia: for n <= -3 the effective potential has no
        # minimum, so there is no bound orbit at all — the body spirals in.
        status, closed = "UNSTABLE — spirals into the Sun", False
    elif abs(frac - round(frac)) < 1e-3 or abs(frac * 2 - round(frac * 2)) < 1e-3:
        status, closed = "CLOSED", True
    else:
        status, closed = "precesses forever", False
    results[n_exp] = (tj, dphi, frac, closed)
    name = {-2.0: " ← GRAVITY", 1.0: " ← spring (Hooke)"}.get(n_exp, "")
    print(f"   n = {n_exp:+.1f}          {dphi:7.4f}   {theory:7.4f}     "
          f"{frac:7.4f}   {status}{name}")

print()
print("* theory is the near-circular limit Δφ = 2π/√(3+n); we launched at e = 0.45,")
print("  which is why the precessing cases differ by a few per cent. The two CLOSED")
print("  cases match exactly — and that exactness is the content of the theorem.")
print()
print("Only n = −2 and n = +1 close. This is Bertrand's theorem (1873), and it is")
print("the reason 'the orbit of Mars is an ellipse' was a discoverable fact.")
print("For n ≤ −3 there are no bound orbits at all: Ehrenfest's 1917 argument that")
print("stable planetary systems (and stable atoms) exist only in 3 spatial dimensions.")
print()

# --- how much precession does a tiny deviation cause? ---
print("Now the question that matters for ch04–ch05:")
print("how far from −2 can the exponent be before Tycho would have noticed?\n")
RAD2ARCSEC = 180 / np.pi * 3600
ORBITS_PER_CENTURY = 415.2                 # Mercury: 100 yr / 87.969 d

# A force slightly STRONGER than 1/r² (exponent below −2) makes the perihelion
# advance in the direction of motion — which is the sign Mercury actually shows.
print("   n              precession per orbit     per century for Mercury")
print("                  (arcsec)                 (arcsec)  ← 415.2 orbits/century")
for dn in (0.0, 1e-7, 1e-6, 1e-5, 1e-4):
    n_exp = -2.0 - dn
    # near-circular apsidal angle: Δφ = 2π/sqrt(3+n); precession = Δφ − 2π
    prec_arc = (2 * np.pi / np.sqrt(3 + n_exp) - 2 * np.pi) * RAD2ARCSEC
    print(f"  −2 − {dn:.0e}     {prec_arc:12.5f}         {prec_arc*ORBITS_PER_CENTURY:14.2f}")

# Invert it: what exponent would reproduce the observed 43"/century?
prec_per_orbit = 43.0 / ORBITS_PER_CENTURY / RAD2ARCSEC        # radians
delta = prec_per_orbit / np.pi                                  # from Δφ ≈ 2π(1 + δ/2)
print()
print(f"Mercury's observed anomaly is 43″/century → {43/ORBITS_PER_CENTURY:.4f}″/orbit")
print(f"→ an effective exponent of  n = −2 − {delta:.2e}")
check = (2 * np.pi / np.sqrt(3 + (-2 - delta)) - 2 * np.pi) * RAD2ARCSEC * ORBITS_PER_CENTURY
print(f"   (check: that exponent gives back {check:.2f}″/century ✓)")
print("A deviation in the 7th decimal place of the exponent. Astronomy caught it.")
print("No laboratory experiment on Earth has ever come close to that sensitivity.")

# --- visualization ---
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3)

order = [-2.0, 1.0, -1.5, -1.0, -2.5, -3.0]
for i, n_exp in enumerate(order):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    tj, dphi, frac, closed = results[n_exp]
    pts = tj[::40]
    ax.plot(pts[:, 0], pts[:, 1], lw=0.7,
            color="crimson" if closed else "steelblue", alpha=0.85)
    ax.plot(0, 0, "*", color="orange", ms=15, zorder=6)
    ax.set_aspect("equal")
    tag = {-2.0: "GRAVITY", 1.0: "Hooke spring"}.get(n_exp, "")
    verdict = ("CLOSED" if closed else
               "UNSTABLE — spirals in" if np.isnan(frac) else "precesses forever")
    ax.set_title(f"F ∝ r^{n_exp:+g}   {tag}\n"
                 f"Δφ/2π = {frac:.4f}  →  {verdict}",
                 fontsize=10, color="crimson" if closed else "steelblue")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("crimson" if closed else "lightgray")
        s.set_linewidth(2.2 if closed else 1.0)

fig.suptitle("Bertrand's theorem, integrated: of all central force laws, only "
             "F ∝ 1/r² and F ∝ r close.\n"
             "The planets are drawn as frozen ellipses because we happen to live in "
             "one of the two lucky universes —\n"
             "and Mercury's tiny 43″/century precession is the universe admitting "
             "it is not quite the pure case.",
             fontsize=11.5, y=0.99)

plt.tight_layout(rect=[0, 0, 1, 0.93])
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
