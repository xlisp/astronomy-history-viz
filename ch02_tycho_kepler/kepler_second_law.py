"""Equal areas in equal times — Kepler's accidental discovery of angular momentum.

Kepler found this law BEFORE the ellipse (it is "second" only by publication
order). He had no idea what it meant. He thought the Sun swept planets along with
a rotating magnetic "species" that weakened with distance.

The real content, invisible for 80 years until Newton: the areal rate dA/dt is
half the angular momentum per unit mass, and it is constant for ANY central force,
whatever its strength law. Kepler's second law is not about gravity at all — it is
about the fact that a force pointing at the Sun exerts no torque about the Sun.

    r × F = 0   ⟹   dL/dt = 0   ⟹   dA/dt = L/2m = constant

This is the first conservation law in physics, discovered 250 years before anyone
had the word "conservation", by a man looking for musical harmony in the heavens.

    phenomenon:   a planet races at perihelion and crawls at aphelion
    simulation:   integrate a real orbit (Mercury, e = 0.206, most eccentric of the
                  classical planets) with a symplectic integrator
    dissection:   compute swept area per time step; compute L = r × v with torch
    formula:      dA/dt = |r × v|/2 = L/2m — check it is constant to 1e-15, and
                  check it stays constant even when we change gravity to 1/r³.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

GM = 4 * np.pi**2          # AU³/yr² — gravitational parameter of the Sun in these units
A, E = 0.387, 0.2056       # Mercury


def integrate(power=-2.0, n_steps=20000, t_end=1.0):
    """Velocity-Verlet (symplectic) integration of a central force F ∝ r^power.

    Symplectic matters: a naive Euler step leaks energy and the orbit spirals,
    which would fake a violation of the very law we are testing.
    """
    dt = t_end / n_steps
    r0 = A * (1 - E)                                  # start at perihelion
    v0 = np.sqrt(GM * (1 + E) / (A * (1 - E)))        # vis-viva at perihelion
    pos = torch.tensor([r0, 0.0], dtype=torch.float64)
    vel = torch.tensor([0.0, v0], dtype=torch.float64)

    def accel(p):
        r = torch.linalg.norm(p)
        return -GM * p / r * r ** power               # r^power * unit vector

    traj, times = [pos.clone()], [0.0]
    a = accel(pos)
    for i in range(n_steps):
        vel = vel + 0.5 * dt * a
        pos = pos + dt * vel
        a = accel(pos)
        vel = vel + 0.5 * dt * a
        traj.append(pos.clone())
        times.append((i + 1) * dt)
    return torch.stack(traj), torch.tensor(times), dt


traj, times, dt = integrate(power=-2.0, t_end=A**1.5)   # one full period, by Kepler III

# --- dissection: swept area per step, computed as a cross product ---
# The triangle swept between r_i and r_{i+1} has area |r_i × r_{i+1}| / 2.
r1, r2 = traj[:-1], traj[1:]
swept = 0.5 * torch.abs(r1[:, 0] * r2[:, 1] - r1[:, 1] * r2[:, 0])
dA_dt = swept / dt

# --- the same quantity from angular momentum ---
vel_est = (traj[1:] - traj[:-1]) / dt
L = r1[:, 0] * vel_est[:, 1] - r1[:, 1] * vel_est[:, 0]     # specific angular momentum

r_mag = torch.linalg.norm(r1, dim=1)
print(f"Mercury: a = {A} AU, e = {E}, period = {A**1.5:.4f} yr")
print(f"  distance varies  {r_mag.min():.4f} → {r_mag.max():.4f} AU   "
      f"(ratio {r_mag.max()/r_mag.min():.2f}×)")
print(f"  speed varies     {torch.linalg.norm(vel_est, dim=1).min():.3f} → "
      f"{torch.linalg.norm(vel_est, dim=1).max():.3f} AU/yr   "
      f"(ratio {(torch.linalg.norm(vel_est,dim=1).max()/torch.linalg.norm(vel_est,dim=1).min()):.2f}×)")
print()
print(f"  areal rate dA/dt = {dA_dt.mean():.8f} ± {dA_dt.std():.2e} AU²/yr")
print(f"  relative variation = {(dA_dt.std()/dA_dt.mean()):.2e}   ← CONSTANT")
print(f"  L/2 = {(L/2).mean():.8f}   →  dA/dt and L/2 agree to "
      f"{(dA_dt - L/2).abs().max():.2e}")
print()

# --- the punchline: it is not about the inverse square at all ---
print("Now break gravity and see what survives:")
print("  force law        dA/dt variation      orbit closes?")
for power, label in [(-2.0, "1/r²  (Newton)"), (-3.0, "1/r³  (invented)"),
                     (-1.0, "1/r   (invented)"), (1.0, "r     (a spring!)")]:
    tj, _, d = integrate(power=power, t_end=A**1.5)
    a1, a2 = tj[:-1], tj[1:]
    sw = 0.5 * torch.abs(a1[:, 0] * a2[:, 1] - a1[:, 1] * a2[:, 0]) / d
    closes = torch.linalg.norm(tj[-1] - tj[0]) < 0.02
    print(f"  {label:18s} {(sw.std()/sw.mean()):.2e}            "
          f"{'yes' if closes else 'no — it precesses'}")
print("\nEqual areas holds for EVERY central force. It is a statement about")
print("torque, not about gravity. Kepler measured a symmetry and did not know it.")
print("Noether would explain it in 1918: rotational symmetry ⟹ angular momentum.")

# --- visualization ---
fig = plt.figure(figsize=(15, 6.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1])

# Panel 1: the swept wedges, visibly equal in area, wildly different in shape
ax = fig.add_subplot(gs[0, 0])
ax.plot(traj[:, 0], traj[:, 1], color="black", lw=1.4)
n_wedge = 12
step = (len(traj) - 1) // n_wedge
chunk = len(traj) // 40                    # each wedge covers the same TIME
cmap = plt.cm.plasma(np.linspace(0, 0.9, n_wedge))
for i in range(n_wedge):
    s = i * step
    seg = traj[s:s + chunk + 1]
    poly = torch.cat([torch.zeros(1, 2, dtype=torch.float64), seg], dim=0)
    ax.fill(poly[:, 0], poly[:, 1], color=cmap[i], alpha=0.75, lw=0)
ax.plot(0, 0, "*", color="orange", ms=22, zorder=6)
ax.set_aspect("equal")
ax.set_xlabel("AU")
ax.set_title("Twelve wedges, each swept in the SAME time.\n"
             "Long and thin near aphelion, short and fat near perihelion —\n"
             "identical area.")
ax.grid(alpha=0.3)

# Panel 2: r and v vary hugely, their product-with-sin doesn't
ax = fig.add_subplot(gs[0, 1])
ax.plot(times[:-1], r_mag, color="steelblue", lw=2, label="distance r (AU)")
ax.plot(times[:-1], torch.linalg.norm(vel_est, dim=1) / 10, color="firebrick", lw=2,
        label="speed |v| / 10 (AU/yr)")
ax.plot(times[:-1], dA_dt * 10, color="darkgreen", lw=3,
        label="areal rate dA/dt × 10  ← flat")
ax.set_xlabel("time (yr)")
ax.set_title("Everything oscillates by a factor of 2–3.\n"
             "Their combination |r × v|/2 does not move at all.")
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3)

# Panel 3: the conservation, in units of machine epsilon
ax = fig.add_subplot(gs[0, 2])
rel = (dA_dt - dA_dt.mean()) / dA_dt.mean()
ax.plot(times[:-1], rel, color="darkgreen", lw=1.5)
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("time (yr)")
ax.set_ylabel("(dA/dt − mean) / mean")
ax.set_title(f"Conserved to {rel.abs().max():.1e} — the residual is\n"
             "integrator round-off, not physics.\n"
             "This is the first conservation law ever found.")
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.grid(alpha=0.3)

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
