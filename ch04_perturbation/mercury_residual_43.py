"""43 arcseconds per century: the smallest number ever to destroy a theory.

Le Verrier, fresh from predicting Neptune, turned the same machinery on Mercury in
1859. Mercury's perihelion advances by about 574″ per century relative to the fixed
stars. He added up the tugs of every known planet and got ~532″. The remaining ~43″
would not go away.

He did the obvious thing — the thing that had just worked spectacularly — and
predicted an unseen inner planet, "Vulcan". Astronomers hunted it for 50 years.
Amateurs reported it; the reports evaporated. Vulcan does not exist.

The residual was not a missing planet. It was a missing THEORY. In November 1915,
Einstein computed the perihelion advance from his brand-new field equations, got
43″, and later wrote that he was so excited he had heart palpitations for three
days. It was the first thing general relativity ever explained — and unlike the
1919 eclipse, it was a *retrodiction* of a number that had been sitting in the
literature, unexplained, for 56 years.

    phenomenon:   Mercury's perihelion creeps forward, ~574″ per century
    simulation:   integrate Mercury under the Sun plus each planet in turn
    dissection:   track the Laplace–Runge–Lenz vector, which points at perihelion
                  and is exactly conserved for a pure 1/r² force — so any rotation
                  of it is, by construction, a measurement of non-Keplerian physics
    formula:      sum the Newtonian budget, subtract from observation, and stare
                  at what is left. → ch05 explains it.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

GM_SUN = 4 * np.pi**2
A_MERC, E_MERC = 0.387098, 0.205630
T_MERC = A_MERC ** 1.5
RAD2ARCSEC = 180 / np.pi * 3600

# perturbing planets: (name, mass in solar masses, semi-major axis in AU)
PERTURBERS = [
    ("Venus",   2.4478e-6,  0.723332),
    ("Earth",   3.0035e-6,  1.000000),      # Earth + Moon
    ("Mars",    3.2272e-7,  1.523710),
    ("Jupiter", 9.5459e-4,  5.202887),
    ("Saturn",  2.8580e-4,  9.536676),
    ("Uranus",  4.3662e-5, 19.189165),
]
# Le Verrier / modern accepted Newtonian contributions, arcsec per century
ACCEPTED = {"Venus": 277.9, "Earth": 90.0, "Mars": 2.5, "Jupiter": 153.6,
            "Saturn": 7.3, "Uranus": 0.14}
OBSERVED_EXCESS = 574.10        # arcsec/century, after removing equinox precession

T_SPAN = 100.0                  # one century
N_STEPS = 600_000               # ~1450 steps per Mercury orbit


def run(masses, axes):
    """Integrate Mercury under the Sun + a set of perturbers, batched over systems.

    Each batch element gets its own list of perturbers via a mass matrix, so the
    per-planet breakdown and the all-together run cost one pass.
    """
    B = len(masses)
    m = torch.tensor(masses, dtype=torch.float64)                  # (B, P)
    a_p = torch.tensor(axes, dtype=torch.float64)                  # (P,)
    T_p = a_p ** 1.5
    dt = T_SPAN / N_STEPS

    pos = torch.zeros(B, 2, dtype=torch.float64)
    vel = torch.zeros(B, 2, dtype=torch.float64)
    pos[:, 0] = A_MERC * (1 - E_MERC)                              # start at perihelion
    vel[:, 1] = np.sqrt(GM_SUN * (1 + E_MERC) / (A_MERC * (1 - E_MERC)))

    def accel(p, t):
        r = torch.linalg.norm(p, dim=1, keepdim=True)
        acc = -GM_SUN * p / r**3
        ang = 2 * np.pi * t / T_p                                  # (P,)
        q = torch.stack([a_p * torch.cos(ang), a_p * torch.sin(ang)], dim=1)  # (P,2)
        d = q[None, :, :] - p[:, None, :]                          # (B,P,2)
        dn = torch.linalg.norm(d, dim=2, keepdim=True)
        # direct pull on Mercury + indirect term (the Sun is pulled too)
        contrib = GM_SUN * m[:, :, None] * (d / dn**3
                                            - q[None, :, :] / a_p[None, :, None]**3)
        return acc + contrib.sum(dim=1)

    peri_lon = torch.empty(N_STEPS // 500 + 1, B, dtype=torch.float64)
    a = accel(pos, 0.0)
    j = 0
    for i in range(N_STEPS + 1):
        if i % 500 == 0:
            # Laplace-Runge-Lenz vector: points from the Sun to perihelion.
            # Exactly conserved under a pure 1/r² force, so its rotation is a
            # direct readout of everything that is NOT pure Kepler.
            r = torch.linalg.norm(pos, dim=1, keepdim=True)
            L = pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]
            ex = vel[:, 1] * L / GM_SUN - pos[:, 0] / r.squeeze(1)
            ey = -vel[:, 0] * L / GM_SUN - pos[:, 1] / r.squeeze(1)
            peri_lon[j] = torch.atan2(ey, ex)
            j += 1
        if i == N_STEPS:
            break
        vel = vel + 0.5 * dt * a
        pos = pos + dt * vel
        a = accel(pos, (i + 1) * dt)
        vel = vel + 0.5 * dt * a
    return peri_lon


# batch: [no perturbers] + [each planet alone] + [all together]
P = len(PERTURBERS)
mass_rows = [[0.0] * P]
for i in range(P):
    row = [0.0] * P
    row[i] = PERTURBERS[i][1]
    mass_rows.append(row)
mass_rows.append([p[1] for p in PERTURBERS])
axes = [p[2] for p in PERTURBERS]

print(f"integrating Mercury for {T_SPAN:.0f} years, {N_STEPS:,} steps, "
      f"{len(mass_rows)} systems in parallel…")
lon = run(mass_rows, axes)
lon = torch.from_numpy(np.unwrap(lon.numpy(), axis=0))
times = torch.linspace(0, T_SPAN, lon.shape[0], dtype=torch.float64)

# Precession rate = slope of the perihelion longitude.
# `times` is in YEARS, so the fitted slope is rad/yr → ×RAD2ARCSEC×100 = ″/century.
design = torch.stack([times, torch.ones_like(times)], dim=1)
rates = torch.linalg.lstsq(design, lon).solution[0] * RAD2ARCSEC * 100

drift_numeric = rates[0].item()
print(f"\nintegrator's own spurious drift (no perturbers at all) = "
      f"{drift_numeric:+.1f}″/century")
print("  Velocity-Verlet conserves energy but still rotates the LRL vector slightly")
print("  through truncation error. Every run below uses identical steps and initial")
print("  conditions, so differencing against this baseline cancels it exactly.\n")

print("  planet     simulated    accepted    note")
print("             (″/century)  (″/century)")
total_sim = 0.0
sim_vals = {}
for i, (name, mass, a_p) in enumerate(PERTURBERS):
    v = rates[i + 1].item() - drift_numeric
    sim_vals[name] = v
    total_sim += v
    print(f"  {name:9s}  {v:9.2f}   {ACCEPTED[name]:9.2f}    "
          f"{'closest planet — biggest tug' if name == 'Venus' else ''}"
          f"{'most massive planet' if name == 'Jupiter' else ''}")

all_together = rates[-1].item() - drift_numeric
print(f"\n  sum of the parts      = {total_sim:8.2f}″/century")
print(f"  all planets at once   = {all_together:8.2f}″/century  "
      f"(perturbations add almost linearly)")
print(f"  accepted Newtonian total = {sum(ACCEPTED.values()):8.2f}″/century")
print(f"\n  Our circular-coplanar model omits the perturbers' own eccentricities and")
print(f"  inclinations, so it overshoots by {100*(total_sim/sum(ACCEPTED.values())-1):.0f}%. "
      f"Le Verrier did this by hand, to six")
print(f"  digits, with full Laplace–Lagrange secular theory, and got 526–532″.")
print(f"  Either way the conclusion is identical, and that is the point: no plausible")
print(f"  Newtonian bookkeeping reaches {OBSERVED_EXCESS:.0f}″.\n")

newtonian = sum(ACCEPTED.values())
residual = OBSERVED_EXCESS - newtonian
print("THE BUDGET")
print(f"  observed advance (vs fixed stars)   {OBSERVED_EXCESS:8.2f}″/century")
print(f"  explained by Newtonian planets      {newtonian:8.2f}″/century")
print(f"  ────────────────────────────────────────────────────")
print(f"  UNEXPLAINED                         {residual:8.2f}″/century")
print()
print(f"43 arcseconds per century is 0.0119° per century — the width of a human hair")
print(f"seen from 50 metres, accumulating over a hundred years. Le Verrier's response")
print(f"was to invent a planet. The correct response was to invent a new theory of")
print(f"space and time. Both are reasonable. Only one was right. → ch05")

# --- visualization ---
fig = plt.figure(figsize=(15, 8.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])

# Panel 1: the perihelion drifting
ax = fig.add_subplot(gs[0, :2])
for i, (name, _, _) in enumerate(PERTURBERS):
    ax.plot(times, (lon[:, i + 1] - lon[0, i + 1]) * RAD2ARCSEC,
            lw=1.6, label=f"{name} alone")
ax.plot(times, (lon[:, -1] - lon[0, -1]) * RAD2ARCSEC, color="black", lw=2.6,
        label="all planets")
ax.axhline(OBSERVED_EXCESS, color="crimson", ls="--", lw=2.2,
           label=f"OBSERVED = {OBSERVED_EXCESS:.0f}″")
ax.set_xlabel("years")
ax.set_ylabel("perihelion advance (arcsec)")
ax.set_title("Mercury's perihelion, tracked via the Laplace–Runge–Lenz vector.\n"
             "Newton's planets get most of the way there — and stop short.")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)

# Panel 2: the exaggerated rosette
ax = fig.add_subplot(gs[0, 2])
th = np.linspace(0, 2 * np.pi, 500)
for k in range(9):
    rot = k * 0.28
    rr = A_MERC * (1 - E_MERC**2) / (1 + E_MERC * np.cos(th - rot))
    ax.plot(rr * np.cos(th), rr * np.sin(th), lw=1.0,
            color=plt.cm.plasma(k / 9), alpha=0.85)
ax.plot(0, 0, "*", color="orange", ms=18)
ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("The motion, exaggerated ~50 000×.\nThe ellipse itself slowly turns.\n"
             "In reality: 43″/century of it is unexplained.")

# Panel 3: the budget as a waterfall
ax = fig.add_subplot(gs[1, :2])
labels = ["observed\nexcess"] + [p[0] for p in PERTURBERS] + ["UNEXPLAINED"]
vals = [OBSERVED_EXCESS] + [-ACCEPTED[p[0]] for p in PERTURBERS] + [residual]
running, bottoms, heights, colors = OBSERVED_EXCESS, [], [], []
bottoms.append(0); heights.append(OBSERVED_EXCESS); colors.append("black")
for p in PERTURBERS:
    v = ACCEPTED[p[0]]
    running -= v
    bottoms.append(running); heights.append(v); colors.append("steelblue")
bottoms.append(0); heights.append(residual); colors.append("crimson")
ax.bar(labels, heights, bottom=bottoms, color=colors, alpha=0.9)
for i, (b, h) in enumerate(zip(bottoms, heights)):
    ax.text(i, b + h + 12, f"{h:.1f}″", ha="center", fontsize=9,
            weight="bold" if i in (0, len(labels) - 1) else "normal")
ax.set_ylabel("arcsec / century")
ax.set_title("Le Verrier's ledger, 1859. Every known planet is accounted for.\n"
             "The last bar is the one that ended classical physics.")
ax.tick_params(axis="x", rotation=20)
ax.grid(alpha=0.3, axis="y")

# Panel 4: simulated vs accepted
ax = fig.add_subplot(gs[1, 2])
nms = [p[0] for p in PERTURBERS]
x = np.arange(len(nms))
ax.bar(x - 0.2, [sim_vals[n] for n in nms], 0.4, label="this simulation", color="steelblue")
ax.bar(x + 0.2, [ACCEPTED[n] for n in nms], 0.4, label="accepted value", color="darkgreen")
ax.set_xticks(x); ax.set_xticklabels(nms, rotation=30)
ax.set_ylabel("arcsec / century")
ax.set_yscale("symlog", linthresh=1)
ax.set_title("Circular-coplanar N-body vs the\nfull secular theory. Close enough\n"
             "to show 43″ cannot hide in here.")
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
