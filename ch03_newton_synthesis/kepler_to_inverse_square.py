"""No Kepler, no gravity: differentiating an ellipse twice gives you 1/r².

This is the single most important inference in the history of physics, and it is
mechanically simple. Newton did not guess the inverse-square law. He DEDUCTED it
from Kepler's three laws, each doing one job:

    Kepler II (equal areas)  ⟹  the force points at the Sun (central, no torque)
    Kepler I  (ellipse)      ⟹  its magnitude goes as 1/r²
    Kepler III (T² = a³)     ⟹  the constant GM is the SAME for every planet,
                                 i.e. the force is UNIVERSAL, not per-planet

Halley asked Newton in August 1684 what curve a 1/r² force would produce. Newton
answered "an ellipse" immediately — he had done it years earlier and lost the
paper. Halley made him redo it; the redo became the Principia (1687).

Here we do the forward direction — the one Newton actually did first — with
autograd. Kepler's equation M = E − e·sin E is solved by Newton's own iterative
method, and that solver is *differentiable*, so gradients flow straight through it.

    phenomenon:   a planet on a Keplerian ellipse, moving by the equal-area rule
    simulation:   position r(t) from Kepler's equation, solved with Newton's method
    dissection:   acceleration = d²r/dt², obtained by calling autograd twice
    formula:      fit log|a| vs log r  →  slope = −2.000.  Nothing was assumed
                  about forces; the exponent falls out of the geometry.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

GM_TRUE = 4 * np.pi**2         # AU³/yr², the Sun in these units


def position(t, a, e, T):
    """Keplerian position at time t — fully differentiable w.r.t. t.

    Newton's method on Kepler's equation. Each iteration is a differentiable
    expression, so torch can push d/dt through the whole solver. Newton would have
    found this delightful: his root-finder inside his own fluxion calculus.
    """
    M = 2 * np.pi * t / T
    E = M + e * torch.sin(M)                     # a good first guess
    for _ in range(50):
        E = E - (E - e * torch.sin(E) - M) / (1 - e * torch.cos(E))
    x = a * (torch.cos(E) - e)
    y = a * np.sqrt(1 - e**2) * torch.sin(E)
    return torch.stack([x, y])


def acceleration(t, a, e, T):
    """d²r/dt², by differentiating the position twice. This is the whole trick."""
    t = t.clone().requires_grad_(True)
    p = position(t, a, e, T)
    v = torch.stack([torch.autograd.grad(p[k], t, create_graph=True)[0] for k in (0, 1)])
    acc = torch.stack([torch.autograd.grad(v[k], t, create_graph=True)[0] for k in (0, 1)])
    return p.detach(), v.detach(), acc.detach()


# --- Mercury: the most eccentric classical planet, so r varies the most ---
A, E, T = 0.387098, 0.2056, 0.387098**1.5

ts = torch.linspace(0.001, T * 0.999, 300, dtype=torch.float64)
pos, vel, acc = [], [], []
for tv in ts:
    p, v, ac = acceleration(tv, A, E, T)
    pos.append(p); vel.append(v); acc.append(ac)
pos, vel, acc = torch.stack(pos), torch.stack(vel), torch.stack(acc)

r = torch.linalg.norm(pos, dim=1)
a_mag = torch.linalg.norm(acc, dim=1)

# --- test 1 (from Kepler II): is the acceleration CENTRAL? ---
# The angle between the acceleration vector and the direction back to the Sun.
cosang = (-(pos * acc).sum(1)) / (r * a_mag)
misalign = torch.rad2deg(torch.arccos(cosang.clamp(-1, 1)))
print("TEST 1 — is the force directed at the Sun?  (Kepler's 2nd law says yes)")
print(f"  max misalignment between a and −r̂ = {misalign.max():.2e} degrees")
print("  → perfectly central. No tangential component anywhere in the orbit.\n")

# --- test 2 (from Kepler I): how does |a| depend on r? ---
logr, loga = torch.log(r), torch.log(a_mag)
design = torch.stack([logr, torch.ones_like(logr)], dim=1)
slope, intercept = torch.linalg.lstsq(design, loga.unsqueeze(1)).solution.squeeze(1)
print("TEST 2 — fit  log|a| = n·log r + c   (no physics assumed, just the geometry)")
print(f"  n = {slope.item():.9f}          ← THE INVERSE SQUARE LAW")
print(f"  deviation from −2 = {abs(slope.item() + 2):.2e}")
print(f"  exp(c) = {torch.exp(intercept).item():.6f} AU³/yr²   (GM of the Sun = "
      f"{GM_TRUE:.6f})")
print(f"  r varies {r.min():.4f} → {r.max():.4f} AU, and |a| varies "
      f"{a_mag.min():.2f} → {a_mag.max():.2f} AU/yr² — a {a_mag.max()/a_mag.min():.2f}× swing")
print(f"  check: (r_max/r_min)² = {(r.max()/r.min())**2:.4f}   matches the swing exactly\n")

# --- test 3 (from Kepler III): is GM the same for every planet? ---
print("TEST 3 — is it the SAME force for every planet?  (Kepler's 3rd law)")
print("  planet     a (AU)     recovered GM = |a|·r²  (AU³/yr²)")
others = [("Mercury", 0.387098, 0.2056), ("Venus", 0.723332, 0.0068),
          ("Earth", 1.0, 0.0167), ("Mars", 1.523710, 0.0934),
          ("Jupiter", 5.202887, 0.0489), ("Saturn", 9.536676, 0.0565)]
gms = []
for nm, aa, ee in others:
    TT = aa**1.5
    _, _, ac1 = acceleration(torch.tensor(TT * 0.3, dtype=torch.float64), aa, ee, TT)
    p1 = position(torch.tensor(TT * 0.3, dtype=torch.float64), aa, ee, TT)
    gm = (torch.linalg.norm(ac1) * (torch.linalg.norm(p1) ** 2)).item()
    gms.append(gm)
    print(f"  {nm:9s} {aa:8.4f}    {gm:.6f}")
print(f"  spread across all six = {np.std(gms)/np.mean(gms):.2e}")
print("  → ONE constant governs all of them. The force is not a property of the")
print("    planet; it is a property of the SUN. That word is 'universal'.\n")

print("Newton's leap, in one line:  a = GM/r²  and  F = ma  ⟹  F = GMm/r².")
print("Kepler supplied every ingredient and never saw it. He died in 1630;")
print("Newton was born in 1642.")

# --- visualization ---
fig = plt.figure(figsize=(15, 8.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])

# Panel 1: the orbit with acceleration arrows
ax = fig.add_subplot(gs[0, 0])
ax.plot(pos[:, 0], pos[:, 1], color="black", lw=1.8)
skip = 12
sc = 0.00028                                    # keeps the longest arrow inside the orbit
ax.quiver(pos[::skip, 0], pos[::skip, 1], acc[::skip, 0] * sc, acc[::skip, 1] * sc,
          color="crimson", width=0.006, scale=1, scale_units="xy", angles="xy")
ax.plot(0, 0, "*", color="orange", ms=22, zorder=6)
ax.set_aspect("equal")
ax.set_title("d²r/dt² from autograd (red arrows).\nEvery one points at the Sun, and they\n"
             "grow dramatically as r shrinks.")
ax.set_xlabel("AU")
ax.grid(alpha=0.3)

# Panel 2: the log-log fit — the inverse square, measured
ax = fig.add_subplot(gs[0, 1])
ax.plot(logr, loga, "o", color="black", ms=4, label="|a| computed by autograd")
ax.plot(logr, slope * logr + intercept, "-", color="crimson", lw=2,
        label=f"slope n = {slope.item():.6f}")
ax.set_xlabel("ln r")
ax.set_ylabel("ln |a|")
ax.set_title("Take the log again (ch02's trick).\nThe slope IS the force exponent: −2.")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel 3: |a|·r² is flat — the conserved thing is GM
ax = fig.add_subplot(gs[0, 2])
ax.plot(ts, a_mag * r**2, color="darkgreen", lw=2.5, label="|a|·r²")
ax.axhline(GM_TRUE, color="crimson", ls="--", lw=1.6, label="4π² = GM of the Sun")
ax.set_ylim(GM_TRUE * 0.98, GM_TRUE * 1.02)
ax.set_xlabel("time (yr)")
ax.set_title("|a|·r² is a constant of the motion.\nThat constant is the Sun's mass.")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Bottom-left: what other exponents would look like on the log plot
ax = fig.add_subplot(gs[1, 0])
rr = torch.linspace(0.3, 0.47, 50, dtype=torch.float64)
for n_try, style in [(-1.0, ":"), (-2.0, "-"), (-3.0, "--")]:
    ax.plot(torch.log(rr), n_try * torch.log(rr) + intercept.item(), style,
            lw=2 if n_try == -2 else 1.4,
            color="crimson" if n_try == -2 else "gray",
            label=f"n = {n_try:g}" + ("  ← reality" if n_try == -2 else ""))
ax.plot(logr, loga, "o", color="black", ms=3.5, zorder=5)
ax.set_xlabel("ln r")
ax.set_ylabel("ln |a|")
ax.set_title("Mercury's eccentricity gives a 52% lever arm in r —\n"
             "easily enough to tell n = −2 from n = −1 or −3.")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Bottom-middle: GM recovered per planet
ax = fig.add_subplot(gs[1, 1])
nms = [o[0] for o in others]
ax.bar(nms, gms, color="steelblue", alpha=0.85)
ax.axhline(GM_TRUE, color="crimson", ls="--", lw=2, label="4π²")
ax.set_ylim(GM_TRUE * 0.9, GM_TRUE * 1.1)
ax.set_ylabel("recovered |a|·r²  (AU³/yr²)")
ax.set_title("Kepler III in disguise: the same GM for\nevery planet ⟹ the force is universal.")
ax.legend(fontsize=9)
ax.tick_params(axis="x", rotation=30)
ax.grid(alpha=0.3, axis="y")

# Bottom-right: the logical chain
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
ax.text(0.0, 1.0, "The deduction, in full", fontsize=13, weight="bold", va="top")
ax.text(0.0, 0.87, (
    "Tycho 1576–1601\n"
    "   20 years of 1′ naked-eye positions\n"
    "        ↓\n"
    "Kepler 1609, 1619\n"
    "   II. equal areas   → force is CENTRAL\n"
    "   I.  ellipse       → magnitude ∝ 1/r²\n"
    "   III. T² = a³      → same GM for all\n"
    "        ↓\n"
    "Newton 1687\n"
    "   F = G·M·m/r²,  and the SAME law\n"
    "   makes the apple fall (see moon_test.py)\n"
    "        ↓\n"
    "everything after: tides, precession,\n"
    "Neptune, spacecraft navigation,\n"
    "and the anomaly that killed it (ch04–05)\n\n"
    "Remove Kepler and the chain has no\n"
    "first link. That is why astronomy is\n"
    "the mother science."),
    fontsize=9.2, va="top", family="monospace")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
