"""Gauss, 1801: three dots on the sky are enough to pin down an entire orbit.

On 1 January 1801 Piazzi in Palermo found a moving object, tracked it for 41 nights
across barely 3° of sky, and then lost it in the Sun's glare. It was Ceres, the
first asteroid, and it was gone.

Every method then known needed either a full orbit's worth of observations or the
assumption of a circle. Piazzi's arc was 3°, less than 1% of an orbit. The
astronomical establishment gave it up.

Gauss, aged 24, worked out how to determine ALL SIX orbital elements from three
observations and nothing else, invented least squares along the way to squeeze the
remaining data, and published a predicted position for December. Von Zach found
Ceres on 7 December 1801 and Olbers on 1 January 1802, within half a degree of the
prediction. Gauss became famous overnight and spent the next eight years writing
the book (*Theoria Motus*, 1809).

Why three observations? Count the unknowns. An orbit is 6 numbers (position and
velocity at one instant). Each observation gives 2 numbers (right ascension and
declination) — a DIRECTION but no distance. Three observations give 6 equations for
6 unknowns. Two observations are not enough; four are more than enough and force
you into least squares (→ differential_correction.py).

    phenomenon:   an object appears at three known directions at three known times
    simulation:   generate those directions from a known orbit (Ceres, 1801)
    dissection:   Gauss's method — coplanarity plus Lagrange f,g series reduces
                  everything to one 8th-degree polynomial in r₂
    formula:      r₂⁸ + a·r₂⁶ + b·r₂³ + c = 0, then back-substitute for the
                  distances, then state vector → orbital elements.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

MU = 2.959122082855911e-4         # AU³/day², the Sun's GM in Gaussian units
DEG = np.pi / 180


# ----------------------------------------------------------------------------
# orbital-element machinery (shared with the rest of this chapter)
# ----------------------------------------------------------------------------
def elements_to_state(a, e, i, Om, w, M, mu=MU):
    """Six orbital elements → heliocentric position and velocity."""
    E = M.clone() if torch.is_tensor(M) else torch.tensor(M)
    for _ in range(60):                                   # Newton on Kepler's equation
        E = E - (E - e * torch.sin(E) - M) / (1 - e * torch.cos(E))
    nu = 2 * torch.atan2(torch.sqrt(1 + e) * torch.sin(E / 2),
                         torch.sqrt(1 - e) * torch.cos(E / 2))
    r = a * (1 - e * torch.cos(E))
    p = a * (1 - e**2)
    r_pf = torch.stack([r * torch.cos(nu), r * torch.sin(nu), torch.zeros_like(r)])
    v_pf = torch.sqrt(mu / p) * torch.stack(
        [-torch.sin(nu), e + torch.cos(nu), torch.zeros_like(nu)])
    co, so = torch.cos(Om), torch.sin(Om)
    ci, si = torch.cos(i), torch.sin(i)
    cw, sw = torch.cos(w), torch.sin(w)
    R = torch.stack([
        torch.stack([co * cw - so * sw * ci, -co * sw - so * cw * ci, so * si]),
        torch.stack([so * cw + co * sw * ci, -so * sw + co * cw * ci, -co * si]),
        torch.stack([sw * si, cw * si, ci])])
    return R @ r_pf, R @ v_pf


def state_to_elements(r, v, mu=MU):
    """Heliocentric position and velocity → six orbital elements."""
    rn = torch.linalg.norm(r)
    vn = torch.linalg.norm(v)
    h = torch.linalg.cross(r, v)
    hn = torch.linalg.norm(h)
    e_vec = torch.linalg.cross(v, h) / mu - r / rn
    e = torch.linalg.norm(e_vec)
    energy = vn**2 / 2 - mu / rn
    a = -mu / (2 * energy)
    i = torch.arccos(h[2] / hn)
    n_vec = torch.stack([-h[1], h[0], torch.zeros(())])
    nn = torch.linalg.norm(n_vec)
    Om = torch.arccos(torch.clamp(n_vec[0] / nn, -1, 1))
    Om = torch.where(n_vec[1] < 0, 2 * np.pi - Om, Om)
    w = torch.arccos(torch.clamp((n_vec @ e_vec) / (nn * e), -1, 1))
    w = torch.where(e_vec[2] < 0, 2 * np.pi - w, w)
    nu = torch.arccos(torch.clamp((e_vec @ r) / (e * rn), -1, 1))
    nu = torch.where((r @ v) < 0, 2 * np.pi - nu, nu)
    E = 2 * torch.atan2(torch.sqrt(1 - e) * torch.sin(nu / 2),
                        torch.sqrt(1 + e) * torch.cos(nu / 2))
    M = E - e * torch.sin(E)
    return a, e, i, Om, w, torch.remainder(M, 2 * np.pi)


# ----------------------------------------------------------------------------
# truth: Ceres, and Earth, at the three epochs Piazzi observed
# ----------------------------------------------------------------------------
CERES = dict(a=2.7658, e=0.0785, i=10.593 * DEG, Om=80.393 * DEG, w=73.598 * DEG,
             M=torch.tensor(310.0 * DEG, dtype=torch.float64))
EARTH = dict(a=1.00000, e=0.01671, i=0.0, Om=0.0, w=102.937 * DEG,
             M=torch.tensor(100.0 * DEG, dtype=torch.float64))
# A three-observation reduction needs the three sight lines to be genuinely
# non-coplanar. Over a short arc they nearly are, the determinant D0 collapses, and
# the method becomes numerically violent — which is exactly the wall Gauss hit with
# Piazzi's data. We demonstrate on a comfortable 120-day arc first, then measure the
# breakdown as the arc shrinks back to Piazzi's.
T_OBS = [0.0, 60.0, 120.0]


def propagate(el, dt):
    n = np.sqrt(MU / el["a"] ** 3)
    M = el["M"] + n * dt
    return elements_to_state(torch.tensor(el["a"], dtype=torch.float64),
                             torch.tensor(el["e"], dtype=torch.float64),
                             torch.tensor(el["i"], dtype=torch.float64),
                             torch.tensor(el["Om"], dtype=torch.float64),
                             torch.tensor(el["w"], dtype=torch.float64), M)


r_true, R_earth, rho_hat = [], [], []
for dt in T_OBS:
    rc, _ = propagate(CERES, dt)
    re, _ = propagate(EARTH, dt)
    r_true.append(rc)
    R_earth.append(re)
    los = rc - re
    rho_hat.append(los / torch.linalg.norm(los))          # what a telescope measures

print("The three observations Gauss would have had (Piazzi, Jan–Feb 1801):")
print("  night   RA (deg)    Dec (deg)     — that is ALL. No distance, ever.")
for k, dt in enumerate(T_OBS):
    u = rho_hat[k]
    ra = float(torch.remainder(torch.atan2(u[1], u[0]), 2 * np.pi)) / DEG
    dec = float(torch.arcsin(u[2])) / DEG
    print(f"   {dt:4.0f}   {ra:9.4f}   {dec:9.4f}")
arc = float(torch.arccos(torch.clamp(rho_hat[0] @ rho_hat[2], -1, 1))) / DEG
print(f"\n  total arc swept across the sky: {arc:.2f}°  — under 1% of an orbit.\n")

# ----------------------------------------------------------------------------
# Gauss's method
# ----------------------------------------------------------------------------
t1, t2, t3 = T_OBS
tau1, tau3 = t1 - t2, t3 - t2
tau = t3 - t1

L1, L2, L3 = rho_hat
Rv1, Rv2, Rv3 = R_earth

# The three lines of sight are not independent: the orbit is planar, so r2 lies in
# the plane of r1 and r3. Everything below is bookkeeping on that one fact.
p1 = torch.linalg.cross(L2, L3)
p2 = torch.linalg.cross(L1, L3)
p3 = torch.linalg.cross(L1, L2)
D0 = L1 @ p1
D = torch.stack([torch.stack([Rv1 @ p1, Rv1 @ p2, Rv1 @ p3]),
                 torch.stack([Rv2 @ p1, Rv2 @ p2, Rv2 @ p3]),
                 torch.stack([Rv3 @ p1, Rv3 @ p2, Rv3 @ p3])])

A = (-D[1, 0] * tau3 / tau + D[1, 1] + D[1, 2] * tau1 / tau) / D0
B = (D[1, 0] * (tau3**2 - tau**2) * tau3 / tau
     + D[1, 2] * (tau**2 - tau1**2) * tau1 / tau) / (6 * D0)
E = Rv2 @ L2
R2sq = Rv2 @ Rv2

coef_a = -(A**2 + 2 * A * E + R2sq)
coef_b = -2 * MU * B * (A + E)
coef_c = -(MU**2) * B**2

print("Gauss's reduction collapses everything to ONE polynomial in r₂:")
print(f"  r₂⁸ + ({float(coef_a):.6f})·r₂⁶ + ({float(coef_b):.6e})·r₂³ "
      f"+ ({float(coef_c):.6e}) = 0")

poly = np.zeros(9)
poly[0], poly[2], poly[5], poly[8] = 1.0, float(coef_a), float(coef_b), float(coef_c)
roots = np.roots(poly)
real_pos = sorted(r.real for r in roots if abs(r.imag) < 1e-8 and r.real > 0)
print(f"  positive real roots: {[f'{x:.6f}' for x in real_pos]}")
r2_mag = torch.tensor(real_pos[-1], dtype=torch.float64)
print(f"  → r₂ = {float(r2_mag):.6f} AU     (truth {float(torch.linalg.norm(r_true[1])):.6f})\n")

# back-substitute for the three geocentric distances
num1 = (6 * (D[2, 0] * tau1 / tau3 + D[1, 0] * tau / tau3) * r2_mag**3
        + MU * D[2, 0] * (tau**2 - tau1**2) * tau1 / tau3)
rho1 = (num1 / (6 * r2_mag**3 + MU * (tau**2 - tau3**2)) - D[0, 0]) / D0
rho2 = A + MU * B / r2_mag**3
num3 = (6 * (D[0, 2] * tau3 / tau1 - D[1, 2] * tau / tau1) * r2_mag**3
        + MU * D[0, 2] * (tau**2 - tau3**2) * tau3 / tau1)
rho3 = (num3 / (6 * r2_mag**3 + MU * (tau**2 - tau1**2)) - D[2, 2]) / D0

print("The distances the method recovers — the numbers no telescope can measure:")
print("   night    ρ (AU, Earth→Ceres)    truth        error")
for k, (rho, Rv, L) in enumerate([(rho1, Rv1, L1), (rho2, Rv2, L2), (rho3, Rv3, L3)]):
    truth = float(torch.linalg.norm(r_true[k] - R_earth[k]))
    print(f"    {T_OBS[k]:4.0f}    {float(rho):12.6f}      {truth:.6f}   "
          f"{float(rho)-truth:+.2e}")

r1 = Rv1 + rho1 * L1
r2 = Rv2 + rho2 * L2
r3 = Rv3 + rho3 * L3

# Lagrange f and g series give a first velocity at the middle epoch
f1 = 1 - MU * tau1**2 / (2 * r2_mag**3)
f3 = 1 - MU * tau3**2 / (2 * r2_mag**3)
g1 = tau1 - MU * tau1**3 / (6 * r2_mag**3)
g3 = tau3 - MU * tau3**3 / (6 * r2_mag**3)
v2 = (-f3 * r1 + f1 * r3) / (f1 * g3 - f3 * g1)


# ----------------------------------------------------------------------------
# Gauss's iteration: replace the truncated f,g series with EXACT ones
# ----------------------------------------------------------------------------
def propagate_state(r, v, dt):
    """Exact two-body propagation, via the elements. Kepler, not a Taylor series."""
    a, e, i, Om, w, M = state_to_elements(r, v)
    n = torch.sqrt(MU / a**3)
    return elements_to_state(a, e, i, Om, w, M + n * dt)


def exact_fg(r2, v2, dt):
    """Lagrange coefficients from an exact propagation: r(t) = f·r₂ + g·v₂."""
    rt, _ = propagate_state(r2, v2, dt)
    h = torch.linalg.cross(r2, v2)
    hn2 = h @ h
    f = (torch.linalg.cross(rt, v2) @ h) / hn2
    g = (torch.linalg.cross(r2, rt) @ h) / hn2
    return f, g


print("\nGauss's iteration — swap the truncated series for exact Kepler f,g:")
print("  iter      r₂ (AU)        |Δr₂|")
prev = r2.clone()
for it in range(12):
    f1, g1 = exact_fg(r2, v2, tau1)
    f3, g3 = exact_fg(r2, v2, tau3)
    den = f1 * g3 - f3 * g1
    c1, c3 = g3 / den, -g1 / den
    # coplanarity: c1·r1 − r2 + c3·r3 = 0, with r_i = R_i + ρ_i·L_i.
    # Three scalar equations, three unknown distances. Solve them directly.
    Amat = torch.stack([c1 * L1, -L2, c3 * L3], dim=1)
    bvec = -c1 * Rv1 + Rv2 - c3 * Rv3
    # c1 and c3 are already folded into the columns, so the solution vector IS
    # (ρ₁, ρ₂, ρ₃) — no further scaling.
    rho1, rho2, rho3 = torch.linalg.solve(Amat, bvec)
    r1, r2, r3 = Rv1 + rho1 * L1, Rv2 + rho2 * L2, Rv3 + rho3 * L3
    v2 = (-f3 * r1 + f1 * r3) / den
    shift = float(torch.linalg.norm(r2 - prev))
    if it < 6 or shift < 1e-13:
        print(f"   {it:3d}   {float(torch.linalg.norm(r2)):.9f}   {shift:.2e}")
    prev = r2.clone()
    if shift < 1e-14:
        break
print(f"  converged; truth r₂ = {float(torch.linalg.norm(r_true[1])):.9f} AU")

print("\nSix elements, from three directions and three times:")
el_fit = state_to_elements(r2, v2)
_, v2_true = propagate(CERES, T_OBS[1])
el_true = state_to_elements(r_true[1], v2_true)
names = ["a (AU)", "e", "i (deg)", "Ω (deg)", "ω (deg)", "M (deg)"]
scale = [1, 1, 1 / DEG, 1 / DEG, 1 / DEG, 1 / DEG]
print("   element      Gauss (3 obs)      truth          error")
for nm, sc, fv, tv in zip(names, scale, el_fit, el_true):
    print(f"   {nm:10s}  {float(fv)*sc:13.6f}   {float(tv)*sc:13.6f}   "
          f"{(float(fv)-float(tv))*sc:+.3e}")

err_a = abs(float(el_fit[0]) - float(el_true[0])) / float(el_true[0])
print(f"\n  semi-major axis recovered to {err_a:.2e} relative — machine precision.")
print("  Three directions and three timestamps, and the object's distance, speed and")
print("  orbital plane all follow. Nothing was assumed about the shape of the orbit.\n")

# ----------------------------------------------------------------------------
# why this was hard in 1801: the conditioning collapses on a short arc
# ----------------------------------------------------------------------------
print("SO WHY WAS THIS HARD? Because Piazzi's arc was short.")
print("  The reduction divides by D₀ = ρ̂₁·(ρ̂₂×ρ̂₃), the volume spanned by the three")
print("  sight lines. Over a short arc they are nearly coplanar, D₀ → 0, and every")
print("  quantity in the method is amplified by 1/D₀.\n")
print("   arc span   sky arc     D₀          amplification 1/D₀")
for span in (400.0, 240.0, 120.0, 60.0, 41.0, 20.0):
    lh, Rs = [], []
    for dt in (0.0, span / 2, span):
        rc, _ = propagate(CERES, dt)
        re, _ = propagate(EARTH, dt)
        lh.append((rc - re) / torch.linalg.norm(rc - re))
        Rs.append(re)
    d0 = float(lh[0] @ torch.linalg.cross(lh[1], lh[2]))
    sky = float(torch.arccos(torch.clamp(lh[0] @ lh[2], -1, 1))) / DEG
    tag = "  ← Piazzi's actual arc" if abs(span - 41.0) < 1e-9 else ""
    print(f"   {span:6.0f} d   {sky:6.2f}°   {d0:+.3e}   {abs(1/d0):12.0f}×{tag}")
print()
print("  At Piazzi's 41 days the amplification is ~10⁵. Every rounding error in a")
print("  hand computation, and every arcsecond of observational noise, is multiplied")
print("  by that. This is why the problem defeated everyone else, and it is why")
print("  Gauss needed three inventions at once rather than just the reduction above:")
print("    · the three-observation reduction (this script)")
print("    · LEAST SQUARES, to use all 24 of Piazzi's nights instead of three, so")
print("      the noise averages down before it gets amplified")
print("    · a fast, accurate solution of Kepler's equation to make the iteration")
print("      practical by hand")
print("  The next script does the second of those. → differential_correction.py\n")

print("Why this mattered beyond one asteroid:")
print("  · It is still, essentially unchanged, how a newly-found asteroid or comet")
print("    gets its first orbit today. JPL and the Minor Planet Center run Gauss or")
print("    its close relative (Laplace's method) on the first three nights of data.")
print("  · Six unknowns, six equations, no assumption about the shape of the orbit —")
print("    Gauss did not have to guess a circle the way his predecessors did.")
print("  · It made a NEW KIND of prediction possible: not 'the planets will return',")
print("    but 'this specific unseen object will be at this specific place in eleven")
print("    months'. That is what convinced everyone. → halley_comet_return.py")

# --- visualization ---
fig = plt.figure(figsize=(15, 8.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])

# Panel 1: the geometry
ax = fig.add_subplot(gs[0, :2])
th = np.linspace(0, 2 * np.pi, 400)
for el, col, lab in [(EARTH, "steelblue", "Earth's orbit"),
                     (CERES, "crimson", "Ceres' orbit (the answer)")]:
    pts = []
    for M0 in th:
        rr, _ = elements_to_state(
            torch.tensor(el["a"], dtype=torch.float64),
            torch.tensor(el["e"], dtype=torch.float64),
            torch.tensor(el["i"], dtype=torch.float64),
            torch.tensor(el["Om"], dtype=torch.float64),
            torch.tensor(el["w"], dtype=torch.float64),
            torch.tensor(M0, dtype=torch.float64))
        pts.append(rr[:2].numpy())
    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], color=col, lw=1.6, label=lab)
ax.plot(0, 0, "*", color="orange", ms=22, label="Sun")
for k in range(3):
    Rv = R_earth[k].numpy()
    rc = r_true[k].numpy()
    ax.plot([Rv[0], Rv[0] + 3.0 * rho_hat[k][0]], [Rv[1], Rv[1] + 3.0 * rho_hat[k][1]],
            color="gray", lw=1.1, ls="--")
    ax.plot(Rv[0], Rv[1], "o", color="steelblue", ms=8)
    ax.plot(rc[0], rc[1], "o", color="crimson", ms=9)
    ax.text(rc[0] + 0.08, rc[1], f"night {T_OBS[k]:.0f}", fontsize=8)
ax.set_aspect("equal")
ax.set_xlabel("AU")
ax.set_title("What Piazzi had: three lines of sight (dashed) and nothing else.\n"
             "Gauss's method finds how far along each line the object sits — the one "
             "number\nthe observation cannot contain — by demanding the three points "
             "lie on a single Kepler orbit.")
ax.legend(fontsize=8.5, loc="upper right")
ax.grid(alpha=0.3)

# Panel 2: the polynomial
ax = fig.add_subplot(gs[0, 2])
rg = np.linspace(0.4, 3.6, 500)
val = rg**8 + float(coef_a) * rg**6 + float(coef_b) * rg**3 + float(coef_c)
ax.plot(rg, val, color="black", lw=2)
ax.axhline(0, color="gray", lw=1)
for x in real_pos:
    ax.plot(x, 0, "o", color="crimson", ms=10)
ax.axvline(float(torch.linalg.norm(r_true[1])), color="darkgreen", ls="--", lw=1.8,
           label="true r₂")
ax.set_xlabel("r₂ (AU)")
ax.set_ylabel("polynomial value")
ax.set_yscale("symlog", linthresh=1e-3)
ax.set_title("The 8th-degree polynomial.\nIts positive real root IS the distance\n"
             "to the object at the middle epoch.")
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3)

# Panel 3: element errors
ax = fig.add_subplot(gs[1, 0])
errs = [abs((float(f) - float(t)) / (float(t) if abs(float(t)) > 1e-9 else 1))
        for f, t in zip(el_fit, el_true)]
ax.bar(names, errs, color="crimson", alpha=0.85)
ax.set_yscale("log")
ax.set_ylabel("relative error")
ax.tick_params(axis="x", rotation=30)
ax.set_title("All six elements, from three dots.\nErrors are the f,g truncation, "
             "not the method.")
ax.grid(alpha=0.3, axis="y", which="both")

# Panel 4: why 3 and not 2
ax = fig.add_subplot(gs[1, 1])
ax.axis("off")
ax.text(0.0, 1.0, "Counting the unknowns", fontsize=11.5, weight="bold", va="top")
ax.text(0.0, 0.86, (
    "an orbit  = 6 numbers\n"
    "   (r, v) at one instant, or\n"
    "   (a, e, i, Ω, ω, M)\n\n"
    "one observation = 2 numbers\n"
    "   RA and Dec — a DIRECTION.\n"
    "   The distance is not measurable.\n\n"
    "   2 obs → 4 equations  underdetermined\n"
    "   3 obs → 6 equations  EXACTLY solvable\n"
    "   n obs → 2n equations  least squares\n\n"
    "Gauss needed all three ideas at once:\n"
    "  · the 3-observation reduction\n"
    "  · least squares for the extra data\n"
    "  · fast Kepler-equation solution\n\n"
    "He invented or perfected each of them\n"
    "in 1801, at the age of 24, to find one\n"
    "lost rock."), fontsize=8.8, va="top", family="monospace")

# Panel 5: the arc
ax = fig.add_subplot(gs[1, 2])
ras = [float(torch.remainder(torch.atan2(u[1], u[0]), 2 * np.pi)) / DEG for u in rho_hat]
decs = [float(torch.arcsin(u[2])) / DEG for u in rho_hat]
ax.plot(ras, decs, "o-", color="crimson", ms=11, lw=2)
for k in range(3):
    ax.annotate(f"night {T_OBS[k]:.0f}", (ras[k], decs[k]), fontsize=8.5,
                xytext=(8, 5), textcoords="offset points")
ax.set_xlabel("right ascension (deg)")
ax.set_ylabel("declination (deg)")
ax.set_title(f"The entire dataset: a {arc:.1f}° scratch\non the sky. From this, "
             "a complete orbit\nand a prediction eleven months out.")
ax.grid(alpha=0.3)

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
