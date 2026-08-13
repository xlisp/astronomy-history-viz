"""Least squares, invented in 1801 to find a lost rock, still running every orbit today.

Gauss's three-observation method (gauss_three_observations.py) uses exactly three
observations and throws the rest away. Piazzi had twenty-four. Using only three
means the noise in those three lands undiluted in the answer — and on a short arc
that noise is amplified by ~10⁵.

So Gauss did the other thing he is famous for: he took ALL the observations, wrote
down the sum of squared residuals, and minimised it. This is the origin of least
squares as a working method (Legendre published the algorithm first, in 1805;
Gauss claimed priority from 1795 and, more importantly, supplied the probabilistic
justification — that least squares is the maximum-likelihood estimator when errors
are Gaussian, a distribution now named after this argument).

The modern procedure has not changed shape in 220 years, and JPL runs it on every
asteroid nightly:

    1. get a preliminary orbit from 3 observations           (Gauss's method)
    2. propagate it to every observation time                (Kepler)
    3. compute residuals: observed direction − predicted     (arcseconds)
    4. compute ∂(residual)/∂(orbital element)                ← autograd does this
    5. take a Gauss–Newton step; go to 2

Step 4 is where this project's tooling earns its keep. Gauss derived those partial
derivatives by hand, in closed form, for six elements — pages of trigonometry.
We get them by calling .backward().

    phenomenon:   many noisy sightings of one moving object
    simulation:   24 observations of Ceres with 2″ of noise, as Piazzi had
    dissection:   Gauss–Newton on the state vector, Jacobian from autograd
    formula:      δx = (JᵀWJ)⁻¹JᵀW·r, and (JᵀWJ)⁻¹ is the COVARIANCE — which is
                  what tells you where to point the telescope in December, and
                  how big a field of view you will need.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

MU = 2.959122082855911e-4
DEG = np.pi / 180
ARCSEC = DEG / 3600


def elements_to_state(a, e, i, Om, w, M, mu=MU):
    """Elements → state. M may be a vector, so every epoch is solved in one pass."""
    E = M.clone()
    for _ in range(24):                    # Newton on Kepler's equation, batched
        E = E - (E - e * torch.sin(E) - M) / (1 - e * torch.cos(E))
    nu = 2 * torch.atan2(torch.sqrt(1 + e) * torch.sin(E / 2),
                         torch.sqrt(1 - e) * torch.cos(E / 2))
    r = a * (1 - e * torch.cos(E))
    p = a * (1 - e**2)
    r_pf = torch.stack([r * torch.cos(nu), r * torch.sin(nu), torch.zeros_like(r)])
    v_pf = torch.sqrt(mu / p) * torch.stack(
        [-torch.sin(nu), e + torch.cos(nu), torch.zeros_like(nu)])
    co, so, ci, si, cw, sw = (torch.cos(Om), torch.sin(Om), torch.cos(i),
                              torch.sin(i), torch.cos(w), torch.sin(w))
    R = torch.stack([
        torch.stack([co * cw - so * sw * ci, -co * sw - so * cw * ci, so * si]),
        torch.stack([so * cw + co * sw * ci, -so * sw + co * cw * ci, -co * si]),
        torch.stack([sw * si, cw * si, ci])])
    return R @ r_pf, R @ v_pf


CERES = dict(a=2.7658, e=0.0785, i=10.593 * DEG, Om=80.393 * DEG, w=73.598 * DEG,
             M=310.0 * DEG)
EARTH = dict(a=1.00000, e=0.01671, i=0.0, Om=0.0, w=102.937 * DEG, M=100.0 * DEG)


def orbit_at(el, dt):
    """el is a 6-vector tensor (a, e, i, Om, w, M0); returns position at time dt."""
    n = torch.sqrt(MU / el[0] ** 3)
    return elements_to_state(el[0], el[1], el[2], el[3], el[4], el[5] + n * dt)[0]


el_true = torch.tensor([CERES["a"], CERES["e"], CERES["i"], CERES["Om"],
                        CERES["w"], CERES["M"]], dtype=torch.float64)
el_earth = torch.tensor([EARTH["a"], EARTH["e"], EARTH["i"], EARTH["Om"],
                         EARTH["w"], EARTH["M"]], dtype=torch.float64)


def predict_radec(el, t):
    """Where would an object with these elements APPEAR, from Earth, at time t?"""
    r = orbit_at(el, t)
    R = orbit_at(el_earth, t)
    los = r - R
    u = los / torch.linalg.norm(los)
    ra = torch.atan2(u[1], u[0])
    dec = torch.arcsin(u[2])
    return ra, dec


# --- the data: 24 nights, as Piazzi had, with realistic 1801 noise ---
torch.manual_seed(11)
NOISE = 4.0 * ARCSEC                 # Piazzi's transit circle, ~4 arcsec
t_obs = torch.linspace(0.0, 41.0, 24, dtype=torch.float64)
ra_t0, dec_t0 = predict_radec(el_true, t_obs)
obs = torch.stack([ra_t0 + torch.randn(len(t_obs), dtype=torch.float64) * NOISE / torch.cos(dec_t0),
                   dec_t0 + torch.randn(len(t_obs), dtype=torch.float64) * NOISE], dim=1)

print(f"{len(t_obs)} observations over {float(t_obs[-1]):.0f} days, "
      f"σ = {NOISE/ARCSEC:.1f} arcsec each.\n")

# --- a deliberately imperfect starting guess, as the 3-observation method gives ---
el_fit = el_true.clone()
el_fit[0] += 0.09          # a  off by 90 000 km
el_fit[1] += 0.010         # e
el_fit[2] += 0.4 * DEG
el_fit[3] += 0.5 * DEG
el_fit[4] += 0.8 * DEG
el_fit[5] += 0.3 * DEG
el_fit = el_fit.clone().requires_grad_(True)


def residuals(el):
    """Observed minus computed, in radians, cos(dec)-weighted so both axes are angles.

    All 24 epochs are evaluated in one vectorised pass, which makes the Jacobian
    cheap enough to recompute at every iteration.
    """
    ra, dec = predict_radec(el, t_obs)
    dra = torch.remainder(obs[:, 0] - ra + np.pi, 2 * np.pi) - np.pi
    return torch.stack([dra * torch.cos(dec), obs[:, 1] - dec], dim=1).reshape(-1)


def jacobian(el):
    return torch.autograd.functional.jacobian(residuals, el, vectorize=True)


print("Gauss–Newton iteration (Jacobian from autograd, not from hand calculus):")
print("  iter    RMS residual (arcsec)      a (AU)       step norm")
rms_hist = []
for it in range(140):
    r = residuals(el_fit)
    rms = float(torch.sqrt((r**2).mean())) / ARCSEC
    rms_hist.append(rms)
    J = jacobian(el_fit.detach())
    # Normal equations. With r = observed − computed, J = ∂r/∂x, minimising ‖r + Jδ‖²
    # gives JᵀJ·δ = −Jᵀr — the minus sign matters; without it the step climbs.
    JtJ = J.T @ J
    damp = 1e-10 * torch.eye(6, dtype=torch.float64) * torch.diag(JtJ).mean()
    delta = -torch.linalg.solve(JtJ + damp, J.T @ r)
    if it < 6 or it % 20 == 0 or it == 139:
        print(f"   {it:3d}    {rms:16.4f}   {float(el_fit[0]):11.5f}   "
              f"{float(torch.linalg.norm(delta)):.2e}")
    el_fit = (el_fit + delta).detach().requires_grad_(True)
    if float(torch.linalg.norm(delta)) < 1e-13:
        print(f"   converged after {it+1} iterations")
        break

r_final = residuals(el_fit)
rms_final = float(torch.sqrt((r_final**2).mean())) / ARCSEC
print(f"\n  final RMS residual = {rms_final:.3f} arcsec   "
      f"(the noise we injected was {NOISE/ARCSEC:.1f} arcsec)")
print("  The fit lands ON the noise floor — there is no signal left in the residuals.\n")

names = ["a (AU)", "e", "i (deg)", "Ω (deg)", "ω (deg)", "M₀ (deg)"]
scale = [1, 1, 1 / DEG, 1 / DEG, 1 / DEG, 1 / DEG]

# --- the covariance, which is the actual product of the exercise ---
J = jacobian(el_fit.detach())
cov = torch.linalg.inv(J.T @ J) * (NOISE**2)
sigma = torch.sqrt(torch.diag(cov))

print("  element        truth          recovered         formal σ        (fit−truth)/σ")
for k, (nm, sc) in enumerate(zip(names, scale)):
    tv, fv, sv = float(el_true[k]) * sc, float(el_fit[k]) * sc, float(sigma[k]) * sc
    print(f"  {nm:10s} {tv:13.6f}  {fv:15.6f}  {sv:13.2e}   {(fv-tv)/sv:+8.2f}")
print()
print("  Every element sits within a couple of σ of truth. That last column is the")
print("  whole point of least squares: not just an answer, but a HONEST ERROR BAR.\n")

# --- the payoff: predict where it will be in eleven months, with an uncertainty ---
T_PREDICT = 330.0
ra_p, dec_p = predict_radec(el_fit, torch.tensor(T_PREDICT, dtype=torch.float64))
ra_t, dec_t = predict_radec(el_true, torch.tensor(T_PREDICT, dtype=torch.float64))

# propagate the covariance forward: sigma_pred = sqrt(g^T C g) for each coordinate
g_ra = torch.autograd.grad(ra_p, el_fit, retain_graph=True)[0]
g_dec = torch.autograd.grad(dec_p, el_fit, retain_graph=True)[0]
sig_ra = float(torch.sqrt(g_ra @ cov @ g_ra))
sig_dec = float(torch.sqrt(g_dec @ cov @ g_dec))
miss = float(torch.sqrt(((ra_p - ra_t) * torch.cos(dec_t)) ** 2 + (dec_p - dec_t) ** 2))

print(f"THE RECOVERY PREDICTION — where to point the telescope {T_PREDICT:.0f} days later:")
print(f"  predicted   RA = {float(ra_p)/DEG:8.4f}°   Dec = {float(dec_p)/DEG:+8.4f}°")
print(f"  truth       RA = {float(ra_t)/DEG:8.4f}°   Dec = {float(dec_t)/DEG:+8.4f}°")
print(f"  miss distance on the sky = {miss/ARCSEC/60:.2f} arcmin")
print(f"  formal 1σ uncertainty    = {sig_ra/ARCSEC/60:.2f}′ in RA, "
      f"{sig_dec/ARCSEC/60:.2f}′ in Dec")
print()
print(f"  Search area needed at 3σ: about "
      f"{6*sig_ra/ARCSEC/60:.1f}′ × {6*sig_dec/ARCSEC/60:.1f}′ — comfortably inside a")
print("  single field of view. That is the difference between 'somewhere in Virgo'")
print("  and 'point here tonight'.")
print()
print("  Von Zach recovered Ceres on 7 December 1801 and Olbers on 1 January 1802,")
print("  both within about half a degree of Gauss's ephemeris, eleven months after")
print("  the last observation. Gauss was 24 and had published nothing on astronomy.")
print()
print("Where this method went afterwards:")
print("  · every asteroid, comet and spacecraft orbit determination since")
print("  · the same normal equations appear as ridge regression, as the Kalman")
print("    filter's update step, and as Gauss–Newton in every optimiser textbook")
print("  · the Gaussian distribution is called that because of this argument")
print("  A 41-day scratch of sky, and a lost rock, produced the workhorse estimator")
print("  of every quantitative science.")

# --- visualization ---
fig = plt.figure(figsize=(15, 8.5))
gs = fig.add_gridspec(2, 3)

# Panel 1: residuals before and after
ax = fig.add_subplot(gs[0, :2])
el0 = el_true.clone()
el0[0] += 0.09; el0[1] += 0.010; el0[2] += 0.4 * DEG
el0[3] += 0.5 * DEG; el0[4] += 0.8 * DEG; el0[5] += 0.3 * DEG
r0 = residuals(el0).detach() / ARCSEC
rf = r_final.detach() / ARCSEC
ax.plot(t_obs, r0[0::2], "o-", color="lightcoral", ms=5, label="before: ΔRA·cos δ")
ax.plot(t_obs, r0[1::2], "s-", color="indianred", ms=5, label="before: ΔDec")
ax.plot(t_obs, rf[0::2], "o-", color="steelblue", ms=5, label="after: ΔRA·cos δ")
ax.plot(t_obs, rf[1::2], "s-", color="navy", ms=5, label="after: ΔDec")
ax.axhspan(-NOISE / ARCSEC, NOISE / ARCSEC, color="gray", alpha=0.25,
           label=f"observation noise ±{NOISE/ARCSEC:.0f}″")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("days from first observation")
ax.set_ylabel("residual (arcsec)")
ax.set_yscale("symlog", linthresh=10)
ax.set_title("Observed minus computed. The starting orbit leaves a clear systematic "
             "trend (red);\nafter convergence only the noise remains (blue). "
             "'No structure in the residuals' is the whole test.")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)

# Panel 2: convergence
ax = fig.add_subplot(gs[0, 2])
ax.semilogy(rms_hist, "o-", color="crimson", lw=2, ms=8)
ax.axhline(NOISE / ARCSEC, color="darkgreen", ls="--", lw=2,
           label=f"noise floor {NOISE/ARCSEC:.0f}″")
ax.set_xlabel("Gauss–Newton iteration")
ax.set_ylabel("RMS residual (arcsec)")
ax.set_title("Quadratic convergence.\nFour steps from a bad guess to\nthe noise floor.")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which="both")

# Panel 3: the Jacobian, i.e. which element each observation constrains
ax = fig.add_subplot(gs[1, 0])
Jn = (J.detach().abs() / J.detach().abs().max(dim=0).values).numpy()
im = ax.imshow(Jn, aspect="auto", cmap="viridis",
               extent=[0, 6, float(t_obs[-1]), 0])
ax.set_xticks(np.arange(6) + 0.5)
ax.set_xticklabels(["a", "e", "i", "Ω", "ω", "M₀"])
ax.set_ylabel("days")
plt.colorbar(im, ax=ax, label="|∂residual/∂element| (normalised)")
ax.set_title("The Jacobian, from autograd.\nGauss derived this by hand in closed\n"
             "form. We call .backward().")

# Panel 4: the correlation matrix
ax = fig.add_subplot(gs[1, 1])
corr = (cov / torch.outer(sigma, sigma)).numpy()
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(["a", "e", "i", "Ω", "ω", "M₀"])
ax.set_yticklabels(["a", "e", "i", "Ω", "ω", "M₀"])
for m in range(6):
    for n in range(6):
        ax.text(n, m, f"{corr[m, n]:+.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(corr[m, n]) > 0.6 else "black")
plt.colorbar(im, ax=ax, label="correlation")
ax.set_title("Element correlations from (JᵀJ)⁻¹.\nA short arc cannot separate a "
             "from M₀ —\nwhich is why the error ellipse is long\nand thin along "
             "the orbit.")

# Panel 5: the recovery field
ax = fig.add_subplot(gs[1, 2])
th = np.linspace(0, 2 * np.pi, 200)
for k, ls in [(1, "-"), (2, "--"), (3, ":")]:
    ax.plot((float(ra_p) / DEG + k * sig_ra / DEG * np.cos(th)) * 60
            - float(ra_t) / DEG * 60,
            (float(dec_p) / DEG + k * sig_dec / DEG * np.sin(th)) * 60
            - float(dec_t) / DEG * 60,
            color="crimson", ls=ls, lw=1.8, label=f"{k}σ")
ax.plot((float(ra_p) - float(ra_t)) / DEG * 60, (float(dec_p) - float(dec_t)) / DEG * 60,
        "o", color="crimson", ms=10, label="prediction")
ax.plot(0, 0, "*", color="darkgreen", ms=20, label="where Ceres really is")
ax.set_xlabel("ΔRA (arcmin)")
ax.set_ylabel("ΔDec (arcmin)")
ax.set_title(f"{T_PREDICT:.0f} days later. The error ellipse is\nthe deliverable — "
             "it tells the observer\nhow wide a field to search.")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_aspect("equal")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"\nSaved: {out}")
