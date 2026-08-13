"""Neptune: a planet discovered with a pen, and Newton's theory at its zenith.

Uranus was found by accident (Herschel, 1781). Within 40 years it was misbehaving:
Bouvard's 1821 tables could not fit both the old pre-discovery sightings and the
new ones. By 1845 the error had reached 2 arcminutes — small, but 100× the
observational precision, and growing.

Two people independently drew the same conclusion: there is another planet out
there. Le Verrier in Paris and Adams in Cambridge each solved the inverse problem —
given the residuals, where must the perturber be? On 23 September 1846 Galle
pointed the Berlin telescope where Le Verrier's letter said, and found Neptune
within 1° on the first night.

This is the most spectacular prediction in the history of science, and it is what
makes the failure in mercury_residual_43.py so devastating: the SAME method,
applied to Mercury by the SAME man (Le Verrier, 1859), predicted a planet
"Vulcan" that does not exist. Newton's theory passed its greatest test and failed
its next one, and the difference between those two outcomes is general relativity.

    phenomenon:   Uranus drifts off its predicted longitude. Against the full
                  historical baseline (including pre-discovery sightings back to
                  1690, when Flamsteed catalogued it as a star) the residual
                  reached ~2′; over the 1781–1846 window simulated here the
                  detrended curvature alone is ~0.5′ — still 16× the error bar
    simulation:   integrate Uranus under Sun + an unseen outer perturber
    dissection:   the inverse problem — scan (a, phase) for the perturber, with
                  the mass solved by least squares at each point
    formula:      the residual is linear in the perturber's mass for small masses,
                  so the search factorizes into a 2-D scan and a 1-D linear solve.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

GM_SUN = 4 * np.pi**2                 # AU³/yr²
A_URANUS, T_URANUS = 19.189165, 84.0148
A_NEP_TRUE, M_NEP_TRUE = 30.069923, 5.1503e-5      # AU, in solar masses
PHASE_NEP_TRUE = 2.10                              # rad, Neptune's longitude at t = 0

T_SPAN, N_STEPS = 65.0, 30000        # 1781 (Herschel) → 1846 (Galle)
M_REF = 5.0e-5                       # reference perturber mass for the linear response


def integrate_uranus(a_pert, phase_pert, m_pert):
    """Uranus under Sun + a perturber on a prescribed circular orbit.

    All candidate perturbers are integrated simultaneously as a batch, so the whole
    search costs one pass. Velocity-Verlet keeps the energy honest over 65 years.
    """
    a_pert = torch.as_tensor(a_pert, dtype=torch.float64).flatten()
    phase_pert = torch.as_tensor(phase_pert, dtype=torch.float64).flatten()
    m_pert = torch.as_tensor(m_pert, dtype=torch.float64).flatten()
    B = len(a_pert)
    T_pert = a_pert ** 1.5                            # Kepler III fixes the period
    dt = T_SPAN / N_STEPS

    pos = torch.zeros(B, 2, dtype=torch.float64)
    vel = torch.zeros(B, 2, dtype=torch.float64)
    pos[:, 0] = A_URANUS
    vel[:, 1] = np.sqrt(GM_SUN / A_URANUS)            # circular start, good enough

    def accel(p, t):
        r = torch.linalg.norm(p, dim=1, keepdim=True)
        a_sun = -GM_SUN * p / r**3
        ang = 2 * np.pi * t / T_pert + phase_pert
        q = torch.stack([a_pert * torch.cos(ang), a_pert * torch.sin(ang)], dim=1)
        d = q - p
        dn = torch.linalg.norm(d, dim=1, keepdim=True)
        # direct term + indirect term (the Sun is also pulled by the perturber)
        a_pert_on_u = GM_SUN * m_pert[:, None] * (d / dn**3 - q / a_pert[:, None]**3)
        return a_sun + a_pert_on_u

    out = torch.empty(N_STEPS + 1, B, 2, dtype=torch.float64)
    out[0] = pos
    a = accel(pos, 0.0)
    for i in range(N_STEPS):
        vel = vel + 0.5 * dt * a
        pos = pos + dt * vel
        a = accel(pos, (i + 1) * dt)
        vel = vel + 0.5 * dt * a
        out[i + 1] = pos
    return out


times = torch.linspace(0, T_SPAN, N_STEPS + 1, dtype=torch.float64)


def longitude(traj):
    return torch.from_numpy(np.unwrap(torch.atan2(traj[..., 1], traj[..., 0]).numpy(),
                                      axis=0))


# --- phenomenon: the truth (Neptune exists) minus the model (it does not) ---
truth = integrate_uranus([A_NEP_TRUE], [PHASE_NEP_TRUE], [M_NEP_TRUE])[:, 0, :]
unperturbed = integrate_uranus([A_NEP_TRUE], [PHASE_NEP_TRUE], [0.0])[:, 0, :]
resid_true = (longitude(truth) - longitude(unperturbed))

# Astronomers of 1840 could not tell a constant drift from a slightly wrong period,
# so they fitted out any linear trend. Only the CURVATURE is real information.
design = torch.stack([times, torch.ones_like(times)], dim=1)
coef = torch.linalg.lstsq(design, resid_true.unsqueeze(1)).solution
observed = resid_true - (design @ coef).squeeze(1)
OBS_NOISE = 2.0 / 3600 * np.pi / 180                  # 2 arcsec, meridian-circle era
torch.manual_seed(7)
observed_noisy = observed + torch.randn(len(observed), dtype=torch.float64) * OBS_NOISE

arcmin = 180 / np.pi * 60
print(f"Uranus' longitude residual after removing any linear trend:")
print(f"  peak-to-peak = {(observed.max()-observed.min())*arcmin:.2f} arcmin "
      f"over {T_SPAN:.0f} years")
print(f"  observational precision of the 1840s ≈ 2″ = {2/60:.3f}′")
print(f"  → the anomaly is {(observed.max()-observed.min())*arcmin/(2/60):.0f}× the "
      f"error bar. Something is out there.\n")

# --- dissection: the inverse problem ---
# For small m the response is linear, so integrate at a reference mass and let a
# 1-D least-squares solve pick the amplitude. That reduces a 3-D search to 2-D.
a_grid = torch.linspace(20.0, 45.0, 46, dtype=torch.float64)
ph_grid = torch.linspace(0, 2 * np.pi, 25, dtype=torch.float64)[:-1]
AA, PP = torch.meshgrid(a_grid, ph_grid, indexing="ij")
flat_a, flat_p = AA.flatten(), PP.flatten()

print(f"scanning {len(flat_a)} candidate perturbers (a × phase), "
      f"mass solved analytically at each…")
pert = integrate_uranus(flat_a, flat_p, torch.full_like(flat_a, M_REF))
base = integrate_uranus(flat_a, flat_p, torch.zeros_like(flat_a))
model = longitude(pert) - longitude(base)                      # (N_STEPS+1, B)
# detrend each candidate the same way the data was detrended
c = torch.linalg.lstsq(design, model).solution
model = model - design @ c

# best mass for each candidate:  m* = M_REF · <model, obs> / <model, model>
num = (model * observed_noisy[:, None]).sum(0)
den = (model * model).sum(0)
scale = num / den.clamp(min=1e-30)
m_best = scale * M_REF
chi2 = ((model * scale - observed_noisy[:, None]) ** 2).sum(0)
# a negative mass is unphysical — a perturber cannot push
chi2 = torch.where(m_best > 0, chi2, torch.full_like(chi2, float("inf")))

k = int(torch.argmin(chi2))
print(f"\nBEST FIT — the planet the residuals demand:")
print(f"  semi-major axis  a = {flat_a[k]:.2f} AU        (Neptune, true: {A_NEP_TRUE:.2f})")
print(f"  mass             m = {m_best[k]:.3e} M_sun   (Neptune, true: {M_NEP_TRUE:.3e})")
print(f"  mass in Earths     = {m_best[k]*332946:.1f} M_earth  "
      f"(true: {M_NEP_TRUE*332946:.1f})")
print(f"  phase at t=0     = {flat_p[k]:.2f} rad        (true: {PHASE_NEP_TRUE:.2f})")

# where do we point the telescope? the perturber's sky longitude in 1846
T_best = flat_a[k] ** 1.5
lon_1846 = (2 * np.pi * T_SPAN / T_best + flat_p[k]) % (2 * np.pi)
lon_true = (2 * np.pi * T_SPAN / A_NEP_TRUE**1.5 + PHASE_NEP_TRUE) % (2 * np.pi)
err_deg = abs(np.rad2deg(float(lon_1846 - lon_true)))
err_deg = min(err_deg, 360 - err_deg)
print(f"\n  → POINT THE TELESCOPE AT heliocentric longitude "
      f"{np.rad2deg(float(lon_1846)):.1f}°")
print(f"    Neptune is actually at {np.rad2deg(float(lon_true)):.1f}° — "
      f"we are off by {err_deg:.1f}°")
print(f"    (Le Verrier's real 1846 letter was off by 1°. Galle found it in 30 minutes.)")
print()
print("Note what was inferred here from a 2-arcminute wobble: the existence, mass,")
print("distance and sky position of a body nobody had ever seen. Newton's theory was")
print("never more powerful than on 23 September 1846.")
print("Thirteen years later the same method, in the same hands, broke. → ch04/mercury")

# --- visualization ---
fig = plt.figure(figsize=(15, 8.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])

# Panel 1: the residual that started it
ax = fig.add_subplot(gs[0, :2])
yrs = 1781 + times.numpy()
ax.plot(yrs, observed_noisy * arcmin, ".", color="gray", ms=1.5, alpha=0.5,
        label="observed residual (with 2″ noise)")
ax.plot(yrs, observed * arcmin, color="black", lw=2, label="true residual")
ax.plot(yrs, (model[:, k] * scale[k]) * arcmin, "--", color="crimson", lw=2,
        label=f"best-fit unseen planet: a = {flat_a[k]:.1f} AU, "
              f"m = {m_best[k]*332946:.0f} M⊕")
ax.axhspan(-2 / 60, 2 / 60, color="steelblue", alpha=0.25,
           label="observational error bar (±2″)")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("year")
ax.set_ylabel("Uranus longitude residual (arcmin)")
ax.set_title(f"The anomaly that produced a planet.\n"
             f"{(observed.max()-observed.min())*arcmin:.2f}′ of curvature — "
             f"{(observed.max()-observed.min())*arcmin/(2/60):.0f}× the error bar — "
             f"and no way to explain it\ninside the known solar system.")
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3)

# Panel 2: the χ² landscape of the inverse problem
ax = fig.add_subplot(gs[0, 2])
chi2_map = chi2.reshape(len(a_grid), len(ph_grid))
finite = torch.isfinite(chi2_map)
im = ax.pcolormesh(np.rad2deg(ph_grid.numpy()), a_grid.numpy(),
                   torch.log10(chi2_map / chi2[torch.isfinite(chi2)].min()).numpy(),
                   cmap="viridis_r", shading="auto")
ax.plot(np.rad2deg(float(flat_p[k])), float(flat_a[k]), "*", color="crimson", ms=22,
        label="best fit")
ax.plot(np.rad2deg(PHASE_NEP_TRUE), A_NEP_TRUE, "o", mfc="none", color="white",
        ms=16, mew=2.5, label="Neptune (truth)")
plt.colorbar(im, ax=ax, label="log₁₀ (χ² / χ²_min)")
ax.set_xlabel("perturber phase at 1781 (deg)")
ax.set_ylabel("perturber semi-major axis (AU)")
ax.set_title("The inverse problem, solved by scanning.\n"
             "One deep minimum — the data pins the planet down.")
ax.legend(fontsize=8, loc="upper right")

# Panel 3: how well is the distance constrained?
ax = fig.add_subplot(gs[1, 0])
best_per_a = torch.where(torch.isfinite(chi2_map), chi2_map,
                         torch.full_like(chi2_map, float("inf"))).amin(dim=1)
ax.semilogy(a_grid, best_per_a / best_per_a.min(), "o-", color="crimson", lw=2)
ax.axvline(A_NEP_TRUE, color="darkgreen", ls="--", lw=2, label="Neptune's true a")
ax.set_xlabel("assumed perturber a (AU)")
ax.set_ylabel("χ² / χ²_min  (best phase & mass)")
ax.set_title("Distance is well constrained.\n(Adams and Le Verrier both leaned on\n"
             "the Titius–Bode rule here and got a ≈ 36 —\nwrong, yet the sky "
             "position was still right.)")
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3, which="both")

# Panel 4: mass degeneracy
ax = fig.add_subplot(gs[1, 1])
ax.plot(a_grid, [float(m_best.reshape(len(a_grid), -1)[i]
                       [int(torch.argmin(chi2_map[i]))]) * 332946
                 for i in range(len(a_grid))], "o-", color="steelblue", lw=2)
ax.axhline(M_NEP_TRUE * 332946, color="darkgreen", ls="--", lw=2,
           label=f"Neptune = {M_NEP_TRUE*332946:.0f} M⊕")
ax.axvline(A_NEP_TRUE, color="darkgreen", ls=":", lw=1.5)
ax.set_xlabel("assumed perturber a (AU)")
ax.set_ylabel("best-fit mass (Earth masses)")
ax.set_title("Mass and distance trade off:\na farther planet must be heavier to\n"
             "tug equally hard. Astronomy is full of\nthis kind of degeneracy.")
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3)

# Panel 5: the sky map — where to point
ax = fig.add_subplot(gs[1, 2], projection="polar")
th = np.linspace(0, 2 * np.pi, 200)
ax.plot(th, np.full_like(th, A_URANUS), color="steelblue", lw=1.2, label="Uranus orbit")
ax.plot(th, np.full_like(th, A_NEP_TRUE), color="darkgreen", lw=1.2, ls="--",
        label="Neptune orbit")
ax.plot(float(lon_1846), float(flat_a[k]), "*", color="crimson", ms=24,
        label="predicted position")
ax.plot(float(lon_true), A_NEP_TRUE, "o", color="darkgreen", ms=12,
        label="Neptune, actual")
lon_u = float(longitude(truth)[-1] % (2 * np.pi))
ax.plot(lon_u, A_URANUS, "o", color="steelblue", ms=10, label="Uranus, 1846")
ax.set_rmax(38)
ax.set_title(f"23 September 1846.\nPrediction misses by {err_deg:.1f}°.\n"
             "Galle found it the same night.", pad=18)
ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(-0.15, -0.15))

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
