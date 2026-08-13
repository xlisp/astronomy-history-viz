"""1998: two rival teams set out to measure the deceleration and found acceleration.

Everyone knew the expansion must be slowing down — gravity is attractive, and that
is the only long-range force acting on cosmic scales. The open question was only
whether it slowed enough to eventually recollapse. Two teams (Perlmutter's Supernova
Cosmology Project; Riess and Schmidt's High-z Team) raced to measure the deceleration
parameter using Type Ia supernovae as standard candles out to z ~ 1.

Both found the distant supernovae were FAINTER than even an empty, coasting universe
predicts. Fainter means farther. Farther means the expansion has been speeding up.

Both teams spent months hunting for the error, because the result was absurd.
Neither found one. Nobel Prize, 2011.

The cause is written into general relativity as the term Einstein added in 1917 to
hold the universe static, then abandoned as his "greatest blunder" once Hubble
showed it was expanding. It is back, it is 68% of everything, and nobody knows what
it is.

    phenomenon:   distant supernovae are ~25% fainter than a decelerating universe
                  predicts
    simulation:   luminosity distance in three cosmologies, by integrating the
                  Friedmann equation
    dissection:   fit (Ω_m, Ω_Λ) to a supernova Hubble diagram with torch autograd
    formula:      d_L(z) = (1+z)·c/H₀ ∫dz'/E(z'),  E(z) = √(Ω_m(1+z')³ + Ω_Λ)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

C_KMS = 299792.458
H0 = 70.0


def luminosity_distance(z, Om, OL, n_int=600):
    """d_L in Mpc, by integrating 1/E(z) on a trapezoid grid.

    Written so that autograd can differentiate through the integral with respect
    to Om and OL — which is what lets us FIT a cosmology instead of guessing one.
    """
    z = torch.atleast_1d(z)
    # integrate from 0 to each z on a shared normalised grid
    u = torch.linspace(0, 1, n_int, dtype=torch.float64)
    zz = z[:, None] * u[None, :]                     # (N, n_int)
    Ok = 1.0 - Om - OL                               # curvature closes the budget
    E = torch.sqrt(Om * (1 + zz) ** 3 + Ok * (1 + zz) ** 2 + OL)
    integ = torch.trapz(1.0 / E, zz, dim=1)          # comoving distance / (c/H0)
    Dc = C_KMS / H0 * integ
    if abs(float(Ok)) < 1e-8:
        Dm = Dc
    elif float(Ok) > 0:
        rk = C_KMS / H0 / torch.sqrt(torch.as_tensor(Ok))
        Dm = rk * torch.sinh(Dc / rk)
    else:
        rk = C_KMS / H0 / torch.sqrt(torch.as_tensor(-Ok))
        Dm = rk * torch.sin(Dc / rk)
    return (1 + z) * Dm


def distance_modulus(z, Om, OL):
    return 5 * torch.log10(luminosity_distance(z, Om, OL)) + 25


# --- three candidate universes ---
MODELS = [("Ω_m=1.0, Ω_Λ=0   (decelerating, the 1990s default)", 1.0, 0.0, "steelblue"),
          ("Ω_m=0.3, Ω_Λ=0   (open, coasting)", 0.3, 0.0, "darkgreen"),
          ("Ω_m=0.3, Ω_Λ=0.7 (accelerating — reality)", 0.3, 0.7, "crimson")]

z_test = torch.tensor([0.1, 0.3, 0.5, 0.8, 1.0], dtype=torch.float64)
print("Distance modulus μ = m − M predicted by each cosmology:")
print("   z     Ω_m=1.0      Ω_m=0.3      ΛCDM      ΛCDM − Ω_m=1  (magnitudes)")
for i, zv in enumerate(z_test):
    mus = [float(distance_modulus(zv, Om, OL)) for _, Om, OL, _ in MODELS]
    print(f"  {float(zv):.1f}   {mus[0]:8.3f}   {mus[1]:8.3f}   {mus[2]:8.3f}   "
          f"{mus[2]-mus[0]:+8.3f}")
gap = float(distance_modulus(torch.tensor(0.5, dtype=torch.float64), 0.3, 0.7)
            - distance_modulus(torch.tensor(0.5, dtype=torch.float64), 1.0, 0.0))
print()
print(f"At z = 0.5 the gap between accelerating and decelerating is {gap:.2f} mag —")
print(f"a factor of {10**(gap/2.5):.2f} in brightness. Type Ia supernovae are")
print("standardisable to ~0.15 mag, so the measurement was just barely possible.")
print("That is why it took until 1998, and why both teams spent months hunting")
print("for a mistake before publishing.\n")

# --- generate a supernova sample and fit the cosmology back ---
torch.manual_seed(9)
OM_TRUE, OL_TRUE = 0.3, 0.7
z_sn = torch.sort(torch.rand(180, dtype=torch.float64) * 1.1 + 0.02).values
mu_true = distance_modulus(z_sn, OM_TRUE, OL_TRUE)
SN_ERR = 0.18
mu_obs = mu_true + torch.randn(len(z_sn), dtype=torch.float64) * SN_ERR

# fit with autograd, parametrising to keep 0 < Om, OL
p = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)
opt = torch.optim.Adam([p], lr=0.03)
for _ in range(3000):
    opt.zero_grad()
    Om, OL = torch.sigmoid(p[0]) * 1.5, torch.sigmoid(p[1]) * 1.5
    loss = (((distance_modulus(z_sn, Om, OL) - mu_obs) / SN_ERR) ** 2).mean()
    loss.backward()
    opt.step()
Om_fit = float(torch.sigmoid(p[0]) * 1.5)
OL_fit = float(torch.sigmoid(p[1]) * 1.5)
print(f"fitted cosmology from {len(z_sn)} supernovae:")
print(f"  Ω_m = {Om_fit:.3f}   (truth {OM_TRUE})")
print(f"  Ω_Λ = {OL_fit:.3f}   (truth {OL_TRUE})")
print(f"  Neither is recovered precisely, and that is not a bug — supernovae")
print(f"  constrain a COMBINATION, roughly Ω_Λ − 1.4·Ω_m, not the two separately:")
print(f"     truth:  Ω_Λ − 1.4Ω_m = {OL_TRUE - 1.4*OM_TRUE:+.3f}")
print(f"     fit:    Ω_Λ − 1.4Ω_m = {OL_fit - 1.4*Om_fit:+.3f}   ← this IS well measured")
print(f"  The confidence region below is a long diagonal ellipse for exactly this")
print(f"  reason. Breaking the degeneracy needs a second, differently-shaped")
print(f"  constraint — which is what the CMB supplies (cmb_blackbody.py). The two")
print(f"  ellipses cross at Ω_m ≈ 0.3, Ω_Λ ≈ 0.7, and that crossing is the")
print(f"  standard model of cosmology.")
print(f"  What survives regardless: Ω_Λ > 0. The expansion is ACCELERATING.\n")

# --- the deceleration parameter ---
q0_fit = Om_fit / 2 - OL_fit
q0_matter = 0.5
print(f"deceleration parameter  q₀ = Ω_m/2 − Ω_Λ")
print(f"  matter-only universe:  q₀ = {q0_matter:+.3f}  (decelerating, as expected)")
print(f"  measured:              q₀ = {q0_fit:+.3f}  (NEGATIVE — it is speeding up)")
print()
print("Einstein introduced Λ in 1917 to make a static universe, and dropped it after")
print("1929. Reinstating it in 1998 required no new physics — just admitting the term")
print("had been there in the equations all along. What it physically IS remains the")
print("largest open question in physics: the naive quantum-field estimate of the")
print("vacuum energy exceeds the observed value by ~120 orders of magnitude.")

# --- visualization ---
fig = plt.figure(figsize=(15, 8.5))
gs = fig.add_gridspec(2, 3)

zz = torch.linspace(0.01, 1.5, 200, dtype=torch.float64)

# Panel 1: the Hubble diagram
ax = fig.add_subplot(gs[0, :2])
ax.errorbar(z_sn, mu_obs, yerr=SN_ERR, fmt="o", color="black", ms=3.5, alpha=0.55,
            capsize=0, label="Type Ia supernovae")
for lab, Om, OL, col in MODELS:
    ax.plot(zz, distance_modulus(zz, Om, OL), color=col, lw=2.2, label=lab)
ax.set_xlabel("redshift z")
ax.set_ylabel("distance modulus μ = m − M")
ax.set_title("The 1998 Hubble diagram. The supernovae sit ABOVE the decelerating "
             "curve —\nfainter, therefore farther, therefore the expansion sped up "
             "after they exploded.")
ax.legend(fontsize=8.5, loc="lower right")
ax.grid(alpha=0.3)

# Panel 2: residuals — where the signal actually lives
ax = fig.add_subplot(gs[0, 2])
ref = distance_modulus(z_sn, 1.0, 0.0)
ax.errorbar(z_sn, mu_obs - ref, yerr=SN_ERR, fmt="o", color="black", ms=3.5,
            alpha=0.45, capsize=0)
for lab, Om, OL, col in MODELS:
    ax.plot(zz, distance_modulus(zz, Om, OL) - distance_modulus(zz, 1.0, 0.0),
            color=col, lw=2.2)
ax.axhline(0, color="steelblue", lw=1.5)
ax.set_xlabel("redshift z")
ax.set_ylabel("μ − μ(Ω_m=1)")
ax.set_title("Same data, decelerating model subtracted.\nThe whole discovery is this "
             "0.25 mag gap.")
ax.grid(alpha=0.3)

# Panel 3: the confidence region
ax = fig.add_subplot(gs[1, 0])
om_g = torch.linspace(0.05, 0.85, 34, dtype=torch.float64)
ol_g = torch.linspace(0.0, 1.3, 36, dtype=torch.float64)
chi2 = torch.zeros(len(om_g), len(ol_g), dtype=torch.float64)
with torch.no_grad():
    for i, om in enumerate(om_g):
        for j, ol in enumerate(ol_g):
            chi2[i, j] = (((distance_modulus(z_sn, om, ol) - mu_obs) / SN_ERR) ** 2).sum()
dchi = (chi2 - chi2.min()).T
cs = ax.contourf(om_g, ol_g, dchi, levels=[0, 2.3, 6.17, 11.8, 1e9],
                 colors=["#08306b", "#2171b5", "#9ecae1", "#f7fbff"])
ax.plot(OM_TRUE, OL_TRUE, "*", color="crimson", ms=20, label="truth")
ax.plot(Om_fit, OL_fit, "o", color="white", mec="k", ms=9, label="autograd fit")
ol_line = 1 - om_g
ax.plot(om_g, ol_line, "k--", lw=1.5, label="flat universe")
ax.plot(om_g, om_g / 2, color="darkorange", lw=1.8, ls=":", label="q₀ = 0 (no accel.)")
ax.set_xlabel("Ω_m")
ax.set_ylabel("Ω_Λ")
ax.set_ylim(0, 1.3)
ax.set_title("1σ / 2σ / 3σ contours.\nΩ_Λ = 0 sits far outside.")
ax.legend(fontsize=7.5, loc="upper left")

# Panel 4: the scale factor histories
ax = fig.add_subplot(gs[1, 1])
t = np.linspace(-1.0, 1.2, 400)
for lab, Om, OL, col in MODELS:
    # integrate da/dt = H0 a E(1/a - 1) backwards & forwards, crudely but honestly
    a = [1.0]
    dt = t[1] - t[0]
    for _ in range(len(t) - 1):
        aa = max(a[-1], 1e-4)
        Ok = 1 - Om - OL
        Hs = np.sqrt(Om / aa**3 + Ok / aa**2 + OL)
        a.append(max(aa + dt * aa * Hs * (H0 * 1.02271e-3), 1e-4))   # H0 in 1/Gyr
    a = np.array(a)
    now = np.argmin(np.abs(a - 1.0))
    ax.plot(t * 13.8 - t[now] * 13.8, a, color=col, lw=2.2)
ax.axhline(1, color="k", ls=":", lw=1)
ax.axvline(0, color="k", ls=":", lw=1)
ax.set_xlabel("time from today (Gyr)")
ax.set_ylabel("scale factor a(t)")
ax.set_ylim(0, 2.2)
ax.set_title("The three futures. Matter-only (blue) coasts\nto a halt; ΛCDM (red) "
             "runs away exponentially.\nWe are on the red curve.")
ax.grid(alpha=0.3)

# Panel 5: the inventory over cosmic time
ax = fig.add_subplot(gs[1, 2])
a_grid = np.logspace(-3, 0.6, 300)
rho_m = 0.3 / a_grid**3
rho_r = 9.2e-5 / a_grid**4
rho_L = np.full_like(a_grid, 0.7)
tot = rho_m + rho_r + rho_L
ax.plot(a_grid, rho_r / tot, color="darkorange", lw=2.2, label="radiation")
ax.plot(a_grid, rho_m / tot, color="steelblue", lw=2.2, label="matter")
ax.plot(a_grid, rho_L / tot, color="crimson", lw=2.2, label="dark energy Λ")
ax.axvline(1.0, color="k", ls=":", lw=1.5)
ax.text(1.05, 0.5, "now", fontsize=9, rotation=90)
ax.set_xscale("log")
ax.set_xlabel("scale factor a  (a = 1 today)")
ax.set_ylabel("fraction of total energy density")
ax.set_title("Λ is constant while everything else\ndilutes — so it was negligible "
             "for 9 Gyr\nand then took over. We happen to live\nnear the crossover.")
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3, which="both")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
