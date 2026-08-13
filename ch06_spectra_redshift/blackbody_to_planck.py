"""Starlight forced the quantum: fitting a star's colour breaks classical physics.

Classical thermodynamics makes a firm, unambiguous prediction for the spectrum of a
hot body — the Rayleigh–Jeans law, B ∝ T/λ⁴. It is derived from equipartition, one
of the most trusted principles in 19th-century physics. It is also catastrophically
wrong: it diverges at short wavelength, predicting infinite energy from every warm
object. Ehrenfest named it the "ultraviolet catastrophe".

Planck fixed it in December 1900 by assuming energy comes in lumps of hν. He called
it "an act of desperation" and spent years trying to get rid of the assumption.

The reason this is an ASTRONOMY chapter: stars are the best blackbodies in nature.
A lab cavity is an approximation of a star, not the other way round. Stellar
spectra, and later the cosmic microwave background (ch08), are where the Planck
function is tested to the precision that makes it undeniable — the CMB is the most
perfect blackbody ever measured, anywhere.

    phenomenon:   stars have colours, and hotter stars are bluer
    simulation:   Planck spectra for real stars from Betelgeuse to Rigel
    dissection:   fit T to a noisy observed spectrum with torch autograd
    formula:      Wien's λ_max·T = 2.898e−3 m·K falls out of dB/dλ = 0, which we
                  get from autograd rather than from calculus by hand.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

H, C, K_B = 6.62607015e-34, 2.99792458e8, 1.380649e-23


def planck(lam, T):
    """Planck's law: the fix that needed quanta."""
    return (2 * H * C**2 / lam**5) / (torch.exp(H * C / (lam * K_B * T)) - 1)


def rayleigh_jeans(lam, T):
    """The classical prediction. Correct at long wavelength, insane at short."""
    return 2 * C * K_B * T / lam**4


STARS = [("Betelgeuse (M2)", 3600, "#ff6b3d"), ("Sun (G2)", 5772, "#fff4c2"),
         ("Sirius A (A1)", 9940, "#cfe0ff"), ("Rigel (B8)", 12100, "#a8c4ff")]

lam = torch.logspace(-7.3, -5.2, 1200, dtype=torch.float64)   # 50 nm → 6.3 µm

# --- Wien's law, obtained by autograd instead of by hand ---
print("Wien's displacement law, derived numerically (dB/dλ = 0 found with autograd):")
print("  star                T (K)    λ_max (nm)    λ_max·T (m·K)")
for name, T, _ in STARS:
    l = lam.clone().requires_grad_(True)
    B = planck(l, T)
    (grad,) = torch.autograd.grad(B.sum(), l)
    # the peak is where the derivative changes sign
    k = int(torch.where(grad[:-1] * grad[1:] < 0)[0][0])
    lmax = l[k].item()
    print(f"  {name:18s} {T:6d}   {lmax*1e9:9.1f}    {lmax*T:.6e}")
print(f"  accepted Wien constant b = 2.897771e-03 m·K")
print("  → the product is the same for every star: that is what 'displacement' means.\n")

# --- the ultraviolet catastrophe, quantified ---
print("The ultraviolet catastrophe, in numbers (Sun, T = 5772 K):")
print("   λ        Planck            Rayleigh–Jeans     RJ / Planck")
for lam_nm in (10000, 2000, 800, 500, 300, 150, 80):
    l = torch.tensor(lam_nm * 1e-9, dtype=torch.float64)
    p, r = planck(l, 5772.0).item(), rayleigh_jeans(l, 5772.0).item()
    print(f"  {lam_nm:5d} nm  {p:.4e}      {r:.4e}      {r/p:12.2f}")
print("  Classical physics is fine in the infrared and wrong by 10¹⁰ in the")
print("  ultraviolet. Integrated over all λ it predicts INFINITE emitted power.\n")

# total power: Stefan-Boltzmann emerges by integrating Planck
sigma_num = float(np.pi * torch.trapz(planck(lam, 5772.0), lam) / 5772.0**4)
print(f"Integrating Planck over all λ gives Stefan–Boltzmann:")
print(f"  σ (numerical) = {sigma_num:.6e} W m⁻² K⁻⁴")
print(f"  σ (accepted)  = 5.670374e-08 W m⁻² K⁻⁴")
print(f"  Integrating Rayleigh–Jeans instead gives infinity, for any temperature.\n")

# --- inverse problem: measure a star's temperature from its spectrum ---
torch.manual_seed(1)
T_TRUE = 5772.0
lam_obs = torch.linspace(300e-9, 2500e-9, 60, dtype=torch.float64)
clean = planck(lam_obs, T_TRUE)
observed = clean * (1 + 0.05 * torch.randn(len(lam_obs), dtype=torch.float64))

logT = torch.tensor(np.log(3000.0), dtype=torch.float64, requires_grad=True)
scale = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)   # unknown distance
opt = torch.optim.Adam([logT, scale], lr=0.02)
for _ in range(6000):
    opt.zero_grad()
    model = torch.exp(scale) * planck(lam_obs, torch.exp(logT))
    loss = ((torch.log(model) - torch.log(observed)) ** 2).mean()
    loss.backward()
    opt.step()
T_fit = float(torch.exp(logT))
print(f"Fitting a noisy spectrum (5% errors, distance unknown):")
print(f"  recovered T = {T_fit:.1f} K   (truth {T_TRUE:.0f} K, "
      f"error {100*abs(T_fit-T_TRUE)/T_TRUE:.2f}%)")
print("  The distance scale factor is degenerate with luminosity — but NOT with")
print("  temperature, because the SHAPE of the curve depends only on T. That is why")
print("  we can know a star's temperature without knowing how far away it is.")

# --- visualization ---
fig = plt.figure(figsize=(15, 8.5))
gs = fig.add_gridspec(2, 3)

# Panel 1: the catastrophe
ax = fig.add_subplot(gs[0, :2])
for name, T, col in STARS:
    ax.loglog(lam * 1e9, planck(lam, T), color=col, lw=2.4, label=f"{name}, {T} K")
ax.loglog(lam * 1e9, rayleigh_jeans(lam, 5772.0), "--", color="crimson", lw=2,
          label="Rayleigh–Jeans (classical), 5772 K")
ax.axvspan(380, 750, color="gold", alpha=0.18, label="visible")
ax.set_ylim(1e2, 1e16)
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("spectral radiance (W m⁻³ sr⁻¹)")
ax.set_title("The ultraviolet catastrophe. Classical physics (dashed) tracks the "
             "data in the infrared\nand then runs off to infinity. "
             "Planck's quantum hypothesis was invented to bend it back down.")
ax.legend(fontsize=8.5, loc="lower center")
ax.grid(alpha=0.3, which="both")

# Panel 2: Wien's law
ax = fig.add_subplot(gs[0, 2])
Ts = torch.linspace(2000, 30000, 200, dtype=torch.float64)
lmax = 2.897771e-3 / Ts
ax.loglog(Ts, lmax * 1e9, color="black", lw=2.2, label="λ_max = b/T")
for name, T, col in STARS:
    ax.plot(T, 2.897771e-3 / T * 1e9, "o", color=col, ms=12, mec="k", mew=0.6)
    ax.annotate(name.split()[0], (T, 2.897771e-3 / T * 1e9), fontsize=8,
                xytext=(6, 5), textcoords="offset points")
ax.axhspan(380, 750, color="gold", alpha=0.25)
ax.set_xlabel("temperature (K)")
ax.set_ylabel("peak wavelength (nm)")
ax.set_title("Why hot stars look blue.\nThe whole of stellar classification\n"
             "is this one hyperbola.")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which="both")

# Panel 3: the temperature fit
ax = fig.add_subplot(gs[1, 0])
ax.plot(lam_obs * 1e9, observed, "o", color="black", ms=4, label="observed (5% noise)")
lam_f = torch.linspace(300e-9, 2500e-9, 400, dtype=torch.float64)
ax.plot(lam_f * 1e9, float(torch.exp(scale)) * planck(lam_f, T_fit), color="crimson",
        lw=2, label=f"best fit T = {T_fit:.0f} K")
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("radiance")
ax.set_title(f"Measuring a star's temperature.\nTruth {T_TRUE:.0f} K, recovered "
             f"{T_fit:.0f} K.\nDistance never enters.")
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3)

# Panel 4: colour ratio as a thermometer
ax = fig.add_subplot(gs[1, 1])
Tg = torch.linspace(2500, 30000, 300, dtype=torch.float64)
b_band = planck(torch.tensor(445e-9, dtype=torch.float64), Tg)
v_band = planck(torch.tensor(551e-9, dtype=torch.float64), Tg)
bv = -2.5 * torch.log10(b_band / v_band)
ax.plot(Tg, bv, color="steelblue", lw=2.4)
for name, T, col in STARS:
    bb = planck(torch.tensor(445e-9, dtype=torch.float64), torch.tensor(float(T)))
    vv = planck(torch.tensor(551e-9, dtype=torch.float64), torch.tensor(float(T)))
    ax.plot(T, (-2.5 * torch.log10(bb / vv)).item(), "o", color=col, ms=12, mec="k", mew=0.6)
ax.set_xscale("log")
ax.set_xlabel("temperature (K)")
ax.set_ylabel("B − V colour index (uncalibrated)")
ax.set_title("Two filters are enough.\nMeasure a star in blue and in yellow,\n"
             "take the ratio, and you have its temperature.")
ax.grid(alpha=0.3, which="both")

# Panel 5: the timeline
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
ax.text(0.0, 1.0, "Why the quantum came from the sky", fontsize=11.5,
        weight="bold", va="top")
ax.text(0.0, 0.88, (
    "1859  Kirchhoff defines the blackbody\n"
    "      problem — motivated by stars\n"
    "1879  Stefan: L ∝ T⁴, from measuring\n"
    "      the Sun's output\n"
    "1893  Wien: λ_max·T = const\n"
    "1900  Rayleigh–Jeans → infinity\n"
    "1900  Planck: E = hν, 'an act of\n"
    "      desperation'\n"
    "1905  Einstein takes the quanta\n"
    "      seriously (photoelectric)\n"
    "1913  Bohr: quantised atoms explain\n"
    "      the stellar spectral sequence\n"
    "1925  Payne: stars are 98% H and He\n"
    "1990  COBE measures the CMB as the\n"
    "      most perfect blackbody known,\n"
    "      to 50 parts per million\n\n"
    "Every step was driven by the need to\n"
    "explain something glowing overhead."),
    fontsize=8.6, va="top", family="monospace")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
