"""The most perfect blackbody ever measured, found by accident, in a horn antenna.

In 1964 Penzias and Wilson were trying to eliminate noise from a Bell Labs antenna.
They had an excess of about 3 K coming from every direction, day and night, all year.
They cleaned out the pigeons and scrubbed off the droppings ("a white dielectric
material"). The noise stayed.

Thirty miles away at Princeton, Dicke's group was building an instrument to look for
exactly this — the relic heat of a hot early universe, predicted by Gamow, Alpher
and Herman in 1948 and then largely forgotten. Dicke put down the phone and said to
his team: "Boys, we've been scooped."

That accidental 3 K is now the tightest constraint in physics. COBE's 1990 spectrum
matched a Planck curve so precisely that the error bars are smaller than the width
of the plotted line — when John Mather showed it at the AAS meeting, the audience
gave it a standing ovation. It is the strongest evidence that the early universe was
hot, dense and in thermal equilibrium: a steady-state universe has no way to produce
a perfect blackbody at all, let alone one this perfect.

    phenomenon:   a 2.725 K glow, identical in every direction to 1 part in 10⁵
    simulation:   the Planck spectrum at 2.725 K vs COBE/FIRAS measurements
    dissection:   fit T with torch; then extract the dipole and show that removing
                  it reveals structure a thousand times fainter
    formula:      T(z) = T₀(1+z) — the CMB was 3000 K when it was emitted, and
                  expansion alone cooled it by a factor of 1100.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

H, C, K_B = 6.62607015e-34, 2.99792458e8, 1.380649e-23
T_CMB = 2.72548


def planck_nu(nu, T):
    """Planck's law per unit frequency — the form radio astronomers use."""
    return (2 * H * nu**3 / C**2) / (torch.exp(H * nu / (K_B * T)) - 1)


# --- COBE/FIRAS measured 43 points; these are representative, in units of the peak ---
nu = torch.linspace(60e9, 660e9, 400, dtype=torch.float64)
torch.manual_seed(0)
nu_obs = torch.linspace(68e9, 640e9, 43, dtype=torch.float64)
FIRAS_ERR = 5e-5                      # fractional; FIRAS was good to ~50 ppm
flux_obs = planck_nu(nu_obs, T_CMB) * (
    1 + FIRAS_ERR * torch.randn(len(nu_obs), dtype=torch.float64))

# --- fit the temperature ---
T = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
opt = torch.optim.Adam([T], lr=0.01)
for _ in range(6000):
    opt.zero_grad()
    loss = (((planck_nu(nu_obs, T) - flux_obs) / (FIRAS_ERR * flux_obs)) ** 2).mean()
    loss.backward()
    opt.step()
T_fit = float(T)
resid = ((planck_nu(nu_obs, T_fit) - flux_obs) / flux_obs).abs().max()
print(f"fitted CMB temperature T = {T_fit:.5f} K   (FIRAS: 2.72548 ± 0.00057 K)")
print(f"  max fractional residual of this fit = {resid:.2e}")
print(f"  (that residual is just the {FIRAS_ERR:.0e} measurement noise we simulated —")
print(f"   the fit itself is not the interesting part)")
print()
print(f"  What is interesting is the BOUND. FIRAS constrained any departure from a")
print(f"  Planck curve to |ΔI/I| < 5e-5, and the Compton-y distortion to |y| < 1.5e-5.")
print(f"  No laboratory blackbody has ever been shown to be that pure. Any energy")
print(f"  dumped into the radiation after the first month — decaying particles,")
print(f"  early star formation, a non-thermal origin — would have left a distortion")
print(f"  above that bound. The universe was, and stayed, in thermal equilibrium.\n")

# --- Wien peak and photon count ---
nu_peak = 2.821439 * K_B * T_fit / H
print(f"peak frequency = {nu_peak/1e9:.1f} GHz  (λ = {C/nu_peak*1e3:.2f} mm, microwave)")
n_gamma = 16 * np.pi * 1.20206 * (K_B * T_fit / (H * C)) ** 3
print(f"photon density = {n_gamma/1e6:.0f} per cm³ — there are about 400 CMB photons")
print(f"  in every cubic centimetre of your body, right now.\n")

# --- the redshift history: same photons, stretched ---
print("The CMB cools as T ∝ (1+z), because expansion stretches every wavelength:")
print("    z         T (K)        age            what was happening")
epochs = [(1100, "380,000 yr", "recombination — the universe turns transparent"),
          (100, "18 Myr", "the dark ages"),
          (10, "480 Myr", "the first galaxies (JWST looks here)"),
          (1, "5.9 Gyr", "peak of cosmic star formation"),
          (0.0, "13.8 Gyr", "today")]
for z, age, what in epochs:
    print(f"  {z:6.1f}   {T_CMB*(1+z):9.1f}     {age:12s}  {what}")
print()
print("At z = 1100 the CMB was 3000 K — the temperature at which hydrogen stops")
print("being ionised. That is not a coincidence: the universe became transparent")
print("precisely BECAUSE it cooled through that threshold. Every CMB photon has been")
print("travelling unimpeded ever since. It is the oldest light there is.\n")

# --- the anisotropy hierarchy ---
print("The sky, peeled layer by layer:")
levels = [("uniform glow", 2.725, "the Big Bang was hot and thermal"),
          ("dipole", 3.36e-3, "OUR motion at 370 km/s through the CMB rest frame"),
          ("galactic foreground", 1e-3, "our own Milky Way, must be subtracted"),
          ("primordial anisotropy", 1.8e-5, "quantum fluctuations from inflation —\n"
           "                                    the seeds of every galaxy")]
for name, amp, what in levels:
    print(f"  {name:24s} ΔT ≈ {amp:.2e} K   ({amp/2.725:.1e} relative)")
    print(f"  {'':24s} {what}")
print()
print("That last line is the remarkable one: the 1e-5 ripples in this map grew,")
print("under gravity, into galaxies, stars, planets and the people measuring them.")
print("The CMB is a photograph of the initial conditions of everything.")

# --- visualization ---
fig = plt.figure(figsize=(15, 8.5))
gs = fig.add_gridspec(2, 3)

# Panel 1: the FIRAS spectrum
ax = fig.add_subplot(gs[0, :2])
ax.plot(nu / 1e9, planck_nu(nu, T_fit) * 1e20, color="crimson", lw=2.4,
        label=f"Planck curve, T = {T_fit:.4f} K")
ax.errorbar(nu_obs / 1e9, flux_obs * 1e20, yerr=(FIRAS_ERR * flux_obs * 1e20) * 400,
            fmt="o", color="black", ms=5, capsize=3,
            label="COBE/FIRAS (error bars ×400 to be visible)")
ax.set_xlabel("frequency (GHz)")
ax.set_ylabel("intensity (10⁻²⁰ W m⁻² Hz⁻¹ sr⁻¹)")
ax.set_title("The COBE/FIRAS spectrum, 1990. The error bars have been multiplied by "
             "400 so that\nthey can be seen at all. This is the best fit to a "
             "theoretical curve anywhere in physics.")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel 2: what a non-thermal universe would look like
ax = fig.add_subplot(gs[0, 2])
ax.plot(nu / 1e9, planck_nu(nu, T_fit) / planck_nu(nu, T_fit).max(), color="crimson",
        lw=2.4, label="blackbody (observed)")
mix = (planck_nu(nu, 2.4) + planck_nu(nu, 3.2)) / 2
ax.plot(nu / 1e9, mix / mix.max(), color="steelblue", lw=2, ls="--",
        label="two temperatures mixed")
star = planck_nu(nu, 2.725) * (nu / nu_peak) ** (-0.7)
ax.plot(nu / 1e9, star / star.max(), color="darkgreen", lw=2, ls=":",
        label="starlight thermalised by dust")
ax.set_xlabel("frequency (GHz)")
ax.set_ylabel("normalised intensity")
ax.set_title("Why 'perfect' matters. Any mixture of\nsources, or any late-time energy "
             "injection,\ndistorts the curve. Nothing does.\nSteady-state cosmology "
             "cannot make this.")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel 3: the dipole
ax = fig.add_subplot(gs[1, 0])
th = np.linspace(0, np.pi, 100)
ph = np.linspace(0, 2 * np.pi, 200)
TH, PH = np.meshgrid(th, ph, indexing="ij")
dipole = 3.362e-3 * np.cos(TH)
im = ax.pcolormesh(np.rad2deg(PH), np.rad2deg(TH) - 90, dipole * 1e3,
                   cmap="RdBu_r", shading="auto")
plt.colorbar(im, ax=ax, label="ΔT (mK)")
ax.set_xlabel("longitude (deg)")
ax.set_ylabel("latitude (deg)")
ax.set_title("Layer 1: the dipole, ±3.4 mK.\nThis is not cosmology — it is us, moving\n"
             "at 370 km/s. Subtract it.")

# Panel 4: the primordial anisotropy
ax = fig.add_subplot(gs[1, 1])
rng = np.random.default_rng(2)
n = 220
field = rng.normal(0, 1, (n, 2 * n))
kx = np.fft.fftfreq(n)[:, None]
ky = np.fft.fftfreq(2 * n)[None, :]
k = np.sqrt(kx**2 + ky**2) + 1e-9
# a rough acoustic-peak-like filter so the map has the right visual texture
filt = np.exp(-(k / 0.055) ** 2) * (1 + 0.8 * np.cos(k / 0.011))
smoothed = np.real(np.fft.ifft2(np.fft.fft2(field) * filt))
smoothed = smoothed / smoothed.std() * 1.8e-5
im = ax.imshow(smoothed * 1e6, cmap="RdBu_r", extent=[0, 360, -90, 90], aspect="auto")
plt.colorbar(im, ax=ax, label="ΔT (µK)")
ax.set_xlabel("longitude (deg)")
ax.set_title("Layer 2: the primordial ripples, ±18 µK.\nA thousand times fainter. "
             "These are the\nseeds every galaxy grew from.")

# Panel 5: T(z)
ax = fig.add_subplot(gs[1, 2])
zs = np.logspace(-2, 3.3, 200)
ax.loglog(zs, T_CMB * (1 + zs), color="crimson", lw=2.4, label="T = 2.725(1+z)")
for z, age, what in epochs:
    ax.plot(max(z, 0.01), T_CMB * (1 + z), "o", color="black", ms=7)
ax.axhline(3000, color="darkorange", ls="--", lw=1.8,
           label="3000 K — hydrogen recombines")
ax.axvline(1100, color="darkorange", ls=":", lw=1.5)
ax.set_xlabel("redshift z")
ax.set_ylabel("CMB temperature (K)")
ax.set_title("The universe was 3000 K when this light\nleft. Expansion alone cooled "
             "it by 1100×.\nMeasured directly in distant gas clouds —\nit really "
             "was hotter back then.")
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3, which="both")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
