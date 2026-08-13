"""Dark lines in sunlight: chemistry of a place no one can ever visit.

In 1835 Auguste Comte, founding positivism, chose an example of something humanity
could never possibly know:

    "We shall never be able by any means to study the chemical composition [of the
     stars]... every notion of the true mean temperature of the stars will
     necessarily always be concealed from us."

He was wrong within 25 years, and wrong because of something already sitting in a
drawer. Fraunhofer had mapped 574 dark lines in the solar spectrum in 1814 without
knowing what they were. In 1859 Kirchhoff and Bunsen showed that each line is a
FINGERPRINT: a specific element absorbing specific wavelengths.

Then it got stranger. In 1868 Janssen and Lockyer found a line at 587.6 nm in the
Sun that matched no known element. Lockyer named it after the Sun — helium. It was
not found on Earth until 1895, 27 years later. We discovered an element by looking
at a star.

And the lines were the raw data for quantum mechanics: Balmer's 1885 formula for
the hydrogen lines is the empirical fact that Bohr's 1913 atom was built to explain.

    phenomenon:   the solar spectrum is crossed by hundreds of dark lines
    simulation:   a Planck continuum with Voigt-ish absorption at the real
                  Fraunhofer wavelengths
    dissection:   match observed lines against laboratory wavelengths; find that
                  one line matches nothing on Earth
    formula:      Balmer's  1/λ = R(1/2² − 1/n²) — fit R from the lines alone and
                  land on the Rydberg constant, the seed of quantum theory.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

H, C_LIGHT, K_B = 6.62607015e-34, 2.99792458e8, 1.380649e-23

# --- the real Fraunhofer lines: (letter, wavelength nm, element, depth) ---
LINES = [
    ("A", 759.37, "O₂ (Earth's air!)", 0.55), ("B", 686.72, "O₂ (Earth's air!)", 0.45),
    ("C", 656.28, "H α", 0.60),               ("D₁", 589.59, "Na", 0.75),
    ("D₂", 588.99, "Na", 0.78),               ("D₃", 587.56, "He  ← unknown in 1868", 0.30),
    ("E", 526.96, "Fe", 0.50),                ("F", 486.13, "H β", 0.55),
    ("G", 430.79, "Ca / Fe", 0.65),           ("h", 410.17, "H δ", 0.45),
    ("H", 396.85, "Ca⁺", 0.85),               ("K", 393.37, "Ca⁺", 0.88),
]


def planck(lam_nm, T):
    """Spectral radiance vs wavelength — the continuum the lines are carved out of."""
    lam = lam_nm * 1e-9
    return (2 * H * C_LIGHT**2 / lam**5) / (torch.exp(H * C_LIGHT / (lam * K_B * T)) - 1)


lam = torch.linspace(380, 780, 4000, dtype=torch.float64)
T_SUN = 5772.0
continuum = planck(lam, T_SUN)

# --- carve the absorption lines into it ---
spectrum = continuum.clone()
for _, l0, _, depth in LINES:
    width = 0.9 if l0 > 500 else 1.4          # crude, but the shape is not the point
    spectrum = spectrum * (1 - depth * torch.exp(-((lam - l0) / width) ** 2))

print(f"Solar spectrum modelled as a {T_SUN:.0f} K Planck continuum with "
      f"{len(LINES)} absorption lines.\n")
print("  line    λ (nm)     identified as")
for letter, l0, elem, _ in LINES:
    print(f"   {letter:3s}   {l0:7.2f}     {elem}")
print()
print("Note line A and B: those are absorbed by oxygen in OUR OWN atmosphere, not")
print("the Sun's. Distinguishing the two took decades and is still a live problem")
print("in exoplanet atmosphere work today.\n")

# --- Balmer: the hydrogen lines follow an arithmetic pattern ---
# Observed hydrogen lines in the visible, and their (unknown at the time) quantum numbers
balmer_obs = torch.tensor([656.28, 486.13, 434.05, 410.17, 397.01], dtype=torch.float64)
n_upper = torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0], dtype=torch.float64)

# Balmer (1885) found this by trial and error on four numbers. Fit R with torch.
R = torch.tensor(1.0e7, dtype=torch.float64, requires_grad=True)
opt = torch.optim.Adam([R], lr=1e4)
for _ in range(20000):
    opt.zero_grad()
    pred = 1.0 / (R * (1 / 2**2 - 1 / n_upper**2)) * 1e9      # in nm
    loss = ((pred - balmer_obs) ** 2).mean()
    loss.backward()
    opt.step()
R_fit = R.detach().item()
R_TRUE = 1.0967758e7

print("Balmer's formula, fitted to five hydrogen lines:   1/λ = R(1/2² − 1/n²)")
print(f"  fitted Rydberg constant R = {R_fit:.6e} 1/m")
print(f"  accepted value            R = {R_TRUE:.6e} 1/m   "
      f"(off by {100*abs(R_fit-R_TRUE)/R_TRUE:.3f}%)")
print()
print("  n    observed λ    Balmer λ     residual")
pred_final = (1.0 / (R_fit * (1 / 4 - 1 / n_upper**2)) * 1e9)
for i in range(len(balmer_obs)):
    print(f"  {int(n_upper[i])}    {balmer_obs[i]:8.2f}    {pred_final[i]:8.2f}    "
          f"{pred_final[i]-balmer_obs[i]:+7.3f} nm")
print()
print("Balmer had no idea why this worked. Neither did anyone else for 28 years.")
print("In 1913 Bohr wrote down quantised orbits and derived R from first principles:")
print("      R = m_e·e⁴ / (8·ε₀²·h³·c)")
R_bohr = (9.1093837015e-31 * (1.602176634e-19)**4 /
          (8 * (8.8541878128e-12)**2 * H**3 * C_LIGHT))
print(f"  Bohr's expression evaluates to {R_bohr:.6e} 1/m  — the same number.")
print()
print("Quantum mechanics was not born in a laboratory. It was born from a pattern")
print("in starlight that a Swiss schoolteacher noticed in four numbers.")

# --- visualization ---
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1, 1])


def wav_to_rgb(w):
    """Approximate visible-spectrum colour, for drawing the spectrum strip."""
    if w < 440:   r, g, b = -(w - 440) / 60, 0.0, 1.0
    elif w < 490: r, g, b = 0.0, (w - 440) / 50, 1.0
    elif w < 510: r, g, b = 0.0, 1.0, -(w - 510) / 20
    elif w < 580: r, g, b = (w - 510) / 70, 1.0, 0.0
    elif w < 645: r, g, b = 1.0, -(w - 645) / 65, 0.0
    else:         r, g, b = 1.0, 0.0, 0.0
    f = 0.3 + 0.7 * (1 if 420 <= w <= 700 else
                     (w - 380) / 40 if w < 420 else (780 - w) / 80)
    return np.clip([r * f, g * f, b * f], 0, 1)


# Panel 1: the spectrum strip as the eye would see it
ax = fig.add_subplot(gs[0, :])
img = np.zeros((60, len(lam), 3))
rel = (spectrum / continuum).numpy()
for i, w in enumerate(lam.numpy()):
    img[:, i, :] = wav_to_rgb(w) * rel[i]
ax.imshow(img, extent=[380, 780, 0, 1], aspect="auto", origin="lower")
for letter, l0, elem, _ in LINES:
    if 380 <= l0 <= 780:
        ax.annotate(letter, (l0, 1.02), fontsize=9, ha="center", color="black",
                    weight="bold")
        ax.annotate(elem.split("(")[0].split("←")[0].strip(), (l0, -0.30),
                    fontsize=7, ha="center", rotation=90, color="black")
ax.set_ylim(-0.05, 1.0)
ax.set_yticks([])
ax.set_xlabel("wavelength (nm)")
ax.set_title("Fraunhofer's solar spectrum (1814): 574 dark lines in sunlight.\n"
             "He measured them precisely and had no idea what they were. "
             "Kirchhoff & Bunsen explained them in 1859.")

# Panel 2: the continuum with lines carved out
ax = fig.add_subplot(gs[1, :2])
ax.plot(lam, continuum / continuum.max(), color="darkorange", lw=1.8,
        label=f"{T_SUN:.0f} K blackbody continuum")
ax.plot(lam, spectrum / continuum.max(), color="black", lw=1.0,
        label="observed solar spectrum")
he = [l for l in LINES if "He" in l[2]][0]
ax.annotate("D₃ — helium.\nMatched NO element\nknown on Earth in 1868.\nFound on Earth in 1895.",
            xy=(he[1], (spectrum[torch.argmin((lam - he[1]).abs())] / continuum.max()).item()),
            xytext=(628, 0.40), fontsize=9, color="crimson",
            arrowprops=dict(arrowstyle="->", color="crimson", lw=2))
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("relative intensity")
ax.set_title("The lines are missing energy: atoms in the Sun's cooler outer layers "
             "absorbing\nexactly the wavelengths they would emit if heated.")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel 3: Comte's claim
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
ax.text(0.0, 1.0, "Auguste Comte, 1835", fontsize=12, weight="bold", va="top")
ax.text(0.0, 0.88, (
    '"We shall never be able by any\n'
    ' means to study the chemical\n'
    ' composition of the stars."\n\n'
    "1814  Fraunhofer maps 574 lines\n"
    "      (data with no theory)\n"
    "1859  Kirchhoff & Bunsen: each\n"
    "      line = one element\n"
    "1868  helium found in the Sun,\n"
    "      27 years before Earth\n"
    "1885  Balmer's formula\n"
    "1913  Bohr's atom\n"
    "1925  Payne: stars are mostly\n"
    "      hydrogen (her examiners\n"
    "      made her disown it; she\n"
    "      was right)\n\n"
    "Comte picked the one question\n"
    "that would be answered fastest.\n"
    "The data was already in a drawer."),
    fontsize=8.6, va="top", family="monospace")

# Panel 4: the Balmer series
ax = fig.add_subplot(gs[2, 0])
ns = torch.arange(3, 12, dtype=torch.float64)
lam_b = 1.0 / (R_fit * (1 / 4 - 1 / ns**2)) * 1e9
for i, (nn, ll) in enumerate(zip(ns.tolist(), lam_b.tolist())):
    ax.axvline(ll, color=wav_to_rgb(ll) if 380 <= ll <= 780 else "purple", lw=2.6)
    if i < 4:
        ax.text(ll, 0.9 - 0.07 * i, f"n={int(nn)}", fontsize=8, ha="center")
ax.axvline(364.6, color="black", ls="--", lw=1.4)
ax.text(364.6, 0.35, "series limit\nn → ∞", fontsize=7.5, ha="center", rotation=90)
ax.set_xlim(350, 700)
ax.set_yticks([])
ax.set_xlabel("wavelength (nm)")
ax.set_title("The Balmer series: lines crowd toward\na limit. Nothing in classical "
             "physics\nmakes atoms do this.")

# Panel 5: 1/λ is linear in 1/n² — the discovery
ax = fig.add_subplot(gs[2, 1])
inv_n2 = (1 / n_upper**2).numpy()
inv_lam = (1 / (balmer_obs * 1e-9)).numpy()
ax.plot(inv_n2, inv_lam, "o", color="black", ms=10, label="observed lines")
xs = np.linspace(0, 0.13, 50)
ax.plot(xs, R_fit * (0.25 - xs), "-", color="crimson", lw=2,
        label=f"1/λ = R(1/4 − 1/n²)\nR = {R_fit:.4e} /m")
ax.set_xlabel("1 / n²")
ax.set_ylabel("1 / λ  (1/m)")
ax.set_title("Balmer's insight, 1885: plot 1/λ against\n1/n² and it is a straight line.\n"
             "Take the reciprocal — the ch02 trick again.")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel 6: residuals
ax = fig.add_subplot(gs[2, 2])
resid = (pred_final - balmer_obs).numpy()
ax.bar([f"n={int(x)}" for x in n_upper.tolist()], resid, color="steelblue", alpha=0.9)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("Balmer − observed (nm)")
ax.set_title(f"Residuals below 0.1 nm across the series.\n"
             f"An empirical formula with ONE constant,\n"
             f"28 years before anyone knew why.")
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
