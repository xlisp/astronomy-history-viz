"""1666: Newton buys a prism at a country fair and takes white light apart.

Newton was, before anything else, a man who made things with his hands. He ground
his own lenses, cast and polished his own speculum-metal mirrors, built his own
furnace, and blew his own glass. The Principia came out of that workshop as much
as out of the study — and so did this experiment, the one that started
spectroscopy and therefore, eventually, all of ch06.

The received view in 1666 (Aristotle, refined by Descartes) was the MODIFICATION
theory: white light is pure and simple, and a prism *adds* colour to it, the way
dye stains cloth. The colours were held to be a corruption introduced by the glass.

Newton's answer was not an argument. It was a sequence of experiments, ending in
what he called the *experimentum crucis*:

    1. A prism spreads sunlight into an elongated band of colours.
       (Modification theory can live with this.)
    2. Isolate ONE colour with a screen and pass it through a SECOND prism.
       It bends further — but does not change colour, and does not spread.
       ← This kills modification theory. A second helping of glass adds no
         further "corruption", so the glass was never adding anything.
    3. Recombine the whole fan with an inverted prism → white light returns.

Conclusion: white light is a MIXTURE, and each colour has its own fixed
refrangibility. The prism sorts; it does not stain.

    phenomenon:   sunlight through glass makes an elongated coloured band
    simulation:   trace rays through a real prism with real dispersion n(λ)
    dissection:   the deviation angle depends on λ only through n(λ)
    formula:      Snell's law twice; δ = θ₁ + θ₄ − A; at minimum deviation
                  n = sin((A+δ_min)/2) / sin(A/2), which is how you MEASURE n.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# --- Cauchy dispersion, n(λ) = A + B/λ² (λ in nm). Two of Newton's era's glasses. ---
GLASSES = {"crown (BK7-like)": (1.5046, 4200.0), "flint (F2-like)": (1.5892, 10800.0)}
APEX = np.deg2rad(60.0)                      # equilateral prism, as Newton's was


def n_of(lam_nm, glass="crown (BK7-like)"):
    A, B = GLASSES[glass]
    return A + B / lam_nm**2


def deviation(theta1, lam_nm, glass="crown (BK7-like)", apex=APEX):
    """Total deviation of a ray entering at incidence theta1. Snell's law, twice."""
    n = n_of(lam_nm, glass)
    theta2 = torch.asin(torch.sin(theta1) / n)          # into the glass
    theta3 = apex - theta2                              # at the second face, inside
    s = n * torch.sin(theta3)
    theta4 = torch.asin(torch.clamp(s, -1.0, 1.0))      # back out into air
    dev = theta1 + theta4 - apex
    return torch.where(s.abs() <= 1.0, dev, torch.full_like(dev, float("nan")))


lam = torch.linspace(400, 700, 400, dtype=torch.float64)

# --- minimum deviation: the configuration Newton used to measure refrangibility ---
# At minimum deviation the ray passes symmetrically, theta1 = theta4. Find it with
# autograd rather than by solving the trigonometry by hand.
def min_deviation(lam_nm, glass="crown (BK7-like)"):
    th = torch.tensor(0.9, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([th], lr=0.01)
    for _ in range(2000):
        opt.zero_grad()
        deviation(th, lam_nm, glass).backward()
        opt.step()
    return float(deviation(th.detach(), lam_nm, glass)), float(th.detach())


print("Newton's measurement, reproduced: deviation at minimum, per colour")
print("  colour        λ (nm)    n(λ)      δ_min      n from δ_min")
colours = [("violet", 420), ("blue", 470), ("green", 530), ("yellow", 589),
           ("orange", 620), ("red", 680)]
dmins = []
for name, l0 in colours:
    lt = torch.tensor(float(l0), dtype=torch.float64)
    dmin, th1 = min_deviation(lt)
    dmins.append(dmin)
    # invert the classic relation — this is how refractive index is measured to this day
    n_meas = np.sin((APEX + dmin) / 2) / np.sin(APEX / 2)
    print(f"  {name:9s}  {l0:6d}   {float(n_of(lt)):.5f}   {np.rad2deg(dmin):7.3f}°   "
          f"{n_meas:.5f}")

spread = np.rad2deg(dmins[0] - dmins[-1])
print(f"\n  violet is deviated {spread:.3f}° more than red.")
print(f"  Newton projected onto a wall 22 feet (6.7 m) away, so the band was")
print(f"  {6.7*np.tan(np.deg2rad(spread))*39.37:.1f} inches long at minimum deviation.")
print(f"  He actually reported ~13 inches — he was not at minimum deviation, which")
print(f"  makes the dispersion larger. Either way: a round hole made an OBLONG image.")
print(f"  That elongation is the whole discovery. A 'pure' light would stay round.\n")

# --- the experimentum crucis, quantified ---
print("THE EXPERIMENTUM CRUCIS")
print("  Take one colour out of the fan with a slit and send it through a SECOND prism.")
print("  The two theories make quantitatively different predictions:")
print("    modification: more glass ⇒ more corruption ⇒ spread again, by a similar")
print("                  amount, REGARDLESS of how narrow you make the slit")
print("    Newton:       the second prism spreads the beam only by |dδ/dλ|·Δλ, so")
print("                  narrowing the slit drives the new spread to ZERO\n")

# The angular dispersion dδ/dλ, from autograd rather than from differentiating by hand.
lam_g = torch.tensor(530.0, dtype=torch.float64, requires_grad=True)
dev_g = deviation(torch.tensor(0.9, dtype=torch.float64), lam_g)
(ddev_dlam,) = torch.autograd.grad(dev_g, lam_g)
disp_deg_per_nm = abs(float(ddev_dlam)) * 180 / np.pi
full_fan = abs(float(deviation(torch.tensor(0.9, dtype=torch.float64),
                               torch.tensor(400.0, dtype=torch.float64))
                     - deviation(torch.tensor(0.9, dtype=torch.float64),
                                 torch.tensor(700.0, dtype=torch.float64)))) * 180 / np.pi

print(f"  angular dispersion at 530 nm (autograd)  |dδ/dλ| = {disp_deg_per_nm:.5f} °/nm")
print(f"  spread of the full 400–700 nm fan after prism 1   = {full_fan:.3f}°\n")
print("   slit width Δλ    spread after prism 2    as a fraction of the original fan")
for dlam in (100.0, 30.0, 10.0, 3.0, 1.0):
    s2 = 2 * disp_deg_per_nm * dlam
    print(f"     ±{dlam:5.1f} nm        {s2:8.4f}°              {s2/full_fan:8.4f}")
print()
print("  The spread collapses in proportion to the slit width and nothing else.")
print("  Newton could narrow the slit until the second prism's fan was invisible,")
print("  while the beam still bent by a full 39°. Bending without spreading is")
print("  exactly what 'the glass adds nothing' looks like in numbers.")
print("  Newton: 'Light itself is a Heterogeneous mixture of differently refrangible")
print("  Rays.' (Opticks, Book I)\n")

# --- why this doomed the refractor, in Newton's judgement ---
print("The consequence Newton drew — and got half wrong:")
disp = float(n_of(torch.tensor(486.1, dtype=torch.float64))
             - n_of(torch.tensor(656.3, dtype=torch.float64)))
nd = float(n_of(torch.tensor(589.3, dtype=torch.float64)))
abbe = (nd - 1) / disp
print(f"  crown glass: n_d = {nd:.4f}, Abbe number V = (n_d−1)/(n_F−n_C) = {abbe:.1f}")
print(f"  A single lens therefore has a focal length that varies by ~1/V = "
      f"{100/abbe:.1f}% across the spectrum.")
print("  Newton concluded that refracting telescopes were fundamentally hopeless and")
print("  built a REFLECTOR instead (1668) — see chromatic_aberration.py.")
print("  He was wrong that the problem was unfixable: combining crown and flint")
print("  glass cancels the dispersion (Chester Moore Hall 1733, John Dollond 1758).")
print("  But he was right that mirrors would win at large aperture, for a reason he")
print("  could not have known: you cannot cast a 5-metre lens that does not sag.")

# --- visualization ---
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])


def wav_rgb(w):
    if w < 440:   r, g, b = -(w - 440) / 60, 0.0, 1.0
    elif w < 490: r, g, b = 0.0, (w - 440) / 50, 1.0
    elif w < 510: r, g, b = 0.0, 1.0, -(w - 510) / 20
    elif w < 580: r, g, b = (w - 510) / 70, 1.0, 0.0
    elif w < 645: r, g, b = 1.0, -(w - 645) / 65, 0.0
    else:         r, g, b = 1.0, 0.0, 0.0
    return np.clip([r, g, b], 0, 1)


def refract(d, nrm, eta):
    """Vector form of Snell's law. Returns None on total internal reflection."""
    ci = -np.dot(d, nrm)
    s2 = eta**2 * (1 - ci**2)
    if s2 > 1.0:
        return None
    return eta * d + (eta * ci - np.sqrt(1 - s2)) * nrm


def trace(lam_nm, y_in, glass="crown (BK7-like)"):
    """Trace one ray through an equilateral prism; returns the polyline."""
    Av, Bv, Cv = np.array([0, 0.866]), np.array([-0.5, 0.0]), np.array([0.5, 0.0])
    n = float(n_of(torch.tensor(float(lam_nm)), glass))
    d = np.array([1.0, 0.0])
    p0 = np.array([-1.6, y_in])
    # left face A→B, outward normal
    fdir = (Bv - Av) / np.linalg.norm(Bv - Av)
    nrm = np.array([fdir[1], -fdir[0]])
    if np.dot(nrm, np.array([-1.0, 0.0])) < 0:
        nrm = -nrm
    t = np.dot(Av - p0, nrm) / np.dot(d, nrm)
    p1 = p0 + t * d
    d1 = refract(d, nrm, 1.0 / n)
    if d1 is None:
        return None
    # right face A→C, outward normal
    fdir2 = (Cv - Av) / np.linalg.norm(Cv - Av)
    nrm2 = np.array([fdir2[1], -fdir2[0]])
    if np.dot(nrm2, np.array([1.0, 0.0])) < 0:
        nrm2 = -nrm2
    t2 = np.dot(Av - p1, nrm2) / np.dot(d1, nrm2)
    p2 = p1 + t2 * d1
    d2 = refract(d1, -nrm2 if np.dot(d1, nrm2) < 0 else nrm2, n)
    if d2 is None:
        return None
    p3 = p2 + 2.6 * d2
    return np.array([p0, p1, p2, p3])


# Panel 1: the prism fanning white light out
ax = fig.add_subplot(gs[0, :2])
tri = np.array([[0, 0.866], [-0.5, 0], [0.5, 0], [0, 0.866]])
ax.fill(tri[:, 0], tri[:, 1], color="lightsteelblue", alpha=0.55, zorder=1)
ax.plot(tri[:, 0], tri[:, 1], color="steelblue", lw=1.6, zorder=2)
for l0 in range(410, 700, 8):
    path = trace(l0, 0.42)
    if path is not None:
        ax.plot(path[1:, 0], path[1:, 1], color=wav_rgb(l0), lw=1.4, alpha=0.85, zorder=3)
path = trace(550, 0.42)
ax.plot(path[:2, 0], path[:2, 1], color="black", lw=2.4, zorder=4)
ax.text(-1.5, 0.50, "white\nsunlight", fontsize=10, weight="bold")
ax.text(1.35, -0.30, "an OBLONG band,\nnot a round spot", fontsize=10, color="crimson")
ax.set_aspect("equal")
ax.set_xlim(-1.8, 2.6)
ax.set_ylim(-0.9, 1.15)
ax.axis("off")
ax.set_title("Newton's prism, traced with real dispersion n(λ) = A + B/λ².\n"
             "A round hole in the shutter produced an oblong image — the fact that "
             "started everything.")

# Panel 2: deviation vs wavelength
ax = fig.add_subplot(gs[0, 2])
th_fixed = torch.tensor(0.9, dtype=torch.float64)
for glass, style in [("crown (BK7-like)", "-"), ("flint (F2-like)", "--")]:
    dev = np.rad2deg(deviation(th_fixed, lam, glass).numpy())
    ax.plot(lam, dev, style, lw=2.2, color="black" if "crown" in glass else "gray",
            label=glass)
for name, l0 in colours:
    ax.axvline(l0, color=wav_rgb(l0), lw=3, alpha=0.4)
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("deviation (deg)")
ax.set_title("Deviation is a function of λ alone.\nFlint glass disperses ~2.5× more "
             "than crown —\nwhich is exactly what makes an\nachromatic doublet possible.")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel 3: the experimentum crucis
ax = fig.add_subplot(gs[1, 0])
ax.axis("off")
ax.set_title("The experimentum crucis (1666)", fontsize=11, weight="bold")
ax.text(0.0, 0.90, (
    " white ──▶ [prism 1] ──▶ fan of colours\n"
    "                             │\n"
    "                        screen with a\n"
    "                        small hole: keep\n"
    "                        ONLY the green\n"
    "                             │\n"
    "   green ──▶ [prism 2] ──▶ green\n"
    "                        bends further,\n"
    "                        spreads NOT AT ALL\n\n"
    "Modification theory predicted more\n"
    "colour from more glass. It got none.\n\n"
    "Therefore the prism never added colour.\n"
    "It only ever SORTED what was there.\n\n"
    "Newton also recombined the full fan with\n"
    "a second, inverted prism and recovered\n"
    "white light — the reverse operation,\n"
    "which no 'corruption' theory can explain."),
    fontsize=8.8, va="top", family="monospace")

# Panel 4: recombination
ax = fig.add_subplot(gs[1, 1])
xs = np.linspace(0, 1, 300)
for i, l0 in enumerate(range(410, 700, 6)):
    y = 0.5 + 0.32 * np.sin(np.pi * xs) * ((l0 - 550) / 150)
    ax.plot(xs, y, color=wav_rgb(l0), lw=1.3, alpha=0.8)
ax.fill([0.12, 0.24, 0.24], [0.85, 0.85, 0.2], color="lightsteelblue", alpha=0.7)
ax.fill([0.88, 0.76, 0.76], [0.85, 0.85, 0.2], color="lightsteelblue", alpha=0.7)
ax.plot([-0.15, 0], [0.5, 0.5], color="black", lw=3)
ax.plot([1.0, 1.15], [0.5, 0.5], color="black", lw=3)
ax.text(-0.13, 0.57, "white in", fontsize=9)
ax.text(1.0, 0.57, "white out", fontsize=9)
ax.text(0.5, 0.05, "separated in between", fontsize=9, ha="center", style="italic")
ax.set_xlim(-0.2, 1.25); ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title("Split, then recombine.\nThe operation is reversible, so nothing\n"
             "was created or destroyed — only sorted.")

# Panel 5: the lineage
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
ax.text(0.0, 1.0, "What this prism became", fontsize=11.5, weight="bold", va="top")
ax.text(0.0, 0.88, (
    "1666  Newton: white light is a mixture\n"
    "        ↓\n"
    "1668  he builds the first reflector to\n"
    "      escape the dispersion he found\n"
    "        ↓\n"
    "1802  Wollaston sees dark lines\n"
    "1814  Fraunhofer maps 574 of them\n"
    "        ↓\n"
    "1859  Kirchhoff & Bunsen: each line is\n"
    "      an ELEMENT          → ch06\n"
    "1868  helium found in the Sun\n"
    "1885  Balmer's formula\n"
    "1913  Bohr's atom → quantum mechanics\n"
    "1929  Hubble's redshifts → ch07\n\n"
    "Every spectrum ever taken is a\n"
    "descendant of a prism bought at\n"
    "Stourbridge Fair for a few shillings."),
    fontsize=8.6, va="top", family="monospace")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"\nSaved: {out}")
