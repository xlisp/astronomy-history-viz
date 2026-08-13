"""Why Galileo stopped his lens down, and why Newton gave up on lenses entirely.

A lens focuses by refraction, and refraction depends on wavelength (see
newton_prism_experiment.py). So a simple lens has a DIFFERENT FOCAL LENGTH FOR
EVERY COLOUR:

    1/f(λ) = (n(λ) − 1)·(1/R₁ − 1/R₂)      ⟹     Δf/f ≈ 1/V

where V is the Abbe number, ~64 for crown glass. A one-metre lens focuses blue
light about 1.6 cm closer than red. There is no focus plane where the image is
sharp; there is only a "circle of least confusion".

This is not a footnote — it dominated observational astronomy for a century and it
shaped what each of these men could see:

  · Galileo (1609) stopped his 37 mm objective down to ~15 mm with a cardboard
    ring. He threw away 84% of his light on purpose, because chromatic blur was
    worse than the loss. Everything he discovered, he discovered through a
    deliberately crippled aperture.
  · Huygens and Hevelius fought it with absurd focal lengths — Huygens' "aerial"
    telescopes had objectives on a mast up to 64 metres from the eyepiece, with no
    tube at all, because Δf/f is fixed but the ANGULAR blur falls as 1/f.
  · Newton (1668), having just proved dispersion was a property of light rather
    than of glass, concluded refractors were hopeless and built a MIRROR. A mirror
    obeys the law of reflection, which contains no n(λ) at all. Zero chromatic
    aberration, exactly, at every wavelength.

    phenomenon:   a bright star through a simple lens has coloured fringes
    simulation:   trace rays of many wavelengths through a thin lens
    dissection:   compute the blur circle and compare it to the diffraction limit
    formula:      blur ≈ D/(2V) at best focus, independent of focal length —
                  so only stopping down, or achromatising, can help.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

CROWN_A, CROWN_B = 1.5046, 4200.0        # Cauchy n(λ) = A + B/λ², λ in nm
FLINT_A, FLINT_B = 1.5892, 10800.0


def n_crown(lam):
    return CROWN_A + CROWN_B / lam**2


def n_flint(lam):
    return FLINT_A + FLINT_B / lam**2


def abbe(nfun):
    nd = float(nfun(torch.tensor(589.3, dtype=torch.float64)))
    nF = float(nfun(torch.tensor(486.1, dtype=torch.float64)))
    nC = float(nfun(torch.tensor(656.3, dtype=torch.float64)))
    return (nd - 1) / (nF - nC), nd


V_crown, nd_crown = abbe(n_crown)
V_flint, nd_flint = abbe(n_flint)
print(f"crown glass: n_d = {nd_crown:.4f},  Abbe V = {V_crown:.1f}")
print(f"flint glass: n_d = {nd_flint:.4f},  Abbe V = {V_flint:.1f}")
print(f"  flint disperses {V_crown/V_flint:.2f}× more strongly per unit refraction —")
print(f"  that ratio is the whole reason an achromatic doublet can exist.\n")

# --- focal length as a function of colour, for a simple equiconvex lens ---
F_NOMINAL, D_LENS = 1000.0, 50.0            # mm
R = 2 * (nd_crown - 1) * F_NOMINAL          # equiconvex: 1/f = 2(n−1)/R

lam = torch.linspace(400, 700, 300, dtype=torch.float64)
f_lam = R / (2 * (n_crown(lam) - 1))

print("A simple 50 mm f/20 crown-glass objective:")
print("  colour      λ (nm)     focal length (mm)")
for name, l0 in [("violet", 420), ("blue", 486), ("yellow", 589), ("red", 656),
                 ("deep red", 700)]:
    fl = float(R / (2 * (n_crown(torch.tensor(float(l0), dtype=torch.float64)) - 1)))
    print(f"  {name:10s}  {l0:5d}      {fl:9.2f}")
spread_f = float(f_lam.max() - f_lam.min())
# The Abbe number is DEFINED over the F (486.1 nm) to C (656.3 nm) interval, so that
# is the range in which Δf/f = 1/V holds. Over the wider 400–700 nm band the spread
# is about twice as large, because n(λ) curves upward at the violet end.
f_F = float(R / (2 * (n_crown(torch.tensor(486.1, dtype=torch.float64)) - 1)))
f_C = float(R / (2 * (n_crown(torch.tensor(656.3, dtype=torch.float64)) - 1)))
print(f"\n  Δf over the F–C interval (486–656 nm) = {f_C - f_F:.2f} mm")
print(f"    Δf/f = {(f_C - f_F)/F_NOMINAL:.4f}   vs   1/V = {1/V_crown:.4f}   ✓ "
      f"this is what 1/V means")
print(f"  Δf over the full visible (400–700 nm)  = {spread_f:.2f} mm  "
      f"(Δf/f = {spread_f/F_NOMINAL:.4f})")
print(f"    — {spread_f/(f_C-f_F):.1f}× worse, because n(λ) turns up steeply in the violet.\n")

# --- the achromat: cancel dispersion between two glasses ---
# The achromatic condition is on POWER, not focal length:  P₁/V₁ + P₂/V₂ = 0
# with P = 1/f. So f₂ = −f₁·V₁/V₂. The low-dispersion crown does the converging
# work; the high-dispersion flint is a weaker diverging element whose colour error
# exactly cancels the crown's.
f1 = 500.0
f2 = -f1 * V_crown / V_flint
f_combined = 1.0 / (1.0 / f1 + 1.0 / f2)
# Thin-lens powers: 1/f_i = (n_i − 1)·K_i, so K_i is fixed by the shape alone.
K1 = 1.0 / (f1 * (nd_crown - 1))
K2 = 1.0 / (f2 * (nd_flint - 1))


def f_doublet(lam):
    """Focal length of the cemented pair, computed from the two glasses' n(λ)."""
    return 1.0 / ((n_crown(lam) - 1) * K1 + (n_flint(lam) - 1) * K2)


f_dbl = f_doublet(lam)
f_dbl_full = float(f_dbl.max() - f_dbl.min())

print("The fix Newton thought impossible (Hall 1733, Dollond 1758):")
print(f"  crown lens f₁ = {f1:.1f} mm (V={V_crown:.1f}) + flint lens f₂ = {f2:.1f} mm "
      f"(V={V_flint:.1f})")
print(f"  achromatic condition (1/f₁)/V₁ + (1/f₂)/V₂ = "
      f"{(1/f1)/V_crown + (1/f2)/V_flint:.3e}  ≈ 0  ✓")
print(f"  combined focal length = {f_combined:.1f} mm, still converging")
print(f"  residual spread over the whole visible = {f_dbl_full:.3e} mm")
print(f"  (a single crown lens of the same focal length: "
      f"{f_combined*spread_f/F_NOMINAL:.2f} mm)\n")
print("  That residual is EXACTLY zero, and that is a property of our model rather")
print("  than of real glass. Both Cauchy fits here have the single form A + B/λ², so")
print("  the two glasses' dispersions are proportional to the same function of λ;")
print("  cancelling them at two wavelengths cancels them at every wavelength. Real")
print("  glasses need a third term (…+ C/λ⁴) whose shapes do NOT match, and the")
print("  leftover is called the SECONDARY SPECTRUM — about f/2200 for a classical")
print("  crown-flint pair, i.e. ~440× better than the f/64 of a singlet but not")
print("  perfect. Killing it needs exotic glasses (fluorite, ED) — which is why a")
print("  good apochromat still costs more than the telescope around it.\n")

V_ACHROMAT = 2200.0        # empirical secondary-spectrum limit of a crown-flint pair

# --- the blur circle, and the comparison that decided the history ---
def blur_angular_arcsec(D_mm, V, f_mm):
    """Angular diameter of the circle of least confusion: D/(2V)/f radians."""
    return (D_mm / (2 * V)) / f_mm * 206265


def diffraction_arcsec(D_mm, lam_nm=550.0):
    return 1.22 * (lam_nm * 1e-6) / D_mm * 206265


print("The decision every 17th-century observer faced, in numbers:")
print("  instrument                    D (mm)   f (mm)   chromatic   diffraction   ratio")
cases = [
    ("Galileo 1609, full aperture", 37.0, 1330.0),
    ("Galileo 1609, stopped to 15mm", 15.0, 1330.0),
    ("Huygens 1686, aerial 'tubeless'", 190.0, 64000.0),
    ("Hevelius 1673, 140 ft tube", 200.0, 42700.0),
    ("Dollond-type achromat", 100.0, 1000.0),
]
for name, D, f in cases:
    V_use = V_ACHROMAT if "achromat" in name else V_crown
    ch = blur_angular_arcsec(D, V_use, f)
    df = diffraction_arcsec(D)
    print(f"  {name:31s} {D:6.1f}  {f:7.0f}   {ch:8.2f}\"   {df:9.2f}\"   {ch/df:6.1f}×")
print()
print("Read the Galileo rows together. At full 37 mm aperture the colour blur is")
print(f"{blur_angular_arcsec(37, V_crown, 1330)/diffraction_arcsec(37):.0f}× the diffraction limit — the lens is nowhere near its own")
print("theoretical resolution. Stopping down to 15 mm loses 84% of the light but")
print(f"cuts the blur to {blur_angular_arcsec(15, V_crown, 1330):.1f}\", only "
      f"{blur_angular_arcsec(15, V_crown, 1330)/diffraction_arcsec(15):.1f}× diffraction.")
print("Galileo could not have explained why. He found it by trial and error, at the")
print("bench, and it is the reason he could resolve Jupiter's moons at all.\n")

print("Huygens' answer was the opposite: keep the aperture, make f enormous.")
print("  Blur ANGLE = D/(2Vf), so a 64 m focal length beats the colour down by brute")
print("  force. The cost was a 190 mm lens dangling from a mast, aimed at a lens on")
print("  the ground by a taut string, in the dark. It worked. Huygens found Titan")
print("  (1655) and resolved Saturn's rings (1656) this way.\n")

print("But Newton's instinct was still vindicated at scale, for a reason about")
print("MATERIALS rather than optics:")
print("  · a lens is supported only at its rim, so it sags under its own weight;")
print("    a mirror is supported across its whole back")
print("  · light must pass THROUGH a lens, so the glass must be flawless all the")
print("    way through; only a mirror's front surface matters")
print("  · a doublet has 4 surfaces to polish, a mirror has 1")
print("  The largest refractor ever built is the 40-inch Yerkes (1897). Every")
print("  telescope larger than that, for 128 years and counting, has been a mirror.")

# --- visualization ---
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])


def wav_rgb(w):
    if w < 440:   r, g, b = -(w - 440) / 60, 0.0, 1.0
    elif w < 490: r, g, b = 0.0, (w - 440) / 50, 1.0
    elif w < 510: r, g, b = 0.0, 1.0, -(w - 510) / 20
    elif w < 580: r, g, b = (w - 510) / 70, 1.0, 0.0
    elif w < 645: r, g, b = 1.0, -(w - 645) / 65, 0.0
    else:         r, g, b = 1.0, 0.0, 0.0
    return np.clip([r, g, b], 0, 1)


# Panel 1: rays through a simple lens, colour by colour
ax = fig.add_subplot(gs[0, :2])
h = D_LENS / 2
for l0 in range(410, 700, 10):
    fl = float(R / (2 * (n_crown(torch.tensor(float(l0), dtype=torch.float64)) - 1)))
    for y0 in (h, h / 2, -h / 2, -h):
        ax.plot([0, fl * 1.35], [y0, y0 - y0 * 1.35], color=wav_rgb(l0), lw=0.8, alpha=0.55)
    ax.plot([-260, 0], [h, h], color=wav_rgb(l0), lw=0.0)
ax.plot([-260, 0], [h, h], "k-", lw=1.2)
ax.plot([-260, 0], [-h, -h], "k-", lw=1.2)
ax.plot([-260, 0], [0, 0], "k-", lw=1.2)
ax.add_patch(plt.matplotlib.patches.Ellipse((0, 0), 22, D_LENS, color="lightsteelblue",
                                            ec="steelblue", lw=1.5, zorder=5))
f_blue = float(R / (2 * (n_crown(torch.tensor(430.0, dtype=torch.float64)) - 1)))
f_red = float(R / (2 * (n_crown(torch.tensor(680.0, dtype=torch.float64)) - 1)))
ax.axvline(f_blue, color="blue", ls=":", lw=1.5)
ax.axvline(f_red, color="red", ls=":", lw=1.5)
ax.annotate("", xy=(f_blue, -22), xytext=(f_red, -22),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.8))
ax.text((f_blue + f_red) / 2, -27, f"Δf = {f_red-f_blue:.1f} mm", ha="center", fontsize=9)
ax.text(f_blue, 27, "blue focus", fontsize=8, color="blue", ha="center")
ax.text(f_red, 22, "red focus", fontsize=8, color="red", ha="center")
ax.set_xlim(-260, 1150)
ax.set_ylim(-32, 32)
ax.set_xlabel("distance along the axis (mm)")
ax.set_title("A simple lens has no focal plane — it has a focal RANGE.\n"
             "Blue crosses the axis 16 mm before red, and there is no place to put "
             "the eyepiece where the star is a point.")
ax.set_yticks([])

# Panel 2: focal length vs wavelength
ax = fig.add_subplot(gs[0, 2])
ax.plot(lam, f_lam, color="black", lw=2.4)
for l0 in range(410, 700, 10):
    fl = float(R / (2 * (n_crown(torch.tensor(float(l0), dtype=torch.float64)) - 1)))
    ax.plot(l0, fl, "o", color=wav_rgb(l0), ms=6)
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("focal length (mm)")
ax.set_title(f"Δf/f = {spread_f/F_NOMINAL:.4f} ≈ 1/V.\n"
             "This ratio is a property of the GLASS.\nIt does not improve with "
             "better polishing.")
ax.grid(alpha=0.3)

# Panel 3: the trade Galileo actually made
ax = fig.add_subplot(gs[1, 0])
Ds = np.linspace(5, 60, 200)
ax.plot(Ds, blur_angular_arcsec(Ds, V_crown, 1330), color="crimson", lw=2.4,
        label="chromatic blur ∝ D")
ax.plot(Ds, diffraction_arcsec(Ds), color="steelblue", lw=2.4,
        label="diffraction limit ∝ 1/D")
total = np.sqrt(blur_angular_arcsec(Ds, V_crown, 1330)**2 + diffraction_arcsec(Ds)**2)
ax.plot(Ds, total, color="black", lw=2, ls="--", label="quadrature sum")
best = Ds[int(np.argmin(total))]
ax.axvline(best, color="darkgreen", ls=":", lw=2,
           label=f"optimum D = {best:.0f} mm")
ax.axvline(37, color="gray", ls="-", lw=1.5)
ax.text(37.5, 40, "Galileo's\nfull lens", fontsize=8, color="gray")
ax.set_xlabel("aperture D (mm)")
ax.set_ylabel("blur (arcsec)")
ax.set_yscale("log")
ax.set_title("Why stopping down HELPED.\nChromatic blur grows with aperture while\n"
             "diffraction shrinks — so there is an optimum,\nand it is far below "
             "the lens you own.")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

# Panel 4: the mirror has no such curve
ax = fig.add_subplot(gs[1, 1])
ax.plot(lam, f_lam / F_NOMINAL, color="crimson", lw=2.6, label="lens: f varies with λ")
ax.plot(lam, np.ones_like(lam), color="steelblue", lw=2.6,
        label="mirror: f is the same for every λ")
ax.plot(lam, (f_dbl / f_doublet(torch.tensor(589.3, dtype=torch.float64))).numpy(),
        color="darkgreen", lw=2.2, ls="--",
        label="achromatic doublet (2-term model: exact)")
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("f(λ) / f(589 nm)")
ax.set_title("The law of reflection contains no n(λ).\n"
             "Newton's reflector solved the problem\nEXACTLY, not approximately.")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel 5: what each man built
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
ax.text(0.0, 1.0, "Three responses to one law", fontsize=11.5, weight="bold", va="top")
ax.text(0.0, 0.88, (
    "GALILEO (1609) — stop it down\n"
    "  37 mm lens masked to ~15 mm.\n"
    "  Threw away 84% of the light.\n"
    "  Found: Jupiter's moons, Venus'\n"
    "  phases, lunar mountains.\n\n"
    "HUYGENS (1650s) — stretch it out\n"
    "  Blur angle = D/(2Vf), so make f\n"
    "  enormous: objectives on a 64 m\n"
    "  mast, no tube at all.\n"
    "  Found: Titan, Saturn's rings.\n\n"
    "NEWTON (1668) — abandon glass\n"
    "  Ground and polished his own\n"
    "  speculum-metal mirror, 33 mm.\n"
    "  Chromatic aberration: exactly 0.\n"
    "  Every large telescope since is\n"
    "  a descendant.\n\n"
    "All three were hands-on opticians.\n"
    "None of these is a theoretical fix —\n"
    "they are three things to DO at a\n"
    "workbench, and the theory came after."),
    fontsize=8.5, va="top", family="monospace")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"\nSaved: {out}")
