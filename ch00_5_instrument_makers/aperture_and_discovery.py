"""Aperture is destiny: what each generation could discover was set by what it could grind.

Two things scale with the diameter D of a telescope's objective, and between them
they decide what is discoverable:

    light gathered      ∝ D²        → how FAINT an object you can reach
    angular resolution  θ = 1.22λ/D → how FINE a detail you can separate

Both improve monotonically with D, and D is limited by nothing except the ability
to cast, grind, polish and support a large piece of glass or metal. So the history
of astronomical discovery is, to an uncomfortable degree, the history of a
manufacturing craft.

But aperture is only half of it. A second, independent gate is the DETECTOR: the
eye integrates for about a tenth of a second and then discards the photons, while a
photographic plate integrates for hours. That is worth roughly five magnitudes, the
same as a 10× increase in aperture, and it arrived in the 1880s.

The sharpest case in this project is Hubble's 1924 proof that Andromeda is a
separate galaxy (ch07), which needed Cepheids at magnitude ~19 measured repeatedly
for a light curve. Kant had proposed 'island universes' in 1755. The hypothesis
waited 169 years, and what it waited for was not an idea — it was glass, silver
and emulsion.

    phenomenon:   bigger telescopes see fainter and finer
    simulation:   Airy patterns for the same double star at three apertures
    dissection:   limiting magnitude and Rayleigh resolution vs D, for eye and plate
    formula:      θ = 1.22λ/D and m_lim = 6 + 2.5·log₁₀((D/7 mm)²) [+5 on a plate],
                  then check which discovery each combination unlocks.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import j1

LAMBDA = 550e-9            # m
EYE_PUPIL = 7e-3           # m

# (year, name, aperture in m, what it made possible)
SCOPES = [
    (1609, "Galileo's best", 0.037, "Jupiter's moons, Venus' phases, lunar mountains"),
    (1671, "Newton's reflector", 0.033, "proof a mirror works — no colour at all"),
    (1673, "Hevelius 140-ft", 0.200, "the last great aerial refractor"),
    (1789, "Herschel 40-foot", 1.220, "Uranus' moons; 2500 nebulae catalogued"),
    (1845, "Rosse 'Leviathan'", 1.830, "SPIRAL structure in M51 — nebulae have shape"),
    (1897, "Yerkes 40-inch", 1.020, "the largest refractor ever built, and the last"),
    (1917, "Hooker 100-inch", 2.540, "Cepheids in Andromeda (1924); v = H₀d (1929)"),
    (1948, "Hale 200-inch", 5.080, "Baade's two stellar populations → age of universe"),
    (1993, "Keck I", 10.000, "high-z galaxies; first exoplanet host spectroscopy"),
    (2021, "JWST", 6.500, "galaxies at z > 13, only 300 Myr after the Big Bang"),
    (2029, "ELT (under construction)", 39.300, "Earth-like exoplanet atmospheres"),
]


def rayleigh_arcsec(D_m, lam=LAMBDA):
    return 1.22 * lam / D_m * 206265


# A photographic plate integrates for hours; the eye integrates for ~0.1 s and then
# throws the photons away. That difference is worth roughly 5 magnitudes on early
# 20th-century plates, and it is a SECOND instrument gate, independent of aperture.
PHOTOGRAPHIC_GAIN = 5.0
PHOTOGRAPHY_FROM = 1880          # when plates became practical for faint astronomy


def limiting_mag(D_m, photographic=False):
    """Naked eye reaches ~6th magnitude with a 7 mm pupil; add 2.5·log₁₀ of the area."""
    m = 6.0 + 2.5 * np.log10((D_m / EYE_PUPIL) ** 2)
    return m + PHOTOGRAPHIC_GAIN if photographic else m


print("  year  instrument                D (m)   resolution   light grasp   m_lim   m_lim")
print("                                          (arcsec)     (× naked eye)  visual  photo")
for yr, name, D, what in SCOPES:
    photo = yr >= PHOTOGRAPHY_FROM
    mp = f"{limiting_mag(D, True):5.1f}" if photo else "   — "
    print(f"  {yr}  {name:24s} {D:6.2f}   {rayleigh_arcsec(D):9.4f}   "
          f"{(D/EYE_PUPIL)**2:11.0f}   {limiting_mag(D):5.1f}   {mp}")
print()
print("There are TWO instrument gates here, not one. Aperture buys D² photons per")
print("second; the photographic plate (practical for faint work from the 1880s) buys")
print("hours of integration where the eye buys a tenth of a second. Each is worth")
print("about five magnitudes, and a discovery needs both.\n")
print("Note 1897: the Yerkes 40-inch is the largest refractor ever built, and nothing")
print("larger was ever attempted. A lens must be supported at its rim and light must")
print("pass through it, so it sags and it must be flawless throughout. Every telescope")
print("after it is a mirror — Newton's 1668 judgement, vindicated 229 years later.\n")

# --- the specific gate: could you have found Hubble's Cepheids? ---
M_CEPHEID_ANDROMEDA = 19.0
print("THE DECISIVE CASE — Hubble, Andromeda, 1924")
print(f"  A Cepheid in M31 sits at apparent magnitude ~{M_CEPHEID_ANDROMEDA:.0f}, and")
print("  Hubble had to do more than glimpse it once: he needed a light curve, so")
print("  dozens of plates each reaching several magnitudes PAST detection, against")
print("  the bright unresolved background of the galaxy itself.\n")
print("  instrument                D (m)   detector      m_lim    reach m=19?")
for yr, name, D, what in SCOPES:
    photo = yr >= PHOTOGRAPHY_FROM
    ml = limiting_mag(D, photo)
    det = "plate" if photo else "eye"
    verdict = "YES" if ml >= M_CEPHEID_ANDROMEDA else "no"
    margin = f"  (+{ml-M_CEPHEID_ANDROMEDA:.1f} mag of headroom)" if ml >= M_CEPHEID_ANDROMEDA else ""
    print(f"  {name:24s} {D:6.2f}   {det:9s} {ml:7.1f}     {verdict}{margin}")

D_vis = EYE_PUPIL * 10 ** ((M_CEPHEID_ANDROMEDA - 6.0) / 5.0)
D_pho = EYE_PUPIL * 10 ** ((M_CEPHEID_ANDROMEDA - 6.0 - PHOTOGRAPHIC_GAIN) / 5.0)
print(f"\n  minimum aperture, observing visually      = {D_vis:.2f} m")
print(f"  minimum aperture, with photographic plates = {D_pho:.2f} m")
print()
print("  So the honest answer is not 'only the Hooker was big enough'. Rosse's 1.83 m")
print("  had the aperture in 1845 but no usable photography; Yerkes had both by 1897")
print("  and could in principle have reached m = 19. What the Hooker uniquely gave")
print("  Hubble was HEADROOM — enough margin to measure a light curve rather than")
print("  merely register a dot, on a galaxy crowded with unresolved stars.")
print()
print("  Kant proposed 'island universes' in 1755. The hypothesis waited 169 years,")
print("  and what it waited for was not an idea. It was glass, silver and emulsion.\n")

# --- the same argument, for resolution ---
print("And the resolution side, with one example each:")
targets = [("Jupiter's Galilean moons (separation ~2′)", 120.0),
           ("Saturn's ring gap (Cassini division, 0.7″)", 0.7),
           ("a typical visual binary (1″)", 1.0),
           ("Betelgeuse's disc (0.045″)", 0.045),
           ("a Sun-like star's disc at 10 pc (0.001″)", 0.001)]
for tname, size in targets:
    need = 1.22 * LAMBDA / (size / 206265)
    print(f"  {tname:44s} needs D ≥ {need:7.3f} m")
print()
print("Galileo's 37 mm resolves 3.7″ — comfortably enough for Jupiter's moons at 2′,")
print("and nowhere near enough for Betelgeuse. The discoveries he made were exactly")
print("the ones his glass permitted, and no others.")

# --- visualization ---
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])


def airy_image(D_m, sep_arcsec, n=220, fov_arcsec=3.0):
    """Two point sources imaged by a circular aperture of diameter D."""
    x = np.linspace(-fov_arcsec / 2, fov_arcsec / 2, n)
    X, Y = np.meshgrid(x, x)
    img = np.zeros_like(X)
    k = np.pi * D_m / LAMBDA / 206265
    for x0 in (-sep_arcsec / 2, sep_arcsec / 2):
        r = np.sqrt((X - x0) ** 2 + Y**2) + 1e-9
        u = k * r
        img += (2 * j1(u) / u) ** 2
    return img


# Panel 1-3: the same double star through three apertures
for i, (D, label) in enumerate([(0.037, "Galileo 1609, 37 mm"),
                                (0.254, "a good amateur, 254 mm"),
                                (2.540, "Hooker 1917, 2.54 m")]):
    ax = fig.add_subplot(gs[0, i])
    img = airy_image(D, 1.0)
    ax.imshow(img ** 0.4, cmap="inferno", extent=[-1.5, 1.5, -1.5, 1.5])
    ax.set_title(f"{label}\nθ = {rayleigh_arcsec(D):.2f}″  ·  two stars 1.0″ apart",
                 fontsize=9.5)
    ax.set_xlabel("arcsec")
    if i == 0:
        ax.set_ylabel("arcsec")

# Panel 4: aperture vs time
ax = fig.add_subplot(gs[1, 0])
yrs = [s[0] for s in SCOPES]
Ds = [s[2] for s in SCOPES]
refl = [s[1] not in ("Galileo's best", "Hevelius 140-ft", "Yerkes 40-inch") for s in SCOPES]
for yr, D, is_r, s in zip(yrs, Ds, refl, SCOPES):
    ax.plot(yr, D, "o" if is_r else "s", ms=10,
            color="crimson" if is_r else "steelblue")
    ax.annotate(s[1].split("(")[0][:16], (yr, D), fontsize=6.8,
                xytext=(5, -3), textcoords="offset points")
ax.plot([], [], "o", color="crimson", ms=8, label="reflector (mirror)")
ax.plot([], [], "s", color="steelblue", ms=8, label="refractor (lens)")
ax.set_yscale("log")
ax.set_xlabel("year")
ax.set_ylabel("aperture (m)")
ax.set_title("Four centuries of aperture. The last refractor\nwas built in 1897; "
             "everything since is a mirror.")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

# Panel 5: the two scaling laws
ax = fig.add_subplot(gs[1, 1])
Dg = np.logspace(-2.6, 1.7, 200)
ax.loglog(Dg, rayleigh_arcsec(Dg), color="crimson", lw=2.4, label="resolution θ ∝ 1/D")
ax2 = ax.twinx()
ax2.loglog(Dg, (Dg / EYE_PUPIL) ** 2, color="steelblue", lw=2.4,
           label="light grasp ∝ D²")
ax.set_xlabel("aperture D (m)")
ax.set_ylabel("resolution (arcsec)", color="crimson")
ax2.set_ylabel("light grasp (× naked eye)", color="steelblue")
for yr, name, D, what in SCOPES[:1] + SCOPES[4:5] + SCOPES[6:7] + SCOPES[8:9]:
    ax.axvline(D, color="gray", ls=":", lw=1)
    ax.text(D, 4e-3, name.split()[0], rotation=90, fontsize=7, color="gray", va="bottom")
ax.set_title("Both quantities improve without limit.\nThe only ceiling is what you "
             "can build.")
ax.grid(alpha=0.3, which="both")

# Panel 6: the gate
ax = fig.add_subplot(gs[1, 2])
names = [s[1].split("(")[0][:18] for s in SCOPES]
mls = [limiting_mag(s[2], s[0] >= PHOTOGRAPHY_FROM) for s in SCOPES]
cols = ["crimson" if m >= M_CEPHEID_ANDROMEDA else "lightgray" for m in mls]
ax.barh(names, mls, color=cols, alpha=0.9)
ax.axvline(M_CEPHEID_ANDROMEDA, color="black", ls="--", lw=2.2,
           label="Cepheids in Andromeda (m ≈ 19)")
ax.set_xlabel("limiting magnitude (fainter →)")
ax.set_xlim(5, 32)
ax.invert_yaxis()
ax.tick_params(axis="y", labelsize=7.5)
ax.set_title("The instrument gate on cosmology.\nBars use the best detector of their era —\nthe jump at 1880 is photography,\nnot a bigger mirror.")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3, axis="x")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"\nSaved: {out}")
