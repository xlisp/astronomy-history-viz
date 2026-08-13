"""What "grinding a mirror" actually means: a craft carried out at the wavelength of light.

Newton did not commission his telescope. He built it: he alloyed the speculum metal
himself (copper, tin, a little arsenic), cast it, ground it on a pitch lap, polished
it with putty powder, and mounted it in a tube he made. The Royal Society elected
him a Fellow in 1672 on the strength of that object.

The reason this is skilled work rather than mere labour is a piece of geometry.
Grinding two glass discs against each other with abrasive between them naturally
produces a SPHERE — the only surface that can slide against its mate in every
orientation. But a sphere does not focus parallel light to a point; only a
PARABOLOID does. So every mirror maker must start with what nature gives (a sphere)
and then deliberately deform it into what optics demands (a paraboloid).

The size of that deformation is the whole story:

    sphere:      z ≈ r²/(2R) + r⁴/(8R³) + …
    paraboloid:  z  = r²/(2R)                   (with R = 2f)
    difference:  Δz = r⁴/(8R³) = D/(1024 N³)    for a mirror of f-ratio N

For a 200 mm f/8 mirror that is 0.38 micrometres — and the tolerance on it is a
fraction of a wavelength of light. You are hand-sculpting glass to a precision of
tens of nanometres, using your thumbs, over several weeks.

    phenomenon:   a sphere blurs starlight; a paraboloid does not
    simulation:   trace rays off both surfaces and find where they cross the axis
    dissection:   compute the sphere/paraboloid sag difference and compare it to
                  the Rayleigh λ/8 surface tolerance
    formula:      the Foucault knife-edge test converts that sub-micron surface
                  error into a longitudinal displacement of D/(8N) — millimetres,
                  measurable with a screw. That amplification is why amateurs can
                  do this at all.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

LAMBDA = 550e-6            # mm — green light, the wavelength the eye peaks at


def sag_sphere(r, R):
    """Exact spherical sag: R − sqrt(R² − r²)."""
    return R - torch.sqrt(R**2 - r**2)


def sag_parabola(r, R):
    """Paraboloid with the same paraxial curvature: z = r²/(2R), R = 2f."""
    return r**2 / (2 * R)


def marginal_focus(r, R):
    """Where a ray striking a SPHERE at height r crosses the axis.

    The normal at the surface point P points at the centre of curvature C = (0, R).
    Reflecting the incoming ray d = (0, −1) about that normal and intersecting the
    result with the axis gives, after the algebra collapses,

        z_focus = R − R² / (2·sqrt(R² − r²))

    which tends to the paraxial R/2 as r → 0 and walks INWARD as r grows. That walk
    is spherical aberration; to first order it is r²/(4R).
    """
    return R - R**2 / (2 * torch.sqrt(R**2 - r**2))


print("THE PROBLEM: grinding makes a sphere, optics wants a paraboloid.\n")
print("  mirror                  D (mm)   f/N    R = 2f (mm)   Δz sphere→parabola")
print("                                                        (µm)      in λ")
MIRRORS = [("Newton 1668 (speculum)", 33.0, 6.0),
           ("amateur classic", 200.0, 8.0),
           ("Herschel 40-foot 1789", 1220.0, 10.0),
           ("Rosse 'Leviathan' 1845", 1830.0, 9.0),
           ("Hooker 100-inch 1917", 2540.0, 5.0),
           ("Hale 200-inch 1948", 5080.0, 3.3),
           ("Keck segment", 1800.0, 1.75)]
for name, D, N in MIRRORS:
    f = N * D
    R = 2 * f
    r = torch.tensor(D / 2, dtype=torch.float64)
    dz = float(sag_sphere(r, torch.tensor(R)) - sag_parabola(r, torch.tensor(R)))
    print(f"  {name:24s} {D:6.0f}  f/{N:<5.2f} {R:9.0f}     {dz*1000:8.2f}   "
          f"{dz/LAMBDA:7.2f}λ")

print()
print("Now the tolerance. Rayleigh's criterion: the WAVEFRONT must be good to λ/4.")
print("Reflection doubles any surface error, so the SURFACE must be good to λ/8:")
print(f"  λ/8 = {LAMBDA/8*1e6:.1f} nm = {LAMBDA/8*1000:.4f} µm\n")
print("  mirror                  Δz to remove   tolerance λ/8   Δz / tolerance")
for name, D, N in MIRRORS:
    R = 2 * N * D
    r = torch.tensor(D / 2, dtype=torch.float64)
    dz = float(sag_sphere(r, torch.tensor(R)) - sag_parabola(r, torch.tensor(R)))
    tol = LAMBDA / 8
    print(f"  {name:24s} {dz*1000:9.2f} µm   {tol*1000:9.4f} µm   {dz/tol:10.1f}×")
print()
print("Read the last column as: 'how many times over must I get this right?'")
print("You must remove a specific sub-micron amount of glass from specific zones,")
print("and land within a few per cent of it. There is no instrument in 1668, or in")
print("1845, that measures 70 nanometres directly.\n")

# --- the Foucault test: turn 70 nm into 3 mm ---
print("THE TRICK THAT MAKES IT POSSIBLE — Foucault's knife-edge test (1858).")
print()
print("Put a point source at the centre of curvature and look at the returning light")
print("with a knife edge. Different ZONES of a paraboloid have different centres of")
print("curvature, spread longitudinally by")
print()
print("    Δ_long = r²/R = D²/(8f) = D/(8N)")
print()
print("  mirror                  surface error   longitudinal spread   amplification")
for name, D, N in MIRRORS:
    R = 2 * N * D
    r = torch.tensor(D / 2, dtype=torch.float64)
    dz = float(sag_sphere(r, torch.tensor(R)) - sag_parabola(r, torch.tensor(R)))
    long_spread = D / (8 * N)
    print(f"  {name:24s} {dz*1000:9.3f} µm   {long_spread:12.2f} mm      "
          f"{long_spread/dz:10.0f}×")
print()
print("A 0.38 µm difference in the GLASS becomes a 3.1 mm difference in WHERE THE")
print("LIGHT CROSSES — an amplification of about 8000×, and 3 mm is measurable with")
print("a cheap screw gauge. Foucault's test is the reason mirror-making left the")
print("guild and reached the kitchen table; it is still how amateurs figure mirrors")
print("today, unchanged in principle since 1858.\n")

# --- what a sphere actually costs you ---
D_TEST, N_TEST = 200.0, 8.0
R_TEST = 2 * N_TEST * D_TEST
rr = torch.linspace(1.0, D_TEST / 2, 200, dtype=torch.float64)
foci = marginal_focus(rr, torch.tensor(R_TEST))
paraxial = R_TEST / 2
f_test = N_TEST * D_TEST
long_ab = float(paraxial - foci.min())
# A marginal ray converges with slope r/f, so missing focus by long_ab leaves it
# long_ab*r/f off-axis at the paraxial focal plane. Diameter is twice that.
blur = 2 * long_ab * (D_TEST / 2) / f_test
blur_arcsec = blur / f_test * 206265
diff_arcsec = 1.22 * LAMBDA / D_TEST * 206265
print(f"A {D_TEST:.0f} mm f/{N_TEST:.0f} SPHERE, left unparabolised:")
print(f"  paraxial focus at {paraxial:.2f} mm, marginal focus at {float(foci.min()):.2f} mm")
print(f"  longitudinal spherical aberration = {long_ab:.3f} mm"
      f"   (closed form r²/4R = {(D_TEST/2)**2/(4*R_TEST):.3f} mm ✓)")
print(f"  star image diameter at the paraxial focus = {blur*1000:.1f} µm = "
      f"{blur_arcsec:.2f} arcsec")
print(f"  the diffraction limit for this aperture is {diff_arcsec:.2f} arcsec")
print(f"  → the unparabolised sphere is {blur_arcsec/diff_arcsec:.0f}× worse than the "
      f"mirror is capable of.")
print()
print("This is what those weeks of polishing buy. Not a better telescope — the SAME")
print("telescope, finally performing at the limit physics allows.")

# --- visualization ---
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3)

# Panel 1: the two surfaces, difference exaggerated
ax = fig.add_subplot(gs[0, 0])
r_plot = torch.linspace(-D_TEST / 2, D_TEST / 2, 300, dtype=torch.float64)
zs = sag_sphere(r_plot.abs(), torch.tensor(R_TEST))
zp = sag_parabola(r_plot.abs(), torch.tensor(R_TEST))
ax.plot(r_plot, (zs - zp) * 1000, color="crimson", lw=2.6)
ax.axhline(LAMBDA / 8 * 1000, color="darkgreen", ls="--", lw=1.8,
           label=f"λ/8 surface tolerance = {LAMBDA/8*1000:.3f} µm")
ax.axhline(-LAMBDA / 8 * 1000, color="darkgreen", ls="--", lw=1.8)
ax.fill_between(r_plot, -LAMBDA / 8 * 1000, LAMBDA / 8 * 1000,
                color="darkgreen", alpha=0.15)
ax.set_xlabel("distance from mirror centre (mm)")
ax.set_ylabel("sphere − paraboloid (µm)")
ax.set_title(f"The glass you must remove.\n{D_TEST:.0f} mm f/{N_TEST:.0f}: a total of "
             f"{float((zs-zp).max())*1000:.2f} µm at the edge,\nheld to ±"
             f"{LAMBDA/8*1000:.3f} µm everywhere.")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel 2: ray trace off a sphere
ax = fig.add_subplot(gs[0, 1])
for r0 in np.linspace(5, D_TEST / 2, 11):
    rt = torch.tensor(r0, dtype=torch.float64)
    zf = float(marginal_focus(rt, torch.tensor(R_TEST)))
    zsurf = float(sag_sphere(rt, torch.tensor(R_TEST)))
    ax.plot([zsurf, zf], [r0, 0], color="steelblue", lw=0.9)
    ax.plot([zsurf, zf], [-r0, 0], color="steelblue", lw=0.9)
    ax.plot([zsurf - 200, zsurf], [r0, r0], color="gray", lw=0.6, alpha=0.6)
    ax.plot([zsurf - 200, zsurf], [-r0, -r0], color="gray", lw=0.6, alpha=0.6)
zsurf_all = sag_sphere(r_plot.abs(), torch.tensor(R_TEST))
ax.plot(zsurf_all, r_plot, color="black", lw=2.5)
ax.axvline(paraxial, color="crimson", ls=":", lw=1.5)
ax.set_xlim(-220, paraxial * 1.06)
ax.set_xlabel("distance along the axis (mm)")
ax.set_ylabel("height (mm)")
ax.set_title("A SPHERE. Outer rays cross the axis short of\nthe paraxial focus — "
             f"spherical aberration,\n{long_ab:.2f} mm of it. No focus plane exists.")
ax.grid(alpha=0.3)

# Panel 3: zoom on the focus
ax = fig.add_subplot(gs[0, 2])
ax.plot(foci - paraxial, rr, color="steelblue", lw=2.4, label="sphere")
ax.plot(foci - paraxial, -rr, color="steelblue", lw=2.4)
ax.axvline(0, color="crimson", lw=2.4, label="paraboloid (all zones agree)")
ax.set_xlabel("focus position relative to paraxial (mm)")
ax.set_ylabel("zone radius on the mirror (mm)")
ax.set_title("Zone by zone, where the light lands.\nA paraboloid collapses this "
             "curve to a\nvertical line — that is the definition.")
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3)

# Panel 4: the Foucault amplification
ax = fig.add_subplot(gs[1, 0])
names = [m[0].split()[0] for m in MIRRORS]
dzs, longs = [], []
for name, D, N in MIRRORS:
    R = 2 * N * D
    r = torch.tensor(D / 2, dtype=torch.float64)
    dzs.append(float(sag_sphere(r, torch.tensor(R)) - sag_parabola(r, torch.tensor(R))))
    longs.append(D / (8 * N))
x = np.arange(len(names))
ax.bar(x - 0.2, np.array(dzs) * 1000, 0.4, color="crimson", label="surface error (µm)")
ax.bar(x + 0.2, np.array(longs) * 1000, 0.4, color="steelblue",
       label="Foucault longitudinal signal (µm)")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=35, fontsize=8)
ax.set_ylabel("µm (log scale)")
ax.set_title("Foucault's amplification, ~10³–10⁴×.\nThe thing you cannot measure "
             "becomes\nthe thing you can.")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y", which="both")

# Panel 5: knife-edge shadow patterns
ax = fig.add_subplot(gs[1, 1])
n_px = 200
yy, xx = np.mgrid[-1:1:n_px * 1j, -1:1:n_px * 1j]
rad = np.sqrt(xx**2 + yy**2)
mask = rad <= 1
for i, (offset, label) in enumerate([(-1.0, "inside focus"), (0.0, "at focus"),
                                     (1.0, "outside focus")]):
    slope = offset - 1.6 * rad**2          # zonal figure error signature
    shade = 0.5 + 0.45 * np.tanh(4 * (slope * xx / (rad + 1e-6)))
    img = np.where(mask, shade, np.nan)
    ax.imshow(img, extent=[i * 2.2 - 1, i * 2.2 + 1, -1, 1], cmap="gray",
              vmin=0, vmax=1)
    ax.text(i * 2.2, -1.25, label, ha="center", fontsize=8.5)
ax.set_xlim(-1.3, 5.7)
ax.set_ylim(-1.5, 1.3)
ax.axis("off")
ax.set_title("What the maker actually sees: the mirror lit\nunevenly by its own "
             "figure errors. A trained eye\nreads nanometres off these shadows.")

# Panel 6: the lineage
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
ax.text(0.0, 1.0, "Astronomers who ground their own", fontsize=11.5,
        weight="bold", va="top")
ax.text(0.0, 0.88, (
    "GALILEO (1609)\n"
    "  ground and polished his own lenses;\n"
    "  sold telescopes to Venice to fund\n"
    "  the work.\n\n"
    "HUYGENS (1655)\n"
    "  with his brother Constantijn, ground\n"
    "  objectives up to 190 mm; invented the\n"
    "  eyepiece still named after him.\n\n"
    "NEWTON (1668)\n"
    "  alloyed, cast, ground, polished and\n"
    "  mounted the first reflector himself —\n"
    "  and found dispersion along the way.\n\n"
    "HERSCHEL (1780s)\n"
    "  cast over 400 mirrors; polished for\n"
    "  16 hours at a stretch while Caroline\n"
    "  fed him. Found Uranus with one.\n\n"
    "ROSSE (1845)\n"
    "  1.83 m speculum, cast in his own\n"
    "  foundry. Saw the spiral arms of M51 —\n"
    "  the first hint that 'nebulae' had\n"
    "  structure.                     → ch07\n\n"
    "The instrument and the discovery are\n"
    "the same act, done by the same hands."),
    fontsize=8.2, va="top", family="monospace")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"\nSaved: {out}")
