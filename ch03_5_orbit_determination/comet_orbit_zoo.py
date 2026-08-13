"""The general recipe: given a state vector, classify any comet and predict its return.

gauss_three_observations.py turns sightings into a state vector (r, v).
differential_correction.py polishes it. This script is what you do next, and it is
the same three lines for every object in the solar system, from a grain of dust to
an interstellar visitor:

    ε = v²/2 − μ/r                 specific orbital energy
    a = −μ/(2ε)                    semi-major axis  (negative ⇒ unbound)
    e = |(v×h)/μ − r̂|              eccentricity vector magnitude

and then the single question that decides everything:

    e < 1   ellipse    → it comes back, with period T = 2π√(a³/μ)
    e = 1   parabola   → marginally unbound, the boundary case
    e > 1   hyperbola  → it never comes back; it was never bound to the Sun

The sign of one number separates "a member of the solar system" from "a visitor
passing through". In 2017 that number came out greater than 1 for the first time —
1I/ʻOumuamua, e = 1.20 — and a category of object that had been hypothetical since
the 1970s became real. 2I/Borisov followed in 2019 at e = 3.36, unmistakably
interstellar.

    phenomenon:   comets return on wildly different schedules, or never
    simulation:   real orbital elements for a dozen well-known comets
    dissection:   compute ε, a, e, T from a state vector, with autograd for dT/da
    formula:      T = 2π√(a³/μ) for e < 1, and nothing at all for e ≥ 1.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

MU = 4 * np.pi**2                  # AU³/yr², heliocentric


def elements_from_state(r_vec, v_vec, mu=MU):
    """The whole recipe. Works for bound and unbound orbits alike."""
    r = torch.linalg.norm(r_vec)
    v = torch.linalg.norm(v_vec)
    energy = v**2 / 2 - mu / r                              # specific orbital energy
    a = -mu / (2 * energy)                                  # negative if unbound
    h = torch.linalg.cross(r_vec, v_vec)
    e_vec = torch.linalg.cross(v_vec, h) / mu - r_vec / r
    e = torch.linalg.norm(e_vec)
    return energy, a, e


def period_of(a, e, mu=MU):
    """T = 2π√(a³/μ), and nothing at all if the orbit is not closed."""
    if float(e) >= 1.0 or float(a) <= 0:
        return float("inf")
    return float(2 * np.pi * torch.sqrt(a**3 / mu))


# --- check the machinery against a known case before trusting it ---
# Halley at perihelion: q = 0.586 AU, and the vis-viva speed there.
A_H, E_H = 17.834, 0.96714
q_h = A_H * (1 - E_H)
v_peri = np.sqrt(MU * (1 + E_H) / q_h)
r_test = torch.tensor([q_h, 0.0, 0.0], dtype=torch.float64)
v_test = torch.tensor([0.0, v_peri, 0.0], dtype=torch.float64)
eps, a_rec, e_rec = elements_from_state(r_test, v_test)
print("Sanity check — feed Halley's perihelion state into the recipe:")
print(f"  input:  r = {q_h:.4f} AU, v = {v_peri:.4f} AU/yr")
print(f"  ε = v²/2 − μ/r = {float(eps):+.6f} AU²/yr²   (negative ⇒ bound)")
print(f"  a = −μ/(2ε)    = {float(a_rec):.4f} AU   (truth {A_H})")
print(f"  e              = {float(e_rec):.5f}       (truth {E_H})")
print(f"  T = 2π√(a³/μ)  = {period_of(a_rec, e_rec):.3f} yr\n")

# --- the zoo: real objects, real elements ---
# (name, a in AU (negative for hyperbolic), e, observed period in yr or None, note)
COMETS = [
    ("2P/Encke",            2.2183, 0.84833,   3.30, "shortest known period"),
    ("Jupiter-family typ.", 3.5000, 0.60000,   6.55, "captured by Jupiter"),
    ("55P/Tempel–Tuttle",  10.3370, 0.90551,  33.24, "parent of the Leonid meteors"),
    ("1P/Halley",          17.8340, 0.96714,  75.32, "the one that started it"),
    ("109P/Swift–Tuttle",  26.0920, 0.96335, 133.28, "parent of the Perseids"),
    ("C/1995 O1 Hale–Bopp", 186.00, 0.99510,  2533., "naked-eye for 18 months"),
    ("C/1996 B2 Hyakutake", 1700.0, 0.99978,  70000., "long-period, from the Oort cloud"),
    ("C/2020 F3 NEOWISE",   358.50, 0.99921,  6800., "the 2020 naked-eye comet"),
    ("1I/ʻOumuamua",       -1.2723, 1.20113,   None, "INTERSTELLAR — first ever, 2017"),
    ("2I/Borisov",         -0.8510, 3.35640,   None, "INTERSTELLAR — 2019, clearly a comet"),
]

print("The zoo. One formula, applied to everything:")
print("  object                    a (AU)      e       q (AU)    T computed    class")
rows = []
for name, a, e, T_obs, note in COMETS:
    at = torch.tensor(a, dtype=torch.float64)
    et = torch.tensor(e, dtype=torch.float64)
    q = a * (1 - e)                        # works for hyperbolic too: a<0, 1-e<0
    T = period_of(at, et)
    if e < 1:
        cls = ("Jupiter-family" if T < 20 else
               "Halley-type" if T < 200 else "long-period")
    else:
        cls = "HYPERBOLIC"
    Tstr = f"{T:10.2f} yr" if np.isfinite(T) else "     never"
    rows.append((name, a, e, q, T, cls, T_obs, note))
    print(f"  {name:24s} {a:8.2f}  {e:.5f}   {q:6.3f}  {Tstr}   {cls}")

print("\n  Verification against the observed periods:")
print("  object                    T computed     T observed     error")
for name, a, e, q, T, cls, T_obs, note in rows:
    if T_obs is None:
        print(f"  {name:24s}      never       never returns   —")
    else:
        print(f"  {name:24s} {T:11.2f}   {T_obs:11.2f}    {100*(T-T_obs)/T_obs:+7.2f}%")
print()
print("  Kepler's third law, unchanged since 1618, still predicting return dates for")
print("  objects Kepler could not have imagined — including two that are not from here.\n")

# --- the boundary: how close to e = 1 do you have to measure? ---
print("HOW HARD IS THE INTERSTELLAR CALL? The whole claim rests on e > 1.")
print("  Many long-period comets have e = 0.9999-something. Deciding whether a comet")
print("  is bound requires distinguishing e = 0.99999 from e = 1.00001.\n")
print("   object            e            a (AU)        aphelion (AU)   bound?")
for name, e_val in [("typical Oort comet", 0.99995), ("Hyakutake", 0.99978),
                    ("borderline", 0.999999), ("1I/ʻOumuamua", 1.20113),
                    ("2I/Borisov", 3.35640)]:
    if e_val < 1:
        # for a near-parabolic comet with q ~ 1 AU
        a_v = 1.0 / (1 - e_val)
        print(f"   {name:18s} {e_val:.6f}   {a_v:12.1f}   {a_v*(1+e_val):13.1f}   yes")
    else:
        print(f"   {name:18s} {e_val:.6f}   {'unbound':>12s}   {'—':>13s}   NO")
print()
print("  A comet with e = 0.99999 and q = 1 AU has an aphelion 200 000 AU out and a")
print("  period of 45 million years. It is still, technically, ours. ʻOumuamua's")
print("  e = 1.20 was measured to about ±0.002 — far enough above 1 to be certain,")
print("  and that certainty is what made it a discovery rather than a curiosity.\n")

# --- sensitivity of the return date, by autograd ---
print("HOW PRECISELY MUST a BE KNOWN? (autograd on T = 2π√(a³/μ))")
print("  object                a (AU)     dT/da (yr/AU)   δa for a 1-month error")
for name, a, e, T_obs, note in COMETS:
    if e >= 1:
        continue
    at = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    T = 2 * np.pi * torch.sqrt(at**3 / MU)
    T.backward()
    dTda = float(at.grad)
    da_month = (1 / 12) / dTda
    print(f"  {name:24s} {a:7.2f}   {dTda:12.3f}   {da_month:12.2e} AU")
print()
print("  Encke's period is so short that its semi-major axis must be known to 10⁻⁵ AU")
print("  to place its return within a month; Hale-Bopp's tolerance is 10⁻³ AU but its")
print("  return is 2500 years away, so nobody will check. The precision you need is")
print("  set entirely by how far ahead you are trying to predict.")
print()
print("  And for very active comets there is a floor no amount of astrometry beats:")
print("  outgassing jets push the nucleus around. Encke's period drifts by hours per")
print("  orbit from this alone, and Halley's by a few days — a non-gravitational")
print("  acceleration that has to be fitted as extra free parameters. The comet is")
print("  not quite a point mass, and it knows it.")

# --- visualization ---
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])

# Panel 1: the orbits, log-radial
ax = fig.add_subplot(gs[0, 0], projection="polar")
nu = np.linspace(-np.pi * 0.98, np.pi * 0.98, 600)
for name, a, e, T_obs, note in COMETS:
    p = a * (1 - e**2)
    r = p / (1 + e * np.cos(nu))
    m = (r > 0) & (r < 60)
    ax.plot(nu[m], np.log10(r[m]), lw=1.6,
            color="crimson" if e >= 1 else "steelblue",
            alpha=0.9 if e >= 1 else 0.6)
for a_p, nm in [(1.0, "Earth"), (5.2, "Jupiter"), (30.1, "Neptune")]:
    ax.plot(np.linspace(0, 2 * np.pi, 100), np.full(100, np.log10(a_p)),
            color="gray", ls=":", lw=1)
ax.set_rticks([0, 1]); ax.set_yticklabels(["1 AU", "10 AU"])
ax.set_title("Blue = bound (ellipse), red = hyperbolic.\nRadius on a log scale. "
             "The red curves\ncome in once and leave forever.", pad=18, fontsize=10)

# Panel 2: the classification plane
ax = fig.add_subplot(gs[0, 1:])
for name, a, e, q, T, cls, T_obs, note in rows:
    col = {"Jupiter-family": "#2171b5", "Halley-type": "#6baed6",
           "long-period": "#c6dbef", "HYPERBOLIC": "crimson"}[cls]
    ax.scatter(q, e, s=170, color=col, edgecolors="k", lw=0.7, zorder=5)
    ax.annotate(name.split()[0], (q, e), fontsize=7.5, xytext=(8, -3),
                textcoords="offset points")
ax.axhline(1.0, color="crimson", lw=2.4, ls="--")
ax.text(0.05, 1.02, "e = 1  ← the line between 'ours' and 'passing through'",
        fontsize=9.5, color="crimson")
ax.set_xscale("log")
ax.set_xlabel("perihelion distance q (AU)")
ax.set_ylabel("eccentricity e")
ax.set_ylim(0.5, 3.6)
ax.set_title("Every comet ever observed, on two axes. The horizontal line at e = 1 is "
             "the entire\nclassification: below it the object returns, above it "
             "it never does.")
ax.grid(alpha=0.3)

# Panel 3: T vs a, Kepler III again
ax = fig.add_subplot(gs[1, 0])
aa = np.logspace(-0.2, 3.4, 200)
ax.loglog(aa, aa**1.5, color="black", lw=2, label="T = a^(3/2)")
for name, a, e, q, T, cls, T_obs, note in rows:
    if np.isfinite(T):
        ax.plot(a, T, "o", ms=9, color="steelblue")
        ax.annotate(name.split("/")[-1].split()[0], (a, T), fontsize=6.5,
                    xytext=(5, -3), textcoords="offset points")
ax.set_xlabel("semi-major axis a (AU)")
ax.set_ylabel("period T (yr)")
ax.set_title("Kepler's third law across five decades\nof distance. Comets obey it "
             "exactly as\nplanets do — that was Halley's whole point.")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which="both")

# Panel 4: energy is what decides
ax = fig.add_subplot(gs[1, 1])
names, energies, cols = [], [], []
for name, a, e, T_obs, note in COMETS:
    q = a * (1 - e)
    v_p = np.sqrt(MU * (1 + e) / q)
    eps_v = v_p**2 / 2 - MU / q
    names.append(name.split("/")[-1].split()[0][:11])
    energies.append(eps_v)
    cols.append("crimson" if e >= 1 else "steelblue")
ax.barh(names, energies, color=cols, alpha=0.9)
ax.axvline(0, color="black", lw=2)
ax.set_xlabel("specific orbital energy ε = v²/2 − μ/r  (AU²/yr²)")
ax.set_xscale("symlog", linthresh=1e-2)
ax.set_title("The sign of ONE number.\nNegative → bound forever.\nPositive → gone "
             "forever.")
ax.tick_params(axis="y", labelsize=7.5)
ax.grid(alpha=0.3, axis="x")

# Panel 5: the recipe
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
ax.text(0.0, 1.0, "The complete recipe", fontsize=11.5, weight="bold", va="top")
ax.text(0.0, 0.88, (
    "INPUT: ≥3 sightings (RA, Dec, time)\n\n"
    "1. preliminary orbit\n"
    "     Gauss's 3-observation method\n"
    "     → gauss_three_observations.py\n\n"
    "2. refine on all observations\n"
    "     Gauss–Newton least squares\n"
    "     → differential_correction.py\n\n"
    "3. state vector → elements\n"
    "     ε = v²/2 − μ/r\n"
    "     a = −μ/(2ε)\n"
    "     e = |(v×h)/μ − r̂|\n\n"
    "4. classify and predict\n"
    "     e < 1 → T = 2π√(a³/μ)\n"
    "     e ≥ 1 → it never returns\n\n"
    "5. corrections that actually matter\n"
    "     · planetary perturbations\n"
    "       (Clairaut, 1757: 618 days!)\n"
    "     · non-gravitational outgassing\n"
    "     · relativity, for Mercury-like\n"
    "       orbits            → ch05\n\n"
    "This is what the Minor Planet Center\n"
    "runs on every new object, nightly."),
    fontsize=8.2, va="top", family="monospace")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"\nSaved: {out}")
