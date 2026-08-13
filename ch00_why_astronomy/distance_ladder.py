"""The cosmic distance ladder: why one broken rung poisons everything above it.

No single method measures both a nearby star and a distant galaxy. Astronomy
instead *chains* methods: each rung is calibrated by the rung below, and only the
bottom rung (parallax, ch00) is pure geometry. This is the most consequential
error-propagation problem in science — the whole "Hubble tension" of the 2020s is
an argument about rung calibration.

The lesson generalizes far beyond astronomy: when estimates are chained
multiplicatively, RELATIVE errors add in quadrature and the total grows without
bound. This is the same algebra as backprop through a deep network (products of
Jacobians), which is why deep nets have exploding/vanishing gradients.

    phenomenon:   we can only measure ratios of distances, rung to rung
    simulation:   build a 5-rung ladder, each rung calibrated on the last
    dissection:   propagate the error with torch autograd — differentiate the
                  chained distance w.r.t. each rung's calibration constant
    formula:      d = ∏ rᵢ  ⟹  (σ_d/d)² = Σ (σ_i/rᵢ)²   — errors add in log space,
                  which is exactly why the ladder is plotted logarithmically.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# --- the five rungs, with the fractional calibration uncertainty of each ---
rungs = [
    ("Parallax\n(Bessel 1838)",        1e-4,  1e3,   0.01, "pure geometry — no astrophysics"),
    ("Cepheids\n(Leavitt 1912)",       1e2,   3e7,   0.03, "period → luminosity, calibrated by parallax"),
    ("Tully–Fisher / TRGB",            1e6,   1e8,   0.05, "galaxy rotation → luminosity"),
    ("Type Ia supernovae\n(1990s)",    1e6,   3e9,   0.04, "standard candle, calibrated by Cepheids"),
    ("Hubble's law\n(Hubble 1929)",    3e7,   1e11,  0.02, "redshift → distance, calibrated by SNe Ia"),
]

# --- dissection: chain the rungs, then differentiate the chain ---
# Each rung multiplies the distance scale by a calibration factor r_i.
# The final distance to a far galaxy is the PRODUCT of all five.
r = torch.tensor([1.0, 300.0, 40.0, 30.0, 35.0], dtype=torch.float64, requires_grad=True)
frac_err = torch.tensor([f for _, _, _, f, _ in rungs], dtype=torch.float64)

d_total = torch.prod(r)          # the chained distance estimate
d_total.backward()               # ∂d/∂r_i for every rung, in one line

print("chained distance  d = ∏ rᵢ =", f"{d_total.item():.3e}", "(arbitrary units)")
print()
print(" rung                      r_i      ∂d/∂r_i      (∂d/∂r_i)·σ_i / d   ← its share of the error")
contrib = []
for i, (name, *_rest) in enumerate(rungs):
    sigma_i = frac_err[i] * r[i]                       # absolute error on this rung
    share = (r.grad[i] * sigma_i / d_total).item()     # fractional error contributed
    contrib.append(abs(share))
    print(f" {name.splitlines()[0]:24s} {r[i].item():7.1f}  {r.grad[i].item():.3e}   {share:8.4f}")

total_frac = float(np.sqrt(np.sum(np.square(contrib))))
print()
print(f"total fractional error  = sqrt(Σ shares²) = {total_frac:.4f}  →  {total_frac*100:.1f}%")
print(f"check against the closed form sqrt(Σ (σ_i/r_i)²) = "
      f"{float(torch.sqrt((frac_err**2).sum())):.4f}   (identical — as it must be)")
print()
print("A 1% systematic slipped into the Cepheid rung propagates, undiluted, to the")
print("age of the universe. This is why the Hubble tension is a *calibration* fight.")

# --- visualization ---
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 6.5),
                              gridspec_kw={"width_ratios": [1.7, 1]})

colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(rungs)))
for i, ((name, lo, hi, ferr, note), c) in enumerate(zip(rungs, colors)):
    ax.plot([lo, hi], [i, i], lw=13, color=c, solid_capstyle="butt", alpha=0.85)
    ax.text(np.sqrt(lo * hi), i + 0.30, name, ha="center", fontsize=9.5, weight="bold")
    ax.text(np.sqrt(lo * hi), i - 0.34, f"±{ferr*100:.0f}%  ·  {note}",
            ha="center", fontsize=7.5, alpha=0.8, style="italic")
    # the overlap region is where rung i+1 gets calibrated by rung i
    if i + 1 < len(rungs):
        nlo, nhi = rungs[i + 1][1], rungs[i + 1][2]
        olo, ohi = max(lo, nlo), min(hi, nhi)
        if olo < ohi:
            ax.fill_betweenx([i, i + 1], olo, ohi, color="crimson", alpha=0.13)

ax.set_xscale("log")
ax.set_xlim(1e-5, 1e12)
ax.set_ylim(-0.8, len(rungs) - 0.2)
ax.set_yticks([])
ax.set_xlabel("distance (parsec)")
ax.set_title("The cosmic distance ladder — each rung is calibrated in the pink overlap\n"
             "with the rung below; only the bottom one is pure trigonometry")
for x, lab in [(1e-5, "Earth"), (1.3, "α Cen"), (8e3, "galactic\ncentre"),
               (7.8e5, "Andromeda"), (5e9, "quasars")]:
    ax.axvline(x, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.text(x, len(rungs) - 0.45, lab, rotation=90, fontsize=7.5,
            color="gray", va="top", ha="right")

# right panel: the error budget
names = [n.splitlines()[0] for n, *_ in rungs]
ax2.barh(names, np.array(contrib) * 100, color=colors, alpha=0.9)
ax2.axvline(total_frac * 100, color="crimson", ls="--", lw=2,
            label=f"total (quadrature) = {total_frac*100:.1f}%")
ax2.set_xlabel("fractional error contributed to the final distance (%)")
ax2.set_title("Error budget from torch autograd:\n∂d/∂rᵢ tells you which rung to fix first")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3, axis="x")
ax2.invert_yaxis()

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
