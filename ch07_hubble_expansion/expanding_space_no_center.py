"""Everyone sees themselves at the centre — and that is the proof there is no centre.

The commonest misunderstanding about the Big Bang is that it happened SOMEWHERE, and
that galaxies are debris flying away from that place. Hubble's law refutes this, and
the refutation is three lines of algebra that anyone can check.

If space expands uniformly, every separation vector scales by the same factor a(t):

    r_ij(t) = a(t) · r_ij(0)
    v_ij = ṙ_ij = (ȧ/a) · r_ij = H · r_ij

Note what is NOT in that derivation: any special point. The relation v = H·r holds
between EVERY pair, measured by EVERY observer. Shift your origin to any galaxy and
the law comes out identical. Hubble's law is therefore not evidence that we are at
the centre; it is precisely the signature of there being no centre at all.

And only a linear law has this property. Any other v(r) would single out a location —
which is exactly why the linearity in hubble_law_fit.py, not the value of H₀, was the
discovery.

    phenomenon:   everything recedes from us, faster the farther away it is
    simulation:   a uniformly expanding grid of galaxies, no centre anywhere
    dissection:   recompute the observed velocity field from three different
                  galaxies' points of view and compare
    formula:      v = H·r is invariant under change of origin  ⟺  v is linear in r.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

rng = np.random.default_rng(5)
H0 = 70.0                     # km/s/Mpc

# --- a slab of universe: galaxies scattered with no preferred point ---
N = 260
gal = torch.tensor(rng.uniform(-300, 300, (N, 2)), dtype=torch.float64)   # Mpc

# --- uniform expansion: every velocity is H times the position, from ANY origin ---
vel = H0 * gal                                    # km/s, in the "our galaxy at 0" frame

print("Take the same expanding universe and re-measure it from three different homes.\n")
observers = [(0, "us (origin)"), (int(torch.argmax(gal[:, 0])), "a galaxy far to the east"),
             (int(torch.argmin(gal[:, 1])), "a galaxy far to the south")]

fits = []
for idx, label in observers:
    origin = gal[idx]
    rel_pos = gal - origin                        # positions as THEY see them
    rel_vel = vel - vel[idx]                      # velocities as THEY see them
    d = torch.linalg.norm(rel_pos, dim=1)
    # radial velocity = component of relative velocity along the line of sight
    keep = d > 1e-9
    v_rad = (rel_vel[keep] * rel_pos[keep]).sum(1) / d[keep]
    X = d[keep].unsqueeze(1)
    H_fit = float(torch.linalg.lstsq(X, v_rad.unsqueeze(1)).solution)
    resid = float((v_rad - H_fit * d[keep]).abs().max())
    fits.append((label, H_fit, resid, d[keep], v_rad))
    print(f"  observer: {label:26s}  measures H₀ = {H_fit:.6f} km/s/Mpc   "
          f"max deviation from linearity = {resid:.2e} km/s")

print(f"\n  input value was H₀ = {H0}. All three observers recover it exactly, and all")
print(f"  three see every other galaxy running away from THEM. Nobody is at a centre,")
print(f"  and everybody appears to be. This is what 'space expands' means.\n")

# --- what is special is LINEARITY, so break the linearity and watch it fail ---
print("It is specifically the LINEARITY that is origin-invariant. Break it and see:")
r_mag = torch.linalg.norm(gal, dim=1, keepdim=True)
vel_nl = 0.25 * r_mag * gal                       # a made-up law with speed ∝ r²
for idx, label in observers:
    origin = gal[idx]
    rp = gal - origin
    rv = vel_nl - vel_nl[idx]
    d = torch.linalg.norm(rp, dim=1)
    keep = d > 1e-9
    v_rad = (rv[keep] * rp[keep]).sum(1) / d[keep]
    X = d[keep].unsqueeze(1)
    H_fit = float(torch.linalg.lstsq(X, v_rad.unsqueeze(1)).solution)
    scat = float((v_rad - H_fit * d[keep]).std())
    print(f"  observer: {label:26s}  fitted slope = {H_fit:8.1f}   "
          f"scatter = {scat:8.1f} km/s")
print("  Under a nonlinear law each observer gets a different slope and a smeared")
print("  cloud instead of a line. Only v ∝ r survives a change of origin, because")
print("  only a linear map satisfies f(a) − f(b) = f(a − b).\n")

# --- the honest version of the 'explosion' objection ---
print("A caution, because the usual textbook contrast is wrong:")
print("  Debris from a free explosion has r = v·t exactly (nothing decelerates it),")
print("  so it ALSO gives v = r/t — a perfect Hubble law, for every fragment, not")
print("  just the one at the centre. Kinematically the two pictures are identical.")
print("  Hubble's law alone therefore does NOT rule out an explosion.")
print()
print("  What rules it out is different evidence:")
print("    · an explosion has an EDGE and a density peak; the galaxy distribution")
print("      shows neither, out to the limits of every survey")
print("    · the CMB is isotropic to 1 part in 10⁵ (ch08) — from off-centre inside")
print("      an explosion it would be hotter on one side")
print("    · at z > 1 the recession 'speeds' exceed c, which debris cannot do but")
print("      stretching space can (ch06)")
print("    · the CMB blackbody is cooled by exactly 1+z, which is a property of")
print("      stretched wavelengths, not of Doppler-shifted fragments")
print()
print("The corollary people find hardest: the Big Bang did not happen at a point in")
print("space. It happened everywhere, to all of space, at once. The observable")
print("universe was smaller, but it was not somewhere else.")

# --- visualization ---
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3)

# Row 1: the velocity field as seen by three observers
for col, (idx, label) in enumerate(observers):
    ax = fig.add_subplot(gs[0, col])
    origin = gal[idx]
    rel_pos = gal - origin
    rel_vel = vel - vel[idx]
    ax.quiver(rel_pos[:, 0], rel_pos[:, 1], rel_vel[:, 0], rel_vel[:, 1],
              torch.linalg.norm(rel_vel, dim=1), cmap="plasma",
              width=0.004, scale=90000, alpha=0.85)
    ax.plot(0, 0, "*", color="crimson", ms=22, zorder=6)
    ax.set_aspect("equal")
    ax.set_xlim(-620, 620); ax.set_ylim(-620, 620)
    ax.set_title(f"seen from {label}\nH₀ = {fits[col][1]:.2f} km/s/Mpc",
                 fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

fig.text(0.5, 0.955, "The same universe, three different home galaxies. "
         "Each observer sees everything fleeing from themselves.",
         ha="center", fontsize=12.5, weight="bold")

# Row 2 panel 1: all three Hubble diagrams superimposed
ax = fig.add_subplot(gs[1, 0])
for (label, H_fit, _, d, v_rad), col in zip(fits, ["crimson", "steelblue", "darkgreen"]):
    ax.plot(d, v_rad, ".", color=col, ms=3, alpha=0.5, label=f"{label} (H={H_fit:.1f})")
dd = np.linspace(0, 800, 20)
ax.plot(dd, H0 * dd, "k--", lw=1.6, label="input H₀ = 70")
ax.set_xlabel("distance from the observer (Mpc)")
ax.set_ylabel("radial velocity (km/s)")
ax.set_title("Three observers, three Hubble diagrams,\none identical slope. "
             "The law is\ninvariant under change of origin.")
ax.legend(fontsize=7.5)
ax.grid(alpha=0.3)

# Row 2 panel 2: a nonlinear velocity law, seen from two different origins
ax = fig.add_subplot(gs[1, 1])
for (idx, label), col in zip(observers[:2], ["crimson", "steelblue"]):
    origin = gal[idx]
    rp = gal - origin
    rv = vel_nl - vel_nl[idx]
    d = torch.linalg.norm(rp, dim=1)
    keep = d > 1e-9
    v_rad = (rv[keep] * rp[keep]).sum(1) / d[keep]
    ax.plot(d[keep], v_rad, ".", color=col, ms=4, alpha=0.6, label=label)
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("distance from the observer (Mpc)")
ax.set_ylabel("radial velocity (km/s)")
ax.set_title("A NONLINEAR law (v ∝ r²) seen from two\nhomes: different slopes, "
             "smeared clouds,\nand some neighbours approaching.\nThis is what a "
             "centre looks like.")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Row 2 panel 3: the scale factor picture
ax = fig.add_subplot(gs[1, 2])
sub = gal[:26]
for k, (a_scale, col, alpha) in enumerate([(0.55, "#9ecae1", 0.9),
                                           (0.78, "#4292c6", 0.9),
                                           (1.0, "#08306b", 1.0)]):
    ax.plot(sub[:, 0] * a_scale, sub[:, 1] * a_scale, "o", color=col, ms=6,
            alpha=alpha, label=f"a(t) = {a_scale}")
for i in range(len(sub)):
    ax.plot([sub[i, 0] * 0.55, sub[i, 0]], [sub[i, 1] * 0.55, sub[i, 1]],
            color="gray", lw=0.5, alpha=0.5)
ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Expansion = one number a(t) multiplying\nevery separation. The pattern "
             "is identical,\njust bigger. Nothing moves 'through' space.")
ax.legend(fontsize=8.5)

plt.tight_layout(rect=[0, 0, 1, 0.94])
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
