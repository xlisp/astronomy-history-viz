"""Mars walks backwards — the single observation that broke the geocentric universe.

For a few weeks every ~2 years, Mars stops against the stars, reverses, loops,
and resumes. Ptolemy explained it with epicycles: Mars really does loop. Copernicus
explained it with parallax: Mars never loops, WE overtake it on the inside track,
the way a fast car makes a slow one appear to slide backwards.

Both reproduce the observations. That's the point — and the trap. What killed
geocentrism was not this data but the fact that Copernicus needed no free
parameters here: the loop's timing and size fall out of the orbital periods, which
were already known. Ptolemy needed one hand-tuned epicycle per planet.

    phenomenon:   Mars' apparent path on the sky reverses direction
    simulation:   two circular heliocentric orbits (Earth 1.0 AU, Mars 1.524 AU)
    dissection:   project onto the sky as seen from Earth — subtract the vectors
    formula:      the loop happens exactly at OPPOSITION, and its period is the
                  synodic period  1/T_syn = 1/T_earth − 1/T_mars = 1/(2.14 yr)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# --- the two orbits (Copernicus' entire input: radii and periods) ---
A_EARTH, T_EARTH = 1.000, 1.0000     # AU, years
A_MARS,  T_MARS  = 1.524, 1.8808

t = torch.linspace(0, 6.0, 3000, dtype=torch.float64)     # 6 years — long enough for two loops

earth = torch.stack([A_EARTH * torch.cos(2 * np.pi * t / T_EARTH),
                     A_EARTH * torch.sin(2 * np.pi * t / T_EARTH)], dim=1)
mars = torch.stack([A_MARS * torch.cos(2 * np.pi * t / T_MARS),
                    A_MARS * torch.sin(2 * np.pi * t / T_MARS)], dim=1)

# --- dissection: what an Earthbound observer sees is the DIFFERENCE vector ---
rel = mars - earth                                  # geocentric position of Mars
lon = torch.atan2(rel[:, 1], rel[:, 0])             # ecliptic longitude on the sky
lon_unwrapped = torch.from_numpy(np.unwrap(lon.numpy()))

# Retrograde = apparent longitude decreasing. Get the orbital velocities from
# autograd rather than differentiating by hand — this is Newton's fluxion, modernized.
tt = t.clone().requires_grad_(True)
pos_e = torch.stack([A_EARTH * torch.cos(2 * np.pi * tt / T_EARTH),
                     A_EARTH * torch.sin(2 * np.pi * tt / T_EARTH)], dim=1)
pos_m = torch.stack([A_MARS * torch.cos(2 * np.pi * tt / T_MARS),
                     A_MARS * torch.sin(2 * np.pi * tt / T_MARS)], dim=1)
# Each coordinate depends only on its own time sample, so summing before backward
# gives every sample's velocity in one pass.
vel_e = torch.stack([torch.autograd.grad(pos_e[:, k].sum(), tt, retain_graph=True)[0]
                     for k in (0, 1)], dim=1)
vel_m = torch.stack([torch.autograd.grad(pos_m[:, k].sum(), tt, retain_graph=True)[0]
                     for k in (0, 1)], dim=1)
vrel = (vel_m - vel_e).detach()

# Apparent angular velocity of Mars on the sky: dλ/dt = (x·ẏ − y·ẋ) / (x² + y²).
# Negative means it is sliding backwards through the constellations.
dlam_dt = (rel[:, 0] * vrel[:, 1] - rel[:, 1] * vrel[:, 0]) / (rel**2).sum(1)

retro = dlam_dt < 0
print(f"Mars is retrograde {100 * retro.double().mean():.1f}% of the time")

# --- formula: the synodic period predicts when it happens ---
T_syn = 1.0 / (1.0 / T_EARTH - 1.0 / T_MARS)
print(f"synodic period  1/T_syn = 1/{T_EARTH} − 1/{T_MARS}  →  T_syn = {T_syn:.3f} yr")

# find the midpoint of each complete retrograde episode (pair each start with the
# first end that follows it, so a partial episode at either edge is discarded)
edges = np.diff(retro.numpy().astype(int))
starts = list(np.where(edges == 1)[0])
ends = list(np.where(edges == -1)[0])
mids = []
for s in starts:
    later = [e for e in ends if e > s]
    if later:
        mids.append((t[s].item() + t[later[0]].item()) / 2)
if len(mids) > 1:
    print(f"measured spacing between retrograde loops = "
          f"{np.mean(np.diff(mids)):.3f} yr   (predicted {T_syn:.3f} yr)")
# at opposition, Earth–Sun–Mars are aligned: check the Sun-relative angle
for i, mid in enumerate(mids):
    k = int(torch.argmin(torch.abs(t - mid)))
    ang = torch.rad2deg(torch.atan2(mars[k, 1], mars[k, 0])
                        - torch.atan2(earth[k, 1], earth[k, 0])).item() % 360
    print(f"  loop {i+1} centred at t = {mid:.2f} yr → Sun-centred Earth–Mars angle "
          f"= {min(ang, 360-ang):5.1f}°   (0° = opposition)")

# --- visualization ---
fig = plt.figure(figsize=(15, 6.5))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

# Panel 1 — the Copernican truth: two circles, nothing loops
ax1.plot(earth[:, 0], earth[:, 1], color="steelblue", lw=1.5, label="Earth orbit")
ax1.plot(mars[:, 0], mars[:, 1], color="firebrick", lw=1.5, label="Mars orbit")
ax1.plot(0, 0, "*", color="orange", ms=20, label="Sun")
for k in range(0, 3000, 130):                 # sight lines Earth → Mars
    ax1.plot([earth[k, 0], mars[k, 0]], [earth[k, 1], mars[k, 1]],
             color="gray", lw=0.5, alpha=0.45)
ax1.set_aspect("equal")
ax1.set_title("Copernicus: two plain circles.\nNo planet ever reverses.")
ax1.legend(fontsize=8, loc="upper right")
ax1.grid(alpha=0.3)

# Panel 2 — the geocentric appearance: the loop
geo = rel.numpy()
ax2.plot(geo[:, 0], geo[:, 1], color="firebrick", lw=1.2, alpha=0.5)
ax2.scatter(geo[retro, 0], geo[retro, 1], s=9, color="crimson", zorder=5,
            label="retrograde (moving backwards)")
ax2.plot(0, 0, "o", color="steelblue", ms=12, label="Earth (held fixed)")
ax2.set_aspect("equal")
ax2.set_title("Ptolemy's view: the same data, Earth held still.\n"
              "Now Mars visibly loops — the 'epicycle'.")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# Panel 3 — the observable: longitude vs time
ax3.plot(t, torch.rad2deg(lon_unwrapped), color="black", lw=1.8)
ax3.fill_between(t, torch.rad2deg(lon_unwrapped).min(), torch.rad2deg(lon_unwrapped).max(),
                 where=retro, color="crimson", alpha=0.25, label="retrograde episode")
for mid in mids:
    ax3.axvline(mid, color="crimson", ls=":", lw=1)
ax3.set_xlabel("years")
ax3.set_ylabel("ecliptic longitude of Mars (deg, unwrapped)")
ax3.set_title(f"What you actually record at the eyepiece.\n"
              f"Loops repeat every T_syn = {T_syn:.2f} yr — predicted, not fitted.")
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
