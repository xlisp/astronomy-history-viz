"""The apple and the Moon: one calculation that merged heaven and earth.

Before 1666 there were two physics. Terrestrial: things fall, they seek their
natural place, motion decays. Celestial: things move eternally in circles, made of
a fifth element, incorruptible. The two had nothing to do with each other — that
was Aristotle's settled architecture and it lasted 2000 years.

Newton, aged 23 during the plague closure of Cambridge, asked a question nobody
had thought to ask: is the Moon FALLING? If gravity extends past the treetops,
past the clouds, all the way to the Moon, and dilutes as 1/r², then the Moon's
centripetal acceleration should be smaller than g by exactly the factor (R⊕/r)².

He knew all four numbers. So can you. This is the whole calculation.

    phenomenon:   apples fall at 9.81 m/s²; the Moon orbits in 27.32 days
    simulation:   compute the Moon's centripetal acceleration from its orbit alone
    dissection:   take the ratio g / a_moon, and compare to (r_moon/R_earth)²
    formula:      if they match, ONE law governs the apple and the Moon, and the
                  distinction between terrestrial and celestial physics is dead.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# --- the four numbers, all measurable from the ground before 1666 ---
g = 9.80665                     # m/s², measured with pendulums (Galileo, Huygens)
R_EARTH = 6.371e6               # m — Eratosthenes had this to ~1% in 240 BC
R_MOON = 3.844e8                # m — from lunar parallax, known since Hipparchus (~150 BC)
T_MOON = 27.321661 * 86400      # s — sidereal month, known to 5 digits for millennia

print("Four numbers available to any astronomer in 1666:")
print(f"  g            = {g} m/s²           (Galileo's inclined planes, Huygens' pendulum)")
print(f"  R_earth      = {R_EARTH:.4e} m    (Eratosthenes, 240 BC, to ~1%)")
print(f"  r_moon       = {R_MOON:.4e} m    (lunar parallax, Hipparchus ~150 BC)")
print(f"  T_moon       = {T_MOON/86400:.5f} d       (recorded since Babylon)\n")

# --- the Moon's acceleration, purely from kinematics ---
# Circular motion: a = v²/r = 4π²r/T².  No force law used yet — this is geometry.
a_moon = 4 * np.pi**2 * R_MOON / T_MOON**2
print("Step 1 — how fast is the Moon accelerating?  (kinematics only, no gravity)")
print(f"  a_moon = 4π²r/T² = {a_moon:.6e} m/s²")
print(f"  The Moon falls {0.5 * a_moon:.4f} m in the first second — about 1.3 mm.")
print(f"  It never lands because it is also moving sideways at "
      f"{2*np.pi*R_MOON/T_MOON:.0f} m/s.\n")

# --- the two ratios that must agree ---
ratio_measured = g / a_moon
ratio_predicted = (R_MOON / R_EARTH) ** 2

print("Step 2 — the test.")
print(f"  measured   g / a_moon        = {ratio_measured:10.2f}")
print(f"  predicted  (r_moon/R_earth)² = {ratio_predicted:10.2f}")
print(f"  agreement: {100*(1 - abs(ratio_measured-ratio_predicted)/ratio_predicted):.2f}%"
      f"   (discrepancy {100*abs(ratio_measured-ratio_predicted)/ratio_predicted:.2f}%)\n")

# --- what exponent does the data actually demand? Solve for it with autograd. ---
# Model: a(r) = g · (R_earth/r)^p.  Find p such that a(r_moon) = a_moon.
p = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
opt = torch.optim.Adam([p], lr=0.02)
for _ in range(6000):
    opt.zero_grad()
    pred = g * (R_EARTH / R_MOON) ** p
    loss = (torch.log(pred) - np.log(a_moon)) ** 2
    loss.backward()
    opt.step()
print("Step 3 — do not assume the exponent; fit it.")
print(f"  a(r) = g·(R⊕/r)^p  fitted to the Moon gives  p = {p.item():.6f}")
print(f"  The data demands p = 2 to {100*abs(p.item()-2)/2:.2f}%. It is an INVERSE SQUARE.\n")

print("Newton's own words, written in old age about the plague years:")
print('  "...and thereby compared the force requisite to keep the Moon in her Orb')
print('   with the force of gravity at the surface of the earth, and found them')
print('   answer pretty nearly."')
print()
print("'Pretty nearly' is the most consequential understatement in physics.")
print("With that sentence the sky stopped being a different kind of place.")

# --- visualization ---
fig = plt.figure(figsize=(15, 6.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1])

# Panel 1: to-scale picture of the fall
ax = fig.add_subplot(gs[0, 0])
th = np.linspace(0, 2 * np.pi, 300)
ax.plot(np.cos(th) * R_MOON / R_EARTH, np.sin(th) * R_MOON / R_EARTH,
        color="gray", ls="--", lw=1.2, label="Moon's orbit")
ax.add_patch(plt.Circle((0, 0), 1.0, color="steelblue", alpha=0.85))
ax.text(0, 0, "Earth", ha="center", va="center", color="white", fontsize=9, weight="bold")
ax.plot(R_MOON / R_EARTH, 0, "o", color="darkgray", ms=13)
ax.text(R_MOON / R_EARTH, 4, "Moon", ha="center", fontsize=9)
ax.annotate("", xy=(R_MOON / R_EARTH - 9, 0), xytext=(R_MOON / R_EARTH, 0),
            arrowprops=dict(arrowstyle="->", color="crimson", lw=2.5))
ax.text(R_MOON / R_EARTH - 5, -6, "falling\n(1.3 mm/s²)", fontsize=8, color="crimson",
        ha="center")
ax.annotate("", xy=(1, 8), xytext=(1, 14),
            arrowprops=dict(arrowstyle="->", color="crimson", lw=2.5))
ax.text(3.5, 11, "apple falling\n(9.81 m/s²)", fontsize=8, color="crimson")
ax.annotate("", xy=(0, 0), xytext=(R_MOON / R_EARTH, 0),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1))
ax.text(30, 3, f"r = {R_MOON/R_EARTH:.1f} R⊕", fontsize=9)
ax.set_aspect("equal")
ax.set_xlim(-15, 75)
ax.set_ylim(-45, 45)
ax.set_title("The same arrow, 60 Earth-radii apart.\nThe only question is how fast it "
             "dilutes.")
ax.axis("off")
ax.legend(fontsize=8, loc="lower right")

# Panel 2: rival dilution laws, and where the Moon lands
ax = fig.add_subplot(gs[0, 1])
rr = np.logspace(0, 2.1, 300)
for p_try, col, lab in [(1.0, "gray", "1/r   (Kepler's guess)"),
                        (2.0, "crimson", "1/r²  (Newton)"),
                        (3.0, "gray", "1/r³")]:
    ax.loglog(rr, g * rr ** (-p_try), color=col, lw=2.4 if p_try == 2 else 1.3,
              ls="-" if p_try == 2 else ":", label=lab)
ax.plot(1, g, "o", color="darkgreen", ms=13, zorder=6, label="apple (measured)")
ax.plot(R_MOON / R_EARTH, a_moon, "*", color="darkgreen", ms=22, zorder=6,
        label="Moon (measured)")
ax.set_xlabel("distance from Earth's centre  (Earth radii)")
ax.set_ylabel("acceleration (m/s²)")
ax.set_title("Two measurements, 60× apart in distance,\n"
             "3600× apart in acceleration.\nOnly one curve passes through both.")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

# Panel 3: sensitivity — how sharply is p=2 pinned down?
ax = fig.add_subplot(gs[0, 2])
ps = np.linspace(1.5, 2.5, 400)
predicted_a = g * (R_EARTH / R_MOON) ** ps
ax.semilogy(ps, predicted_a, color="steelblue", lw=2.2, label="a_moon predicted by exponent p")
ax.axhline(a_moon, color="crimson", ls="--", lw=2, label="a_moon measured")
ax.axvline(2.0, color="darkgreen", ls=":", lw=2)
ax.plot(p.item(), a_moon, "o", color="crimson", ms=11, zorder=6)
ax.fill_between(ps, a_moon * 0.98, a_moon * 1.02, color="crimson", alpha=0.2,
                label="±2% observational slack")
ax.set_xlabel("assumed exponent p in  a ∝ r^(−p)")
ax.set_ylabel("predicted a_moon (m/s²)")
ax.set_title(f"The exponent is pinned to p = {p.item():.3f}.\n"
             "p = 1.9 or 2.1 misses by 25%. The data\nleaves no room to negotiate.")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
