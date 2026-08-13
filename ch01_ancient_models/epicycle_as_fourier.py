"""Ptolemy's epicycles ARE a Fourier series — which is why they always worked.

An epicycle is a circle whose centre rides on another circle. In complex notation
a stack of epicycles is

    z(t) = Σ_k  c_k · exp(i·ω_k·t)

which is *exactly* a Fourier series, 1600 years before Fourier wrote one down.
This is the most important lesson in the philosophy of science, and it is a
theorem, not an opinion: a deep enough epicycle stack approximates ANY closed
curve to arbitrary accuracy. Ptolemy's model was never refuted by data — it
CANNOT be refuted by data. A model that fits everything predicts nothing.

Modern echo: a wide enough neural network is also a universal approximator.
Fitting the training set is evidence of nothing. Kepler beat Ptolemy the way a
good model beats an overfit one — fewer parameters, and they mean something.

    phenomenon:   Mars' geocentric path is an intricate looping curve
    simulation:   real elliptical orbits for Earth and Mars, solved with Newton's
                  method on Kepler's equation  M = E − e·sin E
    dissection:   take the FFT; add epicycles one at a time, largest first
    formula:      the epicycle amplitudes ARE the Fourier coefficients — verified
                  against an independent torch gradient-descent fit.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# --- orbital elements. T_MARS is rounded to the 15:8 near-resonance (1.875 yr vs
# --- the true 1.8808) so that the geocentric path closes exactly after 15 years.
# --- Without a closed path there is no Fourier series — only leakage.
A_EARTH, T_EARTH, E_EARTH = 1.0000, 1.000, 0.01671
A_MARS,  T_MARS,  E_MARS  = 1.5237, 1.875, 0.09341
WINDOW = 15.0            # years: Earth makes 15 revolutions, Mars exactly 8
N_SAMP = 2048


def kepler_orbit(a, T, e, t):
    """Position in the orbital plane, as a complex number.

    Kepler's equation  M = E − e·sin E  has no closed-form solution for E. Solve it
    the way Newton did — with Newton's method, which he invented partly for this.
    """
    M = 2 * np.pi * t / T                       # mean anomaly: uniform, fictitious
    E = M.clone()                               # initial guess
    for _ in range(60):                         # Newton iteration on f(E) = E − e sinE − M
        E = E - (E - e * torch.sin(E) - M) / (1 - e * torch.cos(E))
    x = a * (torch.cos(E) - e)                  # focus at the origin (the Sun)
    y = a * np.sqrt(1 - e**2) * torch.sin(E)
    return torch.complex(x, y)


t = torch.arange(N_SAMP, dtype=torch.float64) / N_SAMP * WINDOW
earth = kepler_orbit(A_EARTH, T_EARTH, E_EARTH, t)
mars = kepler_orbit(A_MARS, T_MARS, E_MARS, t)
target = mars - earth                            # what an Earthbound observer records

print(f"path closes after {WINDOW:g} yr: |z(0) − z(T)| = "
      f"{abs(target[0] - kepler_orbit(A_MARS, T_MARS, E_MARS, torch.tensor([WINDOW]))[0] + kepler_orbit(A_EARTH, T_EARTH, E_EARTH, torch.tensor([WINDOW]))[0]):.2e} AU")

# --- dissection: the Fourier spectrum of the geocentric path ---
c_fft = torch.fft.fft(target) / N_SAMP
k_all = torch.fft.fftfreq(N_SAMP, d=1.0 / N_SAMP).to(torch.int64)   # integer harmonics of 1/15 yr
order = torch.argsort(-c_fft.abs())                                 # biggest epicycle first

print("\nthe dominant epicycles, largest first  (frequency in cycles/year = k/15)")
print("   rank   k      radius (AU)     period (yr)    what it physically is")
names = {8: "Mars' own orbital motion (T = 15/8 = 1.875 yr)",
         15: "the Earth's orbit, reflected onto the sky (T = 1 yr)",
         16: "1st harmonic of Mars → Mars' ECCENTRICITY",
         30: "1st harmonic of Earth → Earth's eccentricity",
         24: "2nd harmonic of Mars → e² correction",
         -8: "retrograde term from Mars' eccentricity",
         0: "constant offset (the deferent's centre)"}
for rank in range(8):
    idx = int(order[rank])
    k = int(k_all[idx])
    per = WINDOW / abs(k) if k != 0 else float("inf")
    print(f"    {rank+1:2d}  {k:+4d}     {c_fft[idx].abs():.6f}      {per:8.3f}    "
          f"{names.get(k, '')}")

# --- add epicycles one at a time and watch the error fall ---
def reconstruct(n_terms):
    keep = order[:n_terms]
    basis = torch.exp(2j * np.pi * k_all[keep][None, :].to(torch.float64)
                      * t[:, None] / WINDOW)
    return basis @ c_fft[keep], k_all[keep], c_fft[keep]

print("\nepicycles   RMS error (AU)   Tycho-visible?  (his limit ≈ 2′ ≈ 0.0009 AU at 1.5 AU)")
TYCHO_AU = np.tan(2 / 60 * np.pi / 180) * 1.5
results = {}
for n in (1, 2, 3, 5, 9, 17):
    pred, ks, cs = reconstruct(n)
    rms = torch.sqrt(((pred - target).abs() ** 2).mean()).item()
    results[n] = (pred, ks, cs, rms)
    print(f"    {n:2d}       {rms:.6f}        "
          f"{'YES — model is refutable' if rms > TYCHO_AU else 'no — indistinguishable from truth'}")

# --- formula: gradient descent finds the same coefficients the FFT does ---
n = 9
keep = order[:n]
basis = torch.exp(2j * np.pi * k_all[keep][None, :].to(torch.float64) * t[:, None] / WINDOW)
c_learn = torch.zeros(n, dtype=torch.complex128, requires_grad=True)
opt = torch.optim.Adam([c_learn], lr=0.05)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3000, gamma=0.1)
for _ in range(12000):
    opt.zero_grad()
    loss = (((basis @ c_learn) - target).abs() ** 2).mean()
    loss.backward()
    opt.step()
    sched.step()
err = (c_learn.detach() - c_fft[keep]).abs().max().item()
print(f"\nmax |gradient-descent coefficient − FFT coefficient| = {err:.2e}")
print("They converge to the same numbers: 'adding an epicycle' and 'adding a Fourier")
print("mode' are the same act.")
print("Ptolemy was doing harmonic analysis by hand, and could always add one more term.")

# --- visualization ---
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])

for idx_p, n in enumerate((1, 3, 9)):
    ax = fig.add_subplot(gs[0, idx_p])
    pred, ks, cs, rms = results[n]
    ax.plot(target.real, target.imag, color="black", lw=2.2, label="true path of Mars")
    ax.plot(pred.real, pred.imag, color="crimson", lw=1.5, ls="--",
            label=f"{n} epicycle{'s' if n > 1 else ''}")
    # draw the nested circles at one instant — this is the physical machine
    j = 300
    origin = torch.tensor(0j, dtype=torch.complex128)
    for oi in torch.argsort(-cs.abs()).tolist():
        rad = cs[oi].abs()
        circ = origin + rad * torch.exp(2j * np.pi * torch.linspace(0, 1, 120,
                                                                   dtype=torch.float64))
        ax.plot(circ.real, circ.imag, color="steelblue", lw=0.7, alpha=0.5)
        step = cs[oi] * torch.exp(2j * np.pi * ks[oi].to(torch.float64) * t[j] / WINDOW)
        ax.plot([origin.real, (origin + step).real],
                [origin.imag, (origin + step).imag], color="steelblue", lw=1.1)
        origin = origin + step
    ax.plot(0, 0, "o", color="steelblue", ms=7)
    ax.set_aspect("equal")
    ax.set_title(f"{n} epicycle{'s' if n > 1 else ''}   ·   RMS = {rms:.4f} AU")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.25)

# bottom-left: error vs number of epicycles
ax = fig.add_subplot(gs[1, 0])
ns = sorted(results)
ax.semilogy(ns, [results[k][3] for k in ns], "o-", color="crimson", lw=2, ms=8)
ax.axhline(TYCHO_AU, color="darkgreen", ls="--", lw=1.8,
           label="Tycho's precision (2′ ≈ 0.0009 AU)")
ax.set_xlabel("number of epicycles (= Fourier modes)")
ax.set_ylabel("RMS error (AU)")
ax.set_title("Any accuracy is purchasable with enough epicycles.\n"
             "This is a THEOREM — so fitting the data proves nothing.")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

# bottom-middle: the spectrum
ax = fig.add_subplot(gs[1, 1])
sel = (k_all.abs() <= 40)
ks_s = k_all[sel].numpy()
amp_s = c_fft.abs()[sel].numpy()
srt = np.argsort(ks_s)
ax.stem(ks_s[srt] / WINDOW, amp_s[srt], basefmt=" ", linefmt="steelblue", markerfmt="o")
ax.set_yscale("log")
ax.set_ylim(1e-6, 3)
ax.set_xlabel("frequency (cycles / year)")
ax.set_ylabel("epicycle radius (AU)")
ax.set_title("Two fundamentals + their harmonics.\n"
             "The harmonics ARE the eccentricity Ptolemy\nhand-tuned with his 'equant'.")
for kk, lab, col in [(8, "Mars 1/1.875 yr", "firebrick"),
                     (15, "Earth 1/1.0 yr", "steelblue"),
                     (16, "Mars 2nd harm.\n(eccentricity)", "darkgreen")]:
    ax.axvline(kk / WINDOW, color=col, ls=":", lw=1.3)
    ax.text(kk / WINDOW, 1.2, lab, rotation=90, fontsize=7, color=col,
            ha="right", va="top")
ax.grid(alpha=0.3, which="both")

# bottom-right: the moral
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
ax.text(0.0, 1.0, "Why Kepler won", fontsize=13, weight="bold", va="top")
ax.text(0.0, 0.88, (
    "Ptolemy (150 AD)\n"
    "  ~40 hand-tuned parameters; one deferent,\n"
    "  epicycle and equant per planet, each\n"
    "  independent of the others.\n"
    "  Fits the data. Forbids nothing.\n"
    "  Universal approximator → unfalsifiable.\n\n"
    "Kepler (1609–1619)\n"
    "  6 parameters per planet — and they are\n"
    "  physical: a, e, i, Ω, ω, T.\n"
    "  Same fit. Plus T² = a³ ties every planet\n"
    "  to every other — a constraint Ptolemy's\n"
    "  model cannot even express.\n\n"
    "The test was never 'does it fit?'\n"
    "It was 'what would refute it?'\n\n"
    "The same test applies to a neural net today."),
    fontsize=9.5, va="top", family="monospace")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
