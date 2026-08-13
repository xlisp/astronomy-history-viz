"""14 September 2015: the last of Einstein's predictions, and a new sense organ.

Einstein predicted gravitational waves in 1916, then spent decades unsure whether
they were real or a coordinate artefact. In 1936 he and Rosen submitted a paper to
Physical Review titled "Do Gravitational Waves Exist?" arguing they do not; the
referee found an error, Einstein withdrew the paper in fury, and later quietly
conceded the referee was right.

The reason for the doubt is the size of the effect. GW150914 stretched LIGO's 4 km
arms by a few times 10⁻¹⁸ m — a small fraction of the width of a single proton.
Detecting that is like measuring the distance to Alpha Centauri to within the
width of a human hair.

What arrived was not a blip but a CHIRP: two black holes of 36 and 29 solar masses
spiralling together, the frequency and amplitude both sweeping upward as they lose
orbital energy to the waves, ending in a merger and ringdown — all in 0.2 seconds.
The waveform is fully determined by general relativity, so the fit returns the
masses, the distance, and the spins.

Note what has happened epistemically. In ch05 light bending was a TEST of relativity.
Here relativity is assumed, and the waveform is used as an INSTRUMENT to weigh black
holes 1.3 billion light-years away. The same transition, a century later.

    phenomenon:   a rising "chirp" in two detectors 3000 km apart, 7 ms offset
    simulation:   the post-Newtonian inspiral waveform
    dissection:   fit the chirp mass from the frequency sweep with torch autograd
    formula:      df/dt = (96/5)π^(8/3)(GM_c/c³)^(5/3) f^(11/3) — the sweep rate
                  depends on ONE combination of the masses, the chirp mass.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

G, C = 6.67430e-11, 2.99792458e8
M_SUN = 1.98892e30

M1, M2 = 36.0 * M_SUN, 29.0 * M_SUN                     # GW150914, as published
M_CHIRP_TRUE = (M1 * M2) ** 0.6 / (M1 + M2) ** 0.2
print(f"GW150914 as reported: m₁ = {M1/M_SUN:.0f} M☉, m₂ = {M2/M_SUN:.0f} M☉")
print(f"  chirp mass M_c = (m₁m₂)^(3/5)/(m₁+m₂)^(1/5) = {M_CHIRP_TRUE/M_SUN:.2f} M☉")
print(f"  total mass {(M1+M2)/M_SUN:.0f} M☉ → final black hole ~62 M☉,")
print(f"  meaning ~3 M☉ was converted to gravitational waves in 0.2 s.")
print(f"  Peak power ~3.6e49 W — briefly more than all the light of every star")
print(f"  in the observable universe, combined.\n")


def chirp_waveform(t, M_c, t_c=0.0, phi_c=0.0, D=410e6 * 3.0857e16):
    """Leading-order post-Newtonian inspiral: h(t), frequency sweeping up.

    tau = t_c − t is the time to coalescence. Both the frequency and the amplitude
    are powers of tau, which is why the signal looks like a chirp that grows.
    """
    tau = torch.clamp(t_c - t, min=1e-4)
    theta = (G * M_c / C**3)
    f = (1 / np.pi) * (5.0 / (256 * tau)) ** (3.0 / 8) * theta ** (-5.0 / 8)
    # phase accumulates as the integral of 2*pi*f dt
    phase = -2 * (5 * G * M_c / C**3) ** (-5.0 / 8) * tau ** (5.0 / 8) + phi_c
    amp = (4 / D) * (G * M_c / C**2) ** (5.0 / 3) * (np.pi * f / C) ** (2.0 / 3)
    return amp * torch.cos(phase), f, amp


SR = 4096
t = torch.arange(-0.22, -0.002, 1.0 / SR, dtype=torch.float64)
h_clean, f_true, amp_true = chirp_waveform(t, M_CHIRP_TRUE)

torch.manual_seed(4)
NOISE = 1.1e-21                     # roughly LIGO's sensitivity in band
h_obs = h_clean + NOISE * torch.randn(len(t), dtype=torch.float64)
snr = float(torch.sqrt((h_clean**2).sum()) / NOISE)
print(f"simulated strain: peak h = {float(h_clean.abs().max()):.2e}")
print(f"  LIGO arm length 4 km → arm stretches by "
      f"{float(h_clean.abs().max())*4000:.2e} m")
print(f"  a proton is 8.4e-16 m across, so that is "
      f"{8.4e-16/(float(h_clean.abs().max())*4000):.0f}× smaller than a proton")
print(f"  matched-filter SNR ≈ {snr:.0f}   (GW150914 was 24)\n")

# --- dissection: recover the chirp mass by matched filtering ---
def quadrature_snr(data, M_c, t_c):
    """Phase-maximised correlation — the actual LIGO statistic.

    A template with the wrong phase correlates to zero even when the mass is
    exactly right, so the raw correlation surface is violently multi-modal. Real
    pipelines build the two quadratures (cos and sin) and take the magnitude,
    which maximises over the unknown phase analytically and leaves a smooth
    surface in the parameters we actually care about.
    """
    _, _, amp = chirp_waveform(t, M_c, t_c)
    tau = torch.clamp(t_c - t, min=1e-4)
    phase = -2 * (5 * G * M_c / C**3) ** (-5.0 / 8) * tau ** (5.0 / 8)
    hc, hs = amp * torch.cos(phase), amp * torch.sin(phase)
    d = data - data.mean()
    nc = torch.sqrt((hc * hc).sum() + 1e-60)
    ns = torch.sqrt((hs * hs).sum() + 1e-60)
    return torch.sqrt(((d * hc).sum() / nc) ** 2 + ((d * hs).sum() / ns) ** 2)


# Step 1: scan a template bank, exactly as the search pipeline does.
mc_grid = torch.linspace(10, 60, 260, dtype=torch.float64) * M_SUN
tc_grid = torch.linspace(-0.02, 0.06, 160, dtype=torch.float64)
best = (-1.0, None, None)
with torch.no_grad():
    for mc in mc_grid:
        for tc_try in tc_grid:
            s = float(quadrature_snr(h_obs, mc, tc_try))
            if s > best[0]:
                best = (s, float(mc), float(tc_try))
_, mc0, tc0 = best
print(f"template-bank search over {len(mc_grid)}×{len(tc_grid)} templates:")
print(f"  loudest template: M_c = {mc0/M_SUN:.2f} M☉, t_c = {tc0*1000:.1f} ms")

# Step 2: refine that template with autograd.
logMc = torch.tensor(np.log(mc0), dtype=torch.float64, requires_grad=True)
tc = torch.tensor(tc0, dtype=torch.float64, requires_grad=True)
opt = torch.optim.Adam([logMc, tc], lr=1e-3)
for _ in range(3000):
    opt.zero_grad()
    (-quadrature_snr(h_obs, torch.exp(logMc), tc)).backward()
    opt.step()
Mc_fit, tc_fit = float(torch.exp(logMc)), float(tc)
print(f"  after gradient refinement:")
print(f"  recovered chirp mass = {Mc_fit/M_SUN:.2f} M☉   "
      f"(truth {M_CHIRP_TRUE/M_SUN:.2f} M☉, "
      f"error {100*abs(Mc_fit-M_CHIRP_TRUE)/M_CHIRP_TRUE:.1f}%)")
print(f"  Only the chirp mass is well measured from the inspiral — the individual")
print(f"  masses stay degenerate until the merger and ringdown are included.\n")

# --- frequency evolution ---
print("the sweep, in numbers:")
print("  time to merger    frequency     orbital separation")
for tt in (-0.2, -0.1, -0.05, -0.02, -0.008):
    tau = -tt
    f_gw = float((1 / np.pi) * (5.0 / (256 * tau)) ** (3.0 / 8)
                 * (G * M_CHIRP_TRUE / C**3) ** (-5.0 / 8))
    f_orb = f_gw / 2
    sep = (G * (M1 + M2) / (2 * np.pi * f_orb) ** 2) ** (1 / 3)
    print(f"    {tau*1000:6.0f} ms       {f_gw:6.1f} Hz     {sep/1000:8.0f} km  "
          f"({sep/(2*G*(M1+M2)/C**2):.1f} Schwarzschild radii)")
print()
print("In the final milliseconds two objects of 60-odd solar masses are orbiting")
print(f"each other about {float(f_true[-1])/2:.0f} times a second, at a good fraction of the")
print("speed of light, separated by a few hundred kilometres. There is no weak-field")
print("approximation that describes this; it required numerical relativity, which")
print("took from 1964 to 2005 to get working. The templates were ready just in time.\n")

print("What the new sense organ found in its first decade:")
for name, what in [("GW150914 (2015)", "black holes exist, and merge, and are heavier "
                    "than expected"),
                   ("GW170817 (2017)", "a neutron star merger seen in gravitational waves "
                    "AND light —\n                    it made gold, and gave an "
                    "independent H₀"),
                   ("GW190521 (2019)", "a 142 M☉ black hole, in the 'forbidden' "
                    "mass gap"),
                   ("~300 events by 2025", "a whole population, and a new way to "
                    "measure the universe")]:
    print(f"  {name:22s} {what}")

# --- visualization ---
fig = plt.figure(figsize=(15, 8.5))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.1])

# Panel 1: the raw strain
ax = fig.add_subplot(gs[0, :])
ax.plot(t, h_obs * 1e21, color="gray", lw=0.7, alpha=0.8, label="detector output + noise")
ax.plot(t, h_clean * 1e21, color="crimson", lw=1.8, label="the signal")
ax.set_ylabel("strain h (×10⁻²¹)")
ax.set_xlim(-0.22, 0)
ax.legend(fontsize=9, loc="upper left")
ax.set_title("GW150914, simulated. Two black holes, 1.3 billion light-years away, "
             "in their last 0.2 seconds.\n"
             "The whole signal is buried in noise until you know what shape to "
             "look for.")
ax.grid(alpha=0.3)

# Panel 2: the recovered template
ax = fig.add_subplot(gs[1, :])
model, _, _ = chirp_waveform(t, torch.tensor(Mc_fit), torch.tensor(tc_fit))
ax.plot(t, h_clean * 1e21, color="crimson", lw=2.2, label="true signal")
ax.plot(t, model.detach() * 1e21, color="black", lw=1.2, ls="--",
        label=f"matched-filter template, M_c = {Mc_fit/M_SUN:.1f} M☉")
ax.set_ylabel("strain h (×10⁻²¹)")
ax.set_xlabel("time from merger (s)")
ax.set_xlim(-0.22, 0)
ax.legend(fontsize=9, loc="upper left")
ax.set_title("Matched filtering: slide a family of general-relativity waveforms "
             "against the data\nand keep the one that correlates best. "
             "The template's parameters ARE the measurement.")
ax.grid(alpha=0.3)

# Panel 3: frequency sweep
ax = fig.add_subplot(gs[2, 0])
ax.plot(t, f_true, color="crimson", lw=2.4)
ax.set_xlabel("time from merger (s)")
ax.set_ylabel("gravitational-wave frequency (Hz)")
ax.set_yscale("log")
ax.set_title(f"The chirp: {float(f_true[0]):.0f} Hz to {float(f_true[-1]):.0f} Hz in 0.2 s\n(the real event reached ~250 Hz at merger).\nThe sweep "
             "RATE is\nwhat encodes the chirp mass.")
ax.grid(alpha=0.3, which="both")

# Panel 4: spectrogram-like view
ax = fig.add_subplot(gs[2, 1])
nfft = 256
spec = []
for i in range(0, len(t) - nfft, 16):
    seg = h_obs[i:i + nfft] * torch.hann_window(nfft, dtype=torch.float64)
    spec.append(torch.abs(torch.fft.rfft(seg)))
spec = torch.stack(spec).T.numpy()
freqs = np.fft.rfftfreq(nfft, 1 / SR)
tt = t[: spec.shape[1] * 16 : 16].numpy()
ax.pcolormesh(tt, freqs, np.log10(spec + 1e-30), cmap="viridis", shading="auto")
ax.set_ylim(20, 400)
ax.set_xlabel("time from merger (s)")
ax.set_ylabel("frequency (Hz)")
ax.set_title("The same data as a spectrogram —\nthe now-famous upward-sweeping "
             "arc\nthat announced the discovery.")

# Panel 5: how small is the effect
ax = fig.add_subplot(gs[2, 2])
scales = [("LIGO arm", 4000), ("human hair", 8e-5), ("atom", 1e-10),
          ("proton", 8.4e-16), ("GW150914\narm stretch", float(h_clean.abs().max()) * 4000)]
names = [s[0] for s in scales]
vals = [s[1] for s in scales]
cols = ["steelblue"] * 4 + ["crimson"]
ax.barh(names, vals, color=cols, alpha=0.9)
ax.set_xscale("log")
ax.set_xlabel("length (m)")
for i, v in enumerate(vals):
    ax.text(v * 2, i, f"{v:.1e}", va="center", fontsize=8)
ax.set_title("The measurement problem, to scale.\nLIGO measures a 4 km ruler to a\n"
             "thousandth of a proton width.")
ax.grid(alpha=0.3, axis="x", which="both")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
