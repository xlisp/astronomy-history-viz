# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project intent

A Python project that re-teaches physics by retracing the **astronomical observations that forced it into existence**: Tycho's 20 years of naked-eye positions → Kepler's 8 arcminutes → Newton's inverse square → Le Verrier's Neptune (and his failed Vulcan) → Einstein's 43″ → Leavitt's ruler → Hubble's redshift → dark matter, dark energy, gravitational waves.

The load-bearing thesis, stated in the README and embodied in every script: **astronomy is the mother science because it is the most precise one.** Physics revolutions happen when observational precision crosses the size of the reigning theory's residual. Every chapter is organized around one such residual.

This is the physics counterpart to the sibling project `../math-history-viz` (same author, same pedagogy). That project traces where mathematical *tools* came from; this one traces where those tools were first *forced into use*. Cross-references between the two are welcome and already present in several docstrings.

## Architecture

Code is organized by **historical episode**, not by technique. Each chapter is a self-contained directory of runnable scripts; each script writes a PNG next to itself.

```
ch00_why_astronomy/        precision vs revolution, parallax, the distance ladder
ch00_5_instrument_makers/  Newton's prism, chromatic aberration, grinding a
                           paraboloid, aperture as the gate on discovery
ch01_ancient_models/       retrograde motion, epicycles = Fourier series
ch02_tycho_kepler/         the 8 arcminutes, equal areas, T² = a³
ch03_newton_synthesis/     Kepler's laws → 1/r², the Moon test, Bertrand's theorem
ch03_5_orbit_determination/ Gauss's 3-observation method, least-squares refinement,
                           Halley's return, the general recipe for any comet
ch04_perturbation/         Neptune (the triumph), Mercury's 43″ (the failure)
ch05_general_relativity/   Mercury precession, light deflection, lensing
ch06_spectra_redshift/     Fraunhofer lines, blackbody → quantum, Doppler
ch07_hubble_expansion/     Leavitt's law, v = H₀d, why there is no centre
ch08_modern_cosmos/        dark matter, dark energy, the CMB, LIGO chirps
```

Decimal-numbered directories are inserted chapters, matching the sibling project's
convention. They sort before their integer neighbour alphabetically; the README
states the intended reading order, so do not renumber to "fix" the sort.

Two threads run orthogonally to the historical spine and should be reinforced, not
diluted, when adding material:
  · **ch00_5** — instrument-making as the actual driver of precision. Galileo,
    Newton, Huygens, Herschel and Rosse ground their own optics; Newton's prism
    and his reflector came out of the same workshop. Discovery is gated by craft.
  · **ch03_5** — turning observations into orbits. This is where Newton's laws
    become a prediction machine, and where least squares was invented.

Chapters 3 and 5 are the keystones — they carry the central claim that Kepler's laws are a *necessary* input to gravitation, and that only astronomy could adjudicate Newton vs Einstein. When adding to them, prefer deepening the derivation chain over introducing new topics.

## Pedagogical rules (all three apply to every script)

1. **Prefer code to formulas.** Express the physics as runnable PyTorch, not LaTeX. An acceleration is `torch.autograd.grad(...)` called twice, not `d²r/dt²` in a docstring. Where a formula is historically named, keep it to one line and immediately follow it with the code that computes it.

2. **Every script must produce a visualization.** No script is complete if it only prints numbers. Save to `Path(__file__).with_suffix(".png")` at 120 dpi. The plot is the deliverable.

3. **Phenomenon → simulation → dissection → formula.** Lead with the observation, simulate it, take it apart, and let the formula fall out as the explanation. Never the reverse. Every module docstring carries an explicit four-line block in exactly this shape — match it.

## Hard requirements specific to this repo

**Reproduce the historical number, and print the comparison.** This project's credibility rests on the fact that `mercury_precession_gr.py` actually prints `42.99″/century` next to the observed `43.11 ± 0.45`. Never hardcode a famous result into a print statement — compute it, then print it beside the accepted value. The README's "数值验证一览" table is the contract; if you change a script, re-run it and update that table.

**Prose in the code must match the numbers the code prints.** Several bugs found during the initial build were of this exact kind (a docstring claiming ~2′ while the script printed 0.5′; a "factor of 2" sentence sitting above output showing 0.9939). Re-read the printed output against your own comments before considering a script done.

**When a simplification changes the answer, say so in the output, not just in a comment.** `mercury_residual_43.py` uses circular coplanar orbits and overshoots the accepted Newtonian total by 4%; it prints that fact and explains why the conclusion is unaffected. Do the same rather than quietly tuning constants to match.

**Watch the units and the era's data conventions.** Real errors caught during the build: (a) the "mean distance to the Sun" in popular tables is `a(1+e²/2)`, not `a`, which fakes a 0.5% violation of Kepler III for Saturn; (b) Cepheid *parallaxes* in mas are not *distances* in kpc; (c) `Δf/f ≈ 1/V` holds over the F–C interval that *defines* the Abbe number, not over the whole visible band; (d) the achromatic condition is on POWER (`P₁/V₁ + P₂/V₂ = 0`), not focal length — inverting it silently produces a diverging lens; (e) `datetime.date` cannot represent BC years, so historical epochs are carried as decimal years. Prefer JPL osculating elements and state the source in a comment.

**Check the sign on every descent step and every optical convention.** Two bugs of this kind survived a first read: a Gauss–Newton update missing its minus sign (it climbed instead of descending), and a `c1`/`c3` scaling applied twice in the Gauss iteration. Both produced plausible-looking output for one iteration before diverging. When an iteration diverges or goes NaN, suspect a sign or a double-applied factor before suspecting the physics.

**Report ill-conditioning as a result, not as a failure.** Gauss's method on Piazzi's real 41-day arc has `1/D₀ ≈ 10⁵`, and a short-arc least-squares fit leaves ω and M₀ individually undetermined while pinning their sum. Both facts are the historically interesting content — they are why Gauss needed three inventions rather than one. Demonstrate the method on a well-conditioned case, then measure the breakdown explicitly, rather than quietly choosing parameters that hide it.

**Symplectic integrators for orbits, always.** Velocity-Verlet, not Euler or RK4. A non-symplectic integrator leaks energy and fakes a precession — which is precisely the signal several scripts are trying to measure. Where numerical drift remains (it does, in `mercury_residual_43.py`), difference against an identically-integrated baseline and say that this is what you are doing.

## Lean on autograd where it is doing real work

PyTorch is not decoration here. Gradients flow through a Newton-method solver for Kepler's equation (`kepler_to_inverse_square.py`), through a Friedmann integral (`supernova_dark_energy.py`), through a matched-filter statistic (`gw_chirp_ligo.py`), and through an entire orbit-determination pipeline (`differential_correction.py`, via `torch.autograd.functional.jacobian` with `vectorize=True`). They also produce Wien's law from `dB/dλ = 0` (`blackbody_to_planck.py`), angular dispersion `dδ/dλ` (`newton_prism_experiment.py`), the return-date sensitivity `dT/da` (`halley_comet_return.py`), and an error budget from `∂d/∂rᵢ` (`distance_ladder.py`). Prefer autograd over hand-derived derivatives — it is thematically the point (Newton's fluxions, modernized; Gauss hand-derived the orbit-determination Jacobian over several pages) and it removes a class of algebra errors.

## Performance

Batch across parameter grids rather than looping in Python — `orbit_shape_vs_exponent.py` went from minutes to 8 seconds by integrating all six force laws as one `(B, 2)` tensor, and `neptune_from_residuals.py` scans 1104 candidate planets in one pass. Three scripts are legitimately slow because they are doing real integration (`mercury_residual_43.py` ~2 min, `mercury_precession_gr.py` ~1 min, `neptune_from_residuals.py` ~35 s); these are flagged in the README and should stay flagged if their runtime changes.

## Style

- English docstrings, comments, print output, and plot labels; **Chinese only in `README.md`.**
- Module docstring opens with a one-line hook naming the date and the person, then the historical context, then the four-line phenomenon/simulation/dissection/formula block.
- Section comments in the body: `# --- phenomenon: ... ---`, `# --- dissection: ... ---`.
- Imports: `from pathlib import Path`, blank line, then `matplotlib.pyplot as plt`, `numpy as np`, `torch`.
- `torch.float64` throughout — several results depend on the 8th significant figure.
- Seed every RNG (`torch.manual_seed`, `np.random.default_rng(n)`) so figures are reproducible.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python ch03_newton_synthesis/kepler_to_inverse_square.py    # the keystone derivation
python ch05_general_relativity/mercury_precession_gr.py     # the keystone verification
python ch03_5_orbit_determination/gauss_three_observations.py  # observations → orbit
python ch00_5_instrument_makers/newton_prism_experiment.py     # the craft thread

# run everything and check nothing regressed (takes ~6 minutes)
for f in ch*/*.py; do python "$f" > /dev/null || echo "FAIL $f"; done
```
