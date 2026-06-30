# Spatial SIR Model with Superspreaders

A Monte-Carlo reproduction of Figures 1–5 from

> R. Fujie & T. Odagaki, **"Effects of superspreaders in spread of epidemic"**,
> *Physica A* **374** (2007) 843–852.

The model is a standard SIR epidemic placed on a **2-D continuous space** in which
the infection probability `w(r)` depends on the distance `r` between two
individuals. Two kinds of superspreader are studied:

| Model | Superspreader trait | Infection probability `w(r)` |
|-------|---------------------|------------------------------|
| **Strong infectiousness** | Infects with high probability at the *same* range as a normal person | constant `w₀` for `r ≤ r₀` (exponent α = 0) |
| **Hub** | Has *more social contacts*, i.e. a longer range | `w₀(1 − r/rₙ)²` for `r ≤ rₙ`, with `rₙ = √6·r₀` |

A normal individual always uses `w(r) = w₀(1 − r/r₀)²` for `r ≤ r₀` (α = 2).

---

## Repository layout

```
spatial-sir-superspreader/
├── README.md
├── src/
│   ├── sir_superspreader_simulation.py   # SHARED engine (importable, no plotting)
│   ├── member1_percolation.py            # Member 1 runner — Figs 1-5 (percolation)
│   ├── member2_speed.py                  # Member 2 runner — Figs 6-7  (to be added)
│   ├── member3_epidemic_curve.py         # Member 3 runner — Figs 8,15 (to be added)
│   └── member4_distribution.py           # Member 4 runner — Figs 9-14 (to be added)
├── figures/                              # all generated PNGs land here
│   ├── figure1_strong_wr.png
│   ├── figure2_hub_wr.png
│   ├── figure3_strong_percolation.png
│   ├── figure4_hub_percolation.png
│   └── figure5_critical_density.png
└── theory/
    └── effects_of_superspreaders_in_spread_of_epidemic.pdf
```

The code is split into two layers so the whole team shares one simulator:

* **`sir_superspreader_simulation.py`** — the **shared engine**. Defines the
  constants, the three `w(r)` functions, the `SIRSuperspreaderSimulation` class,
  **and the shared sweep driver `MonteCarloSweep`** — the generic "scan a
  `(model, N, λ)` grid, run `TRIALS` trials at each point, in parallel" machinery.
  It contains **no** plotting logic. Every member runs *the same* simulation code
  path through `MonteCarloSweep` and overrides only *what to measure* per trial
  (see *Shared engine API* below).
* **`memberN_*.py`** — one runner per member. Each imports the engine,
  instantiates (or subclasses) `MonteCarloSweep` for the configurations relevant
  to its figures, and draws the plots. Member 1's runner is provided; members 2–4
  add their own. **No member re-implements the sweep** — that is what keeps every
  member's runs the *same underlying realisations*.

---

## Shared engine API (for all team members)

Run one trial and read whatever observable you need off the returned
`TrialResult`:

```python
from sir_superspreader_simulation import SIRSuperspreaderSimulation, make_rng

sim = SIRSuperspreaderSimulation(model="hub", n=477, lam=0.4)
res = sim.simulate(make_rng(2007, 0, 0),          # reproducible generator
                   record_timeseries=True,        # per-step series  (members 2, 3)
                   record_network=True)           # who-infected-whom (member 4)
```

`TrialResult` fields and who uses them:

| Field | Meaning | Member |
|-------|---------|--------|
| `percolated`, `percolation_step` | reached the top edge? at which step? | 1 |
| `new_infections_per_step` | new S→I count per time step (index 0 = seed) | 3 (epidemic curve) |
| `front_distance_per_step` | `r_f(t)`: farthest ever-infected from the seed | 2 (propagation speed) |
| `infection_log` | list of `(infector, infectee, step)` — a forest rooted at the seed | 4 (network maps) |
| `secondary_infections` | how many each individual directly infected | 4 (secondary-infection histogram) |
| `positions`, `is_super` | fixed coordinates and superspreader flags | 4 (scatter / colouring) |
| `final_states`, `total_infected`, `num_steps` | end-of-run summary | all |

Recording flags (`record_timeseries`, `record_network`) only control which
fields get filled — **they do not change the epidemic**, so the same seed yields
an identical run whether or not you record. Member 1's high-throughput sweep
uses the lean `sim.run_trial(rng) -> bool` wrapper (recording off, stops early
on percolation).

### Running a sweep — `MonteCarloSweep` (the shared driver)

Don't write your own trial loop or seeding — instantiate the engine's
`MonteCarloSweep`. Its **defaults are Member 1's percolation sweep**, and you
customise it by overriding just two hooks:

| Hook | Default (Member 1) | Override to… |
|------|--------------------|--------------|
| `measure_trial(self, sim, rng)` | `sim.run_trial(rng, …)` → percolation flag | read a different observable off the trial |
| `reduce_trials(self, values)` | `np.mean` → percolation probability | aggregate differently (e.g. element-wise mean of curves) |

```python
from multiprocessing import Pool
from sir_superspreader_simulation import MonteCarloSweep

# Member 3 — epidemic curve: read new_infections_per_step, average element-wise.
class EpidemicCurveSweep(MonteCarloSweep):
    def measure_trial(self, sim, rng):
        return sim.simulate(rng, record_timeseries=True).new_infections_per_step
    def reduce_trials(self, values):
        width = max(map(len, values))
        padded = [v + [0] * (width - len(v)) for v in values]
        return list(np.mean(padded, axis=0))

sweep = EpidemicCurveSweep(trials=1000, lambdas=(0.4,), n_values=[477])
densities, curve_by_lambda = sweep.run("strong", pool=None)  # or pass a Pool
```

`run(model, pool)` returns `(densities, result_by_lambda)`, where
`result_by_lambda[λ]` is a list with one reduced result per `N`. The seeding,
the parallel map, and the density axis are all inherited unchanged — so the
realisations Member 3 sees at `(strong, λ=0.4, N=477)` are the *same* ones any
other member sees there.

**Reproducibility.** `MonteCarloSweep` seeds every trial through one shared
convention, so you normally never touch the RNG yourself. Under the hood it keys
each generator on the **physical configuration**:

```python
from sir_superspreader_simulation import trial_rng, model_id, lambda_key

rng = trial_rng(model_id(model), lambda_key(lam), n, trial=t)   # what the driver does
```

This expands to `make_rng(PROJECT_SEED, …keys…, trial)`, backed by
`numpy.random.SeedSequence` — deterministic across machines and processes (do
**not** seed from Python's built-in `hash`, which is salted per process). Two
properties make it work for the team:

* **Keyed on the physics, not on indices or a member tag.** Because the key is
  `(model, λ, N)` itself, *any* member evaluating the same point gets the **same**
  layout and the **same** epidemic — they just read a different observable. N=477
  is keyed as the number 477, so it lines up even though it isn't on Member 1's
  grid.
* **Each trial is seeded independently** (the trial number is the last key).
  A lone shared stream would make trial `t` depend on how many random numbers
  trials `0…t-1` consumed, which *differs* between a run that stops early at
  percolation (Member 1) and one that runs to extinction (Members 2–4).
  Per-trial seeding makes trial `t` identical regardless.

One alignment note: the shared default is a **single bottom seed**
(`seed_band=None`) — Members 2–4 use this, and Member 1's percolation sweep
(Figs 3–5) uses the same single-seed setup.

The lower-level `make_rng(*int_keys)` is still available if you ever need a
generator outside this convention. Helpers `density_from_n` / `n_from_density`
convert between `N` and `ρπr₀²`.

> **Do not edit the engine's model physics** (constants, `w(r)`, the update rule,
> the x-only periodic / y-open strip geometry) without flagging the team — every
> figure depends on it. Add new *observables* rather than changing existing ones.

---

## The model in detail

* **Space.** `N` individuals are placed on an `L × L` continuous box with
  `L = 10·r₀`. Constants: `r₀ = 1.0`, `w₀ = 1.0`, recovery probability `γ = 1.0`
  (recovery is instantaneous — an infected individual is infectious for exactly
  one time step).
* **States.** Each individual is Susceptible, Infected, or Recovered.
* **Update rule (synchronous).** In one Monte-Carlo step every *currently*
  infected individual tries to infect each susceptible neighbour with probability
  `w(r)` (an independent Bernoulli draw per pair). Newly infected individuals are
  queued and **cannot** spread during the same step. At the end of the step the
  previously infected individuals all recover (`γ = 1`).
* **Percolation.** One initial infected (the *seed*) is placed at the bottom edge
  (`y = 0`, random `x`); the remaining `N − 1` are placed at random and a fraction
  `λ` of all individuals are superspreaders. A trial is **percolated** if the
  infection chain reaches the top, i.e. some infected individual ends up with
  `y > L − r₀`.
* **Density.** With `ρ = N/L²` and `L = 10·r₀`, the dimensionless density plotted
  on the x-axis is `ρπr₀² = N·π/100`.

### Three implementation notes worth knowing

1. **Boundary conditions are periodic in *x* only.** The system is a *strip*:
   the transverse `x`-axis wraps (period `L`), but the propagation `y`-axis is
   **open**, so the bottom (`y = 0`) and top (`y = L`) are genuinely distinct.
   Wrapping `y` as well would make the seed at `y = 0` toroidally adjacent to the
   top edge and let it percolate in a single step (we observed `λ = 0`
   "percolating" at `ρπr₀² ≈ 7.5` instead of the theoretical `≈ 27`). The strip
   geometry is the standard setup for a bottom-to-top percolation measurement and
   is what reproduces the paper.

2. **Seeding: a single bottom seed.** *One* infected individual is placed at the
   bottom edge (`y = 0`, random `x`), faithful to the paper's wording ("an
   initial-infected individual"); the engine default (`seed_band=None`). This is
   the same setup Members 2–4 use, so all members share the same realisations.

3. **Density grid.** `N_MIN = 15` (`ρπr₀² ≈ 0.47`), so the sweep covers the full
   low-density rise the paper's Figs 3–4 show (down into the `ρπr₀² ∈ (0, 5)`
   band) and every `λ`'s 0.5 crossing — including the low-`ρ_c` hub points at
   high `λ` — is captured.

---

## Requirements

* Python ≥ 3.9
* `numpy` ≥ 2.0 (uses `np.trapezoid`), `scipy` (neighbour search via
  `scipy.spatial.cKDTree`), `matplotlib`

```bash
pip install -r requirements.txt
```

---

## How to run

From the repository root:

```bash
python src/member1_percolation.py
```

Figures are written to `figures/` regardless of the working directory (the output
path is resolved relative to the script). Progress is reported through the
`logging` module, e.g.:

```
[INFO] __main__: Sweep strong: 360 configurations x 1000 trials ...
[INFO] __main__: Saved .../figures/figure3_strong_percolation.png
```

### What gets produced

| Figure | File | Content |
|--------|------|---------|
| 1 | `figure1_strong_wr.png` | `w(r)/w₀` vs `r/r₀` — strong model (normal vs superspreader). *Static, no simulation.* |
| 2 | `figure2_hub_wr.png` | `w(r)/w₀` vs `r/r₀` — hub model. *Static, no simulation.* |
| 3 | `figure3_strong_percolation.png` | Percolation probability vs density `ρπr₀²` for `λ = 0 … 1` — strong model. |
| 4 | `figure4_hub_percolation.png` | Same as Fig. 3 for the hub model. |
| 5 | `figure5_critical_density.png` | Critical density `ρ_c πr₀²` (where percolation probability = 0.5) vs `λ`, with the `R₀ = Rc` theory curves, strong vs hub. Each `Rc` is **this run's own measured `λ = 1` critical density** (paper Eqs 4–5), so the curve is anchored on the simulation rather than on the paper's constants. |

---

## Tuning fidelity vs. speed

All knobs live in the **Experiment configuration** block near the top of
`src/member1_percolation.py`:

| Constant | Default (fast preview) | Publication quality |
|----------|------------------------|---------------------|
| `TRIALS` | `200` | `1000` (as in the paper) |
| `N_STEP` | `30` (~26 density points) | `15` (~51 points, smoother curves) |
| `N_MIN`, `N_MAX` | `150`, `900` | unchanged (per task spec) |
| `LAMBDAS` | `(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)` | unchanged |

The fast-preview defaults finish in roughly **30 s on 16 cores**. The
publication-quality settings (`TRIALS = 1000`, `N_STEP = 15`) do ~10× the work —
a few minutes on a many-core machine. The sweep is embarrassingly parallel and
uses `multiprocessing.Pool()` (one worker per CPU by default).

Results are reproducible: each `(model, N, λ)` configuration and trial derives a
unique, deterministic seed from `PROJECT_SEED` via `trial_rng` (see
*Reproducibility* above).
