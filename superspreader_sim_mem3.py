import numpy as np
import matplotlib.pyplot as plt
import csv
import time

# =============================================================================
# CONFIG -- everything you might want to tweak lives here
# =============================================================================
CONFIG = {
    # ---- physical model constants (paper uses w0=1, gamma=1, r0=1) ----
    "r0": 1.0,          # infection cutoff radius (length unit)
    "w0": 1.0,          # max infection probability
    "gamma": 1.0,       # recovery probability per timestep (paper: fixed to 1)
    "L_over_r0": 10.0,  # box size L = 10 * r0 (periodic boundary conditions)
    "alpha_normal": 2,  # exponent in normal infectiousness w(r)=(1-r/r0)^alpha
    "hub_radius_factor": np.sqrt(6.0),  # superspreader cutoff in hub model = sqrt(6)*r0

    # ---- Monte Carlo settings ----
    # Paper averages over 1000 runs. 1000 is accurate but slow on a laptop;
    # 200-300 already gives smooth curves for a report figure. Raise this for
    # your final numbers, lower it while you are debugging.
    "num_runs_fig8": 1000,
    "num_runs_fig15": 1000,
    "random_seed": 42,           # set to None for non-reproducible runs

    # ---- Figure 8 settings (rho*pi*r0^2 = 20.0, lambda = 0.2) ----
    "fig8_rho_pi_r0sq": 20.0,
    "fig8_lambda": 0.2,
    "fig8_max_timesteps": 40,

    # ---- Figure 15 settings (N = 477, rho*pi*r0^2 = 15.0, lambda = 0.4) ----
    "fig15_N": 477,
    "fig15_lambda": 0.4,
    "fig15_max_timesteps": 26,
    "fig15_days_per_timestep": 6,  # paper: "1 timestep is fitted to 6 days"

    "output_dir": ".",
}

# =============================================================================
SARS_SINGAPORE_TIMESTEP = list(range(0, 26))
SARS_SINGAPORE_PATIENTS = [
    0, 0, 3, 9, 20, 51, 17, 16, 40, 27, 12, 9, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
]


# =============================================================================
# Core model
# =============================================================================
def periodic_delta(a, b, L):
    """Minimum-image-convention difference for periodic boundary conditions."""
    d = a - b
    d -= L * np.round(d / L)
    return d


def infection_prob_normal(r, r0, w0, alpha):
    """w(r) for a normal individual: constant cutoff r0, power-law decay."""
    out = np.where(r <= r0, w0 * np.power(np.clip(1.0 - r / r0, 0.0, None), alpha), 0.0)
    return out


def infection_prob_strong(r, r0, w0):
    """w(r) for a superspreader in the STRONG INFECTIOUSNESS model: same
    cutoff r0 as normal individuals, but constant (alpha=0) instead of decaying."""
    return np.where(r <= r0, w0, 0.0)


def infection_prob_hub(r, r0, w0, hub_factor):
    """w(r) for a superspreader in the HUB model: same decaying functional
    form as normal individuals, but with a much longer cutoff sqrt(6)*r0."""
    r_star = hub_factor * r0
    out = np.where(r <= r_star, w0 * np.power(np.clip(1.0 - r / r_star, 0.0, None), 2), 0.0)
    return out


def run_single_simulation(N, L, r0, w0, gamma, lam, model, max_steps, rng,
                           alpha_normal, hub_factor):
    S, I, R = 0, 1, 2

    pos = rng.uniform(0.0, L, size=(N, 2))
    pos[0, 1] = 0.0  # initial infected individual sits at the bottom (y=0)

    is_super = rng.random(N) < lam

    state = np.full(N, S, dtype=np.int8)
    state[0] = I

    newly_infected = np.zeros(max_steps, dtype=np.int64)

    for t in range(max_steps):
        I_idx = np.where(state == I)[0]
        if I_idx.size == 0:
            break
        S_idx = np.where(state == S)[0]

        if S_idx.size > 0:
            xi = pos[I_idx, 0][:, None]
            yi = pos[I_idx, 1][:, None]
            xj = pos[S_idx, 0][None, :]
            yj = pos[S_idx, 1][None, :]

            dx = periodic_delta(xi, xj, L)
            dy = periodic_delta(yi, yj, L)
            r = np.sqrt(dx * dx + dy * dy)

            super_rows = is_super[I_idx]

            W = np.empty_like(r)
            # normal infectors (rows where super_rows is False)
            if np.any(~super_rows):
                W[~super_rows, :] = infection_prob_normal(
                    r[~super_rows, :], r0, w0, alpha_normal)
            # superspreader infectors
            if np.any(super_rows):
                if model == "strong":
                    W[super_rows, :] = infection_prob_strong(
                        r[super_rows, :], r0, w0)
                elif model == "hub":
                    W[super_rows, :] = infection_prob_hub(
                        r[super_rows, :], r0, w0, hub_factor)
                elif model == "none":
                    # lambda should be 0 in this case, but guard anyway:
                    # superspreaders (if any) behave as normal individuals
                    W[super_rows, :] = infection_prob_normal(
                        r[super_rows, :], r0, w0, alpha_normal)
                else:
                    raise ValueError(f"unknown model {model!r}")

            escape_prob = np.prod(1.0 - W, axis=0)
            infect_roll = rng.random(S_idx.size)
            newly_infected_idx = S_idx[infect_roll < (1.0 - escape_prob)]
        else:
            newly_infected_idx = np.array([], dtype=int)

        # recovery (gamma=1 in the paper => always recovers same round)
        recover_roll = rng.random(I_idx.size)
        recovered_idx = I_idx[recover_roll < gamma]

        # atomic update
        state[recovered_idx] = R
        state[newly_infected_idx] = I

        newly_infected[t] = newly_infected_idx.size

    return newly_infected


def run_many_simulations(N, L, r0, w0, gamma, lam, model, max_steps,
                          num_runs, alpha_normal, hub_factor, seed):
    """Average the epidemic curve over many Monte Carlo runs."""
    rng = np.random.default_rng(seed)
    total = np.zeros(max_steps, dtype=np.float64)
    for _ in range(num_runs):
        total += run_single_simulation(
            N, L, r0, w0, gamma, lam, model, max_steps, rng,
            alpha_normal, hub_factor)
    return total / num_runs


def n_from_density(rho_pi_r0sq, L_over_r0):
    """Convert the paper's rho*pi*r0^2 density parameter into a population N
    for a box of size L = L_over_r0 * r0."""
    return int(round(rho_pi_r0sq * (L_over_r0 ** 2) / np.pi))


# =============================================================================
# Figure 8: epidemic curves, no-superspreader / strong / hub
# =============================================================================
def make_figure_8(cfg):
    r0 = cfg["r0"]
    L = cfg["L_over_r0"] * r0
    w0 = cfg["w0"]
    gamma = cfg["gamma"]
    alpha_normal = cfg["alpha_normal"]
    hub_factor = cfg["hub_radius_factor"]
    max_steps = cfg["fig8_max_timesteps"]
    num_runs = cfg["num_runs_fig8"]
    lam = cfg["fig8_lambda"]
    N = n_from_density(cfg["fig8_rho_pi_r0sq"], cfg["L_over_r0"])

    print(f"[Fig 8] N={N}, lambda={lam}, runs={num_runs}, max_steps={max_steps}")

    t0 = time.time()
    curve_none = run_many_simulations(
        N, L, r0, w0, gamma, lam=0.0, model="none", max_steps=max_steps,
        num_runs=num_runs, alpha_normal=alpha_normal, hub_factor=hub_factor,
        seed=cfg["random_seed"])
    print(f"  no-superspreader done ({time.time()-t0:.1f}s)")

    t0 = time.time()
    curve_strong = run_many_simulations(
        N, L, r0, w0, gamma, lam=lam, model="strong", max_steps=max_steps,
        num_runs=num_runs, alpha_normal=alpha_normal, hub_factor=hub_factor,
        seed=cfg["random_seed"])
    print(f"  strong model done ({time.time()-t0:.1f}s)")

    t0 = time.time()
    curve_hub = run_many_simulations(
        N, L, r0, w0, gamma, lam=lam, model="hub", max_steps=max_steps,
        num_runs=num_runs, alpha_normal=alpha_normal, hub_factor=hub_factor,
        seed=cfg["random_seed"])
    print(f"  hub model done ({time.time()-t0:.1f}s)")

    timesteps = np.arange(max_steps)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(timesteps, curve_strong, "o-", color="crimson",
            label=f"Strong infectiousness model ($\\lambda$={lam})", markersize=5)
    ax.plot(timesteps, curve_hub, "s-", color="navy",
            label=f"Hub model ($\\lambda$={lam})", markersize=5,
            markerfacecolor="none")
    ax.plot(timesteps, curve_none, "^-", color="darkturquoise",
            label="No superspreaders ($\\lambda$=0.0)", markersize=5)
    ax.set_xlabel("time step")
    ax.set_ylabel("the number of infected")
    ax.set_title(f"Figure 8 reproduction: Epidemic curves (N={N})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{cfg['output_dir']}/fig8_epidemic_curve.png", dpi=150)
    plt.close(fig)

    with open(f"{cfg['output_dir']}/fig8_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "no_superspreader", "strong_model", "hub_model"])
        for i in range(max_steps):
            writer.writerow([i, curve_none[i], curve_strong[i], curve_hub[i]])

    return curve_none, curve_strong, curve_hub


# =============================================================================
# Figure 15: epidemic curve vs. real SARS Singapore data
# =============================================================================
def make_figure_15(cfg):
    r0 = cfg["r0"]
    L = cfg["L_over_r0"] * r0
    w0 = cfg["w0"]
    gamma = cfg["gamma"]
    alpha_normal = cfg["alpha_normal"]
    hub_factor = cfg["hub_radius_factor"]
    max_steps = cfg["fig15_max_timesteps"]
    num_runs = cfg["num_runs_fig15"]
    lam = cfg["fig15_lambda"]
    N = cfg["fig15_N"]

    print(f"[Fig 15] N={N}, lambda={lam}, runs={num_runs}, max_steps={max_steps}")

    t0 = time.time()
    curve_strong = run_many_simulations(
        N, L, r0, w0, gamma, lam=lam, model="strong", max_steps=max_steps,
        num_runs=num_runs, alpha_normal=alpha_normal, hub_factor=hub_factor,
        seed=cfg["random_seed"])
    print(f"  strong model done ({time.time()-t0:.1f}s)")

    t0 = time.time()
    curve_hub = run_many_simulations(
        N, L, r0, w0, gamma, lam=lam, model="hub", max_steps=max_steps,
        num_runs=num_runs, alpha_normal=alpha_normal, hub_factor=hub_factor,
        seed=cfg["random_seed"])
    print(f"  hub model done ({time.time()-t0:.1f}s)")

    t0 = time.time()
    curve_none = run_many_simulations(
        N, L, r0, w0, gamma, lam=0.0, model="none", max_steps=max_steps,
        num_runs=num_runs, alpha_normal=alpha_normal, hub_factor=hub_factor,
        seed=cfg["random_seed"])
    print(f"  no-superspreader done ({time.time()-t0:.1f}s)")

    timesteps = np.arange(max_steps)
    sars_t = np.array(SARS_SINGAPORE_TIMESTEP[:max_steps])
    sars_y = np.array(SARS_SINGAPORE_PATIENTS[:max_steps])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar(sars_t, sars_y, width=0.8, color="orange",
           label="data of SARS in Singapore", zorder=1)
    ax.plot(timesteps, curve_strong, "o", color="red",
            label=f"Strong infectiousness model ($\\lambda$={lam})",
            markersize=6, zorder=2)
    ax.plot(timesteps, curve_hub, "s", color="blue", markerfacecolor="none",
            label=f"Hub model ($\\lambda$={lam})", markersize=7, zorder=3)
    ax.plot(timesteps, curve_none, "^", color="cyan",
            label="($\\lambda$=0.0)", markersize=5, zorder=2)
    ax.set_xlabel("time step")
    ax.set_ylabel("number of patients")
    ax.set_title(f"Figure 15 reproduction: SARS Singapore vs. model (N={N})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{cfg['output_dir']}/fig15_epidemic_curve_sars.png", dpi=150)
    plt.close(fig)

    with open(f"{cfg['output_dir']}/fig15_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "sars_data", "no_superspreader",
                          "strong_model", "hub_model"])
        for i in range(max_steps):
            writer.writerow([i, sars_y[i] if i < len(sars_y) else 0,
                              curve_none[i], curve_strong[i], curve_hub[i]])

    return curve_none, curve_strong, curve_hub


# =============================================================================
if __name__ == "__main__":
    cfg = CONFIG
    print("Running simulations (this may take a few minutes)...")
    make_figure_8(cfg)
    make_figure_15(cfg)
    print("Done. See fig8_epidemic_curve.png and fig15_epidemic_curve_sars.png")
