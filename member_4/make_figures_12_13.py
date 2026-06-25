"""Recreate Figures 12-13 of Fujie & Odagaki (2007): the distribution of the
number of links (secondary infections) on the infection-route network.

For every trial we read off ``secondary_infections`` (the per-individual
"achievement counter": +1 every time that individual successfully infects a
susceptible neighbour - this is exactly what the engine's ``_spread`` /
``simulate`` loop accumulates with ``np.add.at(secondary, chosen_infectors, 1)``
when ``record_network=True``). We keep one count per *ever-infected*
individual (the network only contains nodes that were actually infected) and
pool these counts over many independent trials, mirroring the paper's
"averaging over 1000 Monte Carlo runs". The pooled, normalised histogram of
that pooled array is the secondary-infection probability distribution shown
in Figs. 12-13.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sir_superspreader_simulation import (
    SUSCEPTIBLE,
    SIRSuperspreaderSimulation,
    lambda_key,
    model_id,
    n_from_density,
    trial_rng,
)
from trial_data_io import save_secondary_infection_counts

DATA_DIR = os.path.join("/".join(os.path.dirname(__file__).split("/")[:-2]), "data")
FIG_DIR = os.path.join("/".join(os.path.dirname(__file__).split("/")[:-2]), "figures")

TRIALS = 1000
DENSITY = 15.0


def pooled_secondary_infections(
    model: str, lam: float, n: int, trials: int
) -> np.ndarray:
    """Run ``trials`` independent Monte-Carlo simulations and pool the
    secondary-infection counter (the "achievement" field) from all
    ever-infected individuals across all trials.
    """
    sim = SIRSuperspreaderSimulation(model, n=n, lam=lam)
    chunks = []
    for t in range(trials):
        rng = trial_rng(model_id(model), lambda_key(lam), n, trial=t)
        result = sim.simulate(rng, record_timeseries=False, record_network=True)
        ever_infected = result.final_states != SUSCEPTIBLE
        chunks.append(result.secondary_infections[ever_infected])
    return np.concatenate(chunks)


def probability_histogram(counts: np.ndarray, max_k: int) -> np.ndarray:
    """Return the probability distribution ``prob[k] = P(k secondary
    infections)`` for ``k = 0, 1, ..., max_k``.
    """
    hist = np.bincount(counts, minlength=max_k + 1)[: max_k + 1]
    return hist / hist.sum()


def plot_fig12(prob_no_super: np.ndarray, filename: str):
    """Render Figure 12: secondary-infection distribution for ``lambda=0.0``
    (no superspreaders) using cyan bars.
    """
    k = np.arange(len(prob_no_super))
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.bar(
        k,
        prob_no_super,
        width=0.8,
        color="cyan",
        edgecolor="black",
        linewidth=0.6,
        label=r"$\lambda=0.0$",
    )
    ax.set_xlim(-0.5, 20)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("the number of links")
    ax.set_ylabel("probability")
    ax.set_title(
        r"Fig. 12 — No superspreaders ($\lambda=0.0$, " r"$\rho\pi r_0^2=15.0$)"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)
    print(f"  saved {filename}")


def plot_fig13(prob_strong: np.ndarray, prob_hub: np.ndarray, filename: str):
    """Render Figure 13: side-by-side secondary-infection distributions for
    ``lambda=0.2`` (strong vs. hub models) using red and blue bars.
    """
    max_k = max(len(prob_strong), len(prob_hub)) - 1
    k = np.arange(max_k + 1)
    ps = np.zeros(max_k + 1)
    ps[: len(prob_strong)] = prob_strong
    ph = np.zeros(max_k + 1)
    ph[: len(prob_hub)] = prob_hub

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    width = 0.38
    ax.bar(
        k - width / 2,
        ps,
        width=width,
        color="red",
        edgecolor="black",
        linewidth=0.4,
        label="Strong infectiousness model",
    )
    ax.bar(
        k + width / 2,
        ph,
        width=width,
        color="blue",
        edgecolor="black",
        linewidth=0.4,
        label="Hub model",
    )
    ax.set_xlim(-0.5, 20)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("the number of links")
    ax.set_ylabel("probability")
    ax.set_title(
        r"Fig. 13 — With superspreaders ($\lambda=0.2$, " r"$\rho\pi r_0^2=15.0$)"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)
    print(f"  saved {filename}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    n = n_from_density(DENSITY)
    print(f"N={n} at rho*pi*r0^2={DENSITY}, {TRIALS} trials per configuration")

    print("Fig. 12: lambda=0.0 ...")
    counts0 = pooled_secondary_infections("hub", 0.0, n, TRIALS)
    save_secondary_infection_counts(counts0, DATA_DIR, "figure12_lambda0.0")
    prob0 = probability_histogram(counts0, max_k=20)
    plot_fig12(
        prob0, os.path.join(FIG_DIR, "figure12_link_distribution_no_superspreaders.png")
    )

    print("Fig. 13: lambda=0.2, strong model ...")
    counts_strong = pooled_secondary_infections("strong", 0.2, n, TRIALS)
    save_secondary_infection_counts(
        counts_strong, DATA_DIR, "figure13_strong_lambda0.2"
    )
    prob_strong = probability_histogram(counts_strong, max_k=20)

    print("Fig. 13: lambda=0.2, hub model ...")
    counts_hub = pooled_secondary_infections("hub", 0.2, n, TRIALS)
    save_secondary_infection_counts(counts_hub, DATA_DIR, "figure13_hub_lambda0.2")
    prob_hub = probability_histogram(counts_hub, max_k=20)

    plot_fig13(
        prob_strong,
        prob_hub,
        os.path.join(FIG_DIR, "figure13_link_distribution_with_superspreaders.png"),
    )

    return prob0, prob_strong, prob_hub


if __name__ == "__main__":
    main()
