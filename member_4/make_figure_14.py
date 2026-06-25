"""Recreate Figure 14 of Fujie & Odagaki (2007): the number of direct
secondary patients from probable SARS cases in Singapore (Feb 25-Apr 30,
2003), and compare it with the model link-distributions of Fig. 13.

The original Fig. 14 is a bar chart in the PDF, not a data table, so the
counts below were obtained by manually digitising the published figure
(pixel-measuring bar heights against the axis tick marks) and cross-checked
against the two textual facts the paper/CDC report give directly:

  * "Overall, 162 (81%) probable SARS cases had no evidence of transmission
    to other persons" (CDC MMWR 52(18):405-411, the paper's own source [2]),
    i.e. the k=0 bar.
  * "The SARS patients which infect 12, 21, 23 and 40 persons are
    superspreaders" (Fujie & Odagaki, Section 4) - the four isolated bars in
    the long tail.

Digitised bar heights (number of cases with exactly k direct secondary
cases): the counts sum to exactly 201, the paper's total number of probable
cases, which is a good consistency check on the digitisation.
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

FIG_DIR = os.path.join("/".join(os.path.dirname(__file__).split("/")[:-2]), "figures")

# Manually digitised from Fig. 14 of the paper (see module docstring).
SARS_SECONDARY_CASES = {
    0: 162,
    1: 20,
    2: 7,
    3: 6,
    7: 1,
    12: 1,
    21: 1,
    23: 2,
    40: 1,
}
assert sum(SARS_SECONDARY_CASES.values()) == 201


def model_probability(
    model: str, lam: float, n: int, trials: int, max_k: int
) -> np.ndarray:
    """Compute the secondary-infection probability distribution by running
    ``trials`` independent simulations and pooling ever-infected individuals.
    """
    sim = SIRSuperspreaderSimulation(model, n=n, lam=lam)
    chunks = []
    for t in range(trials):
        rng = trial_rng(model_id(model), lambda_key(lam), n, trial=t)
        result = sim.simulate(rng, record_timeseries=False, record_network=True)
        ever_infected = result.final_states != SUSCEPTIBLE
        chunks.append(result.secondary_infections[ever_infected])
    counts = np.concatenate(chunks)
    hist = np.bincount(counts, minlength=max_k + 1)[: max_k + 1]
    return hist / hist.sum()


def main(prob_strong: np.ndarray | None = None, prob_hub: np.ndarray | None = None):
    os.makedirs(FIG_DIR, exist_ok=True)

    k_max = max(SARS_SECONDARY_CASES)
    counts = np.zeros(k_max + 1, dtype=int)
    for k, c in SARS_SECONDARY_CASES.items():
        counts[k] = c
    total_cases = counts.sum()
    sars_prob = counts / total_cases

    if prob_strong is None or prob_hub is None:
        n = n_from_density(15.0)
        print(
            "Re-deriving model distributions for the comparison panel "
            "(reuse make_figures_12_13.main() if already computed) ..."
        )
        prob_strong = model_probability("strong", 0.2, n, trials=1000, max_k=k_max)
        prob_hub = model_probability("hub", 0.2, n, trials=1000, max_k=k_max)
    else:
        # Reused from Fig. 13, which only computed up to its own max_k.
        # Zero-pad to SARS's k_max so the bar arrays align.
        def _pad(p):
            out = np.zeros(k_max + 1)
            out[: len(p)] = p
            return out

        prob_strong = _pad(prob_strong)
        prob_hub = _pad(prob_hub)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left panel: SARS Singapore bar chart (faithful reproduction) ------
    ax = axes[0]
    ax.bar(np.arange(k_max + 1), counts, width=0.8, color="magenta", edgecolor="none")
    ax.set_xlim(-1, 41)
    ax.set_ylim(0, 180)
    ax.set_xlabel("number of direct secondary cases")
    ax.set_ylabel("number")
    ax.set_title("Fig. 14 — SARS, Singapore\n(Feb 25–Apr 30, 2003), N = 201 cases")
    for k in (12, 21, 23, 40):
        ax.annotate(f"{k}", (k, counts[k] + 4), ha="center", fontsize=8)

    # --- Right panel: model comparison (probability-normalised) -----------
    ax = axes[1]
    width = 0.27
    k = np.arange(k_max + 1)
    ax.bar(
        k - width,
        sars_prob,
        width=width,
        color="magenta",
        edgecolor="black",
        linewidth=0.3,
        label="SARS Singapore (data)",
    )
    ax.bar(
        k,
        prob_strong,
        width=width,
        color="red",
        edgecolor="black",
        linewidth=0.3,
        label="Strong infectiousness model",
    )
    ax.bar(
        k + width,
        prob_hub,
        width=width,
        color="blue",
        edgecolor="black",
        linewidth=0.3,
        label="Hub model",
    )
    ax.set_yscale("log")
    ax.set_ylim(2e-4, 1.0)
    ax.set_xlim(-1, 26)
    ax.set_xlabel("number of direct secondary cases")
    ax.set_ylabel("probability (log scale)")
    ax.set_title(
        "Comparison with model link-distributions\n"
        r"($\lambda=0.2$, $\rho\pi r_0^2=15.0$, cf. Fig. 13)"
    )
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    filename = os.path.join(FIG_DIR, "figure14_sars_comparison.png")
    fig.savefig(filename, dpi=160)
    plt.close(fig)
    print(f"  saved {filename}")


if __name__ == "__main__":
    main()
