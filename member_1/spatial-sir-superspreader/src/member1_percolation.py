"""Execution script reproducing Figures 1-5 of Fujie & Odagaki (2007).

This script imports the simulation engine from
:mod:`sir_superspreader_simulation` and carries out two tasks:

    Task A (Figures 1 & 2)
        Static plots of the infection probability ``w(r)`` for the strong
        infectiousness model and the hub model.  No simulation is run.

    Task B (Figures 3, 4 & 5)
        Monte-Carlo estimation of the percolation probability as a function of
        density and superspreader fraction ``lambda`` for both models, and the
        resulting critical-density curve.

Run with (from the repository root)::

    python src/member1_percolation.py

All tunable parameters live in the "Experiment configuration" block below.
Defaults favour a *fast preview*; raise ``TRIALS`` to 1000 and lower
``N_STEP`` for publication-quality curves.
"""

from __future__ import annotations

import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sir_superspreader_simulation as engine
from sir_superspreader_simulation import (
    R0,
    RN,
    W0,
    MonteCarloSweep,
    w_hub_superspreader,
    w_strong_superspreader,
    w_normal,
)

logger = logging.getLogger(__name__)

# Output directory for all generated figures (``<repo>/figures``), resolved
# relative to this file so the script works from any working directory.
FIGURES_DIR: Path = Path(__file__).resolve().parent.parent / "figures"


def _figure_path(filename: str) -> Path:
    """Return the absolute path of ``filename`` inside the figures directory."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / filename


# --------------------------------------------------------------------------- #
# Experiment configuration.
# --------------------------------------------------------------------------- #
# Number of independent Monte-Carlo trials per (model, N, lambda) point.
# The paper uses 1000; we default to a fast-preview value.
# TRIALS: int = 200
TRIALS: int = 1000

# Population sweep (upper bound and step; lower bound N_MIN is set below).
N_MAX: int = 900
N_STEP: int = 15  # density step ~0.47; lower for a finer grid.

# Superspreader fractions to scan.
LAMBDAS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

# Models to simulate (label -> engine model key).
MODELS: dict[str, str] = {"Strong infectiousness model": "strong", "Hub model": "hub"}


# --------------------------------------------------------------------------- #
# Seeding.
#
# A single infected individual is placed at the bottom edge (y = 0), faithful to
# the paper's wording ("an initial-infected individual").  N_MIN is kept low so
# the sweep covers the full low-density rise (rho*pi*r_0^2 down to ~0.5) that the
# paper's Figs 3-4 show, and every lambda's 0.5 crossing is captured.
# --------------------------------------------------------------------------- #
N_MIN: int = 15

# Plot styling per lambda, chosen to echo the paper's figures.
_LAMBDA_STYLE: dict[float, dict] = {
    0.0: dict(color="red", marker="o", mfc="none"),
    0.2: dict(color="green", marker="o"),
    0.4: dict(color="magenta", marker="s"),
    0.6: dict(color="black", marker="s", mfc="none"),
    0.8: dict(color="cyan", marker="^", mfc="none"),
    1.0: dict(color="gold", marker="^"),
}


# --------------------------------------------------------------------------- #
# Task A: static infection-probability plots (Figures 1 & 2).
# --------------------------------------------------------------------------- #
def plot_strong_model_probability(filename: str = "figure1_strong_wr.png") -> None:
    """Figure 1: w(r)/w_0 vs r/r_0 for the strong infectiousness model."""
    # Sample a little past the cutoff so the superspreader curve is shown
    # dropping to zero at r = r_0, as in w(r) = w_0 for r <= r_0 and 0 otherwise.
    r = np.linspace(0.0, 1.1 * R0, 600)
    fig, ax = plt.subplots(figsize=(5, 4))
    # Superspreader: constant w/w_0 = 1 up to the cutoff, then zero beyond it.
    ax.plot(r / R0, w_strong_superspreader(r) / W0,
            color="orange", lw=2, label="superspreader")
    ax.plot(r / R0, w_normal(r) / W0,
            color="cyan", lw=2, ls="--", label="normal")

    # Shared cutoff at r = r_0: a black dashed vertical line.  The axes run a
    # little past 1.0 so the superspreader line and the cutoff are not flush
    # against the plot margins.
    ax.axvline(R0 / R0, color="black", ls="--", lw=1.3)
    ax.set_xlabel(r"$r / r_0$")
    ax.set_ylabel(r"$w(r) / w_0$")
    ax.set_title("Fig. 1 — Strong infectiousness model")
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.legend()
    fig.tight_layout()
    saved = _figure_path(filename)
    fig.savefig(saved, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", saved)


def plot_hub_model_probability(filename: str = "figure2_hub_wr.png") -> None:
    """Figure 2: w(r)/w_0 vs r/r_0 for the hub model."""
    # Extend past the superspreader cutoff (r_n/r_0 = sqrt(6) ~ 2.449) so the
    # solid line is visibly seen decaying to (and staying at) zero.
    r = np.linspace(0.0, 2.6 * R0, 600)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(r / R0, w_hub_superspreader(r) / W0,
            color="orange", lw=2, label="superspreader")
    ax.plot(r / R0, w_normal(r) / W0,
            color="cyan", lw=2, ls="--", label="normal")

    ax.set_xlabel(r"$r / r_0$")
    ax.set_ylabel(r"$w(r) / w_0$")
    ax.set_title("Fig. 2 — Hub model")
    ax.set_xlim(0, 2.6)
    ax.set_ylim(0, 1.0)
    ax.legend()
    fig.tight_layout()
    saved = _figure_path(filename)
    fig.savefig(saved, dpi=150)
    plt.close(fig)
    logger.info("Saved %s (hub superspreader cutoff r_n/r_0 = %.3f)", saved, RN / R0)


# --------------------------------------------------------------------------- #
# Task B: percolation probability (Figures 3, 4 & 5).
# --------------------------------------------------------------------------- #
def run_percolation_sweep(model: str, pool: Pool,
                          ) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Compute percolation probabilities for one model over the (N, lambda) grid.

    All simulation work is delegated to the shared engine driver
    :class:`MonteCarloSweep`; the default ``measure_trial`` already returns the
    percolation flag and the default ``reduce_trials`` averages it, so Member 1
    needs no overrides - just the grid.  The single-bottom-seed default
    (``seed_band=None``) is used.

    Returns
    -------
    densities : np.ndarray
        Density values ``rho * pi * r_0^2`` corresponding to ``N``.
    prob_by_lambda : dict
        Mapping ``lambda -> array`` of percolation probabilities.
    """
    n_values = np.arange(N_MIN, N_MAX + 1, N_STEP)
    sweep = MonteCarloSweep(trials=TRIALS, lambdas=LAMBDAS, n_values=n_values)
    densities, result_by_lambda = sweep.run(model, pool)

    # The driver returns lists; the plotting / critical-density helpers want arrays.
    prob_by_lambda = {lam: np.asarray(vals, dtype=float)
                      for lam, vals in result_by_lambda.items()}
    return densities, prob_by_lambda


def _critical_density(densities: np.ndarray, probabilities: np.ndarray,
                      threshold: float = 0.5) -> float | None:
    """Density at which the percolation probability first crosses ``threshold``.

    Linearly interpolates between the two bracketing grid points.  Returns
    ``None`` if the curve never crosses the threshold within the sweep.
    """
    below = probabilities < threshold
    for i in range(len(densities) - 1):
        if below[i] and not below[i + 1]:
            x0, x1 = densities[i], densities[i + 1]
            y0, y1 = probabilities[i], probabilities[i + 1]
            if y1 == y0:
                return float(x0)
            return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))
    return None


def plot_percolation_curves(model_key: str, densities: np.ndarray,
                            prob_by_lambda: dict[float, np.ndarray],
                            fig_label: str, filename: str) -> None:
    """Figures 3 / 4: percolation probability vs density for several lambda."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for lam in LAMBDAS:
        style = _LAMBDA_STYLE[lam]
        ax.plot(densities, prob_by_lambda[lam], ls="none", ms=6,
                label=fr"$\lambda={lam:.1f}$", **style)

    ax.set_xlabel(r"$\rho \pi r_0^2$")
    ax.set_ylabel("percolation probability")
    ax.set_title(fig_label, fontsize=9)
    ax.set_xlim(0, 25)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(title=None, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    saved = _figure_path(filename)
    fig.savefig(saved, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", saved)


def _infection_integral(w_func, r_max: float) -> float:
    """Return ``(1/w_0) * integral of w(r) * 2*pi*r dr`` over ``[0, r_max]``.

    This is the mean number of new infections per unit density caused by one
    spreader of the given kind (the integral term in Eq. 3 of the paper).
    """
    r = np.linspace(0.0, r_max, 4000)
    return float(np.trapezoid(w_func(r) * 2.0 * np.pi * r, r) / W0)


def critical_density_theory(model_key: str, lambdas: np.ndarray,
                            rc: float) -> np.ndarray:
    """Analytic critical density ``rho_c*pi*r_0^2`` from the condition R0 = Rc.

    From Eq. (3), ``R0(lambda) = rho * [lambda*I_super + (1-lambda)*I_normal]``.
    Setting ``R0 = Rc`` and multiplying by ``pi*r_0^2`` gives the critical curve.

    Following the paper (Eqs 4 & 5), ``rc`` is the critical basic reproductive
    number, taken as the model's *measured* critical density at ``lambda = 1``
    (see :func:`measured_rc`).  Anchoring the curve on this run's own
    ``lambda = 1`` point is what makes the simulation markers sit on the line,
    rather than pasting the paper's numeric constants which were measured on a
    different run.
    """
    i_normal = _infection_integral(w_normal, R0)
    if model_key == "strong":
        i_super = _infection_integral(w_strong_superspreader, R0)
    else:
        i_super = _infection_integral(w_hub_superspreader, RN)

    mean_per_density = lambdas * i_super + (1.0 - lambdas) * i_normal
    return rc * np.pi * R0 ** 2 / mean_per_density


def measured_rc(points: list[tuple[float, float]]) -> float | None:
    """Critical basic reproductive number Rc = (rho_c pi r_0^2) at ``lambda = 1``.

    This is the paper's definition (Eqs 4 & 5): Rc is the measured critical
    density of the *fully-superspreader* system.  ``points`` is the list of
    ``(lambda, rho_c)`` pairs for one model; returns ``None`` if there is no
    ``lambda = 1`` entry (e.g. that crossing fell outside the swept range).
    """
    for lam, rho_c in points:
        if abs(lam - 1.0) < 1e-9:
            return rho_c
    return None


def plot_critical_density(critical_by_model: dict[str, list[tuple[float, float]]],
                          rc_by_model: dict[str, float],
                          filename: str = "figure5_critical_density.png") -> None:
    """Figure 5: critical density rho_c pi r_0^2 vs lambda for both models.

    Styling mirrors the paper: simulation results are discrete markers (red
    filled circles for the strong model, open blue squares for the hub model)
    and the ``R0 = Rc`` theory curves are lines (solid green / dashed magenta).
    Each model's ``Rc`` (``rc_by_model``) is its own measured ``lambda = 1``
    critical density, so the curve is anchored on this run's data.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title("Fig. 5 — Critical density vs superspreader fraction", fontsize=10)
    lam_grid = np.linspace(0.0, 1.0, 200)

    sim_styles = {
        "strong": dict(color="red", marker="o",
                       label="Strong infectiousness model (simulation)"),
        "hub": dict(color="blue", marker="s", mfc="none",
                    label="Hub model (simulation)"),
    }
    theory_styles = {
        "strong": dict(color="green", ls="-"),
        "hub": dict(color="magenta", ls="--"),
    }

    # Plot each model's simulation markers first, then its theory curve, so the
    # legend reads strong (sim, theory) then hub (sim, theory) as in the paper.
    for model_key in ("strong", "hub"):
        points = critical_by_model.get(model_key, [])
        if points:
            lams, rho_c = zip(*points)
            ax.plot(lams, rho_c, ls="none", ms=7, **sim_styles[model_key])
        rc = rc_by_model.get(model_key)
        if rc is None:
            continue
        ax.plot(lam_grid, critical_density_theory(model_key, lam_grid, rc),
                lw=2, label=fr"$R_0 = R_c$ ($R_c={rc:.1f}$)",
                **theory_styles[model_key])

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\rho_c \pi r_0^2$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    saved = _figure_path(filename)
    fig.savefig(saved, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", saved)


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Constants: r_0=%.3f, w_0=%.3f, L=%.3f, r_n=%.4f",
                R0, W0, engine.L, RN)

    # ----- Task A: static probability plots (no simulation) ----- #
    logger.info("Task A: generating static w(r) plots (Figures 1 & 2).")
    plot_strong_model_probability()
    plot_hub_model_probability()

    # ----- Task B: percolation Monte-Carlo (Figures 3, 4, 5) ----- #
    logger.info("Task B: Monte-Carlo percolation sweep on %d worker(s).", cpu_count())
    figure_bases = {"strong": "figure3_strong_percolation",
                    "hub": "figure4_hub_percolation"}
    figure_titles = {"strong": "Fig. 3 — Strong infectiousness model",
                     "hub": "Fig. 4 — Hub model"}

    with Pool() as pool:
        critical_by_model: dict[str, list[tuple[float, float]]] = {}
        for label, model_key in MODELS.items():
            densities, prob_by_lambda = run_percolation_sweep(model_key, pool)
            plot_percolation_curves(
                model_key, densities, prob_by_lambda,
                figure_titles[model_key],
                f"{figure_bases[model_key]}.png",
            )

            # Collect critical densities (prob = 0.5) for Figure 5.
            points: list[tuple[float, float]] = []
            for lam in LAMBDAS:
                rho_c = _critical_density(densities, prob_by_lambda[lam])
                if rho_c is not None:
                    points.append((lam, rho_c))
                else:
                    logger.warning(
                        "%s, lambda=%.1f: no 0.5 crossing in N=[%d, %d].",
                        label, lam, N_MIN, N_MAX,
                    )
            critical_by_model[model_key] = points

        # Rc per model = measured critical density at lambda=1 (paper Eqs 4 & 5).
        rc_by_model: dict[str, float] = {}
        for model_key, points in critical_by_model.items():
            rc = measured_rc(points)
            if rc is not None:
                rc_by_model[model_key] = rc
                logger.info("Rc[%s] = rho_c*pi*r_0^2 at lambda=1 = %.2f",
                            model_key, rc)
            else:
                logger.warning("Rc[%s] undefined: no lambda=1 crossing.", model_key)

        plot_critical_density(critical_by_model, rc_by_model)

    logger.info("All figures generated.")


if __name__ == "__main__":
    main()
