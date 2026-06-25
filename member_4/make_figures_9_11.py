"""Recreate Figures 9-11 of Fujie & Odagaki (2007): the spatial "route of
infection" maps for the strong-infectiousness model, the hub model, and the
no-superspreader case, all at lambda and rho*pi*r0^2 matching the paper's
captions (lambda=0.2 / 0.0, rho*pi*r0^2=15.0).

For each panel we:
  1. run a single Monte-Carlo trial with network recording on,
  2. SAVE its coordinates + infection log to CSV (trial_data_io.save_network_data),
  3. immediately re-LOAD that CSV (trial_data_io.load_network_data) and draw the
     scatter/annotate plot purely from the on-disk data, exactly matching the
     workflow requested ("save coordinates and infection log ... use this
     data to draw points and arrows").
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
    L,
    SUSCEPTIBLE,
    SIRSuperspreaderSimulation,
    density_from_n,
    lambda_key,
    model_id,
    n_from_density,
    trial_rng,
)
from trial_data_io import load_network_data, save_network_data

DATA_DIR = os.path.join("/".join(os.path.dirname(__file__).split("/")[:-2]), "data")
FIG_DIR = os.path.join("/".join(os.path.dirname(__file__).split("/")[:-2]), "figures")

# Marker styles per infection status, mirroring the paper's legend.
# All individuals use same dot size; distinction is by color (black vs blue)
# and fill pattern (filled vs open). "S" (never infected) is open/black;
# "I" (ever infected, now recovered) is filled/royalblue.
STYLE = {
    "S_super": dict(
        facecolor="black", edgecolor="black", s=20, linewidths=0.6, zorder=3
    ),
    "S_normal": dict(
        facecolor="none", edgecolor="black", s=20, linewidths=0.6, zorder=2
    ),
    "I_super": dict(
        facecolor="royalblue", edgecolor="royalblue", s=20, linewidths=0.6, zorder=4
    ),
    "I_normal": dict(
        facecolor="none", edgecolor="royalblue", s=20, linewidths=0.7, zorder=3
    ),
}
LABELS = {
    "S_super": "S (superspreader)",
    "S_normal": "S (normal)",
    "I_super": "I (superspreader)",
    "I_normal": "I (normal)",
}


def run_and_save(model: str, lam: float, n: int, trial: int, prefix: str):
    """Run one trial with ``record_network=True`` and persist to CSV."""
    sim = SIRSuperspreaderSimulation(model, n=n, lam=lam)
    rng = trial_rng(model_id(model), lambda_key(lam), n, trial=trial)
    result = sim.simulate(rng, record_timeseries=False, record_network=True)
    coord_path, log_path = save_network_data(result, DATA_DIR, prefix)
    print(
        f"  [{prefix}] N={n} total_infected={result.total_infected} "
        f"(superspreaders={int(result.is_super.sum())}) -> {coord_path}"
    )
    return coord_path, log_path


def plot_route_of_infection(
    coord_path: str, log_path: str, title: str, filename: str, has_superspreaders: bool
):
    positions, is_super, final_states, secondary, infection_log = load_network_data(
        coord_path, log_path
    )
    ever_infected = final_states != SUSCEPTIBLE

    fig, ax = plt.subplots(figsize=(6.4, 6.9))

    # Infection-route arrows first, so markers sit on top of the lines.
    # The x-axis is periodic. For arrows that cross the boundary we draw two
    # segments: a tail from the infector to the exit edge, and a head from the
    # re-entry edge to the infectee. This keeps all arrow segments inside [0, L].
    _ap_normal = dict(arrowstyle="-|>", color="black", lw=0.4, alpha=0.6,
                      shrinkA=1.2, shrinkB=1.2, mutation_scale=6)
    _ap_tail   = dict(arrowstyle="-",   color="black", lw=0.4, alpha=0.6,
                      shrinkA=1.2, shrinkB=0)
    _ap_head   = dict(arrowstyle="-|>", color="black", lw=0.4, alpha=0.6,
                      shrinkA=0,   shrinkB=1.2, mutation_scale=6)

    for src, dst, _ in infection_log:
        sx, sy = positions[src]
        dx, dy = positions[dst]
        ddx = dx - sx
        if ddx > L / 2:
            # Crosses from right side to left: exit at x=0, re-enter at x=L
            dx_w = dx - L                        # wrapped destination (negative)
            t = sx / (sx - dx_w)                 # parameter at x=0
            y_exit = sy + t * (dy - sy)
            ax.annotate("", xy=(0, y_exit), xytext=(sx, sy),
                        arrowprops=_ap_tail, zorder=1)
            ax.annotate("", xy=(dx, dy), xytext=(L, y_exit),
                        arrowprops=_ap_head, zorder=1)
        elif ddx < -L / 2:
            # Crosses from left side to right: exit at x=L, re-enter at x=0
            dx_w = dx + L                        # wrapped destination (> L)
            t = (L - sx) / (dx_w - sx)          # parameter at x=L
            y_exit = sy + t * (dy - sy)
            ax.annotate("", xy=(L, y_exit), xytext=(sx, sy),
                        arrowprops=_ap_tail, zorder=1)
            ax.annotate("", xy=(dx, dy), xytext=(0, y_exit),
                        arrowprops=_ap_head, zorder=1)
        else:
            ax.annotate("", xy=(dx, dy), xytext=(sx, sy),
                        arrowprops=_ap_normal, zorder=1)

    if has_superspreaders:
        groups = [
            ("S_super", (~ever_infected) & is_super),
            ("S_normal", (~ever_infected) & ~is_super),
            ("I_super", ever_infected & is_super),
            ("I_normal", ever_infected & ~is_super),
        ]
    else:
        groups = [
            ("S_normal", ~ever_infected),
            ("I_normal", ever_infected),
        ]

    for key, mask in groups:
        ax.scatter(
            positions[mask, 0], positions[mask, 1], label=LABELS[key], **STYLE[key]
        )

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    ax.annotate(
        "",
        xy=(1.18, 0.97),
        xytext=(1.02, 0.97),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0),
    )
    ax.text(
        1.20,
        0.965,
        "route of infection",
        transform=ax.transAxes,
        fontsize=9,
        va="center",
    )
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 0.88),
        frameon=False,
        fontsize=9,
        markerscale=1.0,
        handletextpad=0.6,
        labelspacing=1.1,
    )
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(filename, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {filename}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    density = 15.0
    n = n_from_density(density)
    print(
        f"N individuals at ρπr₀²={density}: N={n} "
        f"(density_from_n check: {density_from_n(n):.2f})"
    )

    # Configuration for each of the three panels (strong model λ=0.2, hub
    # model λ=0.2, hub model λ=0.0 for the no-superspreader case).
    configs = [
        dict(
            model="strong",
            lam=0.2,
            trial=0,
            prefix="figure9_strong",
            title=r"Fig. 9 — Strong infectiousness model ($\lambda=0.2$, "
            r"$\rho\pi r_0^2=15.0$)",
            filename=os.path.join(FIG_DIR, "figure9_route_strong_infectiousness.png"),
            has_super=True,
        ),
        dict(
            model="hub",
            lam=0.2,
            trial=0,
            prefix="figure10_hub",
            title=r"Fig. 10 — Hub model ($\lambda=0.2$, $\rho\pi r_0^2=15.0$)",
            filename=os.path.join(FIG_DIR, "figure10_route_hub_model.png"),
            has_super=True,
        ),
        dict(
            model="hub",
            lam=0.0,
            trial=0,
            prefix="figure11_nosuper",
            title=r"Fig. 11 — No superspreaders ($\lambda=0.0$, "
            r"$\rho\pi r_0^2=15.0$)",
            filename=os.path.join(FIG_DIR, "figure11_route_no_superspreaders.png"),
            has_super=False,
        ),
    ]

    print("Running trials and saving coordinates / infection logs ...")
    for cfg in configs:
        coord_path, log_path = run_and_save(
            cfg["model"], cfg["lam"], n, cfg["trial"], cfg["prefix"]
        )
        plot_route_of_infection(
            coord_path, log_path, cfg["title"], cfg["filename"], cfg["has_super"]
        )


if __name__ == "__main__":
    main()
