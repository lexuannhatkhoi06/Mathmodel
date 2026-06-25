"""Persistence helpers for Member 4's figures (9-14).

This module adds the one missing piece the team's shared engine
(``sir_superspreader_simulation.py``) does not provide on its own: writing the
per-individual coordinates and the chronological infection log of a trial to
disk (and reading them back), so that the spatial "route of infection" maps
(Figs. 9-11) and the secondary-infection counters (Figs. 12-13) are built from
*saved* data rather than only from objects still sitting in memory.

Everything here is a thin wrapper around :class:`TrialResult` - the underlying
random process, infection rule and the ``secondary_infections`` "achievement
counter" array (incremented once per successful I -> S transmission) all
already live in the shared engine; we only persist what it produces.
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sir_superspreader_simulation import SUSCEPTIBLE, TrialResult


def save_network_data(result: TrialResult, outdir: str, prefix: str) -> tuple[str, str]:
    """Persist one trial's coordinates and infection log to CSV files.

    A trial must be run with ``record_network=True`` for ``infection_log`` and
    ``secondary_infections`` to be populated.

    Parameters
    ----------
    result : TrialResult
        The trial result object (must have ``record_network=True``).
    outdir : str
        Output directory (created if missing).
    prefix : str
        Filename prefix (e.g., ``"fig09_strong"``).

    Returns
    -------
    tuple[str, str]
        (coordinates_csv_path, infection_log_csv_path)
    """
    os.makedirs(outdir, exist_ok=True)

    # Write coordinates, status, and secondary-infection counter per individual.
    coord_path = os.path.join(outdir, f"{prefix}_coordinates.csv")
    with open(coord_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["index", "x", "y", "is_super", "final_state", "secondary_infections"]
        )
        secondary = (
            result.secondary_infections
            if result.secondary_infections is not None
            else np.zeros(len(result.positions), dtype=int)
        )
        for i in range(len(result.positions)):
            writer.writerow(
                [
                    i,
                    f"{result.positions[i, 0]:.6f}",
                    f"{result.positions[i, 1]:.6f}",
                    int(result.is_super[i]),
                    int(result.final_states[i]),
                    int(secondary[i]),
                ]
            )

    # Write infection events (chronological transmission record).
    log_path = os.path.join(outdir, f"{prefix}_infection_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["infector", "infectee", "time_step"])
        writer.writerows(result.infection_log)

    return coord_path, log_path


def load_network_data(coord_path: str, log_path: str):
    """Reload coordinates and infection log from CSV (inverse of
    :func:`save_network_data`).

    Returns
    -------
    positions : np.ndarray, shape (N, 2)
        Individual positions (x, y).
    is_super : np.ndarray of bool, shape (N,)
        Superspreader status per individual.
    final_states : np.ndarray of int, shape (N,)
        Final SIR state per individual.
    secondary_infections : np.ndarray of int, shape (N,)
        Secondary infection count per individual.
    infection_log : list[tuple[int, int, int]]
        Chronological transmission events (infector, infectee, timestep).
    """
    positions, is_super, final_states, secondary = [], [], [], []
    with open(coord_path, newline="") as f:
        for row in csv.DictReader(f):
            positions.append((float(row["x"]), float(row["y"])))
            is_super.append(bool(int(row["is_super"])))
            final_states.append(int(row["final_state"]))
            secondary.append(int(row["secondary_infections"]))

    infection_log = []
    with open(log_path, newline="") as f:
        for row in csv.DictReader(f):
            infection_log.append(
                (int(row["infector"]), int(row["infectee"]), int(row["time_step"]))
            )

    return (
        np.array(positions),
        np.array(is_super, dtype=bool),
        np.array(final_states, dtype=int),
        np.array(secondary, dtype=int),
        infection_log,
    )


def save_secondary_infection_counts(
    counts: np.ndarray, outdir: str, prefix: str
) -> str:
    """Persist the pooled secondary-infection counter array (one count per
    ever-infected individual across all trials).  This allows rebuilding
    Figs. 12-13 without re-running the simulation sweep.
    """
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{prefix}_secondary_infection_counts.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["secondary_infections"])
        writer.writerows([[c] for c in counts.tolist()])
    return path


def load_secondary_infection_counts(path: str) -> np.ndarray:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return np.array([int(r["secondary_infections"]) for r in rows], dtype=int)
