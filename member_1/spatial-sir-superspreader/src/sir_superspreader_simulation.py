"""Core Monte-Carlo simulation engine for a 2D continuous-space SIR model
with superspreaders.

This module implements the model of:

    R. Fujie & T. Odagaki,
    "Effects of superspreaders in spread of epidemic",
    Physica A 374 (2007) 843-852.

It is the **shared engine** for the whole team.  Each member extracts a
different metric from *the same* simulation machinery (see :class:`TrialResult`):

    * Member 1 - percolation probability (Figs 1-5):  ``TrialResult.percolated``
    * Member 2 - propagation speed (Figs 6-7):         ``TrialResult.front_distance_per_step``
    * Member 3 - epidemic curve (Figs 8, 15):          ``TrialResult.new_infections_per_step``
    * Member 4 - network & secondary distribution      ``TrialResult.infection_log``,
                 (Figs 9-14):                          ``TrialResult.secondary_infections``,
                                                        ``TrialResult.positions`` / ``is_super``

The module is intentionally free of any *plotting* logic so that it can be
imported by each member's execution script.  It does, however, own the shared
**sweep driver** (:class:`MonteCarloSweep`): the generic "scan a grid of
``(model, N, lambda)`` points, run ``TRIALS`` trials at each, in parallel"
machinery.  Every member runs *the same* simulation code path through it and only
overrides what to *measure* per trial - so all members' runs are the same
underlying realisations (see :class:`MonteCarloSweep`).

Model summary
-------------
* ``N`` individuals with fixed positions on an ``L x L`` continuous space,
  ``L = 10 r_0``.  Boundary conditions are **periodic in the transverse (x)
  axis only**; the propagation (y) axis is open so the bottom (y = 0) and top
  (y = L) are distinct (required for the bottom-to-top percolation measurement).
* Three states: S (susceptible), I (infected), R (recovered).
* Synchronous update: in one time step every *currently* infected individual
  tries to infect each susceptible neighbour with probability ``w(r)`` (an
  independent Bernoulli draw per pair).  Newly infected individuals are queued
  and cannot spread during the same step.  At the end of the step the previously
  infected individuals recover (``gamma = 1`` - infectious for exactly one step).
* A fraction ``lambda`` of all individuals are superspreaders.

Reproducibility
---------------
Build every generator with :func:`trial_rng` (or the lower-level
:func:`make_rng`).  Both rely on :class:`numpy.random.SeedSequence`, which is
deterministic and platform-independent (unlike Python's salted ``hash``), so
every member obtains identical streams from identical keys on any machine.

A Monte-Carlo realisation is addressed by its **physical** configuration only::

    rng = trial_rng(model_id(model), lambda_key(lam), n, trial=t)

(this is what :class:`MonteCarloSweep` does internally).  Because the key is the
physics - model, superspreader fraction and population size - and *not* an array
index or a per-member tag, any member who evaluates the same ``(model, lam, N)``
point gets the *same* spatial layout and the *same* epidemic; they simply read a
different observable off it.  Crucially, **each trial is seeded independently**
(the trial number is the last key), so trial ``t`` is identical regardless of how
many random numbers any other trial consumes - e.g. whether a run stops early at
percolation (Member 1) or proceeds to extinction (Members 2-4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Global model constants (Section 2 / 3 of the paper).
# --------------------------------------------------------------------------- #
R0: float = 1.0                 # normal-individual cutoff radius r_0
W0: float = 1.0                 # maximum infection probability w_0
L: float = 10.0 * R0            # linear system size (L = 10 r_0)
GAMMA: float = 1.0              # recovery probability (instantaneous recovery)
RN: float = np.sqrt(6.0) * R0   # hub-superspreader cutoff r_n = sqrt(6) r_0

# State encoding used throughout the engine.
SUSCEPTIBLE: int = 0
INFECTED: int = 1
RECOVERED: int = 2

# Model identifiers understood by the engine and by :func:`make_rng`.
MODELS: tuple[str, str] = ("strong", "hub")


# --------------------------------------------------------------------------- #
# Static infection-probability functions w(r).
#
# These accept scalars or NumPy arrays and are used both by the simulation
# and by the static probability plots (Figures 1 and 2).
# --------------------------------------------------------------------------- #
def w_normal(r: np.ndarray | float) -> np.ndarray:
    """Normal individual: ``w(r) = w_0 (1 - r/r_0)^2`` for ``r <= r_0``."""
    r = np.asarray(r, dtype=float)
    return np.where(r <= R0, W0 * (1.0 - r / R0) ** 2, 0.0)


def w_strong_superspreader(r: np.ndarray | float) -> np.ndarray:
    """Strong superspreader: constant ``w(r) = w_0`` for ``r <= r_0`` (alpha=0)."""
    r = np.asarray(r, dtype=float)
    return np.where(r <= R0, W0, 0.0)


def w_hub_superspreader(r: np.ndarray | float) -> np.ndarray:
    """Hub superspreader: ``w(r) = w_0 (1 - r/r_n)^2`` for ``r <= r_n``."""
    r = np.asarray(r, dtype=float)
    return np.where(r <= RN, W0 * (1.0 - r / RN) ** 2, 0.0)


# --------------------------------------------------------------------------- #
# Per-kind infection parameters.
#
# Every spreader's probability is written in the unified form
#
#       w(r) = w_0 * (1 - r/scale)^exponent     for r <= cutoff
#
# so that a single vectorised expression covers all four cases:
#   * normal (both models): scale = cutoff = r_0,  exponent = 2
#   * strong superspreader: cutoff = r_0,          exponent = 0  -> constant w_0
#   * hub superspreader:    scale = cutoff = r_n,  exponent = 2
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InfectionKind:
    """Distance-dependent infection parameters for one class of spreader."""

    cutoff: float      # interaction range (neighbours beyond this never infected)
    scale: float       # length scale used inside the (1 - r/scale) term
    exponent: float    # exponent alpha (0 -> constant probability w_0)


# Normal individuals are identical in both models.
_NORMAL_KIND = InfectionKind(cutoff=R0, scale=R0, exponent=2.0)

# Superspreader parameters depend on the model.
_SUPERSPREADER_KIND = {
    "strong": InfectionKind(cutoff=R0, scale=R0, exponent=0.0),
    "hub": InfectionKind(cutoff=RN, scale=RN, exponent=2.0),
}


# The KD-tree "boxsize" used for neighbour search.  The transverse (x) axis is
# periodic with period L; the propagation (y) axis is left open.  Open boundaries
# are emulated by giving y a period of 2*L: since every pairwise |dy| < L, the
# minimum-image term 2*L - |dy| always exceeds |dy|, so no wrapping ever occurs.
KDTREE_BOXSIZE: tuple[float, float] = (L, 2.0 * L)


def make_rng(*keys: int) -> np.random.Generator:
    """Build a reproducible NumPy generator from a tuple of integer keys.

    Example::

        rng = make_rng(PROJECT_SEED, model_id(model), lambda_key(lam), n)

    Most callers should prefer :func:`trial_rng` (the project convention) over
    calling this directly.  Uses :class:`numpy.random.SeedSequence`, which is
    deterministic across
    platforms and processes (Python's built-in ``hash`` is salted per process
    and must *not* be used for seeding shared experiments).
    """
    return np.random.default_rng(np.random.SeedSequence(list(keys)))


def model_id(model: str) -> int:
    """Map a model name to a stable integer (handy as a :func:`trial_rng` key)."""
    return MODELS.index(model)


def lambda_key(lam: float) -> int:
    """Encode a superspreader fraction as a stable integer rng key (0.4 -> 400).

    Multiplying by 1000 and rounding lets any ``lambda`` resolved to three
    decimals act as a deterministic, platform-independent key.
    """
    return int(round(lam * 1000))


# Master seed for every reproducible experiment in this project.  All team
# members derive their generators from this one constant, so identical
# configurations yield identical Monte-Carlo realisations on any machine.
PROJECT_SEED: int = 20070101


def trial_rng(*config_keys: int, trial: int) -> np.random.Generator:
    """Canonical per-trial generator shared by the whole team.

    Returns the generator for trial number ``trial`` of the configuration
    labelled by ``config_keys``.  The result is
    ``make_rng(PROJECT_SEED, *config_keys, trial)``.

    The project convention - used by :class:`MonteCarloSweep` and therefore by
    every member - keys on the **physical configuration**::

        trial_rng(model_id(model), lambda_key(lam), n, trial=t)

    Keying on the physics (not on array indices or a per-member id) is what makes
    realisations *shared*: any member evaluating the same ``(model, lam, N)`` point
    obtains the same layout and the same epidemic, and merely reads a different
    observable.  And because each trial is seeded *independently* (``trial`` is the
    last key), trial ``t`` is identical no matter how many random numbers any other
    trial consumes - e.g. whether the run stops early at percolation or runs to
    extinction.

    Examples
    --------
    >>> for t in range(1000):
    ...     rng = trial_rng(model_id("hub"), lambda_key(0.4), 477, trial=t)
    ...     result = sim.simulate(rng)          # or sim.run_trial(rng)
    """
    return make_rng(PROJECT_SEED, *config_keys, trial)


def toroidal_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance with periodicity on the transverse (x) axis only.

    The system is a strip: periodic boundary conditions wrap the x-axis with
    period ``L``, while the propagation (y) axis has open boundaries so that the
    bottom (y = 0) and top (y = L) are genuinely distinct.  This geometry is
    required for the bottom-to-top percolation measurement; wrapping y as well
    would make the seed at y = 0 adjacent to the top edge and trivialise the
    percolation.

    Parameters
    ----------
    a, b : np.ndarray
        Coordinate arrays.  ``a`` has shape ``(M, 2)``; ``b`` is either
        ``(M, 2)`` or a single point ``(2,)`` that broadcasts against ``a``.

    Returns
    -------
    np.ndarray
        Shape ``(M,)`` array of minimum-image distances.
    """
    delta = np.abs(a - b)
    delta[:, 0] = np.minimum(delta[:, 0], L - delta[:, 0])  # wrap x only
    return np.sqrt(np.einsum("ij,ij->i", delta, delta))


# --------------------------------------------------------------------------- #
# Result container shared by every team member.
# --------------------------------------------------------------------------- #
@dataclass
class TrialResult:
    """All observables produced by a single Monte-Carlo trial.

    A trial always reports the static layout and the percolation outcome.
    The time-series and network fields are populated only when the matching
    recording flag is passed to :meth:`SIRSuperspreaderSimulation.simulate`
    (so that high-throughput percolation sweeps stay fast).

    Attributes
    ----------
    positions : np.ndarray, shape (N, 2)
        Fixed coordinates of every individual.  ``positions[seed_index]`` is the
        seed on the bottom edge.  (Members 4: scatter / network maps.)
    is_super : np.ndarray of bool, shape (N,)
        Superspreader flag per individual.  (Member 4: colour coding.)
    seed_index : int
        Index of the initial infected individual (always 0 here).
    final_states : np.ndarray, shape (N,)
        State of every individual when the epidemic terminated (S/I/R codes).
    total_infected : int
        Number of individuals ever infected (``final_states != S``).
    percolated : bool
        True if the infection reached the top edge (``y > L - r_0``).
        (Member 1.)
    percolation_step : int or None
        Time step at which the top was first reached, else ``None``.
    num_steps : int
        Number of simulated time steps (excluding the initial seed at t = 0).
    new_infections_per_step : list[int]
        Newly infected individuals per time step; index 0 is the seed (= 1).
        Populated when ``record_timeseries=True``.  (Member 3: epidemic curve.)
    front_distance_per_step : list[float]
        ``r_f``: the largest distance from the seed to any ever-infected
        individual, per time step (monotone non-decreasing).  Populated when
        ``record_timeseries=True``.  (Member 2: propagation speed.)
    infection_log : list[tuple[int, int, int]]
        Chronological ``(infector_index, infectee_index, time_step)`` records;
        one entry per individual that got infected (a forest rooted at the
        seed).  Populated when ``record_network=True``.  (Member 4: arrows.)
    secondary_infections : np.ndarray of int, shape (N,)
        Number of individuals each individual directly infected.  Populated
        when ``record_network=True``.  (Member 4: secondary-infection histogram.)
    """

    positions: np.ndarray
    is_super: np.ndarray
    seed_index: int
    final_states: np.ndarray
    total_infected: int
    percolated: bool
    percolation_step: int | None
    num_steps: int
    new_infections_per_step: list[int] = field(default_factory=list)
    front_distance_per_step: list[float] = field(default_factory=list)
    infection_log: list[tuple[int, int, int]] = field(default_factory=list)
    secondary_infections: np.ndarray | None = None


class SIRSuperspreaderSimulation:
    """Stateless configuration holder that runs single Monte-Carlo trials.

    The object stores only the *configuration* (model, population size and
    superspreader fraction).  Each call to :meth:`simulate` draws a fresh random
    spatial layout, so a single instance can be reused for many independent
    trials.

    Examples
    --------
    Percolation only (fast, Member 1)::

        sim = SIRSuperspreaderSimulation("strong", n=500, lam=0.2)
        percolated = sim.run_trial(make_rng(seed))

    Full observables (Members 2-4)::

        result = sim.simulate(make_rng(seed),
                              record_timeseries=True, record_network=True)
        result.new_infections_per_step   # epidemic curve
        result.front_distance_per_step   # propagation front r_f(t)
        result.infection_log             # who infected whom
        result.secondary_infections      # offspring count per individual
    """

    def __init__(self, model: str, n: int, lam: float) -> None:
        if model not in _SUPERSPREADER_KIND:
            raise ValueError(f"Unknown model {model!r}; expected 'strong' or 'hub'.")
        if n < 1:
            raise ValueError("Population size N must be >= 1.")
        if not 0.0 <= lam <= 1.0:
            raise ValueError("Superspreader fraction lambda must be in [0, 1].")

        self.model = model
        self.n = n
        self.lam = lam
        self.normal_kind = _NORMAL_KIND
        self.super_kind = _SUPERSPREADER_KIND[model]

    # ------------------------------------------------------------------ #
    # Public API.
    # ------------------------------------------------------------------ #
    def run_trial(self, rng: np.random.Generator,
                  percolation_margin: float = R0,
                  seed_band: float | None = None) -> bool:
        """Convenience wrapper for Member 1: return only the percolation flag.

        Runs with all optional recording disabled and stops as soon as the top
        is reached, which is the fast path used by the percolation sweep.
        """
        return self.simulate(
            rng,
            record_timeseries=False,
            record_network=False,
            stop_when_percolated=True,
            percolation_margin=percolation_margin,
            seed_band=seed_band,
        ).percolated

    def simulate(
        self,
        rng: np.random.Generator,
        *,
        record_timeseries: bool = True,
        record_network: bool = False,
        stop_when_percolated: bool = False,
        max_steps: int | None = None,
        percolation_margin: float = R0,
        seed_band: float | None = None,
    ) -> TrialResult:
        """Run one Monte-Carlo trial and return a :class:`TrialResult`.

        Parameters
        ----------
        rng : np.random.Generator
            Source of randomness (use :func:`make_rng` for reproducibility).
        record_timeseries : bool, default True
            Record ``new_infections_per_step`` and ``front_distance_per_step``.
        record_network : bool, default False
            Record ``infection_log`` and ``secondary_infections`` (the
            who-infected-whom forest).  Slightly slower.
        stop_when_percolated : bool, default False
            Stop the moment the top is reached (Member 1's fast path).  Leave
            False to always run to extinction (Members 2-4 need the full curve).
        max_steps : int or None
            Optional hard cap on the number of time steps (safety valve).
        percolation_margin : float, default r_0
            "Reach the top" tolerance: a trial percolates once some infected
            individual has ``y > L - percolation_margin``.  The default ``r_0``
            counts the infection as having reached the top edge when it gets
            within one interaction radius of it.
        seed_band : float or None, default None
            Initial-infection geometry.  ``None`` seeds a single individual at
            the bottom (index 0; the default used by members 2-4 for
            propagation / epidemic-curve / network runs).  A float instead
            infects *every* individual within that distance of the bottom edge
            (``y < seed_band``), i.e. the whole bottom boundary - this is the
            standard spanning-percolation setup whose threshold lies on the
            ``R0 = Rc`` line (see Member 1's notes).

        Notes
        -----
        The infection dynamics and the order in which random numbers are drawn
        do **not** depend on the recording flags, so the same ``rng`` seed yields
        identical epidemics regardless of which observables are collected.
        """
        positions, is_super = self._place_individuals(rng)

        # cKDTree with ``boxsize`` performs the neighbour search on the strip:
        # x wraps with period L (transverse periodic boundary) while y stays open.
        tree = cKDTree(positions, boxsize=KDTREE_BOXSIZE)

        states = np.full(self.n, SUSCEPTIBLE, dtype=np.int8)
        states[0] = INFECTED  # individual 0 is the seed placed at the bottom
        if seed_band is not None:
            # Seed the whole bottom boundary (standard spanning percolation).
            states[positions[:, 1] < seed_band] = INFECTED

        new_per_step: list[int] = []
        front_per_step: list[float] = []
        infection_log: list[tuple[int, int, int]] = []
        secondary = np.zeros(self.n, dtype=np.int64) if record_network else None

        if record_timeseries:
            initial = np.flatnonzero(states == INFECTED)
            new_per_step.append(int(initial.size))   # individuals infected at t = 0
            front0 = (toroidal_distance(positions[initial], positions[0]).max()
                      if initial.size else 0.0)
            front_per_step.append(float(front0))

        percolated = False
        percolation_step: int | None = None
        step = 0

        # The seed sits at y = 0, so it is never already "at the top" (that would
        # require the degenerate case L <= r_0, which never happens here).
        while True:
            if max_steps is not None and step >= max_steps:
                break

            infected_idx = np.flatnonzero(states == INFECTED)
            if infected_idx.size == 0:
                break  # epidemic died out

            step += 1
            infectors, targets = self._spread(
                positions, states, is_super, infected_idx, tree, rng
            )

            # Synchronous recovery: every individual that just spread recovers
            # (gamma = 1) regardless of whether it infected anyone.
            states[infected_idx] = RECOVERED

            if targets.size == 0:
                if record_timeseries:
                    new_per_step.append(0)
                    front_per_step.append(front_per_step[-1])
                break  # no new infections -> epidemic stops

            # A susceptible may be reached by several infectors in the same step;
            # keep the first as its (single) infector so the routes form a forest.
            newly, first_occurrence = np.unique(targets, return_index=True)
            states[newly] = INFECTED

            if record_network:
                chosen_infectors = infectors[first_occurrence]
                np.add.at(secondary, chosen_infectors, 1)
                infection_log.extend(
                    (int(src), int(dst), step)
                    for src, dst in zip(chosen_infectors, newly)
                )

            if record_timeseries:
                new_per_step.append(int(newly.size))
                ever_infected = np.flatnonzero(states != SUSCEPTIBLE)
                front = toroidal_distance(
                    positions[ever_infected], positions[0]
                ).max()
                front_per_step.append(float(front))

            if not percolated and np.any(positions[newly, 1] > L - percolation_margin):
                percolated = True
                percolation_step = step
                if stop_when_percolated:
                    break

        return TrialResult(
            positions=positions,
            is_super=is_super,
            seed_index=0,
            final_states=states,
            total_infected=int(np.count_nonzero(states != SUSCEPTIBLE)),
            percolated=percolated,
            percolation_step=percolation_step,
            num_steps=step,
            new_infections_per_step=new_per_step,
            front_distance_per_step=front_per_step,
            infection_log=infection_log,
            secondary_infections=secondary,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers.
    # ------------------------------------------------------------------ #
    def _place_individuals(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Randomly lay out ``N`` individuals and assign superspreader flags.

        Individual 0 is the initial infected seed, placed at the bottom edge
        (``y = 0``) with a random ``x``.  The remaining ``N - 1`` individuals
        are placed uniformly at random.  Each individual is independently a
        superspreader with probability ``lambda``.
        """
        positions = rng.uniform(0.0, L, size=(self.n, 2))
        positions[0, 1] = 0.0  # seed sits exactly on the bottom boundary
        is_super = rng.random(self.n) < self.lam
        return positions, is_super

    def _spread(
        self,
        positions: np.ndarray,
        states: np.ndarray,
        is_super: np.ndarray,
        infected_idx: np.ndarray,
        tree: cKDTree,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the successful ``(infector, target)`` pairs of one time step.

        Infected individuals are processed in two groups (normal / super) so
        that each group can be queried against the tree with a single cutoff
        radius and evaluated with one vectorised ``w(r)`` expression.  Targets
        may repeat (multiple infectors); de-duplication into the set of newly
        infected individuals is the caller's responsibility.
        """
        susceptible_mask = states == SUSCEPTIBLE
        super_flag = is_super[infected_idx]

        infector_chunks: list[np.ndarray] = []
        target_chunks: list[np.ndarray] = []
        for spreaders, kind in (
            (infected_idx[~super_flag], self.normal_kind),
            (infected_idx[super_flag], self.super_kind),
        ):
            if spreaders.size == 0:
                continue
            infectors, targets = self._infect_group(
                positions, susceptible_mask, spreaders, kind, tree, rng
            )
            infector_chunks.append(infectors)
            target_chunks.append(targets)

        if not infector_chunks:
            empty = np.empty(0, dtype=np.intp)
            return empty, empty
        return np.concatenate(infector_chunks), np.concatenate(target_chunks)

    @staticmethod
    def _infect_group(
        positions: np.ndarray,
        susceptible_mask: np.ndarray,
        spreaders: np.ndarray,
        kind: InfectionKind,
        tree: cKDTree,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised infection attempt for one homogeneous group of spreaders.

        Returns the ``(infector, target)`` index arrays of the *successful*
        infections in this group (targets are still susceptible at draw time).
        """
        empty = np.empty(0, dtype=np.intp)

        # Ragged neighbour lists (one per spreader) within this kind's cutoff.
        neighbour_lists = tree.query_ball_point(positions[spreaders], kind.cutoff)
        counts = np.fromiter((len(n) for n in neighbour_lists), dtype=np.intp,
                             count=spreaders.size)
        total = int(counts.sum())
        if total == 0:
            return empty, empty

        # Flatten the (spreader, target) pairs into parallel 1D arrays.
        spreader_pairs = np.repeat(spreaders, counts)
        target_pairs = np.fromiter(
            (t for sub in neighbour_lists for t in sub),
            dtype=np.intp, count=total,
        )

        # Keep only pairs whose target is a *different*, still-susceptible
        # individual (the spreader itself is within its own search radius).
        valid = susceptible_mask[target_pairs] & (target_pairs != spreader_pairs)
        if not valid.any():
            return empty, empty
        spreader_pairs = spreader_pairs[valid]
        target_pairs = target_pairs[valid]

        # Distance-dependent infection probability for every surviving pair.
        distances = toroidal_distance(
            positions[spreader_pairs], positions[target_pairs]
        )
        probabilities = W0 * (1.0 - distances / kind.scale) ** kind.exponent

        hit = rng.random(distances.size) < probabilities
        return spreader_pairs[hit], target_pairs[hit]


# --------------------------------------------------------------------------- #
# Shared sweep driver.
#
# This is the generic "scan a (model, N, lambda) grid, run TRIALS trials at each
# point, in parallel" machinery that every member runs.  Members do NOT
# re-implement it (which would risk diverging); they instantiate it (or subclass
# it) and override only *what to measure* per trial.  Because the seeding,
# layout and dynamics all live in one shared code path, every member's runs are
# the *same* underlying Monte-Carlo realisations.
# --------------------------------------------------------------------------- #
class MonteCarloSweep:
    """Parallel ``(model, N, lambda)`` Monte-Carlo sweep shared by the whole team.

    The default behaviour *is* Member 1's percolation sweep: at each grid point it
    runs ``trials`` independent trials and returns the fraction that percolated.
    Other members reuse the identical simulation path and customise the result by
    overriding two small hooks:

    * :meth:`measure_trial` - the observable read off **one** trial
      (default: the percolation flag).
    * :meth:`reduce_trials` - how the per-trial values at one grid point are
      aggregated (default: the mean, i.e. the percolation probability).

    Everything else - the seeding convention (:func:`trial_rng` on the physical
    ``(model, lam, N)`` config), the parallel map, the density axis - is shared,
    so e.g. Member 3 subclassing this to read ``new_infections_per_step`` gets the
    *same* layouts and epidemics that Member 1 saw at the same grid points.

    Parameters
    ----------
    trials : int
        Number of independent Monte-Carlo trials per grid point.
    lambdas : sequence of float
        Superspreader fractions to scan.
    n_values : sequence of int
        Population sizes ``N`` to scan (the density axis is derived from these).
    seed_band : float or None, default None
        Passed straight to :meth:`SIRSuperspreaderSimulation.run_trial` /
        ``simulate``.  ``None`` is a single bottom seed (the shared default used
        by Members 2-4); a float seeds the whole bottom edge (Member 1's
        ``bottom_edge`` percolation variant).

    Examples
    --------
    Member 1 (default percolation sweep)::

        sweep = MonteCarloSweep(trials=1000, lambdas=(0.0, 0.5, 1.0),
                                n_values=range(150, 901, 15))
        with Pool() as pool:
            densities, prob_by_lambda = sweep.run("strong", pool)

    A member overriding the measurement (illustrative)::

        class FinalSizeSweep(MonteCarloSweep):
            def measure_trial(self, sim, rng):
                return sim.simulate(rng).total_infected   # read a different field
            def reduce_trials(self, values):
                return float(np.mean(values))
    """

    def __init__(self, *, trials: int, lambdas, n_values,
                 seed_band: float | None = None) -> None:
        self.trials = int(trials)
        self.lambdas = tuple(float(x) for x in lambdas)
        self.n_values = np.asarray(list(n_values), dtype=int)
        self.seed_band = seed_band

    # ------------------------------------------------------------------ #
    # Override points (the only methods most members need to touch).
    # ------------------------------------------------------------------ #
    def measure_trial(self, sim: "SIRSuperspreaderSimulation",
                      rng: np.random.Generator):
        """Observable extracted from a single trial.  Default: percolation flag.

        Override to read whatever your figure needs off the trial, e.g.::

            return sim.simulate(rng).new_infections_per_step   # Member 3
        """
        return sim.run_trial(rng, seed_band=self.seed_band)

    def reduce_trials(self, values: list):
        """Aggregate the per-trial values at one grid point.  Default: the mean.

        For the default percolation flag this returns the percolation probability.
        Override for a different reduction (e.g. an element-wise mean of curves).
        """
        return float(np.mean(values))

    # ------------------------------------------------------------------ #
    # Shared machinery (rarely overridden).
    # ------------------------------------------------------------------ #
    def run_configuration(self, task: tuple):
        """Worker: aggregate observable for one ``(model, N, lambda)`` point.

        Picklable for :class:`multiprocessing.Pool`.  Each trial draws its own
        generator from the shared physical-config convention, so the realisation
        for trial ``t`` is identical for any member running this point.
        """
        model, n, lam = task
        sim = SIRSuperspreaderSimulation(model=model, n=n, lam=lam)
        keys = (model_id(model), lambda_key(lam), int(n))
        values = [self.measure_trial(sim, trial_rng(*keys, trial=t))
                  for t in range(self.trials)]
        return self.reduce_trials(values)

    def build_tasks(self, model: str) -> list[tuple]:
        """Enumerate every ``(model, N, lambda)`` configuration in the grid."""
        return [(model, int(n), float(lam))
                for lam in self.lambdas
                for n in self.n_values]

    def run(self, model: str, pool=None) -> tuple[np.ndarray, dict]:
        """Run the whole grid for one model.

        Parameters
        ----------
        model : str
            ``"strong"`` or ``"hub"``.
        pool : multiprocessing.Pool or None
            If given, configurations are mapped across the pool; otherwise the
            sweep runs serially (handy for debugging).

        Returns
        -------
        densities : np.ndarray
            ``rho * pi * r_0^2`` for each ``N`` in ``n_values``.
        result_by_lambda : dict[float, list]
            ``lambda -> list`` of per-grid-point reduced results (one entry per
            ``N``, in ``n_values`` order).
        """
        tasks = self.build_tasks(model)
        logger.info("Sweep %s: %d configurations x %d trials ...",
                    model, len(tasks), self.trials)
        mapper = pool.map if pool is not None else map
        flat = list(mapper(self.run_configuration, tasks))

        densities = np.array([density_from_n(int(n)) for n in self.n_values])
        stride = self.n_values.size
        result_by_lambda = {
            lam: flat[i * stride:(i + 1) * stride]
            for i, lam in enumerate(self.lambdas)
        }
        return densities, result_by_lambda


def density_from_n(n: int) -> float:
    """Return the dimensionless density ``rho * pi * r_0^2`` for ``N`` individuals.

    With ``rho = N / L^2`` and ``L = 10 r_0`` this is ``N * pi / 100``.
    """
    rho = n / (L * L)
    return rho * np.pi * R0 * R0


def n_from_density(density: float) -> int:
    """Inverse of :func:`density_from_n`: nearest integer ``N`` for a density.

    Convenience for members who prefer to specify ``rho * pi * r_0^2`` directly.
    """
    return int(round(density * L * L / (np.pi * R0 * R0)))
