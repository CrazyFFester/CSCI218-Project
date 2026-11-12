"""Dijkstra, A*, Greedy Best-First and Weighted A*."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import inf, isfinite
from time import perf_counter
from typing import Literal
from warnings import warn

from pathfinding.grid import Grid
from pathfinding.heuristics import (
    Heuristic,
    chebyshev,
    euclidean,
    manhattan,
    octile,
    zero,
)
from pathfinding.result import Cell, SearchResult

Algorithm = Literal["dijkstra", "astar", "greedy", "weighted_astar"]

# Far below any real cost difference (cell costs are >= min_cost > 0),
# far above float rounding error on 512x512 paths.
_TIE_MARGIN = 1e-9

_HEURISTICS: dict[str, Heuristic] = {
    "manhattan": manhattan,
    "euclidean": euclidean,
    "octile": octile,
    "chebyshev": chebyshev,
    "zero": zero,
}


@dataclass(frozen=True, slots=True, eq=False)
class _Label:
    cell: Cell
    g: float
    parent: _Label | None


def search(
    grid: Grid,
    start: Cell,
    goal: Cell,
    algorithm: Algorithm = "astar",
    heuristic: str | Heuristic = octile,
    weight: float | None = None,
) -> SearchResult:
    """Find a route using the selected algorithm on a fixed grid."""

    started = perf_counter()

    if algorithm not in ("dijkstra", "astar", "greedy", "weighted_astar"):
        raise ValueError(f"Unknown algorithm: {algorithm!r}")

    if weight is None:
        weight = 2.0 if algorithm == "weighted_astar" else 1.0

    if not isfinite(weight) or weight < 0:
        raise ValueError("Weight must be finite and nonnegative.")

    if algorithm == "weighted_astar" and weight < 1:
        raise ValueError("Weighted A* requires weight >= 1.")

    if isinstance(heuristic, str):
        try:
            estimate = _HEURISTICS[heuristic.lower()]
        except KeyError:
            raise ValueError(f"Unknown heuristic: {heuristic!r}") from None
    elif callable(heuristic):
        estimate = heuristic
    else:
        raise ValueError("Heuristic must be a name or a callable.")

    visited: list[Cell] = []
    peak = 0

    def finish(path: list[Cell], cost: float) -> SearchResult:
        return SearchResult(
            path=path,
            cost=cost,
            expanded=len(visited),
            max_frontier=peak,
            visited_order=visited,
            runtime=perf_counter() - started,
        )

    if not grid.passable(start) or not grid.passable(goal):
        return finish([], inf)

    if algorithm == "dijkstra":
        multiplier = 0.0
    elif algorithm == "greedy":
        multiplier = 1.0
    else:
        multiplier = weight

    # Scale geometric distances by the cheapest traversable terrain.
    scale = grid.min_cost if multiplier else 0.0

    if multiplier and (not isfinite(scale) or scale <= 0):
        raise ValueError("grid.min_cost must be finite and positive.")

    if multiplier and estimate is manhattan and grid.connectivity == 8:
        warn(
            "Manhattan can overestimate with sqrt(2) diagonals. "
            "Use Octile for an admissible estimate.",
            UserWarning,
            stacklevel=2,
        )

    def priority(label: _Label) -> float:
        # Greedy ignores g. Dijkstra ignores h.
        base = 0.0 if algorithm == "greedy" else label.g

        if not multiplier:
            return base

        h = float(estimate(label.cell, goal))

        if not isfinite(h) or h < 0 or (label.cell == goal and h != 0):
            raise ValueError(
                "Heuristic must be finite, nonnegative and zero at the goal."
            )

        score = base + multiplier * (scale * h)

        if not isfinite(score):
            raise ValueError("Search priority overflow.")

        return score

    serial = count()
    first = _Label(start, 0.0, None)
    best = {start: first}

    # Break equal priorities by larger g, then earlier insertion.
    heap = [(priority(first), -first.g, next(serial), first)]
    peak = 1

    while heap:
        _, _, _, current = heappop(heap)

        # Skip entries that a cheaper route has replaced.
        if best[current.cell] is not current:
            continue

        visited.append(current.cell)

        if current.cell == goal:
            path: list[Cell] = []
            label: _Label | None = current

            while label is not None:
                path.append(label.cell)
                label = label.parent

            return finish(path[::-1], current.g)

        for cell, step_cost in grid.neighbours(current.cell):
            # Grid.neighbours already gives the complete movement cost.
            if not isfinite(step_cost) or step_cost <= 0:
                raise ValueError("Edge costs must be finite and positive.")

            candidate = current.g + step_cost

            if not isfinite(candidate):
                raise ValueError("Path cost overflow.")

            previous = best.get(cell)

            # Rounding on long paths can make an equal-cost route look cheaper by
            # several ulps; treat anything closer than _TIE_MARGIN as a tie.
            if previous is None or candidate < previous.g - _TIE_MARGIN:
                improved = _Label(cell, candidate, current)
                best[cell] = improved

                heappush(
                    heap,
                    (priority(improved), -candidate, next(serial), improved),
                )

                peak = max(peak, len(heap))

    return finish([], inf)


def dijkstra(grid: Grid, start: Cell, goal: Cell) -> SearchResult:
    return search(grid, start, goal, algorithm="dijkstra")


def astar(
    grid: Grid,
    start: Cell,
    goal: Cell,
    heuristic: Heuristic = octile,
    weight: float = 1.0,
) -> SearchResult:
    # Weight 1 gives standard A*. Higher weights give Weighted A*.
    return search(
        grid,
        start,
        goal,
        algorithm="astar",
        heuristic=heuristic,
        weight=weight,
    )


def greedy_best_first(
    grid: Grid,
    start: Cell,
    goal: Cell,
    heuristic: Heuristic = octile,
) -> SearchResult:
    return search(
        grid,
        start,
        goal,
        algorithm="greedy",
        heuristic=heuristic,
    )
