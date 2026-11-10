"""Temporary stand-ins so the visualiser and benchmark can be built before the
real grid and planners are merged. Delete once grid.py and search.py are done.

Stub map (S = start, G = goal, # = wall):

    S . . # .
    . # . # .
    . # . . .
    . # # # .
    . . . . G
"""

import math
from collections.abc import Iterator

import numpy as np

from pathfinding.grid import Grid
from pathfinding.result import Cell, SearchResult

STUB_START: Cell = (0, 0)
STUB_GOAL: Cell = (4, 4)

_WALLS = np.array(
    [
        [0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ],
    dtype=bool,
)


class StubGrid(Grid):
    """4-connected, unit cost, no diagonals."""

    def __init__(self):
        super().__init__(np.where(_WALLS, math.inf, 1.0), connectivity=4)

    def passable(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and math.isfinite(self.cost[cell])

    def neighbours(self, cell: Cell) -> Iterator[tuple[Cell, float]]:
        r, c = cell
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (r + dr, c + dc)
            if self.passable(nxt):
                yield nxt, float(self.cost[nxt])


def stub_grid() -> StubGrid:
    return StubGrid()


def stub_result() -> SearchResult:
    """A hard-coded, valid result on the stub map from STUB_START to STUB_GOAL."""
    path = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]
    visited_order = [
        (0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (3, 0), (1, 2), (4, 0), (2, 2),
        (4, 1), (2, 3), (4, 2), (2, 4), (4, 3), (1, 4), (3, 4), (4, 4),
    ]
    return SearchResult(
        path=path,
        cost=8.0,
        expanded=len(visited_order),
        max_frontier=3,
        visited_order=visited_order,
        runtime=0.0,
    )
