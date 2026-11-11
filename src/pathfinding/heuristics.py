"""Geometric distances for pathfinding on a grid."""

from collections.abc import Callable
from math import hypot, sqrt

from pathfinding.result import Cell

Heuristic = Callable[[Cell, Cell], float]
_DIAGONAL_EXTRA = sqrt(2.0) - 1.0


def _gaps(cell: Cell, goal: Cell) -> tuple[int, int]:
    return abs(cell[0] - goal[0]), abs(cell[1] - goal[1])


def zero(cell: Cell, goal: Cell) -> float:
    return 0.0


def manhattan(cell: Cell, goal: Cell) -> float:
    rows, cols = _gaps(cell, goal)
    return float(rows + cols)


def euclidean(cell: Cell, goal: Cell) -> float:
    return hypot(*_gaps(cell, goal))


def octile(cell: Cell, goal: Cell) -> float:
    rows, cols = _gaps(cell, goal)
    return max(rows, cols) + _DIAGONAL_EXTRA * min(rows, cols)


def chebyshev(cell: Cell, goal: Cell) -> float:
    return float(max(_gaps(cell, goal)))