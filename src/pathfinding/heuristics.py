"""Heuristics for unit cell cost. Owner: Aymen (C).

Each takes (cell, goal) and returns an estimated distance.
astar multiplies the value by grid.min_cost so it stays admissible on weighted terrain.
"""

from collections.abc import Callable

from pathfinding.result import Cell

Heuristic = Callable[[Cell, Cell], float]


def zero(cell: Cell, goal: Cell) -> float:
    return 0.0


def manhattan(cell: Cell, goal: Cell) -> float:
    raise NotImplementedError


def euclidean(cell: Cell, goal: Cell) -> float:
    raise NotImplementedError


def octile(cell: Cell, goal: Cell) -> float:
    raise NotImplementedError


def chebyshev(cell: Cell, goal: Cell) -> float:
    raise NotImplementedError
