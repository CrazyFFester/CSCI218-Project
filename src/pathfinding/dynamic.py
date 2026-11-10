"""Moving obstacles and replanning support. Owner: Shazeel (B)."""

import numpy as np

from pathfinding.grid import Grid
from pathfinding.result import Cell


def path_blocked(grid: Grid, path: list[Cell]) -> bool:
    """True if any cell of path is no longer passable."""
    raise NotImplementedError


def move_obstacles(grid: Grid, obstacles: list[Cell], rng: np.random.Generator) -> tuple[Grid, list[Cell]]:
    """Move each obstacle one cell per tick; return the new grid and positions."""
    raise NotImplementedError
