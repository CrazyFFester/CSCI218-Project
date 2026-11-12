"""Seeded map generators. Same seed + arguments => identical map on every machine."""
from __future__ import annotations

import math
import random
from typing import Iterable, Sequence

import numpy as np

from .grid import Cell, Grid


def random_grid(rows: int, cols: int, density: float, seed: int,
                connectivity: int = 8, keep_free: Iterable[Cell] = ()) -> Grid:
    """Each cell is a wall with probability `density`. Not guaranteed solvable."""
    if not 0.0 <= density < 1.0:
        raise ValueError("density must be in [0, 1)")
    rng = np.random.default_rng(seed)
    walls = rng.random((rows, cols)) < density
    cost = np.where(walls, math.inf, 1.0)
    for r, c in keep_free:
        cost[r, c] = 1.0
    return Grid(cost, connectivity)


def maze_grid(rows: int, cols: int, seed: int, connectivity: int = 8) -> Grid:
    """Perfect maze (recursive backtracker): every free cell is reachable.

    Corridors sit on odd (row, col) cells, so (1, 1) is always free.
    """
    if rows < 3 or cols < 3:
        raise ValueError("maze needs at least 3x3")
    rng = random.Random(seed)
    cost = np.full((rows, cols), math.inf)
    ny, nx = (rows - 1) // 2, (cols - 1) // 2
    visited = [[False] * nx for _ in range(ny)]
    stack = [(0, 0)]
    visited[0][0] = True
    cost[1, 1] = 1.0
    while stack:
        y, x = stack[-1]
        options = [(y + dy, x + dx) for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                   if 0 <= y + dy < ny and 0 <= x + dx < nx and not visited[y + dy][x + dx]]
        if not options:
            stack.pop()
            continue
        ty, tx = rng.choice(options)
        visited[ty][tx] = True
        cost[2 * ty + 1, 2 * tx + 1] = 1.0
        cost[y + ty + 1, x + tx + 1] = 1.0  # knock out the wall between the two cells
        stack.append((ty, tx))
    return Grid(cost, connectivity)


def scatter_terrain(grid: Grid, seed: int, fraction: float,
                    costs: Sequence[float] = (2.0, 3.0, 5.0)) -> Grid:
    """Give `fraction` of the free cells a higher cost (mud, sand...). Edits in place."""
    rng = random.Random(seed)
    free = grid.free_cells()
    k = int(round(fraction * len(free)))
    for cell in rng.sample(free, k):
        grid.set_cost(cell, rng.choice(list(costs)))
    return grid


def random_start_goal(grid: Grid, seed: int, min_distance: int = 0) -> tuple[Cell, Cell]:
    """Two distinct free cells, at least `min_distance` apart (Chebyshev)."""
    rng = random.Random(seed)
    free = grid.free_cells()
    if len(free) < 2:
        raise ValueError("need at least two free cells")
    for _ in range(10_000):
        a, b = rng.sample(free, 2)
        if max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= min_distance:
            return a, b
    raise ValueError("could not find a start/goal pair that far apart")
