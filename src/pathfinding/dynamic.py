"""Dynamic environment: moving obstacles and terrain cost changes.

Obstacles are walls (cost = inf) that move one cell per tick on a seeded rule.
The terrain underneath is remembered in `base`, so it is restored when they leave.
Always edit the grid through this class (or Grid.set_cost) so Grid.min_cost stays right.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

from .grid import Cell, Grid

_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


@dataclass
class Obstacle:
    pos: Cell
    direction: tuple[int, int]


class DynamicEnvironment:
    """mode='patrol': walk straight, reverse when blocked.
       mode='random': random 4-way step each tick (seeded)."""

    def __init__(self, grid: Grid, n_obstacles: int = 0, seed: int = 0,
                 mode: str = "patrol", protected: Iterable[Cell] = ()):
        if mode not in ("patrol", "random"):
            raise ValueError("mode must be 'patrol' or 'random'")
        self.grid = grid
        self.base = grid.cost.copy()          # terrain without obstacles
        self.rng = random.Random(seed)
        self.mode = mode
        self.protected: set[Cell] = set(protected)   # obstacles never enter these
        self.obstacles: list[Obstacle] = []
        self.tick_count = 0
        candidates = [c for c in grid.free_cells() if c not in self.protected]
        if n_obstacles > len(candidates):
            raise ValueError("more obstacles than free cells")
        for cell in self.rng.sample(candidates, n_obstacles):
            self.obstacles.append(Obstacle(cell, self.rng.choice(_DIRS)))
            grid.set_wall(cell)

    # ---- obstacles ----------------------------------------------------
    def protect(self, *cells: Cell) -> None:
        """Keep cells obstacle-free (e.g. start, goal, the agent's current cell)."""
        self.protected.update(cells)

    def _open(self, cell: Cell, occupied: set[Cell]) -> bool:
        return (self.grid.in_bounds(cell) and math.isfinite(self.base[cell[0], cell[1]])
                and cell not in self.protected and cell not in occupied)

    def _next(self, ob: Obstacle, occupied: set[Cell]) -> Cell:
        r, c = ob.pos
        if self.mode == "patrol":
            for _ in range(2):  # try forward, then reversed
                nxt = (r + ob.direction[0], c + ob.direction[1])
                if self._open(nxt, occupied):
                    return nxt
                ob.direction = (-ob.direction[0], -ob.direction[1])
            return ob.pos
        dirs = list(_DIRS)
        self.rng.shuffle(dirs)
        for dr, dc in dirs:
            if self._open((r + dr, c + dc), occupied):
                return (r + dr, c + dc)
        return ob.pos

    def step(self) -> list[Cell]:
        """Advance one tick. Returns every cell whose cost changed."""
        occupied = {o.pos for o in self.obstacles}
        changed: list[Cell] = []
        for ob in self.obstacles:
            old, new = ob.pos, self._next(ob, occupied)
            if new == old:
                continue
            occupied.discard(old)
            occupied.add(new)
            self.grid.set_cost(old, float(self.base[old[0], old[1]]))
            self.grid.set_wall(new)
            ob.pos = new
            changed += [old, new]
        self.tick_count += 1
        return changed

    # ---- terrain ------------------------------------------------------
    def change_terrain(self, cell: Cell, cost: float) -> None:
        """Permanently change a cell's terrain cost (stays a wall if an obstacle is on it)."""
        self.base[cell[0], cell[1]] = cost
        if cell not in {o.pos for o in self.obstacles}:
            self.grid.set_cost(cell, cost)

    def random_terrain_changes(self, k: int, costs: Sequence[float] = (1.0, 2.0, 5.0)) -> list[Cell]:
        occupied = {o.pos for o in self.obstacles}
        pool = [c for c in self.grid.free_cells() if c not in self.protected and c not in occupied]
        chosen = self.rng.sample(pool, min(k, len(pool)))
        for cell in chosen:
            self.change_terrain(cell, self.rng.choice(list(costs)))
        return chosen

    # ---- queries ------------------------------------------------------
    def path_blocked(self, path: Sequence[Cell]) -> bool:
        return path_blocked(self.grid, path)


def path_blocked(grid: Grid, path: Sequence[Cell]) -> bool:
    """True if `path` is no longer walkable on `grid` (wall on it, or a move became illegal)."""
    if not path:
        return False
    if not grid.passable(path[0]):
        return True
    for a, b in zip(path, path[1:]):
        if b not in {n for n, _ in grid.neighbours(a)}:
            return True
    return False
