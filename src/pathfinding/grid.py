"""Grid world. Owner: Shazeel (B)."""

from collections.abc import Iterator

import numpy as np

from pathfinding.result import Cell


class Grid:
    """2D grid of traversal costs.

    cost[r, c] is the cost of entering cell (r, c); np.inf marks a wall.
    Step cost = cost of the cell being entered x step length (1 straight, sqrt(2) diagonal).
    """

    def __init__(self, cost: np.ndarray, connectivity: int = 8):
        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")
        cost = np.asarray(cost, dtype=float)
        if cost.ndim != 2:
            raise ValueError("cost must be a 2D array")
        if np.any(cost <= 0):
            raise ValueError("cell costs must be positive")
        self.cost = cost
        self.connectivity = connectivity

    @property
    def rows(self) -> int:
        return self.cost.shape[0]

    @property
    def cols(self) -> int:
        return self.cost.shape[1]

    @property
    def min_cost(self) -> float:
        finite = self.cost[np.isfinite(self.cost)]
        return float(finite.min()) if finite.size else 1.0

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def passable(self, cell: Cell) -> bool:
        """True if the cell is inside the grid and not a wall."""
        raise NotImplementedError

    def neighbours(self, cell: Cell) -> Iterator[tuple[Cell, float]]:
        """Yield (neighbour, step_cost) for every legal move from cell.

        4-connected: up, down, left, right.
        8-connected: also diagonals, but only if both orthogonal cells beside
        the diagonal are passable (no corner cutting).
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str, connectivity: int = 8) -> "Grid":
        raise NotImplementedError
