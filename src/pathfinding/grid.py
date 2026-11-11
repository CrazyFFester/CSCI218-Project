"""Grid world: numpy cost array, 4/8-way neighbours with the no-corner-cutting rule.

Cells are (row, col). cost[r, c] is the cost of ENTERING that cell; np.inf = wall.
Step cost = cost of the entered cell * step length (1 straight, sqrt(2) diagonal).
"""
from __future__ import annotations

import math
from typing import Iterator, Sequence

import numpy as np

from .result import Cell

_SQRT2 = math.sqrt(2.0)
# Fixed order => deterministic expansion order on every machine.
_ORTHO = ((-1, 0, 1.0), (0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0))
_DIAG = ((-1, 1, _SQRT2), (1, 1, _SQRT2), (1, -1, _SQRT2), (-1, -1, _SQRT2))


class Grid:
    def __init__(self, cost: np.ndarray, connectivity: int = 8):
        cost = np.array(cost, dtype=float)  # copy
        if cost.ndim != 2 or cost.size == 0:
            raise ValueError("cost must be a non-empty 2D array")
        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")
        if np.isnan(cost).any() or (cost <= 0).any():
            raise ValueError("cell costs must be > 0 (use np.inf for walls)")
        self.cost = cost
        self.connectivity = connectivity
        self._min_cost: float | None = None

    # ---- basic queries -------------------------------------------------
    @property
    def rows(self) -> int:
        return self.cost.shape[0]

    @property
    def cols(self) -> int:
        return self.cost.shape[1]

    @property
    def min_cost(self) -> float:
        """Smallest finite cell cost (1.0 if the grid has no free cell)."""
        if self._min_cost is None:
            finite = self.cost[np.isfinite(self.cost)]
            self._min_cost = float(finite.min()) if finite.size else 1.0
        return self._min_cost

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def passable(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols and math.isfinite(self.cost[r, c])

    def neighbours(self, cell: Cell) -> Iterator[tuple[Cell, float]]:
        """Yield (neighbour, step_cost). Diagonals need both side cells passable."""
        r, c = cell
        cost = self.cost
        for dr, dc, length in _ORTHO:
            nr, nc = r + dr, c + dc
            if self.passable((nr, nc)):
                yield (nr, nc), float(cost[nr, nc]) * length
        if self.connectivity == 8:
            for dr, dc, length in _DIAG:
                nr, nc = r + dr, c + dc
                if (self.passable((nr, nc))
                        and self.passable((r + dr, c))
                        and self.passable((r, c + dc))):
                    yield (nr, nc), float(cost[nr, nc]) * length

    # ---- editing -------------------------------------------------------
    def set_cost(self, cell: Cell, value: float) -> None:
        if not self.in_bounds(cell):
            raise IndexError(f"cell {cell} out of bounds")
        if not (value > 0):
            raise ValueError("cost must be > 0 (use np.inf for a wall)")
        self.cost[cell[0], cell[1]] = value
        self._min_cost = None

    def set_wall(self, cell: Cell) -> None:
        self.set_cost(cell, math.inf)

    def clear(self, cell: Cell) -> None:
        self.set_cost(cell, 1.0)

    def copy(self) -> "Grid":
        return Grid(self.cost, self.connectivity)

    def free_cells(self) -> list[Cell]:
        rs, cs = np.nonzero(np.isfinite(self.cost))
        return [(int(r), int(c)) for r, c in zip(rs, cs)]

    # ---- save / load ---------------------------------------------------
    def save(self, path: str) -> None:
        np.save(path, self.cost)

    @classmethod
    def load(cls, path: str, connectivity: int = 8) -> "Grid":
        return cls(np.load(path), connectivity)

    # ---- helpers for tests / demos -------------------------------------
    @classmethod
    def from_ascii(cls, lines: Sequence[str], connectivity: int = 8) -> "Grid":
        """'#' wall, '.' cost 1, digits 2-9 = that terrain cost."""
        rows = []
        for line in lines:
            row = []
            for ch in line:
                if ch == "#":
                    row.append(math.inf)
                elif ch == ".":
                    row.append(1.0)
                elif ch.isdigit() and ch != "0":
                    row.append(float(ch))
                else:
                    raise ValueError(f"bad map character {ch!r}")
            rows.append(row)
        return cls(np.array(rows), connectivity)

    def __repr__(self) -> str:
        return f"Grid({self.rows}x{self.cols}, connectivity={self.connectivity})"
