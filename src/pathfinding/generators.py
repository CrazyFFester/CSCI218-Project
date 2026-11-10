"""Seeded map generators. Owner: Shazeel (B). Same seed must give the same map."""

from pathfinding.grid import Grid


def random_grid(rows: int, cols: int, density: float, seed: int, connectivity: int = 8) -> Grid:
    """Random walls covering `density` (0.0 to 1.0) of the cells."""
    raise NotImplementedError


def maze_grid(rows: int, cols: int, seed: int, connectivity: int = 4) -> Grid:
    """Perfect maze (exactly one route between any two free cells)."""
    raise NotImplementedError


def weighted_terrain(grid: Grid, low: float, high: float, seed: int) -> Grid:
    """Copy of grid with free cells given random integer costs in [low, high]."""
    raise NotImplementedError
