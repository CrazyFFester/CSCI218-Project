"""Planners. Owner: Aymen (C).

One best-first core on heapq with lazy deletion; every planner below is a thin
wrapper over it. Ties on equal f go to the larger g, then to insertion order.
"""

from pathfinding.grid import Grid
from pathfinding.heuristics import Heuristic, octile
from pathfinding.result import Cell, SearchResult


def astar(grid: Grid, start: Cell, goal: Cell, heuristic: Heuristic = octile, weight: float = 1.0) -> SearchResult:
    """f = g + weight * h. weight > 1 gives Weighted A*."""
    raise NotImplementedError


def dijkstra(grid: Grid, start: Cell, goal: Cell) -> SearchResult:
    """A* with h = 0."""
    raise NotImplementedError


def greedy_best_first(grid: Grid, start: Cell, goal: Cell, heuristic: Heuristic = octile) -> SearchResult:
    """f = h only. Fast, not optimal."""
    raise NotImplementedError
