"""Reference answers from networkx, used to check our planners."""

import math
import random

import networkx as nx

from pathfinding.generators import random_grid, scatter_terrain
from pathfinding.grid import Grid


def to_graph(grid: Grid) -> nx.DiGraph:
    graph = nx.DiGraph()
    for cell in grid.free_cells():
        graph.add_node(cell)
        for neighbour, step_cost in grid.neighbours(cell):
            graph.add_edge(cell, neighbour, weight=step_cost)
    return graph


def optimal_cost(grid: Grid, start, goal) -> float:
    try:
        return nx.dijkstra_path_length(to_graph(grid), start, goal)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return math.inf


def distances_to(grid: Grid, goal) -> dict:
    """True cheapest cost from every cell that can reach goal."""
    return nx.single_source_dijkstra_path_length(to_graph(grid).reverse(), goal)


def path_cost(grid: Grid, path) -> float:
    """Cost of walking path with legal moves only; raises if a move is illegal."""
    total = 0.0
    for here, there in zip(path, path[1:]):
        steps = dict(grid.neighbours(here))
        if there not in steps:
            raise AssertionError(f"illegal move {here} -> {there}")
        total += steps[there]
    return total


def random_cases(count: int, seed: int):
    """Small random maps: mixed sizes, densities, 4/8 connectivity, uniform and weighted terrain."""
    rng = random.Random(seed)
    for case in range(count):
        size = rng.randint(2, 20)
        connectivity = 8 if case % 2 else 4
        grid = random_grid(size, size, rng.uniform(0.0, 0.4), seed * 1000 + case, connectivity)
        if case % 3 == 0:
            scatter_terrain(grid, seed * 1000 + case, 0.4)
        free = grid.free_cells()
        if len(free) < 2:
            continue
        start, goal = rng.sample(free, 2)
        yield grid, start, goal
