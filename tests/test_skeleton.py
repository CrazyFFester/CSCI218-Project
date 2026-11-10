import math

import numpy as np
import pytest

from pathfinding import Grid, SearchResult
from pathfinding.stubs import STUB_GOAL, STUB_START, stub_grid, stub_result


def test_grid_rejects_bad_connectivity():
    with pytest.raises(ValueError):
        Grid(np.ones((3, 3)), connectivity=6)


def test_grid_min_cost_ignores_walls():
    grid = Grid(np.array([[2.0, math.inf], [3.0, 5.0]]))
    assert grid.min_cost == 2.0


def test_search_result_is_frozen():
    result = stub_result()
    with pytest.raises(AttributeError):
        result.cost = 0.0


def test_stub_result_is_a_valid_path_on_stub_grid():
    grid = stub_grid()
    result = stub_result()

    assert isinstance(result, SearchResult)
    assert result.found
    assert result.path[0] == STUB_START
    assert result.path[-1] == STUB_GOAL

    total = 0.0
    for here, nxt in zip(result.path, result.path[1:]):
        steps = dict(grid.neighbours(here))
        assert nxt in steps
        total += steps[nxt]
    assert total == result.cost


def test_stub_visited_order_covers_path():
    result = stub_result()
    assert set(result.path) <= set(result.visited_order)
    assert result.expanded == len(result.visited_order)
    assert result.peak_memory == result.max_frontier + result.expanded
