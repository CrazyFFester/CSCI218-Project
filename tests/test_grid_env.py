import math
import numpy as np
import pytest
from pathfinding.grid import Grid
from pathfinding.generators import random_grid, maze_grid, scatter_terrain, random_start_goal
from pathfinding.dynamic import DynamicEnvironment, path_blocked


def test_neighbours_4_and_8_open():
    g4 = Grid(np.ones((3, 3)), 4)
    g8 = Grid(np.ones((3, 3)), 8)
    assert len(list(g4.neighbours((1, 1)))) == 4
    assert len(list(g8.neighbours((1, 1)))) == 8
    assert len(list(g8.neighbours((0, 0)))) == 3


def test_step_cost_is_entered_cell_times_length():
    g = Grid.from_ascii(["...", ".5.", "..."])
    costs = dict(g.neighbours((0, 0)))
    assert costs[(0, 1)] == 1.0
    assert costs[(1, 1)] == pytest.approx(5 * math.sqrt(2))


def test_no_corner_cutting():
    g = Grid.from_ascii([".#", "#."])
    assert list(g.neighbours((0, 0))) == []
    g2 = Grid.from_ascii([".#", ".."])
    assert (1, 1) not in dict(g2.neighbours((0, 0)))


def test_walls_and_bounds_not_passable():
    g = Grid.from_ascii(["#."])
    assert not g.passable((0, 0)) and g.passable((0, 1))
    assert not g.passable((-1, 0)) and not g.passable((0, 5))


def test_min_cost_updates():
    g = Grid.from_ascii(["23", "#4"])
    assert g.min_cost == 2.0
    g.set_cost((0, 0), 1.0)
    assert g.min_cost == 1.0


def test_bad_inputs():
    with pytest.raises(ValueError):
        Grid(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        Grid(np.ones((2, 2)), 6)
    with pytest.raises(ValueError):
        Grid(np.ones((0, 0)))


def test_one_by_one():
    g = Grid(np.ones((1, 1)))
    assert list(g.neighbours((0, 0))) == []


def test_save_load(tmp_path):
    g = random_grid(10, 12, 0.2, seed=1)
    p = tmp_path / "m.npy"
    g.save(str(p))
    h = Grid.load(str(p))
    assert np.array_equal(g.cost, h.cost)


def test_generators_deterministic():
    assert np.array_equal(random_grid(20, 20, 0.3, 7).cost, random_grid(20, 20, 0.3, 7).cost)
    assert not np.array_equal(random_grid(20, 20, 0.3, 7).cost, random_grid(20, 20, 0.3, 8).cost)
    assert np.array_equal(maze_grid(21, 21, 3).cost, maze_grid(21, 21, 3).cost)


def test_random_grid_keep_free_and_density():
    g = random_grid(100, 100, 0.3, 1, keep_free=[(0, 0), (99, 99)])
    assert g.passable((0, 0)) and g.passable((99, 99))
    assert 0.25 < np.isinf(g.cost).mean() < 0.35


def _reachable(g, start):
    seen, todo = {start}, [start]
    while todo:
        for n, _ in g.neighbours(todo.pop()):
            if n not in seen:
                seen.add(n)
                todo.append(n)
    return seen


@pytest.mark.parametrize("shape", [(21, 21), (20, 30), (3, 3), (11, 40)])
def test_maze_fully_connected(shape):
    g = maze_grid(*shape, seed=5, connectivity=4)
    assert g.passable((1, 1))
    assert _reachable(g, (1, 1)) == set(g.free_cells())


def test_scatter_terrain_and_start_goal():
    g = scatter_terrain(random_grid(30, 30, 0.1, 2), seed=2, fraction=0.3)
    assert (g.cost[np.isfinite(g.cost)] > 1).any()
    a, b = random_start_goal(g, seed=4, min_distance=10)
    assert a != b and g.passable(a) and g.passable(b)
    assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= 10
    assert random_start_goal(g, 4, 10) == (a, b)


def test_obstacles_move_and_restore_terrain():
    g = Grid.from_ascii(["....."])
    env = DynamicEnvironment(g, n_obstacles=1, seed=0, protected=[(0, 4)])
    start = env.obstacles[0].pos
    assert not g.passable(start)
    env.step()
    assert env.obstacles[0].pos != start
    assert g.passable(start)
    assert env.tick_count == 1


def test_obstacles_deterministic_and_respect_protected():
    def run():
        g = random_grid(15, 15, 0.1, 1)
        env = DynamicEnvironment(g, 5, seed=9, mode="random", protected=[(0, 0), (14, 14)])
        trail = []
        for _ in range(30):
            env.step()
            trail.append([o.pos for o in env.obstacles])
            assert all(o.pos not in env.protected for o in env.obstacles)
        return trail
    assert run() == run()


def test_patrol_reverses_at_wall():
    g = Grid.from_ascii(["...."])
    env = DynamicEnvironment(g, 1, seed=0, mode="patrol")
    seen = set()
    for _ in range(12):
        env.step()
        seen.add(env.obstacles[0].pos)
    assert len(seen) >= 2


def test_terrain_change_and_min_cost():
    g = Grid.from_ascii(["333"])
    env = DynamicEnvironment(g)
    env.change_terrain((0, 1), 1.0)
    assert g.min_cost == 1.0 and g.cost[0, 1] == 1.0


def test_path_blocked():
    g = Grid.from_ascii(["...", "...", "..."])
    path = [(0, 0), (1, 1), (2, 2)]
    assert not path_blocked(g, path)
    g.set_wall((1, 1))
    assert path_blocked(g, path)
    g.clear((1, 1))
    g.set_wall((0, 1))  # diagonal (0,0)->(1,1) now cuts a corner
    assert path_blocked(g, path)
    assert not path_blocked(g, [])
