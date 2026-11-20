import numpy as np
import pytest

from pathfinding.generators import random_grid, random_start_goal
from pathfinding.viz import MovingAgent


def make_agent(seed, algorithm="astar"):
    grid = random_grid(15, 15, 0.2, seed)
    start, goal = random_start_goal(grid, seed=seed, min_distance=8)
    return grid, MovingAgent(grid, start, goal, algorithm, n_obstacles=12, seed=seed)


@pytest.mark.parametrize("seed", range(10))
def test_agent_never_stands_on_an_obstacle_and_route_stays_walkable(seed):
    _, agent = make_agent(seed)
    for _ in range(200):
        status = agent.tick()
        assert agent.position not in agent.obstacles
        assert agent.grid.passable(agent.position)
        if agent.route:
            assert agent.route[0] == agent.position
            assert agent.route[-1] == agent.goal
            assert agent.env.path_blocked(agent.route) == (status == "blocked")
        if status == "arrived":
            break


def test_blocked_tick_is_followed_by_a_replan_from_the_current_cell():
    for seed in range(10):
        _, agent = make_agent(seed)
        for _ in range(300):
            status = agent.tick()
            if status == "blocked":
                position = agent.position
                assert agent.tick() in ("replanned", "waiting")
                assert agent.position == position
                if agent.route:
                    assert agent.route[0] == position
                return
            if status == "arrived":
                break
    pytest.fail("no route was ever blocked")


def test_agent_replans_and_reaches_goals():
    replans = arrived = 0
    for seed in range(10):
        _, agent = make_agent(seed)
        for _ in range(300):
            if agent.tick() == "arrived":
                arrived += 1
                break
        replans += agent.replans
    assert replans > 0
    assert arrived >= 8


def test_agent_does_not_change_the_edited_map():
    grid, agent = make_agent(3)
    before = grid.cost.copy()
    for _ in range(50):
        agent.tick()
    assert np.array_equal(grid.cost, before)


@pytest.mark.parametrize("algorithm", ["dijkstra", "astar", "greedy", "weighted_astar"])
def test_agent_works_with_every_planner(algorithm):
    _, agent = make_agent(5, algorithm)
    for _ in range(300):
        if agent.tick() == "arrived":
            break
    assert agent.arrived
