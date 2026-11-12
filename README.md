# CSCI218 Project – Intelligent Pathfinding with Dijkstra and A*

Topic 8, Lab group 1, University of Wollongong in Dubai, Autumn 2026.

An agent finds the cheapest path through a 2D grid with walls, terrain costs and moving obstacles. We compare Dijkstra, A* (four heuristics), Greedy Best-First and Weighted A* on nodes expanded, runtime, memory and path cost.

## Install

Python 3.11 or newer.

```bash
git clone https://github.com/CrazyFFester/CSCI218-Project.git
cd CSCI218-Project
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt -e .
```

We use `pygame-ce` (the maintained fork, same `import pygame`), because plain `pygame` has no Python 3.14 wheels. Do not install `pygame` alongside it.

## Run

```bash
pytest                                        # all tests
python -m pathfinding.viz                     # interactive visualiser
python experiments/benchmark.py configs/<sweep>.json   # one experiment sweep
```

## Layout

| Path | What | Owner |
|---|---|---|
| `src/pathfinding/result.py` | `SearchResult`, the shared return type | Emir |
| `src/pathfinding/grid.py` | `Grid`: neighbours, corner rule, editing, save/load | Shazeel |
| `src/pathfinding/generators.py` | `random_grid`, `maze_grid`, `scatter_terrain`, `random_start_goal` | Shazeel |
| `src/pathfinding/dynamic.py` | `DynamicEnvironment` (moving obstacles, terrain changes), `path_blocked` | Shazeel |
| `src/pathfinding/search.py` | `dijkstra`, `astar`, `greedy_best_first`, Weighted A* via `astar(weight=...)` | Aymen |
| `src/pathfinding/heuristics.py` | `manhattan`, `euclidean`, `octile`, `chebyshev` | Aymen |
| `src/pathfinding/viz.py` | pygame visualiser and editor | Abdul |
| `src/pathfinding/stubs.py` | Fixed 5×5 map and known result, used by tests | Emir |
| `experiments/benchmark.py` | Runs a sweep from `configs/`, writes CSV to `results/` | Sobaan |
| `notebooks/analysis.ipynb` | All tables and figures, read only from `results/` | Sobaan |
| `tests/` | pytest, run by CI on every pull request | Emir |

## Using the code

```python
from pathfinding.generators import random_grid, maze_grid, scatter_terrain, random_start_goal
from pathfinding.search import dijkstra, astar, greedy_best_first
from pathfinding.heuristics import octile, euclidean
from pathfinding.dynamic import DynamicEnvironment

grid = random_grid(128, 128, density=0.2, seed=42)          # or maze_grid(129, 129, seed=42)
scatter_terrain(grid, seed=42, fraction=0.3)                 # optional: costs 2, 3, 5 on 30% of free cells
start, goal = random_start_goal(grid, seed=42, min_distance=64)

result = astar(grid, start, goal)                            # Octile by default
result = astar(grid, start, goal, heuristic=euclidean)
result = astar(grid, start, goal, weight=2.0)                # Weighted A*
result = dijkstra(grid, start, goal)
result = greedy_best_first(grid, start, goal)
print(result.found, result.cost, result.expanded, result.runtime)

env = DynamicEnvironment(grid, n_obstacles=20, seed=42, protected=[start, goal])
env.step()                                                   # obstacles move one cell
if env.path_blocked(result.path):
    result = astar(grid, start, goal)                        # replan
```

- `scatter_terrain` and `DynamicEnvironment` change the grid in place. Use `grid.copy()` to keep the original map.
- Change cells only with `grid.set_wall(cell)`, `grid.set_cost(cell, value)` or `grid.clear(cell)`. Writing to `grid.cost[r, c]` directly leaves `grid.min_cost` stale, and A* can then return a non-optimal path.
- `random_start_goal` does not check that the goal is reachable. When it is not, `result.found` is `False` and `result.cost` is `inf`.

## Shared rules

- Every planner takes `(grid, start, goal)` and returns a `SearchResult`.
- No path: `path` is empty and `cost` is `inf`.
- Step cost = cost of the cell being entered × step length (1 straight, √2 diagonal).
- No corner cutting: a diagonal move needs both orthogonal cells beside it to be passable.
- Ties on equal f go to the larger g, then insertion order, so node counts are the same on every machine.
- `runtime` is `time.perf_counter` seconds for the search only.

## Workflow

1. Never push to `main`. Create a branch: `git checkout -b yourname/feature`.
2. Commit often from your own GitHub account.
3. Push and open a pull request. Another member reviews and merges once the tests pass. Never merge your own pull request.
4. Before starting new work, update your branch: `git checkout main && git pull`, then create the branch.
