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
| `src/pathfinding/grid.py` | Grid, neighbours, corner rule, save/load | Shazeel |
| `src/pathfinding/generators.py` | Seeded random and maze maps | Shazeel |
| `src/pathfinding/dynamic.py` | Moving obstacles, blocked-path check | Shazeel |
| `src/pathfinding/search.py` | Dijkstra, A*, Greedy, Weighted A* | Aymen |
| `src/pathfinding/heuristics.py` | Manhattan, Euclidean, Octile, Chebyshev | Aymen |
| `src/pathfinding/viz.py` | pygame visualiser and editor | Abdul |
| `src/pathfinding/stubs.py` | Temporary 5×5 grid and fake result to build against | Emir |
| `experiments/benchmark.py` | Runs a sweep from `configs/`, writes CSV to `results/` | Sobaan |
| `notebooks/analysis.ipynb` | All tables and figures, read only from `results/` | Sobaan |
| `tests/` | pytest, run by CI on every pull request | Emir |

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
3. Push and open a pull request. Another member reviews and merges once the tests pass.
4. Until `grid.py` and `search.py` are merged, build against `pathfinding.stubs`.
