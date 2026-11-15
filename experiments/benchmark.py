"""Run one experiment sweep from a JSON config file. Owner: Sobaan (D).

Usage:
    python experiments/benchmark.py configs/<sweep>.json

The script creates the same seeded map and start/goal pair for every planner in a
trial, runs one warm-up followed by repeated timed searches, and writes one raw
CSV row per planner/trial to ``results/<sweep>.csv``.

Config format (see files in ``configs/``):
    {
      "name": "sweep1_grid_size",
      "seed": 218,
      "pairs": 50,
      "warmup_runs": 1,
      "timed_runs": 5,
      "base": { ... },
      "vary": {"grid_size": [32, 64, 128]},
      "planners": [ ... ]
    }

``vary`` may contain more than one factor; the Cartesian product is run.
Supported factors are grid_size, rows, cols, obstacle_density, connectivity,
map_type, terrain, terrain_fraction, heuristic, and weight.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

# Allow ``python experiments/benchmark.py ...`` to work from a fresh clone
# even if the package has not yet been installed with ``pip install -e .``.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pathfinding.generators import (  # noqa: E402
    maze_grid,
    random_grid,
    random_start_goal,
    scatter_terrain,
)
from pathfinding.heuristics import (  # noqa: E402
    chebyshev,
    euclidean,
    manhattan,
    octile,
)
from pathfinding.search import astar, dijkstra, greedy_best_first  # noqa: E402


HEURISTICS: dict[str, Callable] = {
    "manhattan": manhattan,
    "euclidean": euclidean,
    "octile": octile,
    "chebyshev": chebyshev,
}

SUPPORTED_VARY_KEYS = {
    "grid_size",
    "rows",
    "cols",
    "obstacle_density",
    "connectivity",
    "map_type",
    "terrain",
    "terrain_fraction",
    "heuristic",
    "weight",
}

CSV_FIELDS = [
    "sweep",
    "configuration_id",
    "pair_id",
    "map_seed",
    "pair_seed",
    "rows",
    "cols",
    "grid_size",
    "map_type",
    "obstacle_density",
    "connectivity",
    "terrain",
    "terrain_fraction",
    "start_row",
    "start_col",
    "goal_row",
    "goal_col",
    "reachable",
    "planner",
    "heuristic",
    "weight",
    "found",
    "path_cost",
    "baseline_cost",
    "suboptimality_ratio",
    "expanded",
    "max_frontier",
    "peak_memory",
    "runtime_ms",
    "path_length",
]


def _load_config(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            config = json.load(fh)
    except FileNotFoundError:
        raise SystemExit(f"Config file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from None

    required = {"base", "planners"}
    missing = required - config.keys()
    if missing:
        raise SystemExit(f"Config is missing required key(s): {', '.join(sorted(missing))}")

    config.setdefault("name", path.stem)
    config.setdefault("seed", 218)
    config.setdefault("pairs", 50)
    config.setdefault("warmup_runs", 1)
    config.setdefault("timed_runs", 5)
    config.setdefault("vary", {})

    if not isinstance(config["base"], dict):
        raise SystemExit("'base' must be a JSON object")
    if not isinstance(config["planners"], list) or not config["planners"]:
        raise SystemExit("'planners' must be a non-empty JSON list")
    if not isinstance(config["vary"], dict):
        raise SystemExit("'vary' must be a JSON object")

    unknown = set(config["vary"]) - SUPPORTED_VARY_KEYS
    if unknown:
        raise SystemExit(f"Unsupported vary key(s): {', '.join(sorted(unknown))}")

    for key, values in config["vary"].items():
        if not isinstance(values, list) or not values:
            raise SystemExit(f"vary.{key} must be a non-empty JSON list")

    for key in ("pairs", "timed_runs"):
        if int(config[key]) < 1:
            raise SystemExit(f"'{key}' must be at least 1")
    if int(config["warmup_runs"]) < 0:
        raise SystemExit("'warmup_runs' cannot be negative")

    return config


def _expand_configurations(base: dict[str, Any], vary: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Return the Cartesian product of all varied factors over the base config."""
    if not vary:
        return [dict(base)]

    keys = list(vary)
    configurations: list[dict[str, Any]] = []
    for values in itertools.product(*(vary[key] for key in keys)):
        item = dict(base)
        item.update(dict(zip(keys, values)))
        configurations.append(item)
    return configurations


def _normalise_environment(raw: dict[str, Any]) -> dict[str, Any]:
    env = dict(raw)

    if "grid_size" in env:
        size = int(env["grid_size"])
        env["rows"] = size
        env["cols"] = size
    else:
        env["rows"] = int(env.get("rows", 128))
        env["cols"] = int(env.get("cols", 128))
        env["grid_size"] = env["rows"] if env["rows"] == env["cols"] else ""

    env.setdefault("map_type", "random")
    env.setdefault("obstacle_density", 0.20)
    env.setdefault("connectivity", 8)
    env.setdefault("terrain", "uniform")
    env.setdefault("terrain_fraction", 0.30)
    env.setdefault("terrain_costs", [2.0, 3.0, 4.0, 5.0])
    env.setdefault("min_distance", max(1, min(env["rows"], env["cols"]) // 2))
    env.setdefault("heuristic", "octile")
    env.setdefault("weight", 2.0)

    env["rows"] = int(env["rows"])
    env["cols"] = int(env["cols"])
    env["connectivity"] = int(env["connectivity"])
    env["obstacle_density"] = float(env["obstacle_density"])
    env["terrain_fraction"] = float(env["terrain_fraction"])
    env["min_distance"] = int(env["min_distance"])

    if env["rows"] < 2 or env["cols"] < 2:
        raise ValueError("rows and cols must both be at least 2")
    if env["connectivity"] not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if env["map_type"] not in ("random", "maze"):
        raise ValueError("map_type must be 'random' or 'maze'")
    if env["terrain"] not in ("uniform", "weighted"):
        raise ValueError("terrain must be 'uniform' or 'weighted'")
    if not 0 <= env["obstacle_density"] < 1:
        raise ValueError("obstacle_density must be in [0, 1)")
    if not 0 <= env["terrain_fraction"] <= 1:
        raise ValueError("terrain_fraction must be in [0, 1]")

    return env


def _make_grid(env: dict[str, Any], map_seed: int):
    if env["map_type"] == "random":
        grid = random_grid(
            env["rows"],
            env["cols"],
            density=env["obstacle_density"],
            seed=map_seed,
            connectivity=env["connectivity"],
        )
    else:
        grid = maze_grid(
            env["rows"],
            env["cols"],
            seed=map_seed,
            connectivity=env["connectivity"],
        )

    if env["terrain"] == "weighted":
        terrain_seed = map_seed + 1_000_003
        scatter_terrain(
            grid,
            seed=terrain_seed,
            fraction=env["terrain_fraction"],
            costs=tuple(float(v) for v in env["terrain_costs"]),
        )

    return grid


def _planner_settings(raw: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("each planner entry must be a JSON object")

    algorithm = str(raw.get("algorithm", raw.get("name", "astar"))).lower()
    aliases = {
        "a*": "astar",
        "a_star": "astar",
        "weighted_a*": "weighted_astar",
        "weighted-a*": "weighted_astar",
        "weighted_astar": "weighted_astar",
        "greedy_best_first": "greedy",
        "greedy-best-first": "greedy",
    }
    algorithm = aliases.get(algorithm, algorithm)

    if algorithm not in {"dijkstra", "astar", "weighted_astar", "greedy"}:
        raise ValueError(f"unknown planner algorithm {algorithm!r}")

    heuristic_name = str(raw.get("heuristic", env.get("heuristic", "octile"))).lower()
    if heuristic_name not in HEURISTICS:
        raise ValueError(f"unknown heuristic {heuristic_name!r}")

    if algorithm == "dijkstra":
        weight = 0.0
        heuristic_name = "zero"
    elif algorithm == "greedy":
        weight = 1.0
    elif algorithm == "weighted_astar":
        weight = float(raw.get("weight", env.get("weight", 2.0)))
        if weight < 1:
            raise ValueError("Weighted A* requires weight >= 1")
    else:
        weight = float(raw.get("weight", 1.0))
        if weight <= 0:
            raise ValueError("A* weight must be > 0")

    default_name = {
        "dijkstra": "dijkstra",
        "astar": "astar",
        "weighted_astar": f"weighted_astar_w{weight:g}",
        "greedy": "greedy",
    }[algorithm]

    return {
        "name": str(raw.get("name", default_name)),
        "algorithm": algorithm,
        "heuristic": heuristic_name,
        "weight": weight,
    }


def _run_once(grid, start, goal, planner: dict[str, Any]):
    algorithm = planner["algorithm"]

    if algorithm == "dijkstra":
        return dijkstra(grid, start, goal)

    heuristic = HEURISTICS[planner["heuristic"]]

    if algorithm == "greedy":
        return greedy_best_first(grid, start, goal, heuristic=heuristic)

    # Standard A* and Weighted A* use the same implementation; weight=1 is A*.
    return astar(
        grid,
        start,
        goal,
        heuristic=heuristic,
        weight=planner["weight"],
    )


def _benchmark_planner(
    grid,
    start,
    goal,
    planner: dict[str, Any],
    warmup_runs: int,
    timed_runs: int,
):
    for _ in range(warmup_runs):
        _run_once(grid, start, goal, planner)

    results = [_run_once(grid, start, goal, planner) for _ in range(timed_runs)]
    first = results[0]

    # Deterministic metrics should be identical on every repeat. Failing here is
    # better than silently writing inconsistent experiment data.
    for other in results[1:]:
        if (
            other.found != first.found
            or other.expanded != first.expanded
            or other.max_frontier != first.max_frontier
            or (first.found and not math.isclose(other.cost, first.cost, rel_tol=0.0, abs_tol=1e-9))
        ):
            raise RuntimeError("planner returned inconsistent deterministic results across repeats")

    median_runtime = statistics.median(result.runtime for result in results)
    return first, median_runtime


def _ratio(cost: float, baseline: float) -> float:
    if not math.isfinite(cost) or not math.isfinite(baseline):
        return math.nan
    if baseline == 0:
        return 1.0 if cost == 0 else math.nan
    return cost / baseline


def run_sweep(config_path: Path, output_path: Path | None = None) -> Path:
    config = _load_config(config_path)
    configurations = _expand_configurations(config["base"], config["vary"])

    sweep_name = str(config["name"])
    pairs = int(config["pairs"])
    warmup_runs = int(config["warmup_runs"])
    timed_runs = int(config["timed_runs"])
    master_seed = int(config["seed"])

    if output_path is None:
        output_name = str(config.get("output", f"{sweep_name}.csv"))
        output_path = ROOT / "results" / output_name
    elif not output_path.is_absolute():
        output_path = ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_trials = len(configurations) * pairs
    total_rows = total_trials * len(config["planners"])
    print(
        f"Sweep: {sweep_name} | configurations: {len(configurations)} | "
        f"pairs/config: {pairs} | planned rows: {total_rows}"
    )
    print(f"Writing: {output_path}")

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()

        completed_trials = 0
        for config_index, raw_env in enumerate(configurations, start=1):
            env = _normalise_environment(raw_env)
            planners = [_planner_settings(raw, env) for raw in config["planners"]]
            unreachable_pairs = 0

            for pair_index in range(pairs):
                # A pair index always receives the same map/pair seeds across factor
                # levels. This keeps comparisons fair (for example, all four
                # heuristics see the same map and coordinates) while map generation
                # still changes naturally when the varied factor changes.
                map_seed = master_seed + pair_index
                pair_seed = master_seed + 500_000_000 + pair_index

                grid = _make_grid(env, map_seed)
                try:
                    start, goal = random_start_goal(
                        grid,
                        seed=pair_seed,
                        min_distance=env["min_distance"],
                    )
                except ValueError as exc:
                    raise RuntimeError(
                        f"Could not choose start/goal for configuration {config_index}, "
                        f"pair {pair_index + 1}: {exc}"
                    ) from exc

                # Dijkstra supplies the optimal reference cost used for the
                # suboptimality ratio. Reuse its full benchmark result when the
                # config already includes Dijkstra; otherwise one reference run
                # is enough because only its deterministic path cost is needed.
                dijkstra_planner = next((p for p in planners if p["algorithm"] == "dijkstra"), None)
                cached_dijkstra = None
                if dijkstra_planner is None:
                    baseline_result = dijkstra(grid, start, goal)
                    baseline_cost = baseline_result.cost
                    baseline_found = baseline_result.found
                else:
                    cached_dijkstra = _benchmark_planner(
                        grid, start, goal, dijkstra_planner, warmup_runs, timed_runs
                    )
                    baseline_cost = cached_dijkstra[0].cost
                    baseline_found = cached_dijkstra[0].found

                if not baseline_found:
                    unreachable_pairs += 1

                for planner in planners:
                    if planner["algorithm"] == "dijkstra" and cached_dijkstra is not None:
                        result, median_runtime = cached_dijkstra
                    else:
                        result, median_runtime = _benchmark_planner(
                            grid, start, goal, planner, warmup_runs, timed_runs
                        )

                    writer.writerow(
                        {
                            "sweep": sweep_name,
                            "configuration_id": config_index,
                            "pair_id": pair_index + 1,
                            "map_seed": map_seed,
                            "pair_seed": pair_seed,
                            "rows": env["rows"],
                            "cols": env["cols"],
                            "grid_size": env["grid_size"],
                            "map_type": env["map_type"],
                            "obstacle_density": env["obstacle_density"],
                            "connectivity": env["connectivity"],
                            "terrain": env["terrain"],
                            "terrain_fraction": env["terrain_fraction"] if env["terrain"] == "weighted" else 0.0,
                            "start_row": start[0],
                            "start_col": start[1],
                            "goal_row": goal[0],
                            "goal_col": goal[1],
                            "reachable": baseline_found,
                            "planner": planner["name"],
                            "heuristic": planner["heuristic"],
                            "weight": planner["weight"],
                            "found": result.found,
                            "path_cost": result.cost,
                            "baseline_cost": baseline_cost,
                            "suboptimality_ratio": _ratio(result.cost, baseline_cost),
                            "expanded": result.expanded,
                            "max_frontier": result.max_frontier,
                            "peak_memory": result.peak_memory,
                            "runtime_ms": median_runtime * 1000.0,
                            "path_length": len(result.path),
                        }
                    )

                completed_trials += 1
                if completed_trials == 1 or completed_trials % 10 == 0 or completed_trials == total_trials:
                    print(f"  completed {completed_trials}/{total_trials} map/pair trials")

            print(
                f"  configuration {config_index}: reachable={pairs - unreachable_pairs}, "
                f"unreachable={unreachable_pairs}"
            )

    print(f"Done. Raw results saved to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="path to a JSON sweep config in configs/")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional CSV path; defaults to results/<config name>.csv",
    )
    args = parser.parse_args()
    run_sweep(args.config, args.output)


if __name__ == "__main__":
    main()
