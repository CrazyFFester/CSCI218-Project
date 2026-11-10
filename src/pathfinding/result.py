from dataclasses import dataclass

Cell = tuple[int, int]


@dataclass(frozen=True)
class SearchResult:
    """What every planner returns.

    path: cells from start to goal inclusive; empty when no path exists.
    cost: sum of step costs along path; math.inf when no path exists.
    expanded: number of nodes popped from the open list and expanded.
    max_frontier: largest size the open list reached.
    visited_order: cells in the order they were expanded (drives the visualiser).
    runtime: search time in seconds (time.perf_counter), excluding map creation.
    """

    path: list[Cell]
    cost: float
    expanded: int
    max_frontier: int
    visited_order: list[Cell]
    runtime: float

    @property
    def found(self) -> bool:
        return bool(self.path)

    @property
    def peak_memory(self) -> int:
        return self.max_frontier + self.expanded
