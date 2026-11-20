import math
import pygame

from pathfinding.dynamic import DynamicEnvironment
from pathfinding.generators import random_grid, random_start_goal
from pathfinding.search import search


class MovingAgent:
    """Walks its planned route while obstacles move; replans from its current cell when the route is blocked.

    Works on its own copy of the grid, so the map being edited is never changed.
    """

    def __init__(self, grid, start, goal, algorithm, n_obstacles=12, seed=7):
        self.grid = grid.copy()
        self.goal = goal
        self.algorithm = algorithm
        self.env = DynamicEnvironment(self.grid, n_obstacles=n_obstacles, seed=seed,
                                      mode="random", protected=[start, goal])
        self.position = start
        self.replans = 0
        self.steps = 0
        self.blocked = False
        self.result = search(self.grid, start, goal, algorithm=algorithm)
        self.route = list(self.result.path)

    @property
    def obstacles(self):
        return {obstacle.pos for obstacle in self.env.obstacles}

    @property
    def arrived(self):
        return self.position == self.goal

    def tick(self):
        """Advance one step and return what happened.

        A blocked route is shown for one tick ("blocked") before the agent replans from its
        current cell on the next tick ("replanned"), so the change is visible on screen.
        """
        if self.arrived:
            return "arrived"

        self.env.protected = {self.position, self.goal}

        if self.blocked or not self.route:
            self.result = search(self.grid, self.position, self.goal, algorithm=self.algorithm)
            if not self.result.found:
                self.route = []
                self.blocked = True
                self.env.step()
                return "waiting"
            self.blocked = False
            self.replans += 1
            self.route = list(self.result.path)
            return "replanned"

        self.env.step()
        if self.env.path_blocked(self.route):
            self.blocked = True
            return "blocked"

        if len(self.route) > 1:
            self.route.pop(0)
            self.position = self.route[0]
            self.steps += 1
        return "arrived" if self.arrived else "moved"


def compare_algorithms(grid, start, goal):
    names = ["dijkstra", "astar", "greedy", "weighted_astar"]

    results = {}

    for name in names:
        results[name] = search(
            grid,
            start,
            goal,
            algorithm=name
        )

    return results

CELL_SIZE = 40

WHITE = (245, 245, 245)
BLACK = (35, 35, 35)
GREEN = (40, 180, 90)
RED = (220, 65, 65)
BLUE = (100, 170, 240)
YELLOW = (255, 210, 70)
PURPLE = (150, 90, 200)
ORANGE = (245, 140, 30)
GRID_LINE = (180, 180, 180)
PANEL_WIDTH = 560
MOVE_MS = 350
STATUS_TEXT = {
    "moved": "moving",
    "blocked": "path blocked!",
    "replanned": "replanned from current cell",
    "waiting": "no path right now, waiting",
    "arrived": "goal reached",
}


def new_map(seed):
    grid = random_grid(rows=15, cols=15, density=0.20, seed=seed, connectivity=8)
    start, goal = random_start_goal(grid, seed=seed, min_distance=8)
    return grid, start, goal


def main() -> None:
    pygame.init()

    map_seed = 42
    grid, start, goal = new_map(map_seed)
    placement_mode = None

    width = grid.cols * CELL_SIZE
    height = grid.rows * CELL_SIZE

    screen = pygame.display.set_mode((width + PANEL_WIDTH, height + 180))
    pygame.display.set_caption("Pathfinding Visualiser")

    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 19)
    clock = pygame.time.Clock()
    button_width = 135
    button_height = 32
    button_gap = 8

    algorithm_buttons = {
        "dijkstra": pygame.Rect(18, height + 135, button_width, button_height),
        "astar": pygame.Rect(18 + 143, height + 135, button_width, button_height),
        "greedy": pygame.Rect(18 + 286, height + 135, button_width, button_height),
        "weighted_astar": pygame.Rect(18 + 429, height + 135, button_width, button_height),
    }
    compare_button = pygame.Rect(width + (PANEL_WIDTH - 220) // 2, 30, 220, 40)
    button_labels = {
        "dijkstra": "Dijkstra",
        "astar": "A*",
        "greedy": "Greedy",
        "weighted_astar": "Weighted A*"
    }

    algorithms = {
        pygame.K_1: "dijkstra",
        pygame.K_2: "astar",
        pygame.K_3: "greedy",
        pygame.K_4: "weighted_astar",
    }

    result = None
    algorithm_name = "None"
    comparison_results = None
    revealed = 0
    last_step = 0
    agent = None
    agent_status = ""
    last_move = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    agent = None
                    if compare_button.collidepoint(event.pos):
                        comparison_results = compare_algorithms(
                            grid,
                            start,
                            goal
                        )

                    for name, button_rect in algorithm_buttons.items():
                        if button_rect.collidepoint(event.pos):
                            algorithm_name = name
                            result = search(
                                grid,
                                start,
                                goal,
                                algorithm=algorithm_name
                            )
                            revealed = 0
                            last_step = pygame.time.get_ticks()

                    x, y = event.pos

                    if 0 <= x < width and 0 <= y < height:
                        col = x // CELL_SIZE
                        row = y // CELL_SIZE
                        cell = (row, col)

                        if placement_mode == "start":
                            if grid.passable(cell) and cell != goal:
                                start = cell
                                placement_mode = None
                                result = None
                                comparison_results = None
                                revealed = 0

                        elif placement_mode == "goal":
                            if grid.passable(cell) and cell != start:
                                goal = cell
                                placement_mode = None
                                result = None
                                comparison_results = None
                                revealed = 0

                        elif cell != start and cell != goal:
                            if grid.passable(cell):
                                grid.set_wall(cell)
                            else:
                                grid.clear(cell)

                            result = None
                            comparison_results = None
                            revealed = 0

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_s:
                    agent = None
                    placement_mode = "start"

                if event.key == pygame.K_g:
                    agent = None
                    placement_mode = "goal"

                if event.key == pygame.K_d:
                    if agent:
                        agent = None
                        result = None
                        revealed = 0
                    else:
                        if algorithm_name not in button_labels:
                            algorithm_name = "astar"
                        agent = MovingAgent(grid, start, goal, algorithm_name)
                        result = agent.result
                        revealed = len(result.visited_order)
                        agent_status = STATUS_TEXT["moved"]
                        last_move = pygame.time.get_ticks()

                if event.key == pygame.K_r:
                    map_seed += 1
                    grid, start, goal = new_map(map_seed)
                    agent = None
                    result = None
                    comparison_results = None
                    revealed = 0
                    placement_mode = None

                if event.key in algorithms:
                    agent = None
                    algorithm_name = algorithms[event.key]
                    result = search(
                        grid,
                        start,
                        goal,
                        algorithm=algorithm_name
                    )
                    revealed = 0
                    last_step = pygame.time.get_ticks()

        if agent and not agent.arrived:
            now = pygame.time.get_ticks()

            if now - last_move >= MOVE_MS:
                agent_status = STATUS_TEXT[agent.tick()]
                result = agent.result
                revealed = len(result.visited_order)
                last_move = now

        if result and revealed < len(result.visited_order):
            now = pygame.time.get_ticks()

            if now - last_step >= 40:
                revealed += 1
                last_step = now

        screen.fill(WHITE)

        draw_grid = agent.grid if agent else grid
        obstacles = agent.obstacles if agent else set()
        visited = set(result.visited_order[:revealed]) if result else set()
        if agent:
            path = set(agent.route)
        else:
            path = set(result.path) if result and revealed == len(result.visited_order) else set()

        for row in range(grid.rows):
            for col in range(grid.cols):
                cell = (row, col)

                rect = pygame.Rect(
                    col * CELL_SIZE,
                    row * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE
                )

                if cell in obstacles:
                    color = PURPLE
                elif math.isinf(float(draw_grid.cost[row, col])):
                    color = BLACK
                elif cell in path:
                    color = YELLOW
                elif cell in visited:
                    color = BLUE
                else:
                    color = WHITE

                if cell == start:
                    color = GREEN
                elif cell == goal:
                    color = RED

                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, GRID_LINE, rect, 1)

        if agent:
            agent_row, agent_col = agent.position
            centre = (agent_col * CELL_SIZE + CELL_SIZE // 2, agent_row * CELL_SIZE + CELL_SIZE // 2)
            pygame.draw.circle(screen, ORANGE, centre, CELL_SIZE // 3)

        if agent:
            stats = (
                f"Moving obstacles | {button_labels[agent.algorithm]} | "
                f"Replans: {agent.replans} | Steps: {agent.steps} | {agent_status}"
            )
        elif result and result.found:
            stats = (
                f"Expanded: {result.expanded} | "
                f"Cost: {result.cost:.2f} | "
                f"Time: {result.runtime * 1000:.3f} ms"
            )
        elif result:
            stats = f"No path found | Expanded: {result.expanded}"
        else:
            stats = "Select an algorithm to begin."

        screen.blit(font.render(stats, True, BLACK), (10, height + 8))

        instructions = "1: Dijkstra  2: A*  3: Greedy  4: Weighted A*"
        screen.blit(font.render(instructions, True, BLACK), (10, height + 38))

        editor_text = "Left click: Add/remove walls"
        screen.blit(font.render(editor_text, True, BLACK), (10, height + 65))

        placement_text = "S: Move start  G: Move goal  D: Moving obstacles on/off  R: New map"
        screen.blit(font.render(placement_text, True, BLACK), (10, height + 90))

        if placement_mode:
            mode_text = f"Click a cell to place {placement_mode}"
            screen.blit(font.render(mode_text, True, RED), (10, height + 110))

        for name, button_rect in algorithm_buttons.items():
            if algorithm_name == name:
                button_color = GREEN
            else:
                button_color = BLUE

            pygame.draw.rect(screen, button_color, button_rect, border_radius=5)

            button_text = font.render(button_labels[name], True, WHITE)
            text_rect = button_text.get_rect(center=button_rect.center)
            screen.blit(button_text, text_rect)
        pygame.draw.rect(screen, GREEN, compare_button, border_radius=5)

        compare_text = font.render("Compare Algorithms", True, WHITE)
        compare_rect = compare_text.get_rect(center=compare_button.center)
        screen.blit(compare_text, compare_rect)

        if comparison_results:
            comparison_y = 100

            title = font.render("Algorithm Comparison", True, BLACK)
            screen.blit(title, (width + 20, comparison_y))

            headers = ["Algorithm", "Expanded", "Cost", "Time (ms)", "Max Frontier"]

            x_positions = [width + 15, width + 150, width + 250, width + 340, width + 445]

            for i, header in enumerate(headers):
                text = small_font.render(header, True, BLACK)
                screen.blit(text, (x_positions[i], comparison_y + 30))

            names = ["dijkstra", "astar", "greedy", "weighted_astar"]

            for i, name in enumerate(names):
                item = comparison_results[name]

                values = [
                    button_labels[name],
                    str(item.expanded),
                    f"{item.cost:.2f}" if item.found else "No path",
                    f"{item.runtime * 1000:.3f}",
                    str(item.max_frontier)
                ]

                for j, value in enumerate(values):
                    text = small_font.render(value, True, BLACK)
                    screen.blit(text, (x_positions[j], comparison_y + 60 + i * 25))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
