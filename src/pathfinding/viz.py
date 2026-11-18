import math
import pygame

from pathfinding.generators import random_grid, random_start_goal
from pathfinding.search import search
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
GRID_LINE = (180, 180, 180)


def main() -> None:
    pygame.init()

    grid = random_grid(
    rows=15,
    cols=15,
    density=0.20,
    seed=42,
    connectivity=8
)

    start, goal = random_start_goal(grid, seed=42)
    placement_mode = None

    width = grid.cols * CELL_SIZE
    height = grid.rows * CELL_SIZE

    screen = pygame.display.set_mode((width + 420, height + 180))
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
    compare_button = pygame.Rect(width + 100, 30, 220, 40)
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
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
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
                    placement_mode = "start"

                if event.key == pygame.K_g:
                    placement_mode = "goal"

                if event.key in algorithms:
                    algorithm_name = algorithms[event.key]
                    result = search(
                        grid,
                        start,
                        goal,
                        algorithm=algorithm_name
                    )
                    revealed = 0
                    last_step = pygame.time.get_ticks()

        if result and revealed < len(result.visited_order):
            now = pygame.time.get_ticks()

            if now - last_step >= 40:
                revealed += 1
                last_step = now

        screen.fill(WHITE)

        visited = set(result.visited_order[:revealed]) if result else set()
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

                if math.isinf(float(grid.cost[row, col])):
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

        if result and result.found:
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

        placement_text = "S: Move start  G: Move goal"
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

            x_positions = [width + 15, width + 115, width + 205, width + 285, width + 365]

            for i, header in enumerate(headers):
                text = small_font.render(header, True, BLACK)
                screen.blit(text, (x_positions[i], comparison_y + 30))

            names = ["dijkstra", "astar", "greedy", "weighted_astar"]

            for i, name in enumerate(names):
                item = comparison_results[name]

                values = [
                    name,
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
