import math
import pygame

from pathfinding.stubs import stub_grid, STUB_START, STUB_GOAL
from pathfinding.search import search


CELL_SIZE = 100

WHITE = (245, 245, 245)
BLACK = (35, 35, 35)
GREEN = (40, 180, 90)
RED = (220, 65, 65)
BLUE = (100, 170, 240)
YELLOW = (255, 210, 70)
GRID_LINE = (180, 180, 180)


def main() -> None:
    pygame.init()

    grid = stub_grid()

    width = grid.cols * CELL_SIZE
    height = grid.rows * CELL_SIZE

    screen = pygame.display.set_mode((width, height + 75))
    pygame.display.set_caption("Pathfinding Visualiser")

    font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()

    algorithms = {
        pygame.K_1: "dijkstra",
        pygame.K_2: "astar",
        pygame.K_3: "greedy",
        pygame.K_4: "weighted_astar",
    }

    result = None
    algorithm_name = "None"
    revealed = 0
    last_step = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                if event.key in algorithms:
                    algorithm_name = algorithms[event.key]
                    result = search(
                        grid,
                        STUB_START,
                        STUB_GOAL,
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

                if cell == STUB_START:
                    color = GREEN
                elif cell == STUB_GOAL:
                    color = RED

                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, GRID_LINE, rect, 1)

        if result:
            stats = (
                f"Expanded: {result.expanded} | "
                f"Cost: {result.cost:.2f} | "
                f"Time: {result.runtime * 1000:.3f} ms"
            )
        else:
            stats = "Select an algorithm to begin."

        screen.blit(font.render(stats, True, BLACK), (10, height + 8))

        instructions = "1: Dijkstra  2: A*  3: Greedy  4: Weighted A*"
        screen.blit(font.render(instructions, True, BLACK), (10, height + 38))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
