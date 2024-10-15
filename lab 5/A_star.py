import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

grid = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
]

start = (8, 2)  # Prince
goal = (2, 5)  # Cinderella


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star(start, goal, grid):
    rows, cols = len(grid), len(grid[0])
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        neighbors = get_neighbors(current, grid, rows, cols)
        for neighbor, move_cost in neighbors:
            tentative_g_score = g_score[current] + move_cost
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None


def get_neighbors(node, grid, rows, cols):
    directions = [
        (0, 1, 10), (1, 0, 10), (0, -1, 10), (-1, 0, 10),  # Right, Down, Left, Up
        (1, 1, 14), (1, -1, 14), (-1, 1, 14), (-1, -1, 14)  # Diagonals
    ]
    neighbors = []
    for d in directions:
        neighbor = (node[0] + d[0], node[1] + d[1])
        if 1 <= neighbor[0] <= rows and 1 <= neighbor[1] <= cols:
            if grid[neighbor[0] - 1][neighbor[1] - 1] == 0:
                neighbors.append((neighbor, d[2]))
    return neighbors


def visualise(path):
    my_colors = ['white', 'black', (0.68, 0.85, 0.90), (0.56, 0.93, 0.56), 'pink']
    cmap = ListedColormap(my_colors)
    fig, ax = plt.subplots()
    grid_np = np.array(grid)

    for (r, c) in path:
        grid_np[r - 1, c - 1] = 2
    grid_np[start[0] - 1, start[1] - 1] = 3
    grid_np[goal[0] - 1, goal[1] - 1] = 4

    ax.imshow(grid_np, cmap=cmap, extent=(0.5, 8.5, 9.5, 0.5))

    ax.set_xticks(np.arange(1, 9, 1))
    ax.set_yticks(np.arange(1, 10, 1))
    ax.set_xticks(np.arange(0.5, 8.5, 1), minor=True)
    ax.set_yticks(np.arange(0.5, 9.5, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.tick_params(axis='both', which='both', length=0, width=0)

    ax.text(start[1], start[0], 'Prince', ha='center', va='center', color='black', fontsize=10)
    ax.text(goal[1], goal[0], 'Cinderella', ha='center', va='center', color='black', fontsize=6)

    plt.tight_layout()
    plt.show()


path = a_star(start, goal, grid)
if path:
    print("Prince is alone. He starts going to Cinderella :(")
    print([(r, c) for r, c in path])
    print("Prince is with Cinderella :)")
    visualise(path)
else:
    print("There is no way for Prince to go to Cinderella :(")
