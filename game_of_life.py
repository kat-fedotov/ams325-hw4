import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import matplotlib.animation as animation

def initialize_grid(size):
    # Initialize a grid with all cells dead
    return np.zeros((size, size), dtype=int)

def place_glider(grid, x, y):
    # Place a Glider pattern at position (x, y) on the grid
    glider = np.array([[0, 1, 0],
                      [0, 0, 1],
                      [1, 1, 1]])
    grid[x:x + 3, y:y + 3] = glider

def evolve(grid):
    # Apply the rules of Conway's Game of Life to evolve the grid for one step
    size = grid.shape[0]
    new_grid = grid.copy()

    for i in range(size):
        for j in range(size):
            cell = grid[i, j]
            live_neighbors = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    ni, nj = (i + dx) % size, (j + dy) % size
                    live_neighbors += grid[ni, nj]

            if cell == 1:
                if live_neighbors < 2 or live_neighbors > 3:
                    new_grid[i, j] = 0
            else:
                if live_neighbors == 3:
                    new_grid[i, j] = 1

    return new_grid

# Global variable to store the grid state
grid = None

def func(frame, im):
    global grid
    # Callable function for animation generation
    # Update the grid data
    grid = evolve(grid)
    im.set_array(grid)
    return [im]

def generate_animation(initial_grid, k):
    global grid
    grid = initial_grid
    fig, ax = plt.subplots()
    im = ax.imshow(grid, cmap='binary', interpolation='nearest')

    ani = FuncAnimation(fig, func, frames=k, fargs=(im,), interval=50, blit=True, repeat=False)

    writer = animation.PillowWriter(fps=10)

    ani.save('game_of_life.gif', writer=writer)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python game_of_life.py k")
        sys.exit(1)

    try:
        k = int(sys.argv[1])
        if k <= 0:
            raise ValueError
    except ValueError:
        print("Please provide a positive integer value for k.")
        sys.exit(1)

    grid_size = 20  # You can adjust the grid size as needed
    initial_grid = initialize_grid(grid_size)

    # Place the Glider pattern near the middle of the grid
    x, y = grid_size // 2 - 1, grid_size // 2 - 1
    place_glider(initial_grid, x, y)

    generate_animation(initial_grid, k)
    plt.show()
