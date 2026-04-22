#!/usr/bin/env python3
"""
Grid Visualizer — matches the Gazebo gazebo_tutorial simulation.
  S = Start  (green)
  G = Goal   (red)
  B = Bonus  (yellow)  — changes every run
  X = Obstacle (grey) — changes every run
  . = Free cell
"""

import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── configuration (must match world file) ──────────────────────────────────
ROWS, COLS = 5, 5            # grid dimensions
START       = (0, 0)         # row, col (matches world pose -2, 2)
GOAL        = (4, 4)         # row, col (matches world pose  2, -2)
NUM_BONUS   = 3
NUM_OBSTACLE = 5

# ── grid value constants ───────────────────────────────────────────────────
FREE     = 0
OBSTACLE = 1
START_V  = 2
GOAL_V   = 3
BONUS    = 4

# ── colours (match Gazebo materials) ──────────────────────────────────────
COLOUR = {
    FREE:     '#f5f5f5',   # light grey — free cell
    OBSTACLE: '#607D8B',   # blue-grey  — obstacle
    START_V:  '#4CAF50',   # green      — start
    GOAL_V:   '#F44336',   # red        — goal
    BONUS:    '#FFC107',   # amber      — bonus point
}


def build_grid():
    grid = np.full((ROWS, COLS), FREE, dtype=int)

    # Place fixed start and goal
    grid[START] = START_V
    grid[GOAL]  = GOAL_V

    # All free cells available for random placement
    occupied = {START, GOAL}
    free_cells = [(r, c) for r in range(ROWS) for c in range(COLS)
                  if (r, c) not in occupied]

    chosen = random.sample(free_cells, NUM_BONUS + NUM_OBSTACLE)
    for r, c in chosen[:NUM_BONUS]:
        grid[r, c] = BONUS
    for r, c in chosen[NUM_BONUS:]:
        grid[r, c] = OBSTACLE

    return grid

def update_robot_position(grid, robot_pos):
    """
    robot_pos: tuple (row, col)
    """
    r, c = robot_pos
    
    # Check if the current cell is a bonus point
    if grid[r, c] == BONUS:
        print(f"💰 Bonus collected at {robot_pos}!")
        grid[r, c] = FREE  # This "vanishes" the bonus
    
    return grid

def plot_grid(grid):
    # Build RGBA image from colour map
    img = np.zeros((ROWS, COLS, 3))
    for val, hex_col in COLOUR.items():
        r, g, b = tuple(int(hex_col.lstrip('#')[i:i+2], 16) / 255
                        for i in (0, 2, 4))
        img[grid == val] = [r, g, b]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, origin='upper')

    # Grid lines
    ax.set_xticks(np.arange(-0.5, COLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, ROWS, 1), minor=True)
    ax.grid(which='minor', color='#9E9E9E', linewidth=1)
    ax.tick_params(which='minor', length=0)

    # Major ticks: show column / row indices
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.set_xticklabels(range(COLS), fontsize=8)
    ax.set_yticklabels(range(ROWS), fontsize=8)

    # Labels inside cells
    LABEL = {START_V: 'S', GOAL_V: 'G', BONUS: 'B', OBSTACLE: 'X'}
    LABEL_COLOUR = {START_V: 'white', GOAL_V: 'white',
                    BONUS: '#333', OBSTACLE: 'white'}
    for r in range(ROWS):
        for c in range(COLS):
            val = grid[r, c]
            if val in LABEL:
                ax.text(c, r, LABEL[val],
                        ha='center', va='center',
                        fontsize=13, fontweight='bold',
                        color=LABEL_COLOUR[val])

    # Legend
    legend_handles = [
        mpatches.Patch(color=COLOUR[START_V],  label='Start (S)'),
        mpatches.Patch(color=COLOUR[GOAL_V],   label='Goal (G)'),
        mpatches.Patch(color=COLOUR[BONUS],    label='Bonus Point (B)'),
        mpatches.Patch(color=COLOUR[OBSTACLE], label='Obstacle (X)'),
        mpatches.Patch(color=COLOUR[FREE],     label='Free Cell'),
    ]
    ax.legend(handles=legend_handles, loc='upper left',
              bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=10)

    ax.set_title('Grid Visualizer — Random Obstacles & Bonus Points',
                 fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    grid = build_grid()
    plot_grid(grid)
