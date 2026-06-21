import json
import os

def print_grid(grid):
    for row in grid:
        print("".join(str(c).replace('0', '.') for c in row))

for i in range(3, 10):
    try:
        with open(f"data/task{i:03d}.json") as f: task = json.load(f)
        print(f"\n--- Task {i:03d} ---")
        ex = task['train'][0]
        print("Input:")
        print_grid(ex['input'])
        print("Output:")
        print_grid(ex['output'])
    except: pass
