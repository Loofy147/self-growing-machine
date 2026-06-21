import json, numpy as np
def print_grid(grid):
    for row in grid: print("".join(str(c) for c in row))
with open("data/task002.json") as f: data = json.load(f)
print("Input:")
print_grid(data['train'][0]['input'])
print("Output:")
print_grid(data['train'][0]['output'])
