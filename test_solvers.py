import torch
import json
import numpy as np
from solvers import HoleFillSolver

def test_task_002():
    with open("data/task002.json") as f:
        task = json.load(f)

    # Task 002: color bound is 3, color fill is 4
    model = HoleFillSolver(color_bound=3, color_fill=4)

    for i, pair in enumerate(task['train']):
        grid_in = pair['input']
        grid_out = pair['output']

        x = torch.zeros(1, 10, 30, 30)
        for r in range(len(grid_in)):
            for c in range(len(grid_in[0])):
                x[0, grid_in[r][c], r, c] = 1.0

        y_true = torch.zeros(1, 10, 30, 30)
        for r in range(len(grid_out)):
            for c in range(len(grid_out[0])):
                y_true[0, grid_out[r][c], r, c] = 1.0

        with torch.no_grad():
            y_pred = model(x)

        if torch.allclose(y_pred, y_true):
            print(f"Task 002 Train {i}: PASS")
        else:
            print(f"Task 002 Train {i}: FAIL")

if __name__ == "__main__":
    test_task_002()
