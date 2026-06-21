import json, torch
import numpy as np
from solvers import *

def get_tensor(grid):
    H, W = len(grid), len(grid[0])
    x = torch.zeros(1, 10, 30, 30)
    for r in range(H):
        for c in range(W):
            color = grid[r][c]
            if 0 <= color < 10: x[0, color, r, c] = 1.0
    return x

task_id = "004"
with open(f"data/task{task_id}.json") as f: task_data = json.load(f)

for dr in range(-2, 3):
    for dc in range(-2, 3):
        m = ShiftModel(dr, dc, len(task_data['train'][0]['input']), len(task_data['train'][0]['input'][0]))
        ok = True
        for pair in task_data['train']:
            x = get_tensor(pair['input'])
            y_true = get_tensor(pair['output'])
            y_pred = m(x)
            if not torch.allclose(y_pred, y_true):
                ok = False; break
        if ok:
            print(f"Shift {dr},{dc} works for TRAIN")
            # Check test
            tok = True
            for pair in task_data['test']:
                if not torch.allclose(m(get_tensor(pair['input'])), get_tensor(pair['output'])):
                    tok = False; break
            if tok: print(f"Shift {dr},{dc} works for TEST too!")
