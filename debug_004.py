import json, torch
import numpy as np
from solvers import ShiftModel

def get_tensor(grid):
    H, W = len(grid), len(grid[0])
    x = torch.zeros(1, 10, 30, 30)
    for r in range(H):
        for c in range(W):
            color = grid[r][c]
            if 0 <= color < 10: x[0, color, r, c] = 1.0
    return x

with open("data/task004.json") as f: data = json.load(f)
H, W = len(data['train'][0]['input']), len(data['train'][0]['input'][0])
m = ShiftModel(0, 1, H, W)
m.eval()

for i, pair in enumerate(data['train']):
    x = get_tensor(pair['input'])
    y_true = get_tensor(pair['output'])
    y_pred = m(x)
    if not torch.allclose(y_pred, y_true):
        print(f"Train {i} FAIL")
        # Find first mismatch
        diff = (y_pred - y_true).abs() > 0.1
        idx = diff.nonzero()
        if len(idx) > 0:
            r, c = idx[0, 2].item(), idx[0, 3].item()
            print(f"Mismatch at {r},{c}: pred_ch={y_pred[0,:,r,c].argmax()}, true_ch={y_true[0,:,r,c].argmax()}")
    else:
        print(f"Train {i} PASS")
