import json, torch
from solvers import HoleFillModel
def get_tensor(grid):
    H, W = len(grid), len(grid[0])
    x = torch.zeros(1, 10, 30, 30)
    for r in range(H):
        for c in range(W):
            color = grid[r][c]
            if 0 <= color < 10: x[0, color, r, c] = 1.0
    return x
with open("data/task002.json") as f: data = json.load(f)
m = HoleFillModel(4)
inp = get_tensor(data['train'][0]['input'])
expected = get_tensor(data['train'][0]['output'])
out = m(inp)
print(f"Equal: {torch.allclose(out, expected)}")
if not torch.allclose(out, expected):
    print(f"Max diff: {(out-expected).abs().max()}")
    # Check if sum matches
    print(f"Sum expected: {expected.sum()}, Sum out: {out.sum()}")
