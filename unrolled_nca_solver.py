import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json, os, sys
import numpy as np
from solvers import UnrolledNCAModel
from onnx_builders import build_unrolled_nca
import onnx

def get_tensor(grid):
    x = torch.zeros(1, 10, 30, 30)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if 0 <= color < 10: x[0, color, r, c] = 1.0
    return x

def train_and_verify(task_data, iters=10):
    train_pairs = task_data['train']
    inputs = [get_tensor(p['input']) for p in train_pairs]
    targets = [get_tensor(p['output']) for p in train_pairs]

    hidden = 16
    model = UnrolledNCAModel(
        torch.randn(hidden, 10, 3, 3) * 0.1,
        torch.zeros(hidden),
        torch.randn(10, hidden, 1, 1) * 0.1,
        torch.zeros(10),
        iters=iters
    )
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    for epoch in range(300):
        optimizer.zero_grad()
        loss = 0
        for x, y in zip(inputs, targets):
            pred = model(x)
            loss += F.mse_loss(pred, y)
        if loss.item() < 1e-6: break
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for subset in ['train', 'test']:
            for p in task_data[subset]:
                x = get_tensor(p['input'])
                y_true = get_tensor(p['output'])
                y_pred = model(x)
                if not torch.allclose((y_pred > 0.5).float(), y_true):
                    return None
    return model

def run_range(start, end):
    os.makedirs("submission", exist_ok=True)
    for i in range(start, end+1):
        if os.path.exists(f"submission/task{i:03d}.onnx"): continue
        try:
            with open(f"data/task{i:03d}.json") as f: data = json.load(f)
            print(f"Task {i}: Trying UnrolledNCA...")
            m = train_and_verify(data)
            if m:
                w1 = m.c1.weight.data.cpu().numpy()
                b1 = m.c1.bias.data.cpu().numpy()
                w2 = m.c2.weight.data.cpu().numpy()
                b2 = m.c2.bias.data.cpu().numpy()
                onnx_model = build_unrolled_nca(w1, b1, w2, b2, m.iters)
                onnx.save(onnx_model, f"submission/task{i:03d}.onnx")
                print(f"Task {i} SOLVED with UnrolledNCA")
        except Exception as e:
            print(f"Error Task {i}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        run_range(int(sys.argv[1]), int(sys.argv[2]))
    else:
        run_range(1, 400)
