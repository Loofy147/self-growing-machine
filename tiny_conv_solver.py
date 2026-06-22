import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import numpy as np
import os
import onnx
from export_utils import export_to_onnx
from scoring_utils import get_points
from onnx_builders import build_color_map

def get_tensor(grid):
    x = torch.zeros(1, 10, 30, 30)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if 0 <= color < 10: x[0, color, r, c] = 1.0
    return x

class TinyConv(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(10, 10, kernel_size=kernel_size, padding=kernel_size//2)
    def forward(self, x):
        return self.conv(x)

def train_and_verify(task_data, task_id, kernel_size=3):
    train_pairs = task_data['train']
    inputs = [get_tensor(p['input']) for p in train_pairs]
    targets = [get_tensor(p['output']) for p in train_pairs]

    model = TinyConv(kernel_size)
    optimizer = optim.Adam(model.parameters(), lr=0.1) # Higher LR for faster convergence

    for epoch in range(500):
        optimizer.zero_grad()
        loss = 0
        for x, y in zip(inputs, targets):
            pred = model(x)
            loss += F.mse_loss(pred, y)
        if epoch % 100 == 0: print(f"Epoch {epoch}, loss {loss.item()}");
        if loss.item() < 1e-7: break
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

def run_conv_search(task_range):
    os.makedirs("submission", exist_ok=True)
    for i in task_range:
        if os.path.exists(f"submission/task{i:03d}.onnx"): continue
        try:
            with open(f"data/task{i:03d}.json") as f: task_data = json.load(f)
        except: continue

        print(f"Trying TinyConv for Task {i}...")
        for ks in [1, 3, 5]:
            model = train_and_verify(task_data, i, ks)
            if model:
                filename = f"submission/task{i:03d}.onnx"
                if ks == 1:
                    mapping = {}
                    w = model.conv.weight.data.squeeze().argmax(dim=0)
                    for src in range(10): mapping[src] = int(w[src].item())
                    onnx.save(build_color_map(mapping), filename)
                    print(f"Task {i:03d} SOLVED with TinyConv (1x1 -> Optimized ColorMap)")
                else:
                    export_to_onnx(model, filename)
                    print(f"Task {i:03d} SOLVED with TinyConv {ks}x{ks}")
                break

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        run_conv_search(range(int(sys.argv[1]), int(sys.argv[2]) + 1))
    else:
        run_conv_search(range(1, 401))
