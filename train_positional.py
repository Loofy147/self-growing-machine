import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json, os, sys
import numpy as np
from solvers import PositionalCNN
from export_utils import export_to_onnx
from onnx_builders import build_positional_cnn

def get_tensor(grid):
    x = torch.zeros(1, 10, 30, 30)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if 0 <= color < 10: x[0, color, r, c] = 1.0
    return x

def train_and_verify(task_data, task_id, hidden=16):
    train_pairs = task_data['train']
    inputs = [get_tensor(p['input']) for p in train_pairs]
    targets = [get_tensor(p['output']) for p in train_pairs]

    H, W = len(task_data['train'][0]['input']), len(task_data['train'][0]['input'][0])
    model = PositionalCNN(H, W, hidden)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1000):
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
    for i in range(start, end+1):
        if os.path.exists(f"submission/task{i:03d}.onnx"): continue
        try:
            with open(f"data/task{i:03d}.json") as f: data = json.load(f)
            print(f"Task {i}: Trying PositionalCNN...")
            m = train_and_verify(data, i)
            if m:
                # Extract weights for build_positional_cnn
                w1 = m.net[0].weight.data.cpu().numpy()
                b1 = m.net[0].bias.data.cpu().numpy()
                w2 = m.net[2].weight.data.cpu().numpy()
                b2 = m.net[2].bias.data.cpu().numpy()
                H, W = len(data['train'][0]['input']), len(data['train'][0]['input'][0])
                onnx_model = build_positional_cnn(w1, b1, w2, b2, H, W)
                import onnx
                onnx.save(onnx_model, f"submission/task{i:03d}.onnx")
                print(f"Task {i} SOLVED with PositionalCNN")
        except: pass

if __name__ == "__main__":
    run_range(int(sys.argv[1]), int(sys.argv[2]))
