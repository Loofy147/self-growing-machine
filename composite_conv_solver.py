import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import os
from export_utils import export_to_onnx

def get_tensor(grid):
    x = torch.zeros(1, 10, 30, 30)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if 0 <= color < 10: x[0, color, r, c] = 1.0
    return x

class CompositeConv(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(10, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden, 10, kernel_size=3, padding=1)
        )
    def forward(self, x):
        return self.net(x)

def train_and_verify(task_data, task_id, hidden=16):
    train_pairs = task_data['train']
    inputs = [get_tensor(p['input']) for p in train_pairs]
    targets = [get_tensor(p['output']) for p in train_pairs]

    model = CompositeConv(hidden)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1000):
        optimizer.zero_grad()
        loss = 0
        for x, y in zip(inputs, targets):
            pred = model(x)
            loss += F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        if loss.item() < 1e-6: break

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

def run_composite_search(task_range):
    os.makedirs("submission", exist_ok=True)
    for i in task_range:
        if os.path.exists(f"submission/task{i:03d}.onnx"): continue
        try:
            with open(f"data/task{i:03d}.json") as f: task_data = json.load(f)
        except: continue

        print(f"Trying CompositeConv for Task {i}...")
        model = train_and_verify(task_data, i)
        if model:
            export_to_onnx(model, f"submission/task{i:03d}.onnx")
            print(f"Task {i:03d} SOLVED with CompositeConv")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        run_composite_search(range(int(sys.argv[1]), int(sys.argv[2]) + 1))
    else:
        run_composite_search(range(1, 400))
