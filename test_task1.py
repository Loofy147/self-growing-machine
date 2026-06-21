import json
import torch
import numpy as np
from solvers import KroneckerModel
from search_solvers import get_tensor

with open("data/task001.json") as f:
    task_data = json.load(f)

model = KroneckerModel()
for subset in ['train', 'test']:
    for i, pair in enumerate(task_data[subset]):
        x = get_tensor(pair['input'])
        y_true = get_tensor(pair['output'])
        y_pred = model(x)
        if not torch.allclose(y_pred, y_true):
            print(f"Task 001 {subset} {i} FAIL")
            exit(1)
print("Task 001 PASS")
