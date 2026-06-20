import sys
import os
# Mocking kaggle environment paths
sys.path.append(os.path.join(os.getcwd(), 'data', 'neurogolf_utils'))

import onnx
import onnxruntime
import neurogolf_utils
import json

def verify_task(task_num, onnx_file):
    print(f"Verifying {onnx_file} for Task {task_num:03d}...")
    with open(f"data/task{task_num:03d}.json") as f:
        examples = json.load(f)

    network = onnx.load(onnx_file)
    # neurogolf_utils expects the path to be /kaggle/input/competitions/neurogolf-2026/
    # We need to monkeypatch it.
    neurogolf_utils._NEUROGOLF_DIR = "data/"

    neurogolf_utils.verify_network(network, task_num, examples)

if __name__ == "__main__":
    verify_task(2, "task002.onnx")
