import os, json
from tiny_conv_solver import train_and_verify
from export_utils import export_to_onnx

os.makedirs("submission", exist_ok=True)
for i in range(1, 401):
    if os.path.exists(f"submission/task{i:03d}.onnx"): continue
    try:
        with open(f"data/task{i:03d}.json") as f: task_data = json.load(f)
    except: continue

    # Try 3x3 then 5x5
    for ks in [3, 5]:
        m = train_and_verify(task_data, i, ks)
        if m:
            export_to_onnx(m, f"submission/task{i:03d}.onnx")
            print(f"Sweep: Task {i} SOLVED with TinyConv {ks}x{ks}")
            break
