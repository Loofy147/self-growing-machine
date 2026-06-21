import os, json, torch, sys
from tiny_conv_solver import train_and_verify as train_tiny
from composite_conv_solver import train_and_verify as train_comp
from export_utils import export_to_onnx

def run():
    os.makedirs("submission", exist_ok=True)
    for i in range(1, 401):
        if os.path.exists(f"submission/task{i:03d}.onnx"): continue
        try:
            with open(f"data/task{i:03d}.json") as f: task_data = json.load(f)
        except: continue

        print(f"Task {i}: Trying TinyConv...")
        sys.stdout.flush()
        m = train_tiny(task_data, i, 3)
        if m:
            export_to_onnx(m, f"submission/task{i:03d}.onnx")
            print(f"Task {i} SOLVED with TinyConv 3x3")
            sys.stdout.flush()
            continue

        # print(f"Task {i}: Trying CompositeConv...")
        # sys.stdout.flush()
        # m = train_comp(task_data, i, 16)
        # if m:
        #    export_to_onnx(m, f"submission/task{i:03d}.onnx")
        #    print(f"Task {i} SOLVED with CompositeConv")
        #    sys.stdout.flush()

if __name__ == "__main__":
    run()
