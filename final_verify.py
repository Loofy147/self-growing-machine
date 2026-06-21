import onnx
import sys
import os
sys.path.append("./data/neurogolf_utils")
import neurogolf_utils
import json
import numpy as np
import onnxruntime as ort

def verify_submission():
    solved = sorted([f for f in os.listdir("submission") if f.endswith(".onnx")])
    print(f"Verifying {len(solved)} models...")

    total_pts = 0
    passed_count = 0

    for f in solved:
        task_id = int(f[4:7])
        filepath = f"submission/{f}"

        # 1. Structural check
        if not neurogolf_utils.check_network(filepath):
            print(f"  {f}: FAILED structural check")
            continue

        # 2. Functional check
        try:
            with open(f"data/task{task_id:03d}.json") as file: task_data = json.load(file)

            # Use onnxruntime directly since verify_network is complex for local headless
            model = onnx.load(filepath)
            sess = ort.InferenceSession(model.SerializeToString())

            ok = True
            for subset in ['train', 'test']:
                for pair in task_data[subset]:
                    inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
                    for r, row in enumerate(pair['input']):
                        for c, col in enumerate(row): inp[0, col, r, c] = 1.0

                    out = sess.run(["output"], {"input": inp})[0]

                    expected = np.zeros((1, 10, 30, 30), dtype=np.float32)
                    for r, row in enumerate(pair['output']):
                        for c, col in enumerate(row): expected[0, col, r, c] = 1.0

                    if not np.allclose((out > 0.5).astype(float), expected):
                        ok = False; break
                if not ok: break

            if ok:
                from scoring_utils import get_points
                pts = get_points(model)
                total_pts += pts
                passed_count += 1
                # print(f"  {f}: PASSED ({pts:.2f} pts)")
            else:
                print(f"  {f}: FAILED functional verification")
        except Exception as e:
            print(f"  {f}: ERROR {e}")

    print(f"Summary: {passed_count}/{len(solved)} passed functional verification.")
    print(f"Total verified points: {total_pts:.2f}")

if __name__ == "__main__":
    verify_submission()
