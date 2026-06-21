import onnxruntime as ort
import numpy as np
import json
import sys
sys.path.append("./data/neurogolf_utils")
import neurogolf_utils

with open("data/task006.json") as f: data = json.load(f)
sess = ort.InferenceSession("submission/task006.onnx")

for pair in data['train'] + data['test']:
    inp = neurogolf_utils.convert_to_numpy(pair)['input']
    out = sess.run(["output"], {"input": inp})[0]
    expected = neurogolf_utils.convert_to_numpy(pair)['output']
    if not np.allclose((out > 0.5).astype(float), expected):
        print("FAIL")
        sys.exit(1)
print("PASS")
