import os
import sys
import onnx
import json

# Mocking kaggle environment paths
sys.path.append(os.path.join(os.getcwd(), 'data', 'neurogolf_utils'))
import neurogolf_utils

def validate():
    neurogolf_utils._NEUROGOLF_DIR = "data/"
    submission_dir = "submission"
    if not os.path.exists(submission_dir):
        print("Submission directory not found.")
        return

    files = sorted([f for f in os.listdir(submission_dir) if f.endswith(".onnx")])
    for f in files:
        path = os.path.join(submission_dir, f)
        print(f"Validating {f}...")
        try:
            # Check basic competition constraints
            if not neurogolf_utils.check_network(path):
                print(f"FAILED check_network: {f}")
                continue

            # Load and sanitize (this is part of their official flow)
            model = onnx.load(path)
            sanitized = neurogolf_utils.sanitize_model(model)
            if not sanitized:
                print(f"FAILED sanitize_model: {f}")
                continue

            # Note: score_network requires a profiling trace which we don't have here
            # unless we run the model. But we can check ops.
            disallowed = ["Loop", "Scan", "NonZero", "Unique", "Script", "Function"]
            for node in model.graph.node:
                if node.op_type in disallowed:
                    print(f"DISALLOWED op {node.op_type} in {f}")

            # Check static shapes
            for input in model.graph.input:
                for dim in input.type.tensor_type.shape.dim:
                    if not dim.HasField("dim_value"):
                        print(f"NON-STATIC shape in input of {f}")

            print(f"Basic validation passed for {f}")
        except Exception as e:
            print(f"Error validating {f}: {e}")

if __name__ == "__main__":
    validate()
