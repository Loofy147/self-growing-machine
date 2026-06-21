import onnx
import sys
import os

# Add neurogolf_utils to path
sys.path.append("./data/neurogolf_utils")
import neurogolf_utils
import onnx_builders

def test_task(builder_name, *args):
    print(f"Testing {builder_name}...")
    builder = getattr(onnx_builders, builder_name)
    model = builder(*args)
    onnx.save(model, "test.onnx")

    if not neurogolf_utils.check_network("test.onnx"):
        print("check_network FAILED")
        return

    # Try sanitizing
    sanitized = neurogolf_utils.sanitize_model(onnx.load("test.onnx"))
    if sanitized is None:
        print("sanitize_model FAILED")
        return

    # We can't easily run score_network without a profiler trace
    # but we can check if onnx.checker and shape_inference pass
    try:
        onnx.checker.check_model(sanitized, full_check=True)
        inferred = onnx.shape_inference.infer_shapes(sanitized, strict_mode=True)
        print("Checker and Shape Inference PASSED")

        # Check the logic in calculate_memory that often fails
        graph = inferred.graph
        init_names = {init.name for init in graph.initializer}
        io_names = {t.name for t in list(graph.input) + list(graph.output)}
        if io_names.intersection(init_names):
            print("Overlap between IO and initializers!")

        tensor_map = {
            t.name: t for t in list(graph.input) + list(graph.value_info) + list(graph.output)
        }
        for node in graph.node:
            for output_name in node.output:
                if output_name and output_name != "output":
                    item = tensor_map.get(output_name)
                    if item is None:
                        print(f"Missing value_info for {output_name}")
                    elif not item.type.HasField("tensor_type"):
                        print(f"Not a tensor_type for {output_name}")

    except Exception as e:
        print(f"Error during validation: {e}")

if __name__ == "__main__":
    test_task("build_identity")
    test_task("build_color_map", {1: 2})
    test_task("build_flip", [2], 10, 10)
    test_task("build_kronecker", 3, 3)
