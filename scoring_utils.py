import math
import onnx
import numpy as np

def calculate_params(model):
    params = 0
    for init in model.graph.initializer:
        params += math.prod(init.dims) if init.dims else 1
    for node in model.graph.node:
        if node.op_type != 'Constant': continue
        for attr in node.attribute:
            if attr.name == 'value':
                params += math.prod(attr.t.dims)
            elif attr.name == 'sparse_value':
                params += math.prod(attr.sparse_tensor.values.dims)
            elif attr.name == 'value_floats':
                params += len(attr.floats)
            elif attr.name == 'value_ints':
                params += len(attr.ints)
    return params

def calculate_static_memory(model):
    try:
        onnx.checker.check_model(model)
        inferred = onnx.shape_inference.infer_shapes(model)
        graph = inferred.graph
    except:
        graph = model.graph

    total_mem = 0
    seen_names = set()
    # EXCLUDE input and output from memory calculation as per competition utils
    for item in list(graph.value_info):
        if item.name in ['input', 'output']: continue
        if item.name in seen_names: continue
        seen_names.add(item.name)
        if not item.type.HasField("tensor_type"): continue
        tensor_type = item.type.tensor_type
        if not tensor_type.HasField("shape"): continue

        num_elements = 1
        for dim in tensor_type.shape.dim:
            if not dim.HasField("dim_value") or dim.dim_value <= 0:
                num_elements = 0; break
            num_elements *= dim.dim_value

        if num_elements > 0:
            elem_type = tensor_type.elem_type
            itemsize = 4 # float32
            if elem_type == onnx.TensorProto.INT64: itemsize = 8
            total_mem += num_elements * itemsize

    return total_mem

def get_cost(model):
    params = calculate_params(model)
    mem = calculate_static_memory(model)
    return params + mem

def get_points(model):
    cost = get_cost(model)
    if cost <= 1: return 25.0
    return max(1.0, 25.0 - math.log(cost))
