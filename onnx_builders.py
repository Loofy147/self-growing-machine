import onnx
from onnx import helper, TensorProto
import numpy as np

def make_base_model(nodes, initializers):
    unique_inits = {}
    for init in initializers: unique_inits[init.name] = init
    graph = helper.make_graph(
        nodes, "g",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        list(unique_inits.values())
    )
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 10)])
    model = onnx.shape_inference.infer_shapes(model)
    return model

def build_identity():
    node = helper.make_node("Identity", ["input"], ["output"])
    return make_base_model([node], [])

def build_color_map(mapping):
    weight_data = np.zeros((10, 10, 1, 1), dtype=np.float32)
    for i in range(10): weight_data[mapping.get(i, i), i, 0, 0] = 1.0
    W = helper.make_tensor("W_cmap_fixed", TensorProto.FLOAT, [10, 10, 1, 1], weight_data.flatten())
    node = helper.make_node("Conv", ["input", "W_cmap_fixed"], ["output"], kernel_shape=[1, 1])
    return make_base_model([node], [W])

def build_flip(dims, H, W):
    nodes, initializers = [], []
    nodes.append(helper.make_node("Slice", ["input", "s1", "e1", "a1"], ["content"]))
    initializers.extend([helper.make_tensor("s1", TensorProto.INT64, [2], [0, 0]),
                        helper.make_tensor("e1", TensorProto.INT64, [2], [H, W]),
                        helper.make_tensor("a1", TensorProto.INT64, [2], [2, 3])])
    curr = "content"
    if 2 in dims:
        nodes.append(helper.make_node("Gather", [curr, "row_idx"], ["row_f"], axis=2))
        initializers.append(helper.make_tensor("row_idx", TensorProto.INT64, [H], list(range(H-1, -1, -1))))
        curr = "row_f"
    if 3 in dims:
        nodes.append(helper.make_node("Gather", [curr, "col_idx"], ["col_f"], axis=3))
        initializers.append(helper.make_tensor("col_idx", TensorProto.INT64, [W], list(range(W-1, -1, -1))))
        curr = "col_f"
    nodes.append(helper.make_node("Pad", [curr], ["output"], mode="constant", pads=[0, 0, 0, 0, 0, 0, 30-H, 30-W]))
    return make_base_model(nodes, initializers)

def build_rotate(k, H, W):
    nodes, initializers = [], []
    nodes.append(helper.make_node("Slice", ["input", "s1", "e1", "a1"], ["content"]))
    initializers.extend([helper.make_tensor("s1", TensorProto.INT64, [2], [0, 0]),
                        helper.make_tensor("e1", TensorProto.INT64, [2], [H, W]),
                        helper.make_tensor("a1", TensorProto.INT64, [2], [2, 3])])
    curr, ch, cw = "content", H, W
    k = k % 4
    if k == 0:
        nodes.append(helper.make_node("Identity", [curr], ["rot_final"]))
    elif k == 1:
        nodes.append(helper.make_node("Transpose", [curr], ["t1"], perm=[0, 1, 3, 2]))
        nodes.append(helper.make_node("Gather", ["t1", "idx1"], ["rot_final"], axis=2))
        initializers.append(helper.make_tensor("idx1", TensorProto.INT64, [W], list(range(W-1, -1, -1))))
        ch, cw = W, H
    elif k == 2:
        nodes.append(helper.make_node("Gather", [curr, "idx2r"], ["f2r"], axis=2))
        initializers.append(helper.make_tensor("idx2r", TensorProto.INT64, [H], list(range(H-1, -1, -1))))
        nodes.append(helper.make_node("Gather", ["f2r", "idx2c"], ["rot_final"], axis=3))
        initializers.append(helper.make_tensor("idx2c", TensorProto.INT64, [W], list(range(W-1, -1, -1))))
    elif k == 3:
        nodes.append(helper.make_node("Transpose", [curr], ["t3"], perm=[0, 1, 3, 2]))
        nodes.append(helper.make_node("Gather", ["t3", "idx3"], ["rot_final"], axis=3))
        initializers.append(helper.make_tensor("idx3", TensorProto.INT64, [H], list(range(H-1, -1, -1))))
        ch, cw = W, H
    nodes.append(helper.make_node("Pad", ["rot_final"], ["output"], mode="constant", pads=[0, 0, 0, 0, 0, 0, 30-ch, 30-cw]))
    return make_base_model(nodes, initializers)

def build_transpose(H, W):
    nodes, initializers = [], []
    nodes.append(helper.make_node("Slice", ["input", "s1", "e1", "a1"], ["content"]))
    initializers.extend([helper.make_tensor("s1", TensorProto.INT64, [2], [0, 0]),
                        helper.make_tensor("e1", TensorProto.INT64, [2], [H, W]),
                        helper.make_tensor("a1", TensorProto.INT64, [2], [2, 3])])
    nodes.append(helper.make_node("Transpose", ["content"], ["trans"], perm=[0, 1, 3, 2]))
    nodes.append(helper.make_node("Pad", ["trans"], ["output"], mode="constant", pads=[0, 0, 0, 0, 0, 0, 30-W, 30-H]))
    return make_base_model(nodes, initializers)

def build_scaling(scale, H, W):
    nodes, initializers = [], []
    nodes.append(helper.make_node("Slice", ["input", "s1", "e1", "a1"], ["content"]))
    initializers.extend([helper.make_tensor("s1", TensorProto.INT64, [2], [0, 0]),
                        helper.make_tensor("e1", TensorProto.INT64, [2], [H, W]),
                        helper.make_tensor("a1", TensorProto.INT64, [2], [2, 3])])
    s = int(scale)
    w_data = np.zeros((10, 1, s, s), dtype=np.float32)
    for i in range(10): w_data[i, 0, :, :] = 1.0
    W_init = helper.make_tensor("W_s_scale", TensorProto.FLOAT, [10, 1, s, s], w_data.flatten())
    nodes.append(helper.make_node("ConvTranspose", ["content", "W_s_scale"], ["scaled"], strides=[s, s], kernel_shape=[s, s], group=10))
    nodes.append(helper.make_node("Pad", ["scaled"], ["output"], mode="constant", pads=[0, 0, 0, 0, 0, 0, 30-H*s, 30-W*s]))
    return make_base_model(nodes, [W_init] + initializers)

def build_tile(rr, rc, H, W):
    nodes, initializers = [], []
    nodes.append(helper.make_node("Slice", ["input", "s1", "e1", "a1"], ["content"]))
    initializers.extend([helper.make_tensor("s1", TensorProto.INT64, [2], [0, 0]),
                        helper.make_tensor("e1", TensorProto.INT64, [2], [H, W]),
                        helper.make_tensor("a1", TensorProto.INT64, [2], [2, 3])])
    nodes.append(helper.make_node("Tile", ["content", "reps"], ["tiled"]))
    initializers.append(helper.make_tensor("reps", TensorProto.INT64, [4], [1, 1, rr, rc]))
    ph, pw = min(30, H*rr), min(30, W*rc)
    nodes.append(helper.make_node("Slice", ["tiled", "s3", "e3", "a3"], ["cropped"]))
    initializers.extend([helper.make_tensor("s3", TensorProto.INT64, [2], [0, 0]),
                        helper.make_tensor("e3", TensorProto.INT64, [2], [ph, pw]),
                        helper.make_tensor("a3", TensorProto.INT64, [2], [2, 3])])
    nodes.append(helper.make_node("Pad", ["cropped"], ["output"], mode="constant", pads=[0, 0, 0, 0, 0, 0, 30-ph, 30-pw]))
    return make_base_model(nodes, initializers)

def build_multi_color_shift(shift_dict, H, W):
    nodes, initializers = [], []
    channel_parts = []
    for color in range(1, 10):
        dr, dc = shift_dict.get(color, (0, 0))
        nodes.append(helper.make_node("Slice", ["input", f"sc{color}", f"ec{color}", f"ac{color}"], [f"ch{color}"]))
        initializers.extend([helper.make_tensor(f"sc{color}", TensorProto.INT64, [1], [color]),
                            helper.make_tensor(f"ec{color}", TensorProto.INT64, [1], [color+1]),
                            helper.make_tensor(f"ac{color}", TensorProto.INT64, [1], [1])])
        ph, pw = [max(0, dr), max(0, -dr)], [max(0, dc), max(0, -dc)]
        nodes.append(helper.make_node("Pad", [f"ch{color}"], [f"p{color}"], mode="constant", pads=[0, 0, ph[0], pw[0], 0, 0, ph[1], pw[1]]))
        nodes.append(helper.make_node("Slice", [f"p{color}", f"s3_{color}", f"e3_{color}", f"a3_{color}"], [f"sh{color}"]))
        initializers.extend([helper.make_tensor(f"s3_{color}", TensorProto.INT64, [2], [max(0, -dr), max(0, -dc)]),
                            helper.make_tensor(f"e3_{color}", TensorProto.INT64, [2], [H+max(0, -dr), W+max(0, -dc)]),
                            helper.make_tensor(f"a3_{color}", TensorProto.INT64, [2], [2, 3])])
        channel_parts.append(f"sh{color}")
    nodes.append(helper.make_node("Concat", channel_parts, ["fg_all"], axis=1))
    nodes.append(helper.make_node("ReduceSum", ["fg_all"], ["fg_mask"], axes=[1], keepdims=1))
    nodes.append(helper.make_node("Sub", ["one", "fg_mask"], ["bg"]))
    initializers.append(helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0]))
    nodes.append(helper.make_node("Concat", ["bg", "fg_all"], ["final"], axis=1))
    nodes.append(helper.make_node("Pad", ["final"], ["output"], mode="constant", pads=[0, 0, 0, 0, 0, 0, 30-H, 30-W]))
    return make_base_model(nodes, initializers)

def build_mirror_completion(mode, H, W):
    nodes, initializers = [], []
    nodes.append(helper.make_node("Slice", ["input", "s1", "e1", "a1"], ["content"]))
    initializers.extend([helper.make_tensor("s1", TensorProto.INT64, [2], [0, 0]),
                        helper.make_tensor("e1", TensorProto.INT64, [2], [H, W]),
                        helper.make_tensor("a1", TensorProto.INT64, [2], [2, 3])])
    curr = "content"
    if 'v' in mode:
        nodes.append(helper.make_node("Gather", [curr, "rev_v"], ["flip_v"], axis=2))
        initializers.append(helper.make_tensor("rev_v", TensorProto.INT64, [H], list(range(H-1, -1, -1))))
        nodes.append(helper.make_node("Max", [curr, "flip_v"], ["mv"]))
        curr = "mv"
    if 'h' in mode:
        nodes.append(helper.make_node("Gather", [curr, "rev_h"], ["flip_h"], axis=3))
        initializers.append(helper.make_tensor("rev_h", TensorProto.INT64, [W], list(range(W-1, -1, -1))))
        nodes.append(helper.make_node("Max", [curr, "flip_h"], ["mh"]))
        curr = "mh"
    nodes.append(helper.make_node("Slice", [curr, "sf", "ef", "af"], ["fg"]))
    initializers.extend([helper.make_tensor("sf", TensorProto.INT64, [1], [1]),
                        helper.make_tensor("ef", TensorProto.INT64, [1], [10]),
                        helper.make_tensor("af", TensorProto.INT64, [1], [1])])
    nodes.append(helper.make_node("ReduceSum", ["fg"], ["fg_sum"], axes=[1], keepdims=1))
    nodes.append(helper.make_node("Sub", ["one", "fg_sum"], ["bg"]))
    initializers.append(helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0]))
    nodes.append(helper.make_node("Concat", ["bg", "fg"], ["res"], axis=1))
    nodes.append(helper.make_node("Pad", ["res"], ["output"], mode="constant", pads=[0, 0, 0, 0, 0, 0, 30-H, 30-W]))
    return make_base_model(nodes, initializers)

def build_subgrid_logic(H, W, rows, cols, op):
    nodes, initializers = [], []
    sh, sw = H // rows, W // cols
    nodes.append(helper.make_node("Slice", ["input", "s1", "e1", "a1"], ["content"]))
    initializers.extend([helper.make_tensor("s1", TensorProto.INT64, [2], [0, 0]),
                        helper.make_tensor("e1", TensorProto.INT64, [2], [H, W]),
                        helper.make_tensor("a1", TensorProto.INT64, [2], [2, 3])])
    nodes.append(helper.make_node("Reshape", ["content", "sh1"], ["r1"]))
    initializers.append(helper.make_tensor("sh1", TensorProto.INT64, [6], [1, 10, rows, sh, cols, sw]))
    nodes.append(helper.make_node("Transpose", ["r1"], ["t1"], perm=[0, 1, 2, 4, 3, 5]))
    nodes.append(helper.make_node("Reshape", ["t1"], ["subgrids"]))
    initializers.append(helper.make_tensor("sh2", TensorProto.INT64, [5], [1, 10, rows*cols, sh, sw]))
    nodes.append(helper.make_node("Slice", ["subgrids", "ss0", "ee0", "aa0"], ["sub0"]))
    initializers.extend([helper.make_tensor("ss0", TensorProto.INT64, [1], [0]),
                        helper.make_tensor("ee0", TensorProto.INT64, [1], [1]),
                        helper.make_tensor("aa0", TensorProto.INT64, [1], [2])])
    nodes.append(helper.make_node("Squeeze", ["sub0"], ["res0"], axes=[2]))
    curr_res = "res0"
    for i in range(1, rows*cols):
        nodes.append(helper.make_node("Slice", ["subgrids", f"ss{i}", f"ee{i}", f"aa{i}"], [f"sub{i}"]))
        initializers.extend([helper.make_tensor(f"ss{i}", TensorProto.INT64, [1], [i]),
                            helper.make_tensor(f"ee{i}", TensorProto.INT64, [1], [i+1]),
                            helper.make_tensor(f"aa{i}", TensorProto.INT64, [1], [2])])
        nodes.append(helper.make_node("Squeeze", [f"sub{i}"], [f"sq{i}"], axes=[2]))
        if op == 'and': nodes.append(helper.make_node("Mul", [curr_res, f"sq{i}"], [f"res{i}"]))
        elif op == 'or': nodes.append(helper.make_node("Max", [curr_res, f"sq{i}"], [f"res{i}"]))
        else:
            nodes.append(helper.make_node("Add", [curr_res, f"sq{i}"], [f"sum{i}"]))
            nodes.append(helper.make_node("Mod", [f"sum{i}", "two"], [f"res{i}"], fmod=1))
            initializers.append(helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0]))
        curr_res = f"res{i}"
    nodes.append(helper.make_node("Slice", [curr_res, "sf", "ef", "af"], ["fg_only"]))
    initializers.extend([helper.make_tensor("sf", TensorProto.INT64, [1], [1]),
                        helper.make_tensor("ef", TensorProto.INT64, [1], [10]),
                        helper.make_tensor("af", TensorProto.INT64, [1], [1])])
    nodes.append(helper.make_node("ReduceSum", ["fg_only"], ["fg_sum"], axes=[1], keepdims=1))
    nodes.append(helper.make_node("Sub", ["one", "fg_sum"], ["bg"]))
    initializers.append(helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0]))
    nodes.append(helper.make_node("Concat", ["bg", "fg_only"], ["final"], axis=1))
    nodes.append(helper.make_node("Pad", ["final"], ["output"], mode="constant", pads=[0, 0, 0, 0, 0, 0, 30-sh, 30-sw]))
    return make_base_model(nodes, initializers)

def build_divider_logic(H, W, op, tc):
    nodes, initializers = [], []
    Wo = (W - 1) // 2
    nodes.append(helper.make_node("Slice", ["input", "sl", "el", "al"], ["left"]))
    initializers.extend([helper.make_tensor("sl", TensorProto.INT64, [2], [0, 0]),
                        helper.make_tensor("el", TensorProto.INT64, [2], [H, Wo]),
                        helper.make_tensor("al", TensorProto.INT64, [2], [2, 3])])
    nodes.append(helper.make_node("Slice", ["input", "sr", "er", "ar"], ["right"]))
    initializers.extend([helper.make_tensor("sr", TensorProto.INT64, [2], [0, Wo+1]),
                        helper.make_tensor("er", TensorProto.INT64, [2], [H, 2*Wo+1]),
                        helper.make_tensor("ar", TensorProto.INT64, [2], [2, 3])])
    for side in ["left", "right"]:
        nodes.append(helper.make_node("Slice", [side, "sf", "ef", "af"], [f"{side}_fg"]))
        initializers.extend([helper.make_tensor("sf", TensorProto.INT64, [1], [1]),
                            helper.make_tensor("ef", TensorProto.INT64, [1], [10]),
                            helper.make_tensor("af", TensorProto.INT64, [1], [1])])
        nodes.append(helper.make_node("ReduceSum", [f"{side}_fg"], [f"{side}_mask"], axes=[1], keepdims=1))
    if op == 'and': nodes.append(helper.make_node("Mul", ["left_mask", "right_mask"], ["res_mask"]))
    elif op == 'or': nodes.append(helper.make_node("Max", ["left_mask", "right_mask"], ["res_mask"]))
    else:
        nodes.append(helper.make_node("Add", ["left_mask", "right_mask"], ["sum_mask"]))
        nodes.append(helper.make_node("Mod", ["sum_mask", "two"], ["res_mask"], fmod=1))
        initializers.append(helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0]))
    w_data = np.zeros((10, 1, 1, 1), dtype=np.float32); w_data[tc, 0, 0, 0] = 1.0; w_data[0, 0, 0, 0] = -1.0
    W_tc = helper.make_tensor("W_tc", TensorProto.FLOAT, [10, 1, 1, 1], w_data.flatten())
    nodes.append(helper.make_node("Conv", ["res_mask", "W_tc"], ["out_fg"], kernel_shape=[1, 1]))
    nodes.append(helper.make_node("Add", ["out_fg", "bg_one"], ["out_final"]))
    bg_data = np.zeros((1, 10, 1, 1), dtype=np.float32); bg_data[0, 0, 0, 0] = 1.0
    BG = helper.make_tensor("bg_one", TensorProto.FLOAT, [1, 10, 1, 1], bg_data.flatten())
    nodes.append(helper.make_node("Pad", ["out_final"], ["output"], mode="constant", pads=[0, 0, 0, 0, 0, 0, 30-H, 30-Wo]))
    return make_base_model(nodes, [W_tc, BG] + initializers)

def build_kronecker(H, W):
    nodes, initializers = [], []
    nodes.append(helper.make_node("Slice", ["input", "s1", "e1", "a1"], ["content"]))
    initializers.extend([helper.make_tensor("s1", TensorProto.INT64, [2], [0, 0]),
                        helper.make_tensor("e1", TensorProto.INT64, [2], [H, W]),
                        helper.make_tensor("a1", TensorProto.INT64, [2], [2, 3])])
    nodes.append(helper.make_node("Slice", ["content", "sm", "em", "am"], ["fg_ch"]))
    initializers.extend([helper.make_tensor("sm", TensorProto.INT64, [1], [1]),
                        helper.make_tensor("em", TensorProto.INT64, [1], [10]),
                        helper.make_tensor("am", TensorProto.INT64, [1], [1])])
    nodes.append(helper.make_node("ReduceSum", ["fg_ch"], ["mask"], axes=[1], keepdims=1))
    nodes.append(helper.make_node("Reshape", ["mask", "sh_m"], ["mask_exp"]))
    initializers.append(helper.make_tensor("sh_m", TensorProto.INT64, [6], [1, 1, H, 1, W, 1]))
    nodes.append(helper.make_node("Reshape", ["content", "sh_c"], ["content_exp"]))
    initializers.append(helper.make_tensor("sh_c", TensorProto.INT64, [6], [1, 10, 1, H, 1, W]))
    nodes.append(helper.make_node("Slice", ["content_exp", "sf", "ef", "af"], ["cf_exp"]))
    initializers.extend([helper.make_tensor("sf", TensorProto.INT64, [1], [1]),
                        helper.make_tensor("ef", TensorProto.INT64, [1], [10]),
                        helper.make_tensor("af", TensorProto.INT64, [1], [1])])
    nodes.append(helper.make_node("Mul", ["mask_exp", "cf_exp"], ["res_f_exp"]))
    nodes.append(helper.make_node("Sub", ["one", "mask_exp"], ["inv_m"]))
    initializers.append(helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0]))
    nodes.append(helper.make_node("Slice", ["content_exp", "s0", "e0", "a0"], ["c0_exp"]))
    initializers.extend([helper.make_tensor("s0", TensorProto.INT64, [1], [0]),
                        helper.make_tensor("e0", TensorProto.INT64, [1], [1]),
                        helper.make_tensor("a0", TensorProto.INT64, [1], [1])])
    nodes.append(helper.make_node("Mul", ["mask_exp", "c0_exp"], ["mul0"]))
    nodes.append(helper.make_node("Add", ["inv_m", "mul0"], ["res0_exp"]))
    nodes.append(helper.make_node("Concat", ["res0_exp", "res_f_exp"], ["res_exp"], axis=1))
    nodes.append(helper.make_node("Reshape", ["res_exp", "shr"], ["res_f"]))
    initializers.append(helper.make_tensor("shr", TensorProto.INT64, [4], [1, 10, H*H, W*W]))
    nodes.append(helper.make_node("Pad", ["res_f"], ["output"], mode="constant", pads=[0, 0, 0, 0, 0, 0, 30-H*H, 30-W*W]))
    return make_base_model(nodes, initializers)

def build_hole_fill(cf):
    nodes, initializers = [], []
    nodes.append(helper.make_node("Slice", ["input", "sf", "ef", "af"], ["fg"]))
    initializers.extend([helper.make_tensor("sf", TensorProto.INT64, [1], [1]),
                        helper.make_tensor("ef", TensorProto.INT64, [1], [10]),
                        helper.make_tensor("af", TensorProto.INT64, [1], [1])])
    nodes.append(helper.make_node("ReduceSum", ["fg"], ["mask"], axes=[1], keepdims=1))
    nodes.append(helper.make_node("Sub", ["one", "mask"], ["inv_mask"]))
    initializers.append(helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0]))
    edge_data = np.zeros((1, 1, 30, 30), dtype=np.float32); edge_data[0, 0, 0, :] = 1.0; edge_data[0, 0, -1, :] = 1.0; edge_data[0, 0, :, 0] = 1.0; edge_data[0, 0, :, -1] = 1.0
    EDGE = helper.make_tensor("edge_mask", TensorProto.FLOAT, [1, 1, 30, 30], edge_data.flatten())
    nodes.append(helper.make_node("Constant", [], ["edge"], value=EDGE))
    nodes.append(helper.make_node("Mul", ["edge", "inv_mask"], ["outside_0"]))
    k_data = np.array([[[[0,1,0],[1,1,1],[0,1,0]]]], dtype=np.float32)
    K = helper.make_tensor("flood_k", TensorProto.FLOAT, [1, 1, 3, 3], k_data.flatten())
    initializers.extend([K, helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0])])
    curr = "outside_0"
    for j in range(1, 31):
        nodes.append(helper.make_node("Conv", [curr, "flood_k"], [f"d{j}"], kernel_shape=[3, 3], padding=[1, 1, 1, 1]))
        nodes.append(helper.make_node("Clip", [f"d{j}", "zero", "one"], [f"c{j}"]))
        nodes.append(helper.make_node("Mul", [f"c{j}", "inv_mask"], [f"outside_{j}"]))
        curr = f"outside_{j}"
    nodes.append(helper.make_node("Sub", ["one", curr], ["not_outside"]))
    nodes.append(helper.make_node("Mul", ["inv_mask", "not_outside"], ["enclosed"]))
    nodes.append(helper.make_node("Slice", ["input", "scf", "ecf", "acf"], ["ch_cf"]))
    initializers.extend([helper.make_tensor("scf", TensorProto.INT64, [1], [cf]),
                        helper.make_tensor("ecf", TensorProto.INT64, [1], [cf+1]),
                        helper.make_tensor("acf", TensorProto.INT64, [1], [1])])
    nodes.append(helper.make_node("Add", ["ch_cf", "enclosed"], ["new_cf"]))
    nodes.append(helper.make_node("Clip", ["new_cf", "zero", "one"], ["final_cf"]))
    parts = []
    for c in range(1, 10):
        if c == cf: parts.append("final_cf")
        else:
            nodes.append(helper.make_node("Slice", ["input", f"s{c}", f"e{c}", f"a{c}"], [f"ch_{c}"]))
            initializers.extend([helper.make_tensor(f"s{c}", TensorProto.INT64, [1], [c]),
                                helper.make_tensor(f"e{c}", TensorProto.INT64, [1], [c+1]),
                                helper.make_tensor(f"a{c}", TensorProto.INT64, [1], [1])])
            parts.append(f"ch_{c}")
    nodes.append(helper.make_node("Concat", parts, ["fg_new"], axis=1))
    nodes.append(helper.make_node("ReduceSum", ["fg_new"], ["fg_new_sum"], axes=[1], keepdims=1))
    nodes.append(helper.make_node("Sub", ["one", "fg_new_sum"], ["bg_new"]))
    nodes.append(helper.make_node("Concat", ["bg_new", "fg_new"], ["output"], axis=1))
    return make_base_model(nodes, initializers)

def build_pattern_extraction(pr, pc, ph, pw, rr, rc, mapping, Ho, Wo):
    nodes, initializers = [], []
    nodes.append(helper.make_node("Slice", ["input", "sp", "ep", "ap"], ["patch"]))
    initializers.extend([helper.make_tensor("sp", TensorProto.INT64, [2], [pr, pc]),
                        helper.make_tensor("ep", TensorProto.INT64, [2], [pr+ph, pc+pw]),
                        helper.make_tensor("ap", TensorProto.INT64, [2], [2, 3])])
    curr = "patch"
    if mapping and any(k != v for k, v in mapping.items()):
        wd = np.zeros((10, 10, 1, 1), dtype=np.float32)
        for i in range(10): wd[mapping.get(i, i), i, 0, 0] = 1.0
        W = helper.make_tensor("W_cmap_pat", TensorProto.FLOAT, [10, 10, 1, 1], wd.flatten())
        nodes.append(helper.make_node("Conv", ["patch", "W_cmap_pat"], ["mapped"], kernel_shape=[1, 1]))
        initializers.append(W); curr = "mapped"
    nodes.append(helper.make_node("Tile", [curr, "reps"], ["tiled"]))
    initializers.append(helper.make_tensor("reps", TensorProto.INT64, [4], [1, 1, rr, rc]))
    nodes.append(helper.make_node("Slice", ["tiled", "sc", "ec", "ac"], ["cropped"]))
    initializers.extend([helper.make_tensor("sc", TensorProto.INT64, [2], [0, 0]),
                        helper.make_tensor("ec", TensorProto.INT64, [2], [Ho, Wo]),
                        helper.make_tensor("ac", TensorProto.INT64, [2], [2, 3])])
    nodes.append(helper.make_node("Pad", ["cropped"], ["output"], mode="constant", pads=[0, 0, 0, 0, 0, 0, 30-Ho, 30-Wo]))
    return make_base_model(nodes, initializers)

def build_constant_fill(color, H, W):
    nodes = []
    data = np.zeros((1, 10, H, W), dtype=np.float32); data[0, color, :, :] = 1.0
    C = helper.make_tensor("const_data", TensorProto.FLOAT, [1, 10, H, W], data.flatten())
    nodes.append(helper.make_node("Constant", [], ["c_out"], value=C))
    nodes.append(helper.make_node("Pad", ["c_out"], ["output"], mode="constant", pads=[0, 0, 0, 0, 0, 0, 30-H, 30-W]))
    return make_base_model(nodes, [])

def build_recolor_foreground(color):
    nodes, initializers = [], []
    nodes.append(helper.make_node("Slice", ["input", "sf", "ef", "af"], ["fg_only"]))
    initializers.extend([helper.make_tensor("sf", TensorProto.INT64, [1], [1]),
                        helper.make_tensor("ef", TensorProto.INT64, [1], [10]),
                        helper.make_tensor("af", TensorProto.INT64, [1], [1])])
    nodes.append(helper.make_node("ReduceSum", ["fg_only"], ["fg_mask"], axes=[1], keepdims=1))
    wd = np.zeros((10, 1, 1, 1), dtype=np.float32); wd[color, 0, 0, 0] = 1.0; wd[0, 0, 0, 0] = -1.0
    W = helper.make_tensor("W_recolor", TensorProto.FLOAT, [10, 1, 1, 1], wd.flatten())
    nodes.append(helper.make_node("Conv", ["fg_mask", "W_recolor"], ["out_fg"], kernel_shape=[1, 1]))
    nodes.append(helper.make_node("Add", ["out_fg", "bg_one"], ["output"]))
    bgd = np.zeros((1, 10, 1, 1), dtype=np.float32); bgd[0, 0, 0, 0] = 1.0
    BG = helper.make_tensor("bg_one", TensorProto.FLOAT, [1, 10, 1, 1], bgd.flatten())
    return make_base_model(nodes, [W, BG] + initializers)
