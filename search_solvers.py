import os, json, torch, onnx, sys
import numpy as np
from solvers import *
from onnx_builders import *
from scoring_utils import get_points

def get_tensor(grid):
    H, W = len(grid), len(grid[0])
    x = torch.zeros(1, 10, 30, 30)
    for r in range(H):
        for c in range(W):
            color = grid[r][c]
            if 0 <= color < 10: x[0, color, r, c] = 1.0
    return x

def test_model(model, task_data):
    try:
        model.eval()
        for subset in ['train', 'test']:
            for pair in task_data[subset]:
                x = get_tensor(pair['input'])
                y_true = get_tensor(pair['output'])
                with torch.no_grad(): y_pred = model(x)
                if not torch.allclose(y_pred, y_true, atol=1e-5): return False
        return True
    except: return False

def find_color_mapping(task_data):
    mapping = {}
    for p in task_data['train']:
        in_g, out_g = np.array(p['input']), np.array(p['output'])
        if in_g.shape != out_g.shape: continue
        for i in range(10):
            mask = (in_g == i)
            if not mask.any(): continue
            u_out = np.unique(out_g[mask])
            if len(u_out) > 1: return None
            val = int(u_out[0])
            if i in mapping and mapping[i] != val: return None
            mapping[i] = val
    return mapping

def find_multi_shift(task_data, H, W):
    shift_dict = {}
    for color in range(1, 10):
        best_shift = None
        for dr in range(-15, 16):
            for dc in range(-15, 16):
                ok = True; found_any = False
                for p in task_data['train']:
                    in_g, out_g = np.array(p['input']), np.array(p['output'])
                    curH, curW = in_g.shape
                    in_mask = (in_g == color)
                    out_mask = (out_g == color)
                    if not in_mask.any() and not out_mask.any(): continue
                    found_any = True
                    shifted_in = np.zeros_like(in_g)
                    r0, r1 = max(0, dr), min(curH, curH+dr)
                    c0, c1 = max(0, dc), min(curW, curW+dc)
                    sr0, sr1 = max(0, -dr), min(curH, curH-dr)
                    sc0, sc1 = max(0, -dc), min(curW, curW-dc)
                    if r1 > r0 and c1 > c0:
                        shifted_in[r0:r1, c0:c1] = in_mask[sr0:sr1, sc0:sc1]
                    if not np.array_equal(shifted_in > 0, out_mask):
                        ok = False; break
                if found_any and ok:
                    best_shift = (dr, dc); break
            if best_shift: break
        if best_shift: shift_dict[color] = best_shift
    return shift_dict

def search_tasks(task_range):
    os.makedirs("submission", exist_ok=True)
    for i in task_range:
        if i % 50 == 0: print(f"Processing Task {i}...")
        try:
            with open(f"data/task{i:03d}.json") as f: task_data = json.load(f)
        except: continue

        H, W = len(task_data['train'][0]['input']), len(task_data['train'][0]['input'][0])
        Ho, Wo = len(task_data['train'][0]['output']), len(task_data['train'][0]['output'][0])
        candidates = []

        if test_model(IdentityModel(), task_data):
            m = build_identity(); candidates.append((m, get_points(m), "id"))

        cmap = find_color_mapping(task_data)
        if cmap and any(k != v for k, v in cmap.items()):
            if test_model(ColorMapModel(cmap, H, W), task_data):
                m = build_color_map(cmap); candidates.append((m, get_points(m), "cmap"))

        if H == Ho and W == Wo:
            for dims in [(2,), (3,), (2, 3)]:
                if test_model(SmartFlipModel(dims, H, W), task_data):
                    m = build_flip(dims, H, W); candidates.append((m, get_points(m), "flip")); break
            for k in [1, 2, 3]:
                if test_model(SmartRotateModel(k, H, W), task_data):
                    m = build_rotate(k, H, W); candidates.append((m, get_points(m), "rot")); break
            s_dict = find_multi_shift(task_data, H, W)
            if s_dict:
                if test_model(MultiColorShiftModel(s_dict, H, W), task_data):
                    m = build_multi_color_shift(s_dict, H, W); candidates.append((m, get_points(m), "shift"))
            for mode in ['v', 'h', 'vh']:
                if test_model(MirrorCompletionModel(mode, H, W), task_data):
                    m = build_mirror_completion(mode, H, W); candidates.append((m, get_points(m), "mirror")); break
            for cf in range(1, 10):
                if test_model(HoleFillModel(cf, H, W), task_data):
                    m = build_hole_fill(cf); candidates.append((m, get_points(m), "hole")); break

        if H % Ho == 0 and W % Wo == 0 and (H > Ho or W > Wo):
            r, c = H // Ho, W // Wo
            for op in ['and', 'or', 'xor']:
                if test_model(SubgridLogicModel(H, W, r, c, op), task_data):
                    m = build_subgrid_logic(H, W, r, c, op); candidates.append((m, get_points(m), "logic")); break

        if W == 2*Wo + 1 and H == Ho:
            for op in ['and', 'or', 'xor']:
                for tc in range(1, 10):
                    m_onnx = build_divider_logic(H, W, op, tc)
                    onnx.save(m_onnx, "temp.onnx")
                    import onnxruntime as ort
                    try:
                        sess = ort.InferenceSession("temp.onnx")
                        ok = True
                        for subset in ['train', 'test']:
                            for pair in task_data[subset]:
                                inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
                                for r, row in enumerate(pair['input']):
                                    for col, color in enumerate(row): inp[0, color, r, col] = 1.0
                                out = sess.run(["output"], {"input": inp})[0]
                                expected = np.zeros((1, 10, 30, 30), dtype=np.float32)
                                for r, row in enumerate(pair['output']):
                                    for col, color in enumerate(row): expected[0, color, r, col] = 1.0
                                if not np.allclose((out > 0.5).astype(float), expected): ok = False; break
                            if not ok: break
                        if ok: candidates.append((m_onnx, get_points(m_onnx), "divider")); break
                    except: continue

        if H == Wo and W == Ho:
            if test_model(TransposeModel(H, W), task_data):
                m = build_transpose(H, W); candidates.append((m, get_points(m), "trans"))

        if Ho % H == 0 and Wo % W == 0 and (Ho > H or Wo > W):
            sr, sc = Ho // H, Wo // W
            if sr == sc:
                if test_model(ScalingModel(sr, H, W), task_data):
                    m = build_scaling(sr, H, W); candidates.append((m, get_points(m), "scale"))
            if test_model(TileModel(sr, sc, H, W), task_data):
                m = build_tile(sr, sc, H, W); candidates.append((m, get_points(m), "tile"))

        for ph in [1, 2, 3, 4, 5]:
            if ph > H: continue
            for pw in [1, 2, 3, 4, 5]:
                if pw > W: continue
                if Ho % ph == 0 and Wo % pw == 0 and (ph < H or pw < W or Ho > H or Wo > W):
                    rr, rc = Ho // ph, Wo // pw
                    m_map = find_color_mapping(task_data) or {}
                    m_torch = PatternExtractionModel(0, 0, ph, pw, rr, rc, m_map, Ho, Wo)
                    if test_model(m_torch, task_data):
                        m = build_pattern_extraction(0, 0, ph, pw, rr, rc, m_map, Ho, Wo)
                        candidates.append((m, get_points(m), "pat")); break
                if candidates and candidates[-1][1] > 20: break
            if candidates and candidates[-1][1] > 20: break

        if Ho == H*H and Wo == W*W and H == 3:
            if test_model(KroneckerModel(), task_data):
                m = build_kronecker(H, W); candidates.append((m, get_points(m), "kron"))

        for color in range(1, 10):
            if test_model(ConstantFillModel(color, Ho, Wo), task_data):
                m = build_constant_fill(color, Ho, Wo); candidates.append((m, get_points(m), "fill")); break

        if candidates:
            best_cand = max(candidates, key=lambda x: x[1])
            onnx.save(best_cand[0], f"submission/task{i:03d}.onnx")
            print(f"Task {i:03d} SOLVED ({best_cand[2]})")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        search_tasks(range(int(sys.argv[1]), int(sys.argv[2]) + 1))
    else:
        search_tasks(range(1, 401))
