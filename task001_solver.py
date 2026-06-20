import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import numpy as np
from export_utils import export_to_onnx

class Task001Model(nn.Module):
    def forward(self, x):
        # 1. Get the 3x3 input block
        p3x3 = x[:, :, :3, :3] # [1, 10, 3, 3]

        # 2. Get the mask of non-zero cells in the input pattern
        # Non-zero means channel 1-9 has a 1.
        mask = torch.sum(p3x3[:, 1:10, :, :], dim=1, keepdim=True) # [1, 1, 3, 3]

        # 3. Kronecker-like expansion
        m_exp = mask.view(1, 1, 3, 1, 3, 1)
        p_exp = p3x3.view(1, 10, 1, 3, 1, 3)

        # Multiply only the foreground channels (1-9) by the mask
        # Channel 0 needs special handling.
        res_fg = m_exp * p_exp[:, 1:10, :, :, :, :] # [1, 9, 3, 3, 3, 3]

        # Channel 0 (background) in the 9x9 grid:
        # A cell (r*3+dr, c*3+dc) is background if:
        #   (mask[r,c] == 0)  OR  (p3x3[dr,dc] has color 0)
        # Wait, the Kronecker logic says:
        # if mask[r,c] == 1, block is p3x3.
        # if mask[r,c] == 0, block is all 0s.

        # Actually, if mask[r,c] == 0, it's color 0?
        # Let's check the Kronecker again.
        # np.kron(grid_in > 0, grid_in)
        # If grid_in[r,c] == 0, then (grid_in[r,c] > 0) is 0, so the block is 0*grid_in = all 0s.
        # Color 0 IS valid.

        # So if mask[r,c] == 0, the whole 3x3 block should be color 0.
        # That means channel 0 = 1, channels 1-9 = 0.

        # If mask[r,c] == 1, the block is p3x3.

        # Correct logic for channel 0:
        # res_c0 = 1 if (mask[r,c] == 0) OR (mask[r,c] == 1 AND p3x3[c0, dr, dc] == 1)
        # res_c0 = (1 - mask[r,c]) + mask[r,c] * p3x3[c0, dr, dc]

        res_c0 = (1.0 - m_exp) + m_exp * p_exp[:, 0:1, :, :, :, :]

        # Combine
        res_all = torch.cat([res_c0, res_fg], dim=1) # [1, 10, 3, 3, 3, 3]
        res_9x9 = res_all.reshape(1, 10, 9, 9)

        # 4. Pad to 30x30
        out = F.pad(res_9x9, (0, 21, 0, 21))

        return out

def create_task001_onnx(filename):
    model = Task001Model()
    model.eval()
    export_to_onnx(model, filename)

if __name__ == "__main__":
    create_task001_onnx("task001.onnx")
