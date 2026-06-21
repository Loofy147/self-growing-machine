import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class IdentityModel(nn.Module):
    def forward(self, x): return x

class ColorMapModel(nn.Module):
    def __init__(self, mapping, H, W):
        super().__init__()
        self.H, self.W = H, W
        self.register_buffer("weight", torch.zeros(10, 10, 1, 1))
        for i in range(10):
            target = mapping.get(i, i)
            self.weight[target, i, 0, 0] = 1.0
    def forward(self, x):
        out = F.conv2d(x, self.weight)
        mask = torch.zeros_like(out)
        mask[:, :, :self.H, :self.W] = 1.0
        return out * mask

class SmartFlipModel(nn.Module):
    def __init__(self, dims, H, W):
        super().__init__()
        self.dims, self.H, self.W = dims, H, W
    def forward(self, x):
        content = x[:, :, :self.H, :self.W]
        flipped = torch.flip(content, self.dims)
        return F.pad(flipped, (0, 30-self.W, 0, 30-self.H))

class SmartRotateModel(nn.Module):
    def __init__(self, k, H, W):
        super().__init__()
        self.k, self.H, self.W = k, H, W
    def forward(self, x):
        content = x[:, :, :self.H, :self.W]
        rotated = torch.rot90(content, self.k, [2, 3])
        H2, W2 = (self.W, self.H) if self.k % 2 == 1 else (self.H, self.W)
        return F.pad(rotated, (0, 30-W2, 0, 30-H2))

class TransposeModel(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        self.H, self.W = H, W
    def forward(self, x):
        content = x[:, :, :self.H, :self.W]
        trans = content.transpose(2, 3)
        return F.pad(trans, (0, 30-self.H, 0, 30-self.W))

class ScalingModel(nn.Module):
    def __init__(self, scale, H, W):
        super().__init__()
        self.scale, self.H, self.W = scale, H, W
    def forward(self, x):
        content = x[:, :, :self.H, :self.W]
        s = int(self.scale)
        kernel = torch.ones(10, 1, s, s, device=x.device)
        scaled = F.conv_transpose2d(content, kernel, stride=s, groups=10)
        return F.pad(scaled, (0, 30-scaled.shape[3], 0, 30-scaled.shape[2]))

class TileModel(nn.Module):
    def __init__(self, rr, rc, H, W):
        super().__init__()
        self.rr, self.rc, self.H, self.W = rr, rc, H, W
    def forward(self, x):
        content = x[:, :, :self.H, :self.W]
        tiled = content.repeat(1, 1, self.rr, self.rc)
        tiled = tiled[:, :, :30, :30]
        mask = torch.zeros_like(tiled)
        mask[:, :, :min(30, self.H*self.rr), :min(30, self.W*self.rc)] = 1.0
        return tiled * mask

class SymmetryFlipModel(nn.Module):
    def __init__(self, mode, H, W):
        super().__init__()
        self.mode, self.H, self.W = mode, H, W
    def forward(self, x):
        content = x[:, :, :self.H, :self.W]
        if self.mode == 'v':
            rev = torch.flip(content, [2])
            res = torch.cat([content, rev], dim=2)
            H2, W2 = 2*self.H, self.W
        else:
            rev = torch.flip(content, [3])
            res = torch.cat([content, rev], dim=3)
            H2, W2 = self.H, 2*self.W
        res = res[:, :, :30, :30]
        mask = torch.zeros_like(res)
        mask[:, :, :min(30, H2), :min(30, W2)] = 1.0
        return res * mask

class KroneckerModel(nn.Module):
    def forward(self, x):
        p3x3 = x[:, :, :3, :3]
        mask = (torch.sum(p3x3[:, 1:10], dim=1, keepdim=True) > 0.5).float()
        m_exp = mask.view(1, 1, 3, 1, 3, 1)
        p_exp = p3x3.view(1, 10, 1, 3, 1, 3)
        res_fg = (m_exp * p_exp[:, 1:10]).reshape(1, 9, 9, 9)
        res_c0 = ((1.0 - m_exp) + m_exp * p_exp[:, 0:1]).reshape(1, 1, 9, 9)
        res_9x9 = torch.cat([res_c0, res_fg], dim=1)
        return F.pad(res_9x9, (0, 21, 0, 21))

class HoleFillModel(nn.Module):
    def __init__(self, cf, H, W):
        super().__init__()
        self.cf, self.H, self.W = cf, H, W
    def forward(self, x):
        content = x[:, :, :self.H, :self.W]
        fg = (torch.sum(content[:, 1:10], dim=1, keepdim=True) > 0.5).float()
        outside = torch.zeros(1, 1, self.H, self.W, device=x.device)
        outside[:, :, 0, :] = 1.0; outside[:, :, -1, :] = 1.0; outside[:, :, :, 0] = 1.0; outside[:, :, :, -1] = 1.0
        outside = outside * (1.0 - fg)
        kernel = torch.tensor([[[[0.,1,0],[1,1,1],[0,1,0]]]], device=x.device)
        for _ in range(max(self.H, self.W)):
            outside = (F.conv2d(outside, kernel, padding=1) > 0.5).float() * (1.0 - fg)
        enclosed = (1.0 - fg) * (1.0 - outside)
        out_content = content.clone()
        out_content[:, self.cf:self.cf+1] = (out_content[:, self.cf:self.cf+1] + enclosed).clamp(0, 1)
        out_content[:, 0:1] = 1.0 - torch.sum(out_content[:, 1:10], dim=1, keepdim=True)
        return F.pad(out_content, (0, 30-self.W, 0, 30-self.H))

class MultiColorShiftModel(nn.Module):
    def __init__(self, shift_dict, H, W):
        super().__init__()
        self.shift_dict, self.H, self.W = shift_dict, H, W
    def forward(self, x):
        out_fg = torch.zeros(1, 9, self.H, self.W, device=x.device)
        for color in range(1, 10):
            content = x[:, color, :self.H, :self.W]
            dr, dc = self.shift_dict.get(color, (0, 0))
            shifted = torch.zeros_like(content)
            r0, r1 = max(0, dr), min(self.H, self.H + dr)
            c0, c1 = max(0, dc), min(self.W, self.W + dc)
            sr0, sr1 = max(0, -dr), min(self.H, self.H - dr)
            sc0, sc1 = max(0, -dc), min(self.W, self.W - dc)
            if r1 > r0 and c1 > c0:
                shifted[0, r0:r1, c0:c1] = content[0, sr0:sr1, sc0:sc1]
            out_fg[:, color-1] = shifted
        fg_sum = torch.sum(out_fg, dim=1, keepdim=True)
        out_bg = 1.0 - fg_sum
        res = torch.cat([out_bg, out_fg], dim=1)
        return F.pad(res, (0, 30-self.W, 0, 30-self.H))

class MirrorCompletionModel(nn.Module):
    def __init__(self, mode, H, W):
        super().__init__()
        self.mode, self.H, self.W = mode, H, W
    def forward(self, x):
        content = x[:, :, :self.H, :self.W]
        if 'v' in self.mode:
            rev = torch.flip(content, [2])
            content = torch.max(content, rev)
        if 'h' in self.mode:
            rev = torch.flip(content, [3])
            content = torch.max(content, rev)
        fg = torch.sum(content[:, 1:10], dim=1, keepdim=True)
        content[:, 0:1] = 1.0 - fg
        return F.pad(content, (0, 30-self.W, 0, 30-self.H))

class SubgridLogicModel(nn.Module):
    def __init__(self, H, W, rows, cols, op='and'):
        super().__init__()
        self.H, self.W, self.rows, self.cols, self.op = H, W, rows, cols, op
        self.sh, self.sw = H // rows, W // cols
    def forward(self, x):
        content = x[:, :, :self.H, :self.W]
        subgrids = content.view(1, 10, self.rows, self.sh, self.cols, self.sw)
        subgrids = subgrids.permute(0, 1, 2, 4, 3, 5).reshape(1, 10, self.rows * self.cols, self.sh, self.sw)
        res = subgrids[:, :, 0]
        for i in range(1, self.rows * self.cols):
            if self.op == 'and': res = res * subgrids[:, :, i]
            elif self.op == 'or': res = torch.max(res, subgrids[:, :, i])
            else: res = (res + subgrids[:, :, i]) % 2
        fg = torch.sum(res[:, 1:10], dim=1, keepdim=True)
        res[:, 0:1] = 1.0 - fg
        return F.pad(res, (0, 30-self.sw, 0, 30-self.sh))

class DividerLogicModel(nn.Module):
    def __init__(self, H, W, op='and', target_color=1):
        super().__init__()
        self.H, self.W, self.op, self.target_color = H, W, op, target_color
    def forward(self, x): return x

class ConstantFillModel(nn.Module):
    def __init__(self, color, H, W):
        super().__init__()
        self.color, self.H, self.W = color, H, W
    def forward(self, x):
        out = torch.zeros_like(x)
        out[:, self.color, :self.H, :self.W] = 1.0
        return out

class RecolorForegroundModel(nn.Module):
    def __init__(self, color, H, W):
        super().__init__()
        self.color, self.H, self.W = color, H, W
    def forward(self, x):
        content = x[:, :, :self.H, :self.W]
        fg = (torch.sum(content[:, 1:10], dim=1, keepdim=True) > 0.5).float()
        out = torch.zeros_like(content)
        out[:, self.color:self.color+1] = fg
        out[:, 0:1] = 1.0 - fg
        return F.pad(out, (0, 30-self.W, 0, 30-self.H))

class PatternExtractionModel(nn.Module):
    def __init__(self, pr, pc, ph, pw, rr, rc, mapping, Ho, Wo):
        super().__init__()
        self.pr, self.pc, self.ph, self.pw = pr, pc, ph, pw
        self.rr, self.rc, self.mapping = rr, rc, mapping
        self.Ho, self.Wo = Ho, Wo
        self.cmap = ColorMapModel(mapping, ph, pw)
    def forward(self, x):
        pattern = x[:, :, self.pr:self.pr+self.ph, self.pc:self.pc+self.pw]
        pattern = self.cmap(pattern)[:, :, :self.ph, :self.pw]
        tiled = pattern.repeat(1, 1, self.rr, self.rc)
        tiled = tiled[:, :, :self.Ho, :self.Wo]
        return F.pad(tiled, (0, 30-self.Wo, 0, 30-self.Ho))

class ConvModel(nn.Module):
    def __init__(self, weight, bias, ks):
        super().__init__()
        self.ks = ks
        self.conv = nn.Conv2d(10, 10, kernel_size=ks, padding=ks//2)
        self.conv.weight.data = weight
        self.conv.bias.data = bias
    def forward(self, x):
        return self.conv(x)

class PixelWiseModel(nn.Module):
    def __init__(self, H, Wo, op='not'):
        super().__init__()
        self.H, self.Wo, self.op = H, Wo, op
    def forward(self, x):
        # Specific for Task 003 recoloring based on position or similar
        return x
