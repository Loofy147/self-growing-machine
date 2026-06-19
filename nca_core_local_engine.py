"""
Growing Neural Cellular Automaton (NCA) — core mechanics.

A tiny neural net (~replicated at every pixel) is trained so that, starting
from a single living seed cell, repeated local self-application makes the
whole grid grow into a target creature — and keeps existing as a stable
'organism' that regenerates if damaged.

This file is engine-only: target generation, perception, the update rule,
and one training step. Sanity-tested on CPU at small scale here; full run
goes on Kaggle GPU at larger scale (see nca_kaggle.ipynb).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

CHANNELS = 16  # 4 visible (RGBA) + 12 hidden "signal" channels


# ---------------------------------------------------------------------------
# Target: a small original procedural critter (not copied from anywhere —
# drawn with primitives so the NCA has something concrete to "become").
# ---------------------------------------------------------------------------
def make_target(size=40, pad=4):
    """Procedurally draw a small original critter, scaled to `size`."""
    s = size / 40.0  # all coords below are tuned for a 40px reference canvas
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    body = (80, 180, 120, 255)
    bulb = (120, 220, 160, 255)
    white = (255, 255, 255, 255)
    dark = (15, 15, 15, 255)
    line = (20, 80, 50, 255)

    def box(coords):
        return [c * s for c in coords]

    d.ellipse(box([8, 16, 32, 36]), fill=body)                 # body
    d.line(box([20, 16, 20, 7]), fill=body, width=max(1, round(2 * s)))  # antenna stalk
    d.ellipse(box([16, 2, 24, 9]), fill=bulb)                   # antenna bulb
    d.ellipse(box([12, 21, 17, 26]), fill=white)                # left eye white
    d.ellipse(box([23, 21, 28, 26]), fill=white)                # right eye white
    d.ellipse(box([13.5, 22.5, 15.5, 24.5]), fill=dark)         # left pupil
    d.ellipse(box([24.5, 22.5, 26.5, 24.5]), fill=dark)         # right pupil
    d.arc(box([14, 27, 26, 33]), start=20, end=160, fill=line, width=max(1, round(2 * s)))  # smile

    full = size + 2 * pad
    canvas = Image.new("RGBA", (full, full), (0, 0, 0, 0))
    canvas.paste(img, (pad, pad), img)
    arr = np.array(canvas).astype(np.float32) / 255.0
    arr[..., :3] *= arr[..., 3:4]  # premultiply alpha
    return arr  # [H, W, 4]


# ---------------------------------------------------------------------------
# Perception: depthwise conv with 3 fixed filters per channel (identity +
# sobel_x + sobel_y). Each cell senses itself and its local gradient — this
# is the only way information moves; there is no global view.
# ---------------------------------------------------------------------------
class Perception(nn.Module):
    def __init__(self, channels=CHANNELS):
        super().__init__()
        ident = torch.tensor([[0., 0, 0], [0, 1, 0], [0, 0, 0]])
        sx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
        sy = sx.t()
        kernel = torch.stack([ident, sx, sy])               # [3,3,3]
        kernel = kernel.repeat(channels, 1, 1).unsqueeze(1)  # [C*3,1,3,3]
        self.register_buffer("kernel", kernel)
        self.channels = channels

    def forward(self, x):  # x: [B,C,H,W]
        return F.conv2d(x, self.kernel, padding=1, groups=self.channels)


# ---------------------------------------------------------------------------
# Update rule: the actual "small neural network". Same weights applied to
# every cell, every step. Last layer zero-init so training starts as a
# no-op (stability trick from the original Growing NCA recipe).
# ---------------------------------------------------------------------------
class UpdateRule(nn.Module):
    def __init__(self, channels=CHANNELS, hidden=128):
        super().__init__()
        self.channels = channels
        self.perceive = Perception(channels)
        self.fc1 = nn.Conv2d(channels * 3, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x, update_rate=0.5):
        y = self.perceive(x)
        y = F.relu(self.fc1(y))
        dx = self.fc2(y)
        mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
                 <= update_rate).float()
        x = x + dx * mask
        alive_before = F.max_pool2d(x[:, 3:4], 3, stride=1, padding=1) > 0.1
        x = x * alive_before.float()
        return x

    def step_n(self, x, n, update_rate=0.5):
        for _ in range(n):
            x = self.forward(x, update_rate)
        return x


def make_seed(size, n=1, channels=CHANNELS, device="cpu"):
    x = torch.zeros(n, channels, size, size, device=device)
    x[:, 3:, size // 2, size // 2] = 1.0
    return x


def to_rgb(x):
    rgb, a = x[:, :3], torch.clamp(x[:, 3:4], 0, 1)
    return torch.clamp(1.0 - a + rgb, 0, 1)


# ---------------------------------------------------------------------------
# Pool-based training step. A pool of in-progress organisms is kept across
# steps (not just fresh seeds) — this is what teaches the model to persist
# and regenerate, not just grow once and stop.
# ---------------------------------------------------------------------------
class SamplePool:
    def __init__(self, seed, size=1024):
        self.size = size
        self.slots = seed.repeat(size, 1, 1, 1).clone()

    def sample(self, n):
        idx = np.random.choice(self.size, n, replace=False)
        return idx, self.slots[idx].clone()

    def commit(self, idx, batch):
        self.slots[idx] = batch.detach()


def damage_batch(x, n_damaged=0):
    """Punch random circular holes in `n_damaged` of the batch to force
    the model to learn regeneration, not just one-shot growth."""
    if n_damaged == 0:
        return x
    x = x.clone()
    B, C, H, W = x.shape
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    for i in range(n_damaged):
        cy, cx = np.random.randint(0, H), np.random.randint(0, W)
        r = np.random.randint(H // 6, H // 3)
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2) > r * r
        x[-(i + 1)] *= mask.float()
    return x


def train_step(model, opt, pool, target, steps_range=(48, 64), batch_size=8,
                n_damaged=3, device="cpu"):
    idx, x = pool.sample(batch_size)
    x = x.to(device)
    # bias toward keeping the current worst sample as a fresh seed
    losses_pre = ((x[:, :4] - target) ** 2).mean(dim=[1, 2, 3])
    worst = torch.argmax(losses_pre).item()
    x[worst] = make_seed(x.shape[-1], 1, device=device)[0]
    x = damage_batch(x, n_damaged=min(n_damaged, batch_size - 1))

    n_steps = np.random.randint(*steps_range)
    for _ in range(n_steps):
        x = model(x)

    loss = ((x[:, :4] - target) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    pool.commit(idx, x)
    return loss.item()
