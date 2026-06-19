import torch
import torch.optim as optim
import numpy as np
from nca_core_local_engine import (
    UpdateRule, SamplePool, make_seed, make_target, train_step, CHANNELS
)
import time

def train():
    device = "cpu" # Force CPU for predictable timing in sandbox
    print(f"Training on {device}")

    SIZE = 32
    PAD = 2
    FULL_SIZE = SIZE + 2 * PAD

    SHAPES = ["critter", "square", "circle", "triangle"]
    DNA_DIM = 4
    DNA_MAP = {s: torch.eye(DNA_DIM)[i].unsqueeze(0) for i, s in enumerate(SHAPES)}

    TARGET_MAP = {}
    for s in SHAPES:
        t_np = make_target(size=SIZE, pad=PAD, shape=s)
        TARGET_MAP[s] = torch.tensor(t_np).permute(2, 0, 1).unsqueeze(0).to(device)

    model = UpdateRule(channels=CHANNELS, hidden=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-3)

    POOLS = {}
    for s in SHAPES:
        def get_seed_func(shape_name=s):
            def seed_func(n=1):
                dna = DNA_MAP[shape_name].repeat(n, 1)
                return make_seed(FULL_SIZE, n=n, dna=dna)
            return seed_func

        POOLS[s] = SamplePool(get_seed_func(s), size=64)

    N_ITERS = 200 # Sanity run
    BATCH_SIZE = 4

    t0 = time.time()
    for i in range(N_ITERS):
        s = np.random.choice(SHAPES)
        target = TARGET_MAP[s]
        pool = POOLS[s]

        loss = train_step(model, optimizer, pool, target,
                          steps_range=(32, 48), batch_size=BATCH_SIZE,
                          n_damaged=1, device=device, noise_std=0.01)

        if i % 50 == 0:
            print(f"Iter {i:4d}, Shape {s:8s}, Loss {loss:.6f}, Time {time.time()-t0:.1f}s")

    print(f"Training finished in {time.time()-t0:.1f}s")
    torch.save(model.state_dict(), "programmable_nca.pt")
    print("Model saved as programmable_nca.pt")

if __name__ == "__main__":
    train()
