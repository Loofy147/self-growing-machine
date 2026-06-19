import torch
import numpy as np
from nca_core_local_engine import (
    UpdateRule, make_seed, make_target, to_rgb, CHANNELS, damage_batch
)
from PIL import Image
import os

def verify():
    device = "cpu"
    SIZE, PAD = 32, 2
    FULL_SIZE = SIZE + 2 * PAD
    SHAPES = ["critter", "square", "circle", "triangle"]
    DNA_DIM = 4
    DNA_MAP = {s: torch.eye(DNA_DIM)[i].unsqueeze(0) for i, s in enumerate(SHAPES)}

    model = UpdateRule(channels=CHANNELS, hidden=64).to(device)
    if os.path.exists("programmable_nca.pt"):
        model.load_state_dict(torch.load("programmable_nca.pt", map_location=device))
        print("Loaded trained model.")
    else:
        print("Warning: Model not found, using unitialized model.")

    model.eval()

    # 1. Verify Open-ended morphogenesis (Growing different shapes)
    print("Verifying open-ended morphogenesis...")
    results = []
    for s in SHAPES:
        dna = DNA_MAP[s]
        x = make_seed(FULL_SIZE, n=1, dna=dna, device=device)
        with torch.no_grad():
            for _ in range(64):
                x = model(x)
        rgb = to_rgb(x)[0].permute(1, 2, 0).numpy()
        results.append(rgb)

    # Save a strip of results
    combined = np.hstack(results)
    Image.fromarray((combined * 255).astype(np.uint8)).save("achievement_morphogenesis.png")
    print("Saved achievement_morphogenesis.png")

    # 2. Verify Robust adaptation (Self-healing under noise)
    print("Verifying robust adaptation (self-healing)...")
    s = "square"
    dna = DNA_MAP[s]
    x = make_seed(FULL_SIZE, n=1, dna=dna, device=device)
    with torch.no_grad():
        for _ in range(64):
            x = model(x)

        # Damage it
        x = damage_batch(x, n_damaged=1)
        rgb_damaged = to_rgb(x)[0].permute(1, 2, 0).numpy()

        # Let it repair with noise
        for _ in range(40):
            x = model(x, noise_std=0.02)
        rgb_repaired = to_rgb(x)[0].permute(1, 2, 0).numpy()

    combined_regen = np.hstack([rgb_damaged, rgb_repaired])
    Image.fromarray((combined_regen * 255).astype(np.uint8)).save("achievement_robustness.png")
    print("Saved achievement_robustness.png")

    print("Verification complete.")

if __name__ == "__main__":
    verify()
