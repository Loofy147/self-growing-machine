import unittest
import torch
import numpy as np
from nca_core_local_engine import (
    UpdateRule, make_seed, make_target, CHANNELS
)

class TestNCA(unittest.TestCase):
    def test_make_target(self):
        for shape in ["critter", "square", "circle", "triangle"]:
            t = make_target(size=20, pad=2, shape=shape)
            self.assertEqual(t.shape, (24, 24, 4))

    def test_make_seed_dna(self):
        dna = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
        seed = make_seed(size=10, n=1, dna=dna)
        self.assertEqual(seed.shape, (1, CHANNELS, 10, 10))
        # Check alpha channel
        self.assertEqual(seed[0, 3, 5, 5], 1.0)
        # Check DNA channels
        torch.testing.assert_close(seed[0, 4:8, 5, 5], dna[0])

    def test_update_rule_forward(self):
        model = UpdateRule(channels=CHANNELS, hidden=16)
        x = torch.zeros(1, CHANNELS, 10, 10)
        x[:, 3, 5, 5] = 1.0
        out = model(x)
        self.assertEqual(out.shape, x.shape)
        # Check that it stays alive (neighboring cells might wake up)
        self.assertTrue(torch.any(out[:, 3, :, :] > 0))

if __name__ == "__main__":
    unittest.main()
