from __future__ import annotations
from dataclasses import dataclass
from typing import List
import torch

@dataclass(frozen=True)
class BundleIndex:
    num_items: int

    @property
    def num_bundles(self) -> int:
        return 2 ** self.num_items

    @property
    def full_mask(self) -> int:
        return (1 << self.num_items) - 1

    def complement_index(self) -> torch.Tensor:
        # For each mask k (agent1 bundle), comp[k] = full_mask ^ k (agent2 bundle)
        comp = torch.tensor([self.full_mask ^ k for k in range(self.num_bundles)], dtype=torch.long)
        return comp

    def masks_tensor(self) -> torch.Tensor:
        # shape [num_bundles, num_items] with bits {0,1}
        masks = torch.zeros((self.num_bundles, self.num_items), dtype=torch.float32)
        for k in range(self.num_bundles):
            for j in range(self.num_items):
                masks[k, j] = 1.0 if (k >> j) & 1 else 0.0
        return masks
