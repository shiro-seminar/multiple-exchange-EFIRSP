from __future__ import annotations
import torch
from .allocations import AllocationIndex


@torch.no_grad()
def oracle_welfare_no_disposal(aidx: AllocationIndex, U_true: torch.Tensor) -> torch.Tensor:
    """Compute first-best welfare by enumerating all allocations (no IR constraint).

    U_true: [B, A, K] utilities for each agent for each allocation.
    Returns:
      best_welfare: [B]
    """
    # Sum utilities across all agents for each allocation
    welfare = U_true.sum(dim=1)  # [B, K]
    best = welfare.max(dim=1).values
    return best
