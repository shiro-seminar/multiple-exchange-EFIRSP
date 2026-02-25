from __future__ import annotations
import torch
from .allocations import AllocationIndex
from .config import Config


@torch.no_grad()
def greedy_iterative_swap(
    cfg: Config,
    aidx: AllocationIndex,
    U_true: torch.Tensor,
    endow_idx: torch.Tensor,
    max_iter: int = 10
) -> torch.Tensor:
    """Greedy Iterative Swap for N agents.

    Starting from endowment, iteratively perform pairwise item swaps between
    agents if they result in a Pareto improvement (strictly better for at least one,
    >= for all others).

    This version tries:
    1. Single-item transfers (item from A to B)
    2. Pairwise swaps (item j from A to B, item k from B to A)

    Args:
        U_true: [B, A, K] True utilities
        endow_idx: [B] Initial allocation index

    Returns:
        final_idx: [B] Final allocation index
    """
    device = U_true.device
    B = U_true.shape[0]
    A, m = cfg.num_agents, cfg.num_items

    # Current allocation
    current_idx = endow_idx.clone()
    
    # Get current utilities
    def get_utils(alloc_idx):
        # alloc_idx: [B]
        alloc_idx_exp = alloc_idx.view(B, 1, 1).expand(B, A, 1)
        utils = U_true.gather(2, alloc_idx_exp).squeeze(2)  # [B, A]
        return utils

    curr_utils = get_utils(current_idx)

    # Precompute all allocations
    all_allocs = aidx.all_allocations_tensor().to(device)  # [K, m]
    powers = torch.tensor([A ** j for j in range(m)], device=device, dtype=torch.long)

    def alloc_to_idx(alloc):
        return (alloc * powers).sum(dim=1)

    for step in range(max_iter):
        improved_any = torch.zeros(B, dtype=torch.bool, device=device)
        
        best_idx = current_idx.clone()
        best_gain = torch.zeros(B, device=device)

        # Get current allocation as [B, m]
        curr_alloc = all_allocs[current_idx.long()]  # [B, m]

        # === Try pairwise swaps: agent i gives item j1, agent k gives item j2 ===
        for i in range(A):
            for k in range(A):
                if i >= k:
                    continue  # Only consider i < k to avoid duplicates
                
                for j1 in range(m):
                    for j2 in range(m):
                        if j1 == j2:
                            continue
                        
                        # Check if agent i owns j1 and agent k owns j2
                        i_owns_j1 = (curr_alloc[:, j1] == i)
                        k_owns_j2 = (curr_alloc[:, j2] == k)
                        valid_swap = i_owns_j1 & k_owns_j2
                        
                        if not valid_swap.any():
                            continue
                        
                        # Swap: j1 goes to k, j2 goes to i
                        cand_alloc = curr_alloc.clone()
                        cand_alloc[:, j1] = k
                        cand_alloc[:, j2] = i
                        cand_idx = alloc_to_idx(cand_alloc)
                        
                        cand_utils = get_utils(cand_idx)
                        diffs = cand_utils - curr_utils  # [B, A]
                        
                        # Pareto improvement
                        all_ok = (diffs >= -1e-5).all(dim=1)
                        some_better = (diffs > 1e-5).any(dim=1)
                        is_pareto = all_ok & some_better
                        
                        valid = valid_swap & is_pareto
                        gain = diffs.sum(dim=1)
                        update = valid & (gain > best_gain)
                        
                        best_idx = torch.where(update, cand_idx, best_idx)
                        best_gain = torch.where(update, gain, best_gain)
                        improved_any = improved_any | update

        # === Also try single-item transfers (for completeness) ===
        for j in range(m):
            for src in range(A):
                for dst in range(A):
                    if src == dst:
                        continue
                    
                    owns_j = (curr_alloc[:, j] == src)
                    
                    if not owns_j.any():
                        continue
                    
                    cand_alloc = curr_alloc.clone()
                    cand_alloc[:, j] = dst
                    cand_idx = alloc_to_idx(cand_alloc)
                    
                    cand_utils = get_utils(cand_idx)
                    diffs = cand_utils - curr_utils
                    
                    all_ok = (diffs >= -1e-5).all(dim=1)
                    some_better = (diffs > 1e-5).any(dim=1)
                    is_pareto = all_ok & some_better
                    
                    valid = owns_j & is_pareto
                    gain = diffs.sum(dim=1)
                    update = valid & (gain > best_gain)
                    
                    best_idx = torch.where(update, cand_idx, best_idx)
                    best_gain = torch.where(update, gain, best_gain)
                    improved_any = improved_any | update

        # Update state
        current_idx = best_idx
        curr_utils = get_utils(current_idx)
        
        if not improved_any.any():
            break
            
    return current_idx
