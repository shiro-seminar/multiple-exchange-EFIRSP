from __future__ import annotations
import torch
from .allocations import AllocationIndex, bundle_utility
from .config import Config


def sample_types(cfg: Config, aidx: AllocationIndex, batch_size: int) -> dict:
    """Sample true types and endowments for N agents.

    Returns dict with:
      v_true: [B, A, m]
      alpha_true: [B, A]
      endow_idx: [B] allocation index for initial endowment
    """
    device = torch.device(cfg.device)
    B, A, m = batch_size, cfg.num_agents, cfg.num_items

    v_true = torch.empty((B, A, m), device=device).uniform_(cfg.v_min, cfg.v_max)
    alpha_true = torch.empty((B, A), device=device).uniform_(cfg.alpha_min, cfg.alpha_max)

    endow_idx = aidx.random_endowment_no_disposal(B, device)

    return {
        "v_true": v_true,
        "alpha_true": alpha_true,
        "endow_idx": endow_idx,
    }


def types_to_allocation_utils(
    cfg: Config,
    aidx: AllocationIndex,
    v: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Compute utilities for every allocation for each agent.

    Args:
      v: [B, A, m]
      alpha: [B, A]
      
    Returns:
      U: [B, A, K], where U[:, i, k] = utility of agent i under allocation k
    """
    device = v.device
    B, A, m = v.shape
    K = aidx.num_allocations

    # Get all agent masks for all allocations
    all_allocs = aidx.all_allocations_tensor().to(device)  # [K, m]
    
    # For each allocation k, compute mask for each agent
    # agent_masks[k, i, j] = 1 if allocation k gives item j to agent i
    agent_masks = torch.zeros((K, A, m), dtype=torch.float32, device=device)
    for i in range(A):
        agent_masks[:, i, :] = (all_allocs == i).float()

    # Compute utilities: U[b, i, k] = bundle_utility(v[b,i], alpha[b,i], agent_masks[k,i])
    # v: [B, A, m], alpha: [B, A]
    # agent_masks: [K, A, m]
    
    # Expand for broadcasting
    v_exp = v.unsqueeze(2)  # [B, A, 1, m]
    alpha_exp = alpha.unsqueeze(2)  # [B, A, 1]
    masks_exp = agent_masks.unsqueeze(0).permute(0, 2, 1, 3)  # [1, A, K, m]
    
    # Additive value
    add = (v_exp * masks_exp).sum(dim=-1)  # [B, A, K]
    
    # Bundle size
    size = masks_exp.sum(dim=-1)  # [1, A, K]
    
    # Synergy
    syn = torch.where(size >= 1, (size - 1) * alpha_exp, torch.zeros_like(add))
    
    U = add + syn  # [B, A, K]
    return U


def outside_option_utils(
    cfg: Config,
    aidx: AllocationIndex,
    U_true: torch.Tensor,
    endow_idx: torch.Tensor,
) -> torch.Tensor:
    """Outside option utility for each agent given endowment allocation.

    Args:
        U_true: [B, A, K] utilities
        endow_idx: [B] endowment allocation index
        
    Returns:
        outside: [B, A]
    """
    B = endow_idx.shape[0]
    A = cfg.num_agents
    
    # Gather utility at endowment for each agent
    endow_idx_exp = endow_idx.view(B, 1, 1).expand(B, A, 1)
    outside = U_true.gather(2, endow_idx_exp).squeeze(2)  # [B, A]
    return outside
