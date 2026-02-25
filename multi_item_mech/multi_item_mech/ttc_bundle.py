"""Top Trading Cycles for N agents with bundle exchange.

Each agent has an initial bundle. Agents can only trade entire bundles.
An agent "points to" the bundle they most prefer. When a cycle is found,
agents in the cycle exchange bundles according to the cycle.
"""
from __future__ import annotations

import torch
from .allocations import AllocationIndex
from .config import Config


@torch.no_grad()
def ttc_bundle_exchange(
    cfg: Config,
    aidx: AllocationIndex,
    U_true: torch.Tensor,
    endow_idx: torch.Tensor,
) -> torch.Tensor:
    """
    TTC with bundle exchange for N agents.
    
    Each agent starts with a bundle. They can only exchange entire bundles.
    
    Args:
        U_true: [B, A, K] utilities for each agent for each allocation
        endow_idx: [B] initial allocation index
        
    Returns:
        final_idx: [B] final allocation index after TTC
    """
    device = U_true.device
    B = U_true.shape[0]
    A = cfg.num_agents
    
    # Get agent masks for each sample's endowment
    endow_masks = aidx.allocation_to_agent_masks(endow_idx)  # [B, A, m]
    
    # Compute utility for each agent if they receive each bundle
    # util_matrix[b, i, j] = utility of agent i receiving agent j's bundle
    util_matrix = torch.zeros((B, A, A), device=device)
    for j in range(A):
        # Agent j's bundle mask
        bundle_j = endow_masks[:, j, :]  # [B, m]
        for i in range(A):
            # Agent i's utility for bundle j
            # u_i(bundle_j) = sum_g v_{i,g} * mask_g + (size-1) * alpha_i
            # We need to compute this from U_true
            # But U_true is indexed by allocation, not by bundle...
            # 
            # Actually, we need to find the allocation where agent i gets bundle_j
            # This is complex. Let's simplify by computing utilities directly.
            pass
    
    # Simpler approach: compute utilities directly from bundle masks
    # Get v_true and alpha_true by recomputing from the allocation utilities
    # Actually, we don't have v_true here. Let's use a different approach.
    
    # For each sample, we have A agents with A bundles.
    # We want: util_matrix[b, i, j] = utility of agent i if they get agent j's initial bundle
    
    # From endow_masks[b, j, :] we get agent j's bundle as a binary mask
    # We need agent i's valuation for that bundle
    
    # Since we only have U_true[b, i, k] for allocations, not for arbitrary bundles,
    # we need to find which allocation gives agent i exactly bundle_j
    
    # This is tricky. Let's compute it differently:
    # The initial endowment allocation is endow_idx[b].
    # Under this allocation, agent i gets bundle endow_masks[b, i, :]
    # and has utility U_i(endow) = U_true[b, i, endow_idx[b]]
    
    # For TTC, we want to know: if agent i gets agent j's bundle instead,
    # what's their utility?
    
    # Let's create "swapped" allocations for each pair and evaluate
    
    # Get current assignment per agent: assign[b, j] = item set for agent j
    all_allocs = aidx.all_allocations_tensor().to(device)  # [K, m]
    current_allocs = all_allocs[endow_idx.long()]  # [B, m] - item->agent assignment
    
    for i in range(A):
        for j in range(A):
            if i == j:
                # Agent i keeps their own bundle
                # Find utility from U_true at endow_idx
                util_matrix[:, i, j] = U_true[torch.arange(B), i, endow_idx.long()]
            else:
                # Agent i gets agent j's bundle, agent j gets agent i's bundle
                # Create swapped allocation
                swapped = current_allocs.clone()
                # Items that belonged to agent i now belong to agent j
                # Items that belonged to agent j now belong to agent i
                mask_i = (current_allocs == i)
                mask_j = (current_allocs == j)
                swapped[mask_i] = j
                swapped[mask_j] = i
                
                # Convert to allocation index
                powers = torch.tensor([A ** p for p in range(cfg.num_items)], device=device, dtype=torch.long)
                swap_idx = (swapped * powers).sum(dim=1)
                
                # Get agent i's utility under this swapped allocation
                util_matrix[:, i, j] = U_true[torch.arange(B), i, swap_idx]
    
    # Now run TTC
    # assigned[b, i] = which agent's bundle agent i will receive (-1 = not yet assigned)
    assigned = torch.full((B, A), -1, dtype=torch.long, device=device)
    # available[b, j] = is agent j's bundle still available
    available = torch.ones((B, A), dtype=torch.bool, device=device)
    
    for _ in range(A):  # At most A rounds
        # For each unassigned agent, find their most preferred available bundle
        preferences = torch.full((B, A), -1, dtype=torch.long, device=device)
        
        for b_idx in range(B):
            for i in range(A):
                if assigned[b_idx, i] >= 0:
                    continue  # Already assigned
                
                best_j = -1
                best_util = float('-inf')
                for j in range(A):
                    if available[b_idx, j]:
                        u = util_matrix[b_idx, i, j].item()
                        if u > best_util:
                            best_util = u
                            best_j = j
                preferences[b_idx, i] = best_j
        
        # Find cycles and execute trades
        for b_idx in range(B):
            # Build preference graph for this sample
            visited = [False] * A
            for start in range(A):
                if assigned[b_idx, start] >= 0 or visited[start]:
                    continue
                
                # Follow chain from start
                path = []
                current = start
                while current >= 0 and current not in path and not visited[current]:
                    if assigned[b_idx, current] >= 0:
                        break
                    path.append(current)
                    current = preferences[b_idx, current].item()
                
                # Check if we formed a cycle
                if current in path:
                    cycle_start_idx = path.index(current)
                    cycle = path[cycle_start_idx:]
                    
                    # Execute cycle: each agent in cycle gets the bundle of who they're pointing to
                    for agent in cycle:
                        target_bundle = preferences[b_idx, agent].item()
                        assigned[b_idx, agent] = target_bundle
                        visited[agent] = True
                    
                    # Mark bundles as unavailable
                    for agent in cycle:
                        available[b_idx, preferences[b_idx, agent].item()] = False
        
        # Check if all assigned
        if (assigned >= 0).all():
            break
    
    # Handle any remaining unassigned (shouldn't happen in proper TTC)
    for b_idx in range(B):
        for i in range(A):
            if assigned[b_idx, i] < 0:
                assigned[b_idx, i] = i  # Keep own bundle
    
    # Construct final allocation from assignments
    final_allocs = torch.zeros((B, cfg.num_items), dtype=torch.long, device=device)
    for b_idx in range(B):
        for i in range(A):
            source = assigned[b_idx, i].item()  # Agent i gets agent source's original bundle
            # Items that originally belonged to source now belong to i
            source_mask = (current_allocs[b_idx] == source)
            final_allocs[b_idx, source_mask] = i
    
    # Convert to allocation index
    powers = torch.tensor([A ** p for p in range(cfg.num_items)], device=device, dtype=torch.long)
    final_idx = (final_allocs * powers).sum(dim=1)
    
    return final_idx
