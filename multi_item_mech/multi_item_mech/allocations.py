from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class AllocationIndex:
    """Manages allocation space for N agents and M items.
    
    An allocation assigns each item to exactly one agent.
    Total allocations = num_agents ^ num_items
    """
    num_agents: int
    num_items: int

    @property
    def num_allocations(self) -> int:
        return self.num_agents ** self.num_items

    def all_allocations_tensor(self) -> torch.Tensor:
        """Generate all possible allocations.
        
        Returns:
            [K, m] tensor where K = A^m, each row assigns items to agents (0..A-1)
        """
        A, m = self.num_agents, self.num_items
        K = self.num_allocations
        
        allocs = torch.zeros((K, m), dtype=torch.long)
        for k in range(K):
            val = k
            for j in range(m):
                allocs[k, j] = val % A
                val //= A
        return allocs

    def allocation_to_agent_masks(self, alloc_idx: torch.Tensor) -> torch.Tensor:
        """Convert allocation indices to per-agent item masks.
        
        Args:
            alloc_idx: [B] allocation indices in [0, K)
            
        Returns:
            [B, A, m] binary masks where masks[b, i, j] = 1 if agent i gets item j
        """
        device = alloc_idx.device
        B = alloc_idx.shape[0]
        A, m = self.num_agents, self.num_items
        
        all_allocs = self.all_allocations_tensor().to(device)  # [K, m]
        allocs = all_allocs[alloc_idx.long()]  # [B, m]
        
        # Create one-hot masks for each agent
        masks = torch.zeros((B, A, m), dtype=torch.float32, device=device)
        for i in range(A):
            masks[:, i, :] = (allocs == i).float()
        return masks

    def agent_masks_to_allocation(self, masks: torch.Tensor) -> torch.Tensor:
        """Convert per-agent masks back to allocation index.
        
        Args:
            masks: [B, A, m] binary masks
            
        Returns:
            [B] allocation indices
        """
        device = masks.device
        B = masks.shape[0]
        A, m = self.num_agents, self.num_items
        
        # Each item is assigned to the agent with mask=1
        # alloc[j] = argmax over agents for item j
        allocs = masks.argmax(dim=1)  # [B, m]
        
        # Convert to index: sum_j alloc[j] * A^j
        powers = torch.tensor([A ** j for j in range(m)], device=device, dtype=torch.long)
        indices = (allocs * powers).sum(dim=1)
        return indices

    def random_endowment_no_disposal(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate random endowments without disposal.
        
        Strategy: Randomly assign each item to agent 0 or 1, agent 2 gets remaining.
        But since we require no disposal, we distribute randomly among first A-1 agents,
        and agent A-1 gets whatever is unassigned.
        
        For simplicity: each item goes to a random agent uniformly.
        
        Returns:
            [B] allocation indices
        """
        A, m = self.num_agents, self.num_items
        
        # Random assignment: each item to one of A agents
        allocs = torch.randint(0, A, (batch_size, m), device=device)
        
        # Convert to index
        powers = torch.tensor([A ** j for j in range(m)], device=device, dtype=torch.long)
        indices = (allocs * powers).sum(dim=1)
        return indices


def bundle_utility(v: torch.Tensor, alpha: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute utility for a bundle given valuations and synergy.
    
    Args:
        v: [..., m] item valuations
        alpha: [...] synergy parameter
        mask: [..., m] binary mask for items in bundle
        
    Returns:
        [...] utility values
        
    Utility: u(S) = sum_{j in S} v_j + (|S| - 1) * alpha if |S| >= 1 else 0
    """
    add = (v * mask).sum(dim=-1)
    size = mask.sum(dim=-1)
    syn = torch.where(size >= 1, (size - 1) * alpha, torch.zeros_like(add))
    return add + syn
