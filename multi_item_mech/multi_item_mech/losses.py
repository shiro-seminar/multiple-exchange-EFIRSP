from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import Config
from .allocations import AllocationIndex
from .allocations import AllocationIndex
from .data_gen import outside_option_utils, types_to_allocation_utils


def compute_ir_mask(
    cfg: Config,
    aidx: AllocationIndex,
    U_report: torch.Tensor,     # [B, A, K]
    endow_idx: torch.Tensor,    # [B]
) -> torch.Tensor:
    """Compute mask of IR-feasible allocations based on reported types."""
    # outside_u: [B, A]
    outside_u = outside_option_utils(cfg, aidx, U_report, endow_idx)
    
    # feasible if U_i(alloc) >= outside_u_i for all i
    # U_report: [B, A, K], outside_u: [B, A] -> [B, A, 1]
    diff = U_report - outside_u.unsqueeze(2)
    feasible = (diff >= -1e-5).all(dim=1)  # [B, K]
    return feasible.float()


def expected_utilities_from_probs(
    aidx: AllocationIndex,
    probs: torch.Tensor,      # [B, K]
    U_true: torch.Tensor,     # [B, A, K]
) -> torch.Tensor:
    """Expected utilities for all agents under allocation lottery probs.
    
    EU_i = sum_k probs[k] * U_true[:, i, k]
    
    Returns: [B, A]
    """
    # probs: [B, K], U_true: [B, A, K]
    # EU: [B, A] = sum over k of probs * U_true
    EU = torch.einsum('bk,bak->ba', probs, U_true)
    return EU


def ir_violation_from_probs(
    cfg: Config,
    aidx: AllocationIndex,
    probs: torch.Tensor,       # [B, K]
    U_true: torch.Tensor,      # [B, A, K]
    endow_idx: torch.Tensor,   # [B]
) -> torch.Tensor:
    """Individual rationality violation (mean over all agents and samples)."""
    EU = expected_utilities_from_probs(aidx, probs, U_true)           # [B, A]
    outside_u = outside_option_utils(cfg, aidx, U_true, endow_idx)    # [B, A]
    return F.relu(outside_u - EU).mean()


def sample_misreports(
    cfg: Config,
    v_true: torch.Tensor,       # [B, A, m]
    alpha_true: torch.Tensor,   # [B, A]
    M: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample additive noises for (v, alpha) misreports."""
    B, A, m = v_true.shape
    device = v_true.device

    v_noise = (2 * torch.rand((B, M, A, m), device=device) - 1) * float(cfg.misreport_noise_v)
    a_noise = (2 * torch.rand((B, M, A), device=device) - 1) * float(cfg.misreport_noise_alpha)
    return v_noise, a_noise


def sp_loss_sampled(
    cfg: Config,
    aidx: AllocationIndex,
    mech,  # Callable: (v_report, alpha_report, endow_idx) -> probs [B, K]
    v_true: torch.Tensor,
    alpha_true: torch.Tensor,
    U_true: torch.Tensor,
    endow_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Approximate dominant-strategy regret (SP violation) by sampling misreports.

    For each agent i:
      - Sample M misreports for (v_i, alpha_i), keep others truthful
      - Recompute allocation under the *reported* types
      - Evaluate utilities under TRUE preferences (U_true)
      - Regret_i = max_{misreport} max(0, EU_i(mis) - EU_i(truth))
    Returns mean over agents.
    """
    device = v_true.device
    B, A, m = v_true.shape
    M = int(cfg.misreport_samples)

    # EU under truthful reports
    # Compute mask for truthful reports
    mask_true = compute_ir_mask(cfg, aidx, U_true, endow_idx)
    probs_true = mech(v_true, alpha_true, endow_idx, mask=mask_true)  # [B, K]
    EU_true = expected_utilities_from_probs(aidx, probs_true, U_true)  # [B, A]

    # Repeat endow for flattened (B*M)
    endow_rep = endow_idx.unsqueeze(1).expand(B, M).reshape(B * M)

    regrets = []
    for i in range(A):
        v_noise, a_noise = sample_misreports(cfg, v_true, alpha_true, M)  # [B,M,A,m], [B,M,A]

        v_mis_i = torch.clamp(
            v_true[:, i, :].unsqueeze(1) + v_noise[:, :, i, :], cfg.v_min, cfg.v_max
        )  # [B, M, m]
        a_mis_i = torch.clamp(
            alpha_true[:, i].unsqueeze(1) + a_noise[:, :, i], cfg.alpha_min, cfg.alpha_max
        )  # [B, M]

        # Build reports (vectorized)
        v_rep = v_true.unsqueeze(1).expand(B, M, A, m).contiguous()
        a_rep = alpha_true.unsqueeze(1).expand(B, M, A).contiguous()
        v_rep = v_rep.clone()
        a_rep = a_rep.clone()
        v_rep[:, :, i, :] = v_mis_i
        a_rep[:, :, i] = a_mis_i

        v_rep_f = v_rep.reshape(B * M, A, m)
        a_rep_f = a_rep.reshape(B * M, A)

        v_rep_f = v_rep.reshape(B * M, A, m)
        a_rep_f = a_rep.reshape(B * M, A)
        
        # Compute mask for misreports
        U_mis_rep = types_to_allocation_utils(cfg, aidx, v_rep_f, a_rep_f)
        mask_mis = compute_ir_mask(cfg, aidx, U_mis_rep, endow_rep)

        probs_mis = mech(v_rep_f, a_rep_f, endow_rep, mask=mask_mis).reshape(B, M, -1)  # [B, M, K]

        # Utilities under TRUE preferences; only allocation changes
        EU_mis_all = expected_utilities_from_probs(
            aidx,
            probs_mis.reshape(B * M, -1),
            U_true.repeat_interleave(M, dim=0),
        )  # [B*M, A]

        EU_mis_i = EU_mis_all.reshape(B, M, A)[:, :, i]  # [B, M]
        gain = EU_mis_i - EU_true[:, i].unsqueeze(1)      # [B, M]
        regret = torch.relu(gain).max(dim=1).values       # [B]
        regrets.append(regret)

    sp = torch.stack(regrets, dim=1).mean()
    return sp


def augmented_loss(
    cfg: Config,
    aidx: AllocationIndex,
    net,
    v_true: torch.Tensor,
    alpha_true: torch.Tensor,
    U_true: torch.Tensor,
    endow_idx: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Augmented Lagrangian objective: -welfare + λ·viol + ρ/2 · viol^2."""
    probs = net(
        v_true,
        alpha_true,
        endow_idx,
        temperature=float(getattr(cfg, "temperature", 1.0)),
        hard=bool(getattr(cfg, "hard_output", False)),
        mask=compute_ir_mask(cfg, aidx, U_true, endow_idx),
    )  # [B, K]

    EU = expected_utilities_from_probs(aidx, probs, U_true)  # [B, A]
    welfare = EU.sum(dim=1).mean()

    ir = ir_violation_from_probs(cfg, aidx, probs, U_true, endow_idx)
    sp = sp_loss_sampled(cfg, aidx, net, v_true, alpha_true, U_true, endow_idx)

    ef_loss = -welfare
    loss = (
        ef_loss
        + cfg.lambda_ir * ir
        + cfg.lambda_sp * sp
        + (cfg.rho / 2) * (ir * ir + sp * sp)
    )

    stats = {
        "welfare": float(welfare.detach().cpu()),
        "ir": float(ir.detach().cpu()),
        "sp": float(sp.detach().cpu()),
        "loss": float(loss.detach().cpu()),
    }
    return loss, stats
