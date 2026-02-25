from __future__ import annotations
import torch
import copy

from .config import Config
from .allocations import AllocationIndex
from .data_gen import sample_types, types_to_allocation_utils, outside_option_utils
from .model import AllocationNet
from .losses import expected_utilities_from_probs, sp_loss_sampled, compute_ir_mask
from .oracle import oracle_welfare_no_disposal
from .baselines import greedy_iterative_swap
from .ttc_bundle import ttc_bundle_exchange


@torch.no_grad()
def ir_feasible_oracle(
    aidx: AllocationIndex,
    U_true: torch.Tensor,
    outside: torch.Tensor,
) -> torch.Tensor:
    """Compute best welfare among IR-feasible allocations."""
    # U_true: [B, A, K], outside: [B, A]
    welfare = U_true.sum(dim=1)  # [B, K]
    
    # Check IR for each allocation: U_true[:, i, k] >= outside[:, i] for all i
    outside_exp = outside.unsqueeze(2)  # [B, A, 1]
    feas = (U_true >= outside_exp).all(dim=1)  # [B, K]
    
    neg_inf = torch.full_like(welfare, -1e18)
    welfare_feas = torch.where(feas, welfare, neg_inf)
    
    best = welfare_feas.max(dim=1).values
    fallback = outside.sum(dim=1)
    best = torch.where(best < -1e17, fallback, best)
    return best


def one_hot_from_argmax(K: int, chosen: torch.Tensor, device: torch.device) -> torch.Tensor:
    probs_det = torch.zeros((chosen.shape[0], K), device=device)
    probs_det.scatter_(1, chosen.view(-1, 1), 1.0)
    return probs_det


def main():
    device = torch.device("cpu")

    ckpt = torch.load("allocation_net.pt", map_location=device)

    cfg = Config()
    if isinstance(ckpt, dict) and "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
        for k, v in ckpt["cfg"].items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    cfg.device = "cpu"

    aidx = AllocationIndex(num_agents=cfg.num_agents, num_items=cfg.num_items)
    net = AllocationNet(cfg, aidx).to(device=device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    N = 5000  # Evaluation samples
    batch = sample_types(cfg, aidx, N)
    v_true = batch["v_true"]
    a_true = batch["alpha_true"]
    endow_idx = batch["endow_idx"]

    U_true = types_to_allocation_utils(cfg, aidx, v_true, a_true)
    outside = outside_option_utils(cfg, aidx, U_true, endow_idx)

    K = aidx.num_allocations
    A = cfg.num_agents

    # SP approximation settings
    cfg_sp_net = copy.deepcopy(cfg)
    cfg_sp_net.misreport_samples = 48
    n_sp_net = 1000

    v_sp_net = v_true[:n_sp_net]
    a_sp_net = a_true[:n_sp_net]
    U_sp_net = U_true[:n_sp_net]
    endow_sp_net = endow_idx[:n_sp_net]

    cfg_sp_greedy = copy.deepcopy(cfg)
    cfg_sp_greedy.misreport_samples = 128
    n_sp_greedy = 100  # Keep small, Greedy is slow

    v_sp_greedy = v_true[:n_sp_greedy]
    a_sp_greedy = a_true[:n_sp_greedy]
    U_sp_greedy = U_true[:n_sp_greedy]
    endow_sp_greedy = endow_idx[:n_sp_greedy]

    # ============ Status quo (keep endowment) ============
    welfare_sq = outside.sum(dim=1).mean().item()
    ir_sq = 0.0
    sp_sq = 0.0

    # ============ Uniform random lottery ============
    probs_uni = torch.full((N, K), 1.0 / K, device=device)
    EU_uni = expected_utilities_from_probs(aidx, probs_uni, U_true)
    welfare_uni = EU_uni.sum(dim=1).mean().item()
    ir_uni = torch.relu(outside - EU_uni).mean().item()

    class UniformMech:
        def __call__(self, v_report, a_report, endow, *args, **kwargs):
            B = v_report.shape[0]
            return torch.full((B, K), 1.0 / K, device=v_report.device)

    sp_uni = sp_loss_sampled(cfg_sp_net, aidx, UniformMech(), v_sp_net, a_sp_net, U_sp_net, endow_sp_net).item()

    # ============ Oracle bounds ============
    welfare_oracle = oracle_welfare_no_disposal(aidx, U_true).mean().item()
    welfare_ir_oracle = ir_feasible_oracle(aidx, U_true, outside).mean().item()

    # ============ NN-WM (report-based welfare maximizer, no constraints) ============
    # Picks welfare-maximizing allocation based on reported types (ignores IR/SP)
    nnwm_chosen = U_true.sum(dim=1).argmax(dim=1)  # [B]
    probs_nnwm = one_hot_from_argmax(K, nnwm_chosen, device)
    EU_nnwm = expected_utilities_from_probs(aidx, probs_nnwm, U_true)
    welfare_nnwm = EU_nnwm.sum(dim=1).mean().item()
    ir_nnwm = torch.relu(outside - EU_nnwm).mean().item()
    
    class NNWMMech:
        def __call__(self, v_report, a_report, endow, *args, **kwargs):
            B = v_report.shape[0]
            U_rep = types_to_allocation_utils(cfg, aidx, v_report, a_report)
            chosen = U_rep.sum(dim=1).argmax(dim=1)
            probs0 = torch.zeros((B, K), device=v_report.device)
            probs0.scatter_(1, chosen.view(-1, 1), 1.0)
            return probs0
    
    sp_nnwm = sp_loss_sampled(cfg_sp_net, aidx, NNWMMech(), v_sp_net, a_sp_net, U_sp_net, endow_sp_net).item()

    # ============ Learned mechanism (lottery) ============
    # ============ Learned mechanism (lottery) ============
    mask = compute_ir_mask(cfg, aidx, U_true, endow_idx)
    probs = net(v_true, a_true, endow_idx, mask=mask)
    EU = expected_utilities_from_probs(aidx, probs, U_true)
    welfare_lot = EU.sum(dim=1).mean().item()
    ir_lot = torch.relu(outside - EU).mean().item()
    sp_lot = sp_loss_sampled(cfg_sp_net, aidx, net, v_sp_net, a_sp_net, U_sp_net, endow_sp_net).item()

    # ============ Learned mechanism (argmax deterministic) ============
    # ============ Learned mechanism (argmax deterministic) ============
    chosen = net.predict_argmax(v_true, a_true, endow_idx, mask=mask)
    probs_det = one_hot_from_argmax(K, chosen, device)
    EU_det = expected_utilities_from_probs(aidx, probs_det, U_true)
    welfare_det = EU_det.sum(dim=1).mean().item()
    ir_det = torch.relu(outside - EU_det).mean().item()

    class NetDet:
        def __call__(self, v_report, a_report, endow, *args, **kwargs):
            mask = kwargs.get('mask', None)
            ch = net.predict_argmax(v_report, a_report, endow, mask=mask)
            probs0 = torch.zeros((v_report.shape[0], K), device=v_report.device)
            probs0.scatter_(1, ch.view(-1, 1), 1.0)
            return probs0

    sp_det = sp_loss_sampled(cfg_sp_net, aidx, NetDet(), v_sp_net, a_sp_net, U_sp_net, endow_sp_net).item()

    zeros = (chosen == 0).float().mean().item()
    uniq = int(torch.unique(chosen).numel())

    # ============ Print results ============
    print("")
    print(f"=== Comparison (N={N}, {A} agents, {cfg.num_items} items, K={K} allocations) ===")
    print(f"{'Method':<40} {'Welfare':>10} {'IR':>10} {'SP':>10}")
    print("-" * 72)
    print(f"{'Status quo (endowment)':<40} {welfare_sq:>10.4f} {ir_sq:>10.4f} {sp_sq:>10.4f}")
    print(f"{'Uniform random lottery':<40} {welfare_uni:>10.4f} {ir_uni:>10.4f} {sp_uni:>10.4f}")
    print(f"{'Oracle (no constraints) [upper bound]':<40} {welfare_oracle:>10.4f} {'N/A':>10} {'N/A':>10}")
    print(f"{'Oracle w/ IR [upper bound]':<40} {welfare_ir_oracle:>10.4f} {'N/A':>10} {'N/A':>10}")
    print(f"{'NN-WM (welfare max, no IR/SP)':<40} {welfare_nnwm:>10.4f} {ir_nnwm:>10.4f} {sp_nnwm:>10.4f}")
    print("-" * 72)
    print(f"{'Learned (lottery)':<40} {welfare_lot:>10.4f} {ir_lot:>10.4f} {sp_lot:>10.4f}")
    print(f"{'Learned (argmax deterministic)':<40} {welfare_det:>10.4f} {ir_det:>10.4f} {sp_det:>10.4f}")
    print("")
    print(f"Argmax stats: P(alloc=0)={zeros:.4f}, unique_allocs={uniq}")

    # ============ TTC Bundle Exchange ============
    try:
        ttc_idx = ttc_bundle_exchange(cfg, aidx, U_true, endow_idx)
        probs_ttc = one_hot_from_argmax(K, ttc_idx, device)
        EU_ttc = expected_utilities_from_probs(aidx, probs_ttc, U_true)
        welfare_ttc = EU_ttc.sum(dim=1).mean().item()
        ir_ttc = torch.relu(outside - EU_ttc).mean().item()
        sp_ttc = 0.0  # TTC is SP by design
        print("")
        print("-" * 72)
        print(f"{'TTC Bundle Exchange':<40} {welfare_ttc:>10.4f} {ir_ttc:>10.4f} {sp_ttc:>10.4f}")
    except Exception as e:
        print(f"\n[Warning] TTC Bundle Exchange failed: {e}")

    # ============ Greedy Swap Baseline ============
    try:
        mask_trade = greedy_iterative_swap(cfg, aidx, U_true, endow_idx, max_iter=15)
        probs_trade = one_hot_from_argmax(K, mask_trade, device)

        EU_trade = expected_utilities_from_probs(aidx, probs_trade, U_true)
        welfare_trade = EU_trade.sum(dim=1).mean().item()
        ir_trade = torch.relu(outside - EU_trade).mean().item()

        class GreedySwapMech:
            def __init__(self, endow_base: torch.Tensor):
                self.endow_base = endow_base

            def __call__(self, v_report: torch.Tensor, a_report: torch.Tensor, *args, **kwargs) -> torch.Tensor:
                Btot = v_report.shape[0]
                B0 = self.endow_base.shape[0]

                if Btot == B0:
                    endow = self.endow_base
                else:
                    assert Btot % B0 == 0, f"Batch mismatch: Btot={Btot}, B0={B0}"
                    rep = Btot // B0
                    endow = self.endow_base.repeat_interleave(rep)

                U_rep = types_to_allocation_utils(cfg, aidx, v_report, a_report)
                chosen = greedy_iterative_swap(cfg, aidx, U_rep, endow, max_iter=15)
                probs0 = torch.zeros((Btot, K), device=v_report.device)
                probs0.scatter_(1, chosen.view(-1, 1).long(), 1.0)
                return probs0

        print("")
        print("-" * 72)
        trade_sp = sp_loss_sampled(
            cfg_sp_greedy,
            aidx,
            GreedySwapMech(endow_sp_greedy),
            v_sp_greedy,
            a_sp_greedy,
            U_sp_greedy,
            endow_sp_greedy,
        ).item()
        print(f"{'Heuristic Trade (Greedy Swap)':<40} {welfare_trade:>10.4f} {ir_trade:>10.4f} {trade_sp:>10.4f}")
        print(f"Gap to Oracle: {(welfare_oracle - welfare_trade):.4f}")

    except Exception as e:
        print(f"\n[Warning] Heuristic baseline failed: {e}")


if __name__ == "__main__":
    main()
