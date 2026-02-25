"""Analyze the learned mechanism: input/output mapping, sensitivity, and patterns.

Usage:
    cd multi_item_mech
    python -m multi_item_mech.analyze_mechanism
"""
from __future__ import annotations

import torch
import copy

from .config import Config
from .allocations import AllocationIndex
from .data_gen import sample_types, types_to_allocation_utils, outside_option_utils
from .model import AllocationNet
from .losses import compute_ir_mask
from .oracle import oracle_welfare_no_disposal


def load_model(path: str = "allocation_net.pt"):
    """Load trained model and config."""
    device = torch.device("cpu")
    ckpt = torch.load(path, map_location=device)

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
    return cfg, aidx, net, device


def items_str(mask_row, m):
    """Convert a binary mask to a human-readable item list like '{0,2,4}'."""
    items = [str(j) for j in range(m) if mask_row[j] > 0.5]
    return "{" + ",".join(items) + "}" if items else "{}"


# ========================================================================
# 1. Sample I/O table
# ========================================================================
@torch.no_grad()
def sample_io_table(cfg, aidx, net, device, n_samples=10):
    """Show n_samples of input → output mapping in detail."""
    A, m, K = cfg.num_agents, cfg.num_items, aidx.num_allocations

    torch.manual_seed(42)
    batch = sample_types(cfg, aidx, n_samples)
    v_true = batch["v_true"]
    a_true = batch["alpha_true"]
    endow_idx = batch["endow_idx"]
    U_true = types_to_allocation_utils(cfg, aidx, v_true, a_true)
    outside = outside_option_utils(cfg, aidx, U_true, endow_idx)

    # Learned allocation
    mask = compute_ir_mask(cfg, aidx, U_true, endow_idx)
    alloc_idx = net.predict_argmax(v_true, a_true, endow_idx, mask=mask)

    # Oracle
    oracle_idx = U_true.sum(dim=1).argmax(dim=1)

    # IR-feasible oracle
    outside_exp = outside.unsqueeze(2)
    feas = (U_true >= outside_exp).all(dim=1)
    welfare_all = U_true.sum(dim=1)
    neg_inf = torch.full_like(welfare_all, -1e18)
    welfare_feas = torch.where(feas, welfare_all, neg_inf)
    ir_oracle_idx = welfare_feas.argmax(dim=1)

    endow_masks = aidx.allocation_to_agent_masks(endow_idx)
    alloc_masks = aidx.allocation_to_agent_masks(alloc_idx)
    oracle_masks = aidx.allocation_to_agent_masks(oracle_idx)
    ir_oracle_masks = aidx.allocation_to_agent_masks(ir_oracle_idx)

    print("=" * 80)
    print("  PART 1: Sample Input → Output Table")
    print("=" * 80)

    for s in range(n_samples):
        print(f"\n{'─' * 75}")
        print(f"  Sample {s}")
        print(f"{'─' * 75}")

        # Input: valuations
        print(f"  【Input】 Valuations v[agent, item]:")
        for i in range(A):
            v_str = " ".join(f"{v_true[s, i, j]:.3f}" for j in range(m))
            print(f"    Agent {i}: [{v_str}]")

        # Input: endowment
        print(f"\n  【Input】 Endowment (index={endow_idx[s].item()}):")
        for i in range(A):
            bundle = items_str(endow_masks[s, i], m)
            u_endow = U_true[s, i, endow_idx[s].long()].item()
            print(f"    Agent {i}: {bundle:>12s}  utility={u_endow:.4f}")
        w_endow = U_true[s, :, endow_idx[s].long()].sum().item()

        # Output: learned allocation
        print(f"\n  【Output】 Learned Allocation (index={alloc_idx[s].item()}):")
        ir_all_ok = True
        for i in range(A):
            bundle = items_str(alloc_masks[s, i], m)
            u_alloc = U_true[s, i, alloc_idx[s].long()].item()
            u_endow = U_true[s, i, endow_idx[s].long()].item()
            ir_status = "OK" if u_alloc >= u_endow - 1e-5 else "VIOLATED"
            if ir_status == "VIOLATED":
                ir_all_ok = False
            print(f"    Agent {i}: {bundle:>12s}  utility={u_alloc:.4f}  (endow={u_endow:.4f}, IR={ir_status})")
        w_alloc = U_true[s, :, alloc_idx[s].long()].sum().item()

        # Comparison: Oracle
        w_oracle = U_true[s, :, oracle_idx[s].long()].sum().item()
        w_ir_oracle = U_true[s, :, ir_oracle_idx[s].long()].sum().item()

        # Oracle bundles
        oracle_bundles = " | ".join(
            f"Ag{i}→{items_str(oracle_masks[s, i], m)}" for i in range(A)
        )
        ir_oracle_bundles = " | ".join(
            f"Ag{i}→{items_str(ir_oracle_masks[s, i], m)}" for i in range(A)
        )

        print(f"\n  【Comparison】")
        print(f"    Endowment welfare  = {w_endow:.4f}")
        print(f"    Learned welfare    = {w_alloc:.4f}  (IR={'ALL OK' if ir_all_ok else 'VIOLATED'})")
        print(f"    IR-Oracle welfare  = {w_ir_oracle:.4f}  [{ir_oracle_bundles}]")
        print(f"    Oracle welfare     = {w_oracle:.4f}  [{oracle_bundles}]")

        # Does learned match oracle?
        match_oracle = alloc_idx[s].item() == oracle_idx[s].item()
        match_ir_oracle = alloc_idx[s].item() == ir_oracle_idx[s].item()
        print(f"    Match Oracle?      = {match_oracle}  |  Match IR-Oracle? = {match_ir_oracle}")


# ========================================================================
# 2. Allocation frequency analysis
# ========================================================================
@torch.no_grad()
def allocation_frequency(cfg, aidx, net, device, n_samples=5000, top_k=20):
    """Show most frequently chosen allocations."""
    A, m, K = cfg.num_agents, cfg.num_items, aidx.num_allocations

    torch.manual_seed(123)
    batch = sample_types(cfg, aidx, n_samples)
    v_true = batch["v_true"]
    a_true = batch["alpha_true"]
    endow_idx = batch["endow_idx"]
    U_true = types_to_allocation_utils(cfg, aidx, v_true, a_true)

    mask = compute_ir_mask(cfg, aidx, U_true, endow_idx)
    alloc_idx = net.predict_argmax(v_true, a_true, endow_idx, mask=mask)

    # Count
    counts = torch.bincount(alloc_idx.long(), minlength=K)
    sorted_vals, sorted_ids = counts.sort(descending=True)

    all_allocs = aidx.all_allocations_tensor()  # [K, m]

    print("\n" + "=" * 80)
    print("  PART 2: Allocation Frequency (top-{})".format(top_k))
    print("=" * 80)
    print(f"  Total samples: {n_samples},  Unique allocations used: {(counts > 0).sum().item()} / {K}")
    print()

    header = f"  {'Rank':>4} {'AllocIdx':>8} {'Count':>6} {'Freq%':>7}  "
    for i in range(A):
        header += f"{'Ag'+str(i)+' items':>12} "
    print(header)
    print("  " + "─" * (len(header) - 2))

    cumulative = 0
    for rank in range(min(top_k, K)):
        idx = sorted_ids[rank].item()
        cnt = sorted_vals[rank].item()
        if cnt == 0:
            break
        cumulative += cnt
        freq = cnt / n_samples * 100

        alloc_row = all_allocs[idx]  # [m]
        agent_items = []
        for i in range(A):
            items_i = [str(j) for j in range(m) if alloc_row[j].item() == i]
            agent_items.append("{" + ",".join(items_i) + "}")

        row = f"  {rank+1:>4} {idx:>8} {cnt:>6} {freq:>6.1f}%  "
        for bundle in agent_items:
            row += f"{bundle:>12} "
        print(row)

    print(f"\n  Top-{top_k} cumulative coverage: {cumulative/n_samples*100:.1f}%")


# ========================================================================
# 3. Sensitivity analysis: vary one valuation
# ========================================================================
@torch.no_grad()
def sensitivity_analysis(cfg, aidx, net, device, target_agent=0, target_item=0, n_points=21):
    """Fix all inputs, vary v[target_agent, target_item] from v_min to v_max.
    Show how the chosen allocation changes."""
    A, m, K = cfg.num_agents, cfg.num_items, aidx.num_allocations

    # Fix a baseline input
    torch.manual_seed(7)
    batch = sample_types(cfg, aidx, 1)
    v_base = batch["v_true"].clone()       # [1, A, m]
    a_base = batch["alpha_true"].clone()   # [1, A]
    endow_idx = batch["endow_idx"]         # [1]

    vals = torch.linspace(cfg.v_min, cfg.v_max, n_points)

    print("\n" + "=" * 80)
    print(f"  PART 3: Sensitivity Analysis — Agent {target_agent}, Item {target_item}")
    print("=" * 80)
    print(f"  Fixed endowment index = {endow_idx[0].item()}")
    print(f"  Fixed valuations (other entries):")
    for i in range(A):
        v_str = " ".join(f"{v_base[0, i, j]:.3f}" for j in range(m))
        marker = " ← varying" if i == target_agent else ""
        print(f"    Agent {i}: [{v_str}]{marker}")
    print()

    all_allocs = aidx.all_allocations_tensor()

    header = f"  {'v_val':>6}  {'AllocIdx':>8}  "
    for i in range(A):
        header += f"{'Ag'+str(i):>12} "
    header += f" {'Welfare':>8}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    prev_alloc = None
    for vi, val in enumerate(vals):
        v_input = v_base.clone()
        v_input[0, target_agent, target_item] = val.item()

        U = types_to_allocation_utils(cfg, aidx, v_input, a_base)
        mask = compute_ir_mask(cfg, aidx, U, endow_idx)
        chosen = net.predict_argmax(v_input, a_base, endow_idx, mask=mask)
        c = chosen[0].item()

        alloc_row = all_allocs[c]
        agent_items = []
        for i in range(A):
            items_i = [str(j) for j in range(m) if alloc_row[j].item() == i]
            agent_items.append("{" + ",".join(items_i) + "}")

        welfare = U[0, :, c].sum().item()

        change_marker = "  ***" if prev_alloc is not None and c != prev_alloc else ""
        prev_alloc = c

        row = f"  {val:.3f}   {c:>8}  "
        for bundle in agent_items:
            row += f"{bundle:>12} "
        row += f"  {welfare:.4f}{change_marker}"
        print(row)


# ========================================================================
# 4. Endowment dependency analysis
# ========================================================================
@torch.no_grad()
def endowment_dependency(cfg, aidx, net, device, n_endowments=30):
    """Fix valuations, vary endowments. Show how allocation changes."""
    A, m, K = cfg.num_agents, cfg.num_items, aidx.num_allocations

    torch.manual_seed(77)
    batch = sample_types(cfg, aidx, 1)
    v_fix = batch["v_true"].expand(n_endowments, -1, -1).clone()
    a_fix = batch["alpha_true"].expand(n_endowments, -1).clone()

    # Sample different endowments
    torch.manual_seed(99)
    endow_idx = aidx.random_endowment_no_disposal(n_endowments, device)

    U = types_to_allocation_utils(cfg, aidx, v_fix, a_fix)
    mask = compute_ir_mask(cfg, aidx, U, endow_idx)
    alloc_idx = net.predict_argmax(v_fix, a_fix, endow_idx, mask=mask)

    # Oracle (same for all since valuations are fixed)
    oracle_idx = U[0].sum(dim=0).argmax().item()

    all_allocs = aidx.all_allocations_tensor()
    endow_masks = aidx.allocation_to_agent_masks(endow_idx)

    print("\n" + "=" * 80)
    print("  PART 4: Endowment Dependency (fixed valuations, varying endowment)")
    print("=" * 80)
    print(f"  Fixed valuations:")
    for i in range(A):
        v_str = " ".join(f"{v_fix[0, i, j]:.3f}" for j in range(m))
        print(f"    Agent {i}: [{v_str}]")
    print(f"  Oracle allocation index = {oracle_idx}")

    oracle_row = all_allocs[oracle_idx]
    oracle_bundles = " | ".join(
        f"Ag{i}→" + "{" + ",".join(str(j) for j in range(m) if oracle_row[j].item() == i) + "}"
        for i in range(A)
    )
    print(f"  Oracle = [{oracle_bundles}]\n")

    header = f"  {'#':>3} {'EndowIdx':>8}  "
    for i in range(A):
        header += f"{'Endow Ag'+str(i):>12} "
    header += f" {'AllocIdx':>8}  "
    for i in range(A):
        header += f"{'Alloc Ag'+str(i):>12} "
    header += f" {'=Oracle?':>8}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    match_count = 0
    for s in range(n_endowments):
        e_idx = endow_idx[s].item()
        a_idx = alloc_idx[s].item()
        is_oracle = a_idx == oracle_idx
        if is_oracle:
            match_count += 1

        endow_bundles = []
        for i in range(A):
            items_i = [str(j) for j in range(m) if endow_masks[s, i, j] > 0.5]
            endow_bundles.append("{" + ",".join(items_i) + "}")

        alloc_row = all_allocs[a_idx]
        alloc_bundles = []
        for i in range(A):
            items_i = [str(j) for j in range(m) if alloc_row[j].item() == i]
            alloc_bundles.append("{" + ",".join(items_i) + "}")

        row = f"  {s:>3} {e_idx:>8}  "
        for b in endow_bundles:
            row += f"{b:>12} "
        row += f"  {a_idx:>8}  "
        for b in alloc_bundles:
            row += f"{b:>12} "
        row += f"  {'YES' if is_oracle else 'no':>8}"
        print(row)

    print(f"\n  Oracle match rate: {match_count}/{n_endowments} ({match_count/n_endowments*100:.1f}%)")
    unique_allocs = len(set(alloc_idx.tolist()))
    print(f"  Unique output allocations: {unique_allocs}")

    if unique_allocs == 1:
        print("  → メカニズムは endowment に依存せず、常に同じアロケーションを出力 (Oracle-like)")
    elif unique_allocs <= 5:
        print("  → メカニズムは endowment にある程度依存しつつ、少数のパターンに収束")
    else:
        print("  → メカニズムは endowment ごとに異なるアソケーションを選択 (endowment-sensitive)")


# ========================================================================
# Main
# ========================================================================
def main():
    cfg, aidx, net, device = load_model()
    print(f"\nLoaded model: {cfg.num_agents} agents, {cfg.num_items} items, K={aidx.num_allocations}")
    print(f"Config: hidden={cfg.hidden}, depth={cfg.depth}, steps={cfg.steps}")

    sample_io_table(cfg, aidx, net, device, n_samples=10)
    allocation_frequency(cfg, aidx, net, device, n_samples=5000, top_k=20)
    sensitivity_analysis(cfg, aidx, net, device, target_agent=0, target_item=0, n_points=21)
    endowment_dependency(cfg, aidx, net, device, n_endowments=30)

    print("\n" + "=" * 80)
    print("  Analysis complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
