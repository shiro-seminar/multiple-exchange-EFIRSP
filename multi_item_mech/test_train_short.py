"""Short training test for 3-agent, 5-item mechanism"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn

from multi_item_mech.config import Config
from multi_item_mech.allocations import AllocationIndex
from multi_item_mech.data_gen import sample_types, types_to_allocation_utils
from multi_item_mech.model import AllocationNet
from multi_item_mech.losses import augmented_loss
from multi_item_mech.oracle import oracle_welfare_no_disposal

def main():
    cfg = Config()
    cfg.steps = 500  # Short test
    torch.manual_seed(cfg.seed)

    aidx = AllocationIndex(num_agents=cfg.num_agents, num_items=cfg.num_items)
    device = torch.device(cfg.device)

    print(f"=== Training Test: {cfg.num_agents} agents, {cfg.num_items} items, {aidx.num_allocations} allocations ===")

    net = AllocationNet(cfg, aidx).to(device=device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    net.cfg.temperature = cfg.temperature
    net.cfg.hard_output = False
    net.cfg.st_alpha = 0.0

    for step in range(1, cfg.steps + 1):
        batch = sample_types(cfg, aidx, cfg.batch_size)
        v_true = batch["v_true"]
        a_true = batch["alpha_true"]
        endow_idx = batch["endow_idx"]

        U_true = types_to_allocation_utils(cfg, aidx, v_true, a_true)

        cfg.misreport_samples = 16  # Keep low for speed

        loss, stats = augmented_loss(cfg, aidx, net, v_true, a_true, U_true, endow_idx)
        
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        opt.step()

        if step % 100 == 0 or step == 1:
            with torch.no_grad():
                oracle = oracle_welfare_no_disposal(aidx, U_true).mean()
            print(
                f"step={step:4d} loss={stats['loss']:.4f} welfare={stats['welfare']:.4f} "
                f"oracle={oracle.item():.4f} IR={stats['ir']:.4f} SP={stats['sp']:.4f}"
            )

    print("\n=== Training test completed successfully ===")

if __name__ == "__main__":
    main()
