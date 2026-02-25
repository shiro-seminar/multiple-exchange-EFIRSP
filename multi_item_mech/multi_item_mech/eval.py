from __future__ import annotations
import torch
from .config import Config
from .bundles import BundleIndex
from .data_gen import sample_types, types_to_bundle_utils
from .model import BundleNet
from .losses import expected_utilities_from_probs
from .oracle import oracle_welfare_no_disposal

@torch.no_grad()
def main():
    cfg = Config()
    bidx = BundleIndex(num_items=cfg.num_items)
    device = torch.device(cfg.device)

    ckpt = torch.load("bundle_net.pt", map_location=device)
    net = BundleNet(cfg, bidx).to(device=device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    batch = sample_types(cfg, bidx, 2000)
    v_true = batch["v_true"]
    a_true = batch["alpha_true"]
    U_true = types_to_bundle_utils(cfg, bidx, v_true, a_true)

    endow_mask1 = batch["endow_mask1"]

    probs = net(v_true, a_true, endow_mask1)
    EU = expected_utilities_from_probs(bidx, probs, U_true)
    welfare = EU.sum(dim=1).mean()
    oracle = oracle_welfare_no_disposal(bidx, U_true).mean()

    chosen = net.predict_argmax(v_true, a_true, endow_mask1)  # deterministic bundle indices for agent1
    print(f"Mean welfare (lottery): {welfare.item():.4f}")
    print(f"Oracle welfare:        {oracle.item():.4f}")
    print(f"Gap:                  {(oracle-welfare).item():.4f}")
    print(f"Example argmax bundles (first 10): {chosen[:10].tolist()}")
    print(f"Example EU (first 5):\n{EU[:5].cpu().numpy()}")

if __name__ == "__main__":
    main()
