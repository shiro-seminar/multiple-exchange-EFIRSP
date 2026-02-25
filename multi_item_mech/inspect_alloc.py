import torch
from collections import Counter

from multi_item_mech.config import Config
from multi_item_mech.bundles import BundleIndex
from multi_item_mech.model import BundleNet
from multi_item_mech.data_gen import sample_types

def main():
    cfg = Config()
    device = torch.device(cfg.device)
    bidx = BundleIndex(num_items=cfg.num_items)

    ckpt = torch.load("bundle_net.pt", map_location=device)
    net = BundleNet(cfg, bidx).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    # compareと同様にテストタイプを生成（再現性のためseed固定）
    torch.manual_seed(0)
    N = 10000
    batch = sample_types(cfg, bidx, N)
    v = batch["v_true"].to(device)
    a = batch["alpha_true"].to(device)

    # 確定配分：agent1が取るbundle index（argmax）
    with torch.no_grad():
        probs = net(v, a)              # [N, K]
        k1 = probs.argmax(dim=1)       # [N]

    k1_list = k1.cpu().tolist()
    cnt = Counter(k1_list)

    masks = bidx.masks_tensor().cpu().int()        # [K, m] (item0..)
    comp  = bidx.complement_index().cpu()          # [K]

    print(f"N={N}, unique bundles={len(cnt)}")
    print("Top 15 deterministic allocations (agent1 bundle index -> mask[item0..])")

    for idx, c in cnt.most_common(15):
        p = c / N
        a1 = masks[idx].tolist()
        idx2 = int(comp[idx].item()) if hasattr(comp[idx], "item") else int(comp[idx])
        a2 = masks[idx2].tolist()
        bits = format(idx, f"0{cfg.num_items}b")   # 見た目は item(m-1)..item0
        print(f"  k={idx:2d}  freq={c:4d}  p={p:6.4f}  a1={a1}  a2={a2}  bits(item{cfg.num_items-1}..0)={bits}")

    print("\nFirst 10 samples (agent1 allocation mask):")
    for i in range(10):
        idx = k1_list[i]
        print(f"  sample {i:2d}: k={idx:2d}  a1={masks[idx].tolist()}")

if __name__ == "__main__":
    main()
