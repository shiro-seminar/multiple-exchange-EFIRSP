"""Quick verification that training pipeline works correctly."""
import sys
sys.path.insert(0, 'multi_item_mech')

from multi_item_mech.config import Config
from multi_item_mech.bundles import BundleIndex
from multi_item_mech.data_gen import sample_types, types_to_bundle_utils
from multi_item_mech.model import BundleNet
from multi_item_mech.losses import augmented_loss
import torch

print("Testing training pipeline...")

cfg = Config()
bidx = BundleIndex(cfg.num_items)
net = BundleNet(cfg, bidx)

batch = sample_types(cfg, bidx, 64)
v = batch['v_true']
a = batch['alpha_true']
e = batch['endow_mask1']
U = types_to_bundle_utils(cfg, bidx, v, a)

loss, stats = augmented_loss(cfg, bidx, net, v, a, U, e)

print(f"✅ Forward pass successful!")
print(f"   loss    = {loss.item():.4f}")
print(f"   welfare = {stats['welfare']:.4f}")
print(f"   IR      = {stats['ir']:.4f}")
print(f"   SP      = {stats['sp']:.4f}")

# Test backward pass
loss.backward()
print("✅ Backward pass successful!")

# Test optimizer step
opt = torch.optim.Adam(net.parameters(), lr=1e-4)
opt.step()
print("✅ Optimizer step successful!")

print("\n🎉 All training pipeline tests passed!")
