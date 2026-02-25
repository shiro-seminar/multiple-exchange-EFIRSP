import torch
from multi_item_mech.multi_item_mech.config import Config
from multi_item_mech.multi_item_mech.bundles import BundleIndex
from multi_item_mech.multi_item_mech.data_gen import sample_types, types_to_bundle_utils
from multi_item_mech.multi_item_mech.model import BundleNet
from multi_item_mech.multi_item_mech.losses import augmented_loss

print("Testing training pipeline...")
cfg = Config()
cfg.device='cpu'
cfg.misreport_samples = 8
bidx = BundleIndex(num_items=cfg.num_items)
net = BundleNet(cfg, bidx)

batch = sample_types(cfg, bidx, 4)
v_true, a_true, endow = batch['v_true'], batch['alpha_true'], batch['endow_mask1']

U_true = types_to_bundle_utils(cfg, bidx, v_true, a_true)
loss, stats = augmented_loss(cfg, bidx, net, v_true, a_true, U_true, endow)

print(f"✅ Training pipeline test PASSED!")
print(f"  Loss:    {stats['loss']:.4f}")
print(f"  Welfare: {stats['welfare']:.4f}")
print(f"  IR:      {stats['ir']:.4f}")
print(f"  SP:      {stats['sp']:.4f}")
