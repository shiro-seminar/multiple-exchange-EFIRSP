import sys
sys.path.insert(0, '.')

from multi_item_mech.allocations import AllocationIndex
from multi_item_mech.data_gen import sample_types, types_to_allocation_utils
from multi_item_mech.config import Config
from multi_item_mech.model import AllocationNet
from multi_item_mech.losses import augmented_loss

cfg = Config()
aidx = AllocationIndex(cfg.num_agents, cfg.num_items)
print(f'Config: {cfg.num_agents} agents, {cfg.num_items} items, {aidx.num_allocations} allocations')

batch = sample_types(cfg, aidx, 4)
print(f'v_true shape: {batch["v_true"].shape}')

U = types_to_allocation_utils(cfg, aidx, batch["v_true"], batch["alpha_true"])
print(f'U_true shape: {U.shape}')

net = AllocationNet(cfg, aidx)
probs = net(batch["v_true"], batch["alpha_true"], batch["endow_idx"])
print(f'probs shape: {probs.shape}')

loss, stats = augmented_loss(cfg, aidx, net, batch["v_true"], batch["alpha_true"], U, batch["endow_idx"])
print(f'Loss: {loss.item():.4f}, welfare: {stats["welfare"]:.4f}')

print('All imports and basic operations OK!')
