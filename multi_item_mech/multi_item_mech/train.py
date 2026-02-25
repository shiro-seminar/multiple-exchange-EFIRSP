from __future__ import annotations

import torch
import torch.nn as nn

from .config import Config
from .allocations import AllocationIndex
from .data_gen import sample_types, types_to_allocation_utils
from .model import AllocationNet
from .losses import augmented_loss
from .oracle import oracle_welfare_no_disposal


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    aidx = AllocationIndex(num_agents=cfg.num_agents, num_items=cfg.num_items)
    device = torch.device(cfg.device)

    net = AllocationNet(cfg, aidx).to(device=device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    milestones = list(getattr(cfg, "lr_milestones", []))
    gamma = float(getattr(cfg, "lr_gamma", 0.5))
    scheduler = None
    if len(milestones) > 0:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=gamma)

    # --- soft_training_only mode: 訓練中は常にsoft (softmax) で学習 ---
    soft_training_only = bool(getattr(cfg, "soft_training_only", True))
    temperature = float(getattr(cfg, "temperature", 1.0))

    for step in range(1, cfg.steps + 1):
        # Set training mode
        if soft_training_only:
            net.cfg.temperature = temperature
            net.cfg.hard_output = False
            net.cfg.st_alpha = 0.0
        else:
            net.cfg.temperature = temperature
            net.cfg.hard_output = True
            net.cfg.st_alpha = 1.0

        batch = sample_types(cfg, aidx, cfg.batch_size)
        v_true = batch["v_true"]
        a_true = batch["alpha_true"]
        endow_idx = batch["endow_idx"]

        # True utilities over allocations
        U_true = types_to_allocation_utils(cfg, aidx, v_true, a_true)

        # Move dynamic misreport_samples logic here
        if step < 20000:
            cfg.misreport_samples = 16
        elif step < 40000:
            cfg.misreport_samples = 32
        else:
            cfg.misreport_samples = 64

        # Compute augmented loss (includes welfare, IR, SP)
        loss, stats = augmented_loss(cfg, aidx, net, v_true, a_true, U_true, endow_idx)
        
        # Extract metrics from stats dict
        welfare = stats['welfare']
        ir = stats['ir']
        sp = stats['sp']

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        opt.step()
        
        # LR scheduler
        if scheduler is not None:
            scheduler.step()
            
        # Augmented Lagrangian dual update
        dual_every = int(getattr(cfg, "dual_update_every", 100))
        if step % dual_every == 0:
            ir_val = float(ir)
            sp_val = float(sp)
            # cfg.lambda_ir = max(0.0, float(cfg.lambda_ir) + float(cfg.rho) * ir_val)  # IR is now zero by mask
            cfg.lambda_sp = max(0.0, float(cfg.lambda_sp) + float(cfg.rho) * sp_val)

            ir_t = float(getattr(cfg, "ir_target", 0.02))
            sp_t = float(getattr(cfg, "sp_target", 0.02))
            if (ir_val > ir_t) or (sp_val > sp_t):
                rho_mult = float(getattr(cfg, "rho_mult", 1.02))
                rho_max = float(getattr(cfg, "rho_max", 500.0))
                cfg.rho = min(float(cfg.rho) * rho_mult, rho_max)

        if step % 200 == 0 or step == 1:
            with torch.no_grad():
                oracle = oracle_welfare_no_disposal(aidx, U_true).mean()
                gap = oracle - welfare
                mode = "SOFT" if soft_training_only else "HARD(ST)"
                tau = float(getattr(net.cfg, "temperature", 1.0))
            print(
                f"step={step:5d} mode={mode:7s} tau={tau:.3f} "
                f"loss={stats['loss']:.4f}  welfare={welfare:.4f}  "
                f"oracle={oracle.item():.4f}  gap={gap:.4f}  "
                f"IR={ir:.4f}  SP={sp:.4f}"
            )

        # ---- Diagnostic: show endowment, allocation, utilities for 1 sample ----
        if step % 1000 == 0 or step == 1:
            with torch.no_grad():
                A = cfg.num_agents
                m = cfg.num_items

                # --- Endowment (initial allocation) ---
                endow_masks = aidx.allocation_to_agent_masks(endow_idx)  # [B, A, m]
                endow_mask_0 = endow_masks[0]  # [A, m] for sample 0

                # --- Realized allocation (argmax from net) ---
                alloc_idx = net.predict_argmax(v_true, a_true, endow_idx)  # [B]
                alloc_masks = aidx.allocation_to_agent_masks(alloc_idx)  # [B, A, m]
                alloc_mask_0 = alloc_masks[0]  # [A, m] for sample 0

                # --- Per-agent utilities ---
                endow_utils_0 = U_true[0, :, endow_idx[0].long()]  # [A]
                alloc_utils_0 = U_true[0, :, alloc_idx[0].long()]  # [A]

                # --- Oracle allocation ---
                oracle_idx = U_true[0].sum(dim=0).argmax()  # best welfare alloc
                oracle_masks_0 = aidx.allocation_to_agent_masks(oracle_idx.unsqueeze(0))[0]
                oracle_utils_0 = U_true[0, :, oracle_idx.long()]

                print("=" * 70)
                print(f"  [Diagnostic] step={step}  (sample 0 / batch)")
                print(f"  Item valuations v[0]: shape=({A}, {m})")
                for i in range(A):
                    v_str = ", ".join(f"{v_true[0, i, j]:.3f}" for j in range(m))
                    print(f"    Agent {i}: [{v_str}]")

                print(f"\n  Endowment (alloc_idx={endow_idx[0].item()}):")
                for i in range(A):
                    items = [j for j in range(m) if endow_mask_0[i, j] > 0.5]
                    print(f"    Agent {i}: items={items}  utility={endow_utils_0[i]:.4f}")

                print(f"\n  Realized Allocation (alloc_idx={alloc_idx[0].item()}):")
                for i in range(A):
                    items = [j for j in range(m) if alloc_mask_0[i, j] > 0.5]
                    print(f"    Agent {i}: items={items}  utility={alloc_utils_0[i]:.4f}")

                print(f"\n  Oracle Allocation (alloc_idx={oracle_idx.item()}):")
                for i in range(A):
                    items = [j for j in range(m) if oracle_masks_0[i, j] > 0.5]
                    print(f"    Agent {i}: items={items}  utility={oracle_utils_0[i]:.4f}")

                # IR check per agent
                ir_ok = ["OK" if alloc_utils_0[i] >= endow_utils_0[i] - 1e-5 else "VIOLATED" for i in range(A)]
                print(f"\n  IR check: {', '.join(f'Agent {i}={s}' for i, s in enumerate(ir_ok))}")

                welfare_endow = endow_utils_0.sum().item()
                welfare_alloc = alloc_utils_0.sum().item()
                welfare_oracle = oracle_utils_0.sum().item()
                print(f"  Welfare: endow={welfare_endow:.4f}  alloc={welfare_alloc:.4f}  oracle={welfare_oracle:.4f}")
                print("=" * 70)

        # Checkpoint save every 5000 steps
        if step % 5000 == 0:
            ckpt_path = f"allocation_net_step{step}.pt"
            torch.save({
                "state_dict": net.state_dict(),
                "cfg": cfg.__dict__,
                "step": step,
                "optimizer": opt.state_dict(),
            }, ckpt_path)
            print(f"[Checkpoint] Saved to {ckpt_path}")

    # save
    path = "allocation_net.pt"
    torch.save({"state_dict": net.state_dict(), "cfg": cfg.__dict__}, path)
    print(f"Saved model to {path}")


if __name__ == "__main__":
    main()
