from dataclasses import dataclass
from typing import Tuple
@dataclass
class Config:
    # Problem
    num_agents: int = 3
    num_items: int = 5
    # Utility model
    v_min: float = 0.0
    v_max: float = 1.0
    alpha_min: float = 0.0      # Pure additive: no synergy
    alpha_max: float = 0.0      # Pure additive: no synergy
    # Endowment / outside option
    random_endowment: bool = True  # endowment is a random bundle for agent 1; agent 2 gets complement
    # Model
    hidden: int = 128
    depth: int = 3
    dropout: float = 0.0

    # Training
    batch_size: int = 512
    steps: int = 50000
    lr: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 0
    
    # ---- Training mode ----
    # soft_training_only: 訓練中は最後までsoftmax確率で訓練し、
    # 推論時のみargmaxで確定化する
    soft_training_only: bool = True
    temperature: float = 1.0    # softmaxの温度パラメータ

    # ---- LR scheduler ----
    lr_milestones: Tuple[int, int] = (35000, 45000)
    lr_gamma: float = 0.5
   
    # Loss weights (Augmented Lagrangian style) 
    lambda_ir: float = 0.0
    lambda_sp: float = 0.0
    rho: float = 5.0           # Lower initial penalty

    # ---- AL update knobs ----
    dual_update_every: int = 200    # 更新頻度を下げる
    rho_mult: float = 1.01         # 非常にゆっくり上げる
    rho_max: float = 100.0          # Lower max
    ir_target: float = 0.10         # More relaxed IR target
    sp_target: float = 0.001
    
    # SP approximation
    misreport_samples: int = 64

    misreport_noise_v: float = 0.35
    misreport_noise_alpha: float = 0.35

    # Device
    device: str = "cpu"
