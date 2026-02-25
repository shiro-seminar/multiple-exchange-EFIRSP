from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .allocations import AllocationIndex


class AllocationNet(nn.Module):
    """
    MLP mechanism for N agents:
      (reported types + endowment) -> distribution over K=A^m allocations.

    Key feature:
      - soft output: ordinary softmax probabilities (differentiable)
      - hard output (Straight-Through):
          forward uses one-hot(MAP) (deterministic allocation)
          backward uses softmax gradient (so training works with SGD)
    """

    def __init__(self, cfg: Config, aidx: AllocationIndex):
        super().__init__()
        self.cfg = cfg
        self.aidx = aidx

        # Input:
        #   v (num_agents * num_items) + alpha (num_agents) + endowment_onehot (K)
        K = aidx.num_allocations
        in_dim = cfg.num_agents * cfg.num_items + cfg.num_agents + K

        h = cfg.hidden
        d = cfg.depth

        layers: list[nn.Module] = []
        for layer_i in range(d):
            layers.append(nn.Linear(in_dim if layer_i == 0 else h, h))
            layers.append(nn.ReLU())
            if getattr(cfg, "dropout", 0.0) and cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))

        # Output logits over allocations
        layers.append(nn.Linear(h, K))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        v_report: torch.Tensor,
        alpha_report: torch.Tensor,
        endow_idx: torch.Tensor,
        temperature: float | None = None,
        hard: bool | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns probs over allocations. Shape: [B, K].

        Args:
          endow_idx:
            Endowment allocation index. Shape: [B] (int/long).
            This is converted to one-hot(K) and appended to the network input.

          temperature:
            Softmax temperature. If None, uses cfg.temperature.

          hard:
            - False: return softmax probabilities (standard)
            - True : return Straight-Through one-hot(MAP) (forward hard, backward soft)
            If None, uses cfg.hard_output if present; otherwise defaults to False.
        """
        if temperature is None:
            temperature = getattr(self.cfg, "temperature", 1.0)

        if hard is None:
            hard = bool(getattr(self.cfg, "hard_output", False))

        B = v_report.shape[0]
        K = self.aidx.num_allocations

        # One-hot encode endowment: [B] -> [B, K]
        endow_idx = endow_idx.long()
        endow_oh = F.one_hot(endow_idx, num_classes=K).to(dtype=v_report.dtype)

        # Flatten inputs and pass through MLP
        x = torch.cat(
            [
                v_report.reshape(B, -1),
                alpha_report.reshape(B, -1),
                endow_oh.reshape(B, -1),
            ],
            dim=1,
        )
        logits = self.net(x)

        if mask is not None:
            # mask: [B, K] where 1.0 = valid, 0.0 = invalid
            # invalid_score = -1e9
            logits = logits + (1.0 - mask) * -1e9

        # Soft distribution (differentiable)
        tau = max(float(temperature), 1e-6)
        y_soft = F.softmax(logits / tau, dim=-1)

        # st_alpha: 0 -> 1 (gradually harden)
        st_alpha = float(getattr(self.cfg, "st_alpha", 1.0 if hard else 0.0))
        st_alpha = max(0.0, min(1.0, st_alpha))

        if (not hard) or st_alpha <= 0.0:
            return y_soft

        # Hard MAP (one-hot) from y_soft
        idx = torch.argmax(y_soft, dim=-1)
        y_hard = torch.zeros_like(y_soft).scatter_(1, idx.view(-1, 1), 1.0)

        # Straight-Through estimator
        y_st = y_hard - y_soft.detach() + y_soft

        # Mix soft and ST-hard
        y = (1.0 - st_alpha) * y_soft + st_alpha * y_st
        return y

    @torch.no_grad()
    def predict_argmax(
        self,
        v_report: torch.Tensor,
        alpha_report: torch.Tensor,
        endow_idx: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Deterministic allocation: returns chosen allocation index. Shape: [B].
        (Pure argmax on logits; no temperature.)
        """
        B = v_report.shape[0]
        K = self.aidx.num_allocations

        endow_idx = endow_idx.long()
        endow_oh = F.one_hot(endow_idx, num_classes=K).to(dtype=v_report.dtype)

        x = torch.cat(
            [
                v_report.reshape(B, -1),
                alpha_report.reshape(B, -1),
                endow_oh.reshape(B, -1),
            ],
            dim=1,
        )
        logits = self.net(x)
        if mask is not None:
            logits = logits + (1.0 - mask) * -1e9
        return torch.argmax(logits, dim=-1)

    @torch.no_grad()
    def predict_onehot(
        self,
        v_report: torch.Tensor,
        alpha_report: torch.Tensor,
        endow_idx: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Deterministic allocation as one-hot distribution. Shape: [B, K].
        """
        idx = self.predict_argmax(v_report, alpha_report, endow_idx, mask=mask)
        onehot = torch.zeros(
            (idx.shape[0], self.aidx.num_allocations),
            device=idx.device,
            dtype=torch.float32,
        )
        onehot.scatter_(1, idx.view(-1, 1), 1.0)
        return onehot


# Backward-compatible alias
BundleNet = AllocationNet
