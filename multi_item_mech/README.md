# multi_item_mech (2 agents, 5 items) — learn an allocation rule with EF/IR/SP losses

This is a minimal, from-scratch scaffold for your project.

Core choice (for the first build):
- No disposal (complete allocation): every item goes to agent 1 or agent 2.
- Two agents, five items → allocations are indexed by agent-1 bundle mask in {0..31}.
- Agent 2 receives the complement bundle.
- Utilities: additive + (optional) bundle-size synergy bonus.

Training:
- We **do not** use supervised labels.
- We sample "true types" from a distribution, feed **reported** types to the network,
  compute EF/IR/SP from the outcome, and update parameters by gradient descent.

Quick start:
```bash
python -m multi_item_mech.train
```

You can tune settings in `multi_item_mech/config.py`.
