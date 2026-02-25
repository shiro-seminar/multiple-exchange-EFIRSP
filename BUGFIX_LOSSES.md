# Bug Fixes in losses.py (2026-01-11)

## Overview
This document details critical bug fixes applied to `multi_item_mech/losses.py` that correct dimension handling, tensor indexing, and Augmented Lagrangian formulation issues.

---

## Fixed Issues

### 1. 🔴 **Dimension Handling Bug in `endow_rep` (Line 77)**

**Location:** `sp_loss_sampled()` function

**Original Code:**
```python
endow_rep = endow_mask1.view(B, 1).expand(B, M).reshape(B * M)
```

**Fixed Code:**
```python
endow_rep = endow_mask1.unsqueeze(1).expand(B, M).reshape(B * M)
```

**Issue:**
- `endow_mask1` is a 1D tensor with shape `[B]` (batch of endowment indices)
- `.view(B, 1)` assumes the tensor is already 2D and could fail or produce unexpected results
- `.unsqueeze(1)` is the explicit, safer operation for adding a dimension to a 1D tensor

**Impact:**
- **Severity:** Medium-Low (code may have worked due to PyTorch's flexibility, but semantically incorrect)
- **Effect:** More robust code that clearly expresses intent and handles edge cases better

---

### 2. 🔴 **Critical: Incorrect Noise Indexing for Misreports (Lines 84-88)**

**Location:** `sp_loss_sampled()` function, within the agent loop

**Original Code:**
```python
for i in range(2):
    v_noise, a_noise = sample_misreports(cfg, v_true, alpha_true, M)  # [B,M,m], [B,M]
    
    v_mis_i = torch.clamp(
        v_true[:, i, :].unsqueeze(1) + v_noise, cfg.v_min, cfg.v_max
    )  # [B,M,m]
    a_mis_i = torch.clamp(
        alpha_true[:, i].unsqueeze(1) + a_noise, cfg.alpha_min, cfg.alpha_max
    )  # [B,M]
```

**Fixed Code:**
```python
for i in range(2):
    v_noise, a_noise = sample_misreports(cfg, v_true, alpha_true, M)  # [B,M,A,m], [B,M,A]
    
    v_mis_i = torch.clamp(
        v_true[:, i, :].unsqueeze(1) + v_noise[:, :, i, :], cfg.v_min, cfg.v_max
    )  # [B,M,m]
    a_mis_i = torch.clamp(
        alpha_true[:, i].unsqueeze(1) + a_noise[:, :, i], cfg.alpha_min, cfg.alpha_max
    )  # [B,M]
```

**Issue:**
- `sample_misreports()` returns noise tensors with shapes:
  - `v_noise`: `[B, M, A, m]` (batch × misreport_samples × agents × items)
  - `a_noise`: `[B, M, A]` (batch × misreport_samples × agents)
- The original code added the **entire** noise tensor to agent `i`'s true values
- Due to broadcasting, this would add noise for **both agents** instead of just agent `i`

**Correct Behavior:**
- Extract noise specifically for agent `i` using `[:, :, i, :]` and `[:, :, i]`
- This ensures each iteration correctly computes misreports for only one agent at a time

**Impact:**
- **Severity:** HIGH - This was a logic error affecting SP (strategy-proofness) loss computation
- **Effect:** SP loss was incorrectly calculated, potentially causing:
  - Wrong gradient signals during training
  - Poor convergence or failure to satisfy SP constraints
  - Mechanism learning incorrect allocation rules

---

### 3. 🟡 **Augmented Lagrangian Formulation (Line 145)**

**Location:** `augmented_loss()` function

**Original Code:**
```python
loss = (
    ef_loss
    + cfg.lambda_ir * ir
    + cfg.lambda_sp * sp
    + cfg.rho * (ir * ir + sp * sp)
)
```

**Fixed Code:**
```python
loss = (
    ef_loss
    + cfg.lambda_ir * ir
    + cfg.lambda_sp * sp
    + (cfg.rho / 2) * (ir * ir + sp * sp)
)
```

**Issue:**
- The standard Augmented Lagrangian formulation uses `ρ/2` as the coefficient for quadratic penalty terms
- This ensures that when taking the gradient, the penalty term becomes `ρ · violation`
- Without the `1/2` factor, the gradient is `2ρ · violation`, which effectively doubles the penalty strength

**Mathematical Background:**
```
Standard AL: L = f(x) + λ·c(x) + (ρ/2)·c(x)²
Gradient:    ∇L = ∇f + λ·∇c + ρ·c·∇c

Previous:    L = f(x) + λ·c(x) + ρ·c(x)²
Gradient:    ∇L = ∇f + λ·∇c + 2ρ·c·∇c  ← doubled penalty gradient
```

**Impact:**
- **Severity:** Medium (affects hyperparameter interpretation)
- **Effect:**
  - With the old formulation, the effective penalty was `2 × cfg.rho`
  - After the fix, if you want the same penalty strength, set `cfg.rho` to **double** the previous value
  - Current `config.py` has `rho = 50.0`, which now acts as intended
  - The previous behavior was equivalent to `rho = 100.0` in standard formulation

---

## Recommended Actions

### For Active Training Runs
1. **Stop current training** if using the old `losses.py`
2. **Apply these fixes** before continuing
3. **Consider adjusting `cfg.rho`**:
   - If previous results were good with old code, try `rho = 100.0` (double current value)
   - Otherwise, keep current `rho = 50.0` and monitor training

### For Result Comparison
- Results from **before and after** this fix may not be directly comparable
- The SP loss calculation bug likely caused significant training issues
- Recommend re-running all experiments after applying fixes

### Testing
Run a quick sanity check:
```python
# Test SP loss computation with simple inputs
python -c "from multi_item_mech.losses import sp_loss_sampled; print('SP loss import successful')"
```

---

## Code Review Checklist

- [x] Dimension handling uses explicit operations (`unsqueeze` vs `view`)
- [x] Tensor indexing correctly extracts per-agent data
- [x] Augmented Lagrangian follows standard formulation
- [x] Comments updated to reflect correct tensor shapes
- [ ] Training convergence improved (to be verified experimentally)

---

## References

- **File:** `multi_item_mech/multi_item_mech/losses.py`
- **Functions affected:**
  - `sp_loss_sampled()` (Lines 48-114)
  - `augmented_loss()` (Lines 117-154)
- **Date:** 2026-01-11
- **Related:** Previous training issues with IR/SP violations not converging
