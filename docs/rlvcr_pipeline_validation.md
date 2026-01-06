# RLVCR Pipeline Validation Checklist

## ✅ Pipeline Components Check

### 1. **Rollout Loop (`rollout_loop.py`)**
- [x] `rlvcr_multi_turn_loop` correctly calls `_generate_alternative_thinking_levels`
- [x] Only processes samples with `episode_reward > 0`
- [x] Expands successful samples from 1 → 4 variants
- [x] Stores `thinking_entropy`, `thinking_cost`, `thinking_group_id`, `is_original` in `non_tensor_batch`
- [x] Uses `dtype=object` for non-tensor data

### 2. **Advantage Computation (`ray_trainer.py`)**
- [x] Correctly identifies RLVCR mode via `config.algorithm.adv_estimator`
- [x] Extracts `thinking_group_id` and `is_original` flags
- [x] Computes action advantages only on original samples
- [x] Maps action advantages to expanded samples
- [x] Computes thinking advantages within groups
- [x] Uses reward shaping: `R = entropy + α * (0.5 - cost/cost_max)`
- [x] Final advantage: `action_adv + β * thinking_adv`

### 3. **Core RLVCR (`core_rlvcr.py`)**
- [x] `compute_rlvcr_outcome_advantage` handles dual advantages
- [x] Proper expansion to token-level advantages
- [x] Correct handling of attention masks

## ⚠️ Configuration Issues to Fix

### Current Issues in `run_alfworld.sh`:

1. **Parameter Name Mismatch**
   ```bash
   # Current (incorrect)
   algorithm.rlvcr.step_advantage_w
   algorithm.rlvcr.thinking_diversity_w
   
   # Should be (correct)
   algorithm.rlvcr.thinking_weight      # β parameter
   algorithm.rlvcr.thinking_cost_alpha  # α parameter
   algorithm.rlvcr.cost_max            # Max tokens for normalization
   ```

2. **Batch Size Configuration**
   ```bash
   # Current
   actor_rollout_ref.actor.ppo_mini_batch_size=8
   actor_rollout_ref.actor.ppo_micro_batch_size=2
   
   # Recommended (to avoid adjust_batch issues)
   actor_rollout_ref.actor.ppo_mini_batch_size=32
   actor_rollout_ref.actor.ppo_micro_batch_size=8
   ```

3. **Response Length**
   ```bash
   # Current
   actor_rollout_ref.rollout.response_length=512
   
   # Recommended (for thinking sequences)
   actor_rollout_ref.rollout.response_length=1024
   ```

4. **Logger Configuration**
   ```bash
   # Current
   trainer.logger=null
   
   # Recommended
   trainer.logger=[console,wandb]
   ```

## 📊 Data Flow Validation

### Input Stage
```python
# Original batch: N samples
batch_size = 8
episode_rewards = [1.0, 0, 2.0, 0, 1.5, 0, 1.0, 0]
# 4 successful, 4 failed
```

### Expansion Stage
```python
# After RLVCR expansion
expanded_batch_size = 20  # 4*4 + 4*1
thinking_group_ids = [0,0,0,0, -1, 2,2,2,2, -1, 4,4,4,4, -1, 6,6,6,6, -1]
is_original = [T,F,F,F, T, T,F,F,F, T, T,F,F,F, T, T,F,F,F, T]
```

### Advantage Stage
```python
# Action advantages (computed on original 8)
action_advs = GRPO([1.0, 0, 2.0, 0, 1.5, 0, 1.0, 0])
# Map to 20 samples

# Thinking advantages (computed within groups)
for group in [0, 2, 4, 6]:
    R_thinking = entropy + α * (0.5 - cost/250)
    thinking_adv = R - mean(R_group)

# Final
final_adv = action_adv + β * thinking_adv
```

## 🔧 Required Code Updates

### 1. Update `ray_trainer.py` to read config properly:
```python
# Add config reading for RLVCR parameters
alpha = self.config.algorithm.rlvcr.get('thinking_cost_alpha', 1.0)
beta = self.config.algorithm.rlvcr.get('thinking_weight', 0.5)
cost_max = self.config.algorithm.rlvcr.get('cost_max', 250)
```

### 2. Ensure `adjust_batch` doesn't delete samples:
```python
# In utils.py
if size_divisor > batch_size:
    # Don't delete all samples
    return data
```

## ✅ Testing Checklist

- [ ] Run with small batch to verify expansion
- [ ] Check wandb logs for advantage distribution
- [ ] Verify thinking costs are reasonable (0-250 range)
- [ ] Confirm entropy values are in [0, 1]
- [ ] Check that failed trajectories aren't expanded
- [ ] Verify action advantages are properly mapped

## 📈 Expected Metrics

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| Thinking Entropy | [0, 1] | Min probability of action tokens |
| Thinking Cost | [0, 250] | Token count of thinking text |
| R_length | [-0.5, 0.5] | Normalized cost reward |
| Action Advantage | [-2, 2] | GRPO normalized |
| Thinking Advantage | [-1, 1] | Group normalized |
| Final Advantage | [-3, 3] | Combined advantages |

## 🚀 Launch Command

```bash
# Use the updated script
bash examples/rlvcr_trainer/run_alfworld_updated.sh
```
