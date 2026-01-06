# RLVCR Wandb Logging Guide

## 📊 Overview

The RLVCR training pipeline now includes comprehensive logging to Weights & Biases (wandb) for monitoring thinking metrics, case studies, and detailed analysis.

## 🔧 Setup

### 1. Enable Logging in Training Script

```bash
# In run_alfworld.sh, ensure these are set:
trainer.logger=[console,wandb]          # Enable wandb logging
rlvcr_save_cases=True                   # Save thinking cases
rlvcr_max_cases_per_epoch=20           # Cases per epoch
```

### 2. Run Training

```bash
bash examples/rlvcr_trainer/run_alfworld.sh
```

## 📈 Logged Metrics

### Core Thinking Metrics

| Metric | Description | Range | Good Values |
|--------|-------------|-------|-------------|
| `rlvcr/entropy_mean` | Average action confidence (min_prob) | [0, 1] | 0.3-0.8 |
| `rlvcr/entropy_std` | Confidence variance | [0, 1] | 0.1-0.3 |
| `rlvcr/cost_mean` | Average thinking tokens | [0, 250] | 30-100 |
| `rlvcr/cost_std` | Cost variance | [0, 250] | 10-50 |

### Performance Metrics

| Metric | Description |
|--------|-------------|
| `rlvcr/success_rate` | Episode success rate |
| `rlvcr/success_entropy_mean` | Confidence on successful episodes |
| `rlvcr/success_cost_mean` | Cost on successful episodes |

### Thinking Level Distribution

| Metric | Description |
|--------|-------------|
| `rlvcr/level_1_ratio` | Proportion using Level 1 (no thinking) |
| `rlvcr/level_2_ratio` | Proportion using Level 2 (basic) |
| `rlvcr/level_3_ratio` | Proportion using Level 3 (detailed) |
| `rlvcr/level_4_ratio` | Proportion using Level 4 (strategic) |

### Advantage Components

| Metric | Description |
|--------|-------------|
| `rlvcr/action_adv_mean` | Average action advantage |
| `rlvcr/thinking_adv_mean` | Average thinking advantage |
| `rlvcr/final_adv_mean` | Combined final advantage |

## 📊 Visualizations

### 1. Distribution Plots
- **Entropy Distribution**: Shows action confidence spread
- **Cost Distribution**: Shows thinking length distribution  
- **Cost vs Confidence**: Scatter plot showing relationship
- **Cost vs Reward**: Performance vs efficiency

### 2. Level Comparison
- **Bar charts** comparing metrics across thinking levels
- **Evolution plots** showing how level usage changes over time

## 💾 Case Studies

### Saved Cases Include:
- **High Reward Cases**: Most successful episodes
- **High Variance Cases**: Diverse thinking patterns
- **Efficient Cases**: High reward/cost ratio

### Case Data:
```python
# Each case contains:
{
    "trajectory_id": "abc123",
    "original_level": 3,
    "action": "go to cabinet 1",
    "prompt": "Current state...",
    "thinking_texts": [...],    # 4 thinking variants
    "entropies": [...],         # 4 confidence scores  
    "costs": [...],             # 4 token costs
    "episode_reward": 1.0,
    "step_idx": 15,
    "epoch": 2
}
```

## 🔍 Monitoring in Wandb

### Key Dashboards to Create:

1. **Training Overview**
   - Success rate trend
   - Average confidence and cost
   - Level distribution pie chart

2. **Efficiency Analysis**  
   - Cost vs success correlation
   - Thinking level effectiveness
   - Token efficiency trends

3. **Case Studies**
   - Table of interesting cases
   - Detailed case breakdowns
   - Pattern analysis

### Example Wandb Queries:

```python
# Filter high-performing cases
df[df['rlvcr/success_rate'] > 0.8]

# Find optimal cost range
optimal_cost = df.groupby(pd.cut(df['rlvcr/cost_mean'], 10))['rlvcr/success_rate'].mean()

# Level effectiveness
level_success = df[['rlvcr/level_1_ratio', 'rlvcr/success_rate']].corr()
```

## 📋 Analysis Checklist

### During Training:
- [ ] Success rate is improving
- [ ] Confidence is stabilizing (not decreasing)
- [ ] Cost distribution is reasonable (not too high/low)
- [ ] Level distribution adapts to task complexity

### Post-Training Analysis:
- [ ] Run analysis script: `python analyze_rlvcr_logs.py --run-path user/run_id`
- [ ] Check cost-success correlation (should be near 0 or negative)
- [ ] Verify thinking level progression makes sense
- [ ] Review saved cases for quality

## 🛠️ Analysis Tools

### 1. Automated Analysis Script
```bash
python examples/rlvcr_trainer/analyze_rlvcr_logs.py \
    --run-path your_username/run_id \
    --project rlvcr_alfworld \
    --output analysis_plots.png
```

### 2. Manual Wandb Analysis
```python
import wandb
api = wandb.Api()
run = api.run("project/run_id")
history = run.history()

# Your custom analysis here
```

## 🎯 Interpretation Guidelines

### Good Signs:
✅ **Decreasing cost over time** - Model learns efficiency
✅ **Stable/increasing confidence** - Model becomes more certain
✅ **Adaptive level usage** - Uses simple thinking for easy tasks
✅ **Negative cost-success correlation** - Efficiency improves performance

### Warning Signs:
⚠️ **Constantly increasing cost** - May need higher cost penalty (α)
⚠️ **Decreasing confidence** - May indicate training instability  
⚠️ **All Level 1 or all Level 4** - Poor level diversity
⚠️ **Positive cost-success correlation** - Inefficient thinking patterns

## 🔧 Tuning Based on Logs

### If Cost Too High (>150 tokens average):
```bash
# Increase cost penalty
RLVCR_THINKING_COST_ALPHA=1.5  # from 1.0
```

### If Confidence Too Low (<0.3):
```bash
# Reduce generation temperature
actor_rollout_ref.rollout.temperature=0.7  # from 0.9
```

### If Poor Level Diversity:
```bash
# Adjust thinking weight
RLVCR_THINKING_WEIGHT=0.3  # from 0.5 (reduce thinking influence)
```

## 📝 Custom Metrics

You can add custom metrics by modifying `rlvcr_logger.py`:

```python
# Add to log_thinking_metrics
custom_metrics = {
    "rlvcr/custom_metric": your_computation,
}
metrics.update(custom_metrics)
```

## 🚀 Next Steps

1. **Start Training** with logging enabled
2. **Monitor** key metrics in wandb dashboard  
3. **Analyze** patterns using provided scripts
4. **Tune** hyperparameters based on insights
5. **Compare** different runs and configurations
