#!/bin/bash
set -x
echo '——————————kill gpu process——————————'
pid=$(ps -ef | grep 'gpu' | grep 'python' | grep -v grep | awk '{print $2}')
echo $pid
kill -9 $pid

ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

# 基础配置 - 保持原有设置
train_data_size=16                    # 保持原来的8
val_data_size=200
group_size=4                         # 保持原来的4

# RLVCR核心配置参数
RLVCR_THINKING_WEIGHT=0           # Beta: Weight for thinking advantages in final advantage
RLVCR_THINKING_COST_ALPHA=0        # Alpha: Weight for thinking cost penalty in reward
RLVCR_COST_MAX=250                   # 保持原来的250
RLVCR_MODE="mean_norm"               # Normalization mode

# RLVCR分批处理优化配置 (性能优化版本)
RLVCR_GENERATION_CHUNK_SIZE=256       # 8→16，减少思维生成batch数量
RLVCR_CONFIDENCE_CHUNK_SIZE=2048       # 32→64，减少置信度计算调用次数

# RLVCR日志配置
RLVCR_SAVE_CASES=True                # Whether to save thinking cases for analysis
RLVCR_MAX_CASES_PER_EPOCH=20         # Maximum cases to save per epoch

python3 -m examples.data_preprocess.prepare \
--mode 'text' \
--train_data_size $train_data_size \
--val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=rlvcr \
    algorithm.rlvcr.enable=True \
    algorithm.rlvcr.thinking_weight=${RLVCR_THINKING_WEIGHT} \
    algorithm.rlvcr.thinking_cost_alpha=${RLVCR_THINKING_COST_ALPHA} \
    algorithm.rlvcr.cost_max=${RLVCR_COST_MAX} \
    algorithm.rlvcr.mode=${RLVCR_MODE} \
    +algorithm.rlvcr.generation_chunk_size=${RLVCR_GENERATION_CHUNK_SIZE} \
    +algorithm.rlvcr.confidence_chunk_size=${RLVCR_CONFIDENCE_CHUNK_SIZE} \
    rlvcr_save_cases=${RLVCR_SAVE_CASES} \
    rlvcr_max_cases_per_epoch=${RLVCR_MAX_CASES_PER_EPOCH} \
    algorithm.use_kl_in_reward=False \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=18000 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="/apdcephfs_cq11/share_1567347/share_info/llm_models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28" \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.1 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    +actor_rollout_ref.actor.kl_loss_coef_annealing=False \
    +actor_rollout_ref.actor.kl_loss_coef_init=0.05 \
    +actor_rollout_ref.actor.kl_loss_coef_final=0.15 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=20000 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=30 \
    env.rollout.n=$group_size \
    env.alfworld.generalization_level=3 \
    +env.alfworld.action_only=False \
    env.alfworld.meta_think=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rlvcr_alfworld' \
    trainer.experiment_name='rlvcr_qwen2.5-7b_alf_wo_cold_start_lr5e7_bs32_kl0.1_1113' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.resume_mode=auto \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=200 \
    trainer.val_before_train=True $@

echo '____启动gpu进程____'
cd /apdcephfs_cq10/share_1567347/share_info/ruihanyang
bash occupy_gpus.sh
