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

python3 -m examples.data_preprocess.prepare \
--mode 'text' \
--train_data_size $train_data_size \
--val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.rlvcr.enable=False \
    algorithm.use_kl_in_reward=False \
    +algorithm.non_think_reward.enable=True \
    +algorithm.non_think_reward.coef=0.5 \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=30000 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/models/llama3.1-8b_binaryCoT-sft_sci_lr2e6_bs16_epoch5_full_1211" \
    actor_rollout_ref.actor.optim.lr=2e-7 \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.2 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=31024 \
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
    env.env_name=sciworld/ScienceWorldEnv \
    env.seed=0 \
    env.max_steps=40 \
    env.rollout.n=$group_size \
    env.sciworld.generalization_level=3 \
    env.sciworld.meta_think=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rlvcr_sciworld' \
    trainer.experiment_name='grpo_llama3.1-8b_sci_binaryCoT_lr2e7_bs32_kl0.2_nonthink0.5_1217' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.resume_mode=auto \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=200 \
    trainer.val_before_train=True $@

echo '____启动gpu进程____'
cd /apdcephfs_cq10/share_1567347/share_info/ruihanyang
bash occupy_gpus.sh
