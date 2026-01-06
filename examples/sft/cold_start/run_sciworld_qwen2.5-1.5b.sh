

ENV=sciworld

python3 -m examples.data_preprocess.cold_start_data \
    --local_dir=$HOME/data/$ENV \
    --data_source=YOUR_DATA_SOURCE

torchrun  --standalone --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/$ENV/train.parquet \
    data.val_files=$HOME/data/$ENV/train.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.max_length=4192 \
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    optim.lr=1e-5 \
    data.micro_batch_size_per_gpu=16 \
    model.partial_pretrain=Qwen/Qwen2.5-1.5B-Instruct \
    trainer.default_hdfs_dir=null \
    trainer.project_name=RLVMR \
    trainer.experiment_name=qwen1.5b_cold-start \
    trainer.total_epochs=5 \
    trainer.default_local_dir=./checkpoints/cold_start/sciworld/qwen1.5b \
    trainer.logger=['console','wandb'] \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=true